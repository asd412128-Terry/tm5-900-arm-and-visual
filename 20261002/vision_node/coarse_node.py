"""
============================================================================
 車載相機粗定位節點：偵測番茄 → 轉成 base 座標 → 終端機列出
============================================================================
 目前階段做到「偵測到的番茄轉換到 base 座標」+「終端機互動選目標」+「把選中的目標
 座標發布給手臂」：訂閱車載相機影像/深度/內參、廣播一次外參 TF、每幀跑
 TomatoDetector，把結果疊在畫面上並節流列印到終端機；沒有鎖定目標時，背景執行緒跑
 跟 fine_node.py 共用的 TargetSelector.prompt_choose_id() 互動選取（見
 _interactive_select_thread），選定後鎖定那顆番茄的內部 key，持續把它的即時座標發布到
 /coarse_target_point 給 arm_task_node.py（見 _publish_target）。
 ★ 2026-09-29：只在 /robot_status=COARSE（手臂確定回到 Home、等目標）時才跑偵測和互動
 選取；選定後持續發座標，直到手臂收下、改發 BUSY 為止（交接），之後整個暫停，畫面
 只顯示原始影像。下次再收到 COARSE 時清掉舊的追蹤結果重新開始（車子可能已經動過）。
 還沒做：算精定位法蘭姿態（那段在 arm_task_node.py，不在這裡）。

 實際運算全部委派給 coarse_detector / coordinates，這支只負責 ROS 訂閱/callback 串接，
 跟 fine_node.py 同一個分工。啟動：python3 -m vision_node.coarse_main
============================================================================
"""

import math
import sys
import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from .coarse_detector import TomatoDetector
from .coarse_tracker import CoarseTomatoTracker
from .config import (BASE_FRAME, COARSE_MODEL_PATH, COARSE_YOLO_CONF, COARSE_YOLO_IMGSZ,
                      COARSE_PRINT_CHANGE_M, DISPLAY_SCALE,
                      SCAN_PRINT_INTERVAL,
                      VEHICLE_CAMERA_EXTRINSIC_IS_PLACEHOLDER, VEHICLE_CAMERA_EXTRINSIC_ROTATION_QUAT,
                      VEHICLE_CAMERA_EXTRINSIC_TRANSLATION, VEHICLE_CAMERA_INFO_TOPIC,
                      VEHICLE_CAMERA_OPTICAL_FRAME, VEHICLE_CAMERA_PARENT_FRAME,
                      VEHICLE_COLOR_TOPIC, VEHICLE_DEPTH_TOPIC, VISION_MODE,
                      YOLO_IOU)
from .coordinates import CoordinateEstimator
from .fine_target_selector import TargetSelector


class CoarseNode(Node):
    """初始化偵測器、TF、ROS 訂閱與內部狀態。"""
    def __init__(self):
        super().__init__('coarse_node')
        self.bridge = CvBridge()

        self.get_logger().info(
            f'執行模式: {VISION_MODE}，模型: {COARSE_MODEL_PATH}，'
            f'imgsz: {COARSE_YOLO_IMGSZ}，信心門檻: {COARSE_YOLO_CONF}')
        self.get_logger().info('正在載入 YOLO 模型（車載相機粗定位，只用番茄類別）...')
        self.detector = TomatoDetector(COARSE_MODEL_PATH, coordinate_estimator=CoordinateEstimator())
        self.tracker = CoarseTomatoTracker()   # 跨幀配對 + 座標取中位數 + 固定 ID
        self.selector = TargetSelector()       # 跟 fine_node.py 共用的終端機互動選取機制
        self._locked_key = None   # 使用者選定的番茄的內部 key（跨幀不變），None = 還沒選
        self._selecting = False   # 互動選取執行緒正在跑，避免同時開兩個搶 stdin
        self._active = False      # /robot_status=COARSE 才 True：只有這時候跑偵測/選取/發座標

        if VEHICLE_CAMERA_EXTRINSIC_IS_PLACEHOLDER:
            self.get_logger().warn(
                '車載相機外參/topic 目前是佔位值（config.py VEHICLE_CAMERA_*），算出來的 base '
                '座標【不準】，只能驗證流程。填入真值後把 VEHICLE_CAMERA_EXTRINSIC_IS_PLACEHOLDER 改成 False。')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.make_camera_tf()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.color_sub = self.create_subscription(Image, VEHICLE_COLOR_TOPIC, self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, VEHICLE_DEPTH_TOPIC, self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, VEHICLE_CAMERA_INFO_TOPIC, self.info_callback, 10)
        # 使用者互動選定的目標番茄座標（見 _interactive_select_thread / _publish_target）
        # 發給 arm_task_node，只有位置沒有姿態——粗定位沒有果梗方向，姿態交給
        # arm_task_node 端 MathUtils.camera_facing_flange_pose 自己算，跟 fine_node.py
        # 發 /target_pose 的分工原則一致（vision 端只給量到的原始幾何，機械手臂的姿態
        # 運算留在 arm 端）。
        self.target_pub = self.create_publisher(PointStamped, '/coarse_target_point', 10)
        self.status_sub = self.create_subscription(String, '/robot_status', self.status_callback, 10)

        self.camera_info = None
        self.latest_depth_img = None
        self.latest_tomatoes = []
        self._last_print = 0.0
        self._last_printed = None    # 上次印出來的 {id: (x, y, z)}；None = 還沒印過（第一次一定印）

        self.get_logger().info(
            f'粗定位節點啟動：訂閱 {VEHICLE_COLOR_TOPIC}，結果轉到 {BASE_FRAME} 座標系。')

    """廣播一次 VEHICLE_CAMERA_PARENT_FRAME → VEHICLE_CAMERA_OPTICAL_FRAME 的固定外參。"""
    def make_camera_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = VEHICLE_CAMERA_PARENT_FRAME
        t.child_frame_id = VEHICLE_CAMERA_OPTICAL_FRAME
        t.transform.translation.x = VEHICLE_CAMERA_EXTRINSIC_TRANSLATION[0]
        t.transform.translation.y = VEHICLE_CAMERA_EXTRINSIC_TRANSLATION[1]
        t.transform.translation.z = VEHICLE_CAMERA_EXTRINSIC_TRANSLATION[2]
        t.transform.rotation.x = VEHICLE_CAMERA_EXTRINSIC_ROTATION_QUAT[0]
        t.transform.rotation.y = VEHICLE_CAMERA_EXTRINSIC_ROTATION_QUAT[1]
        t.transform.rotation.z = VEHICLE_CAMERA_EXTRINSIC_ROTATION_QUAT[2]
        t.transform.rotation.w = VEHICLE_CAMERA_EXTRINSIC_ROTATION_QUAT[3]
        self.tf_static_broadcaster.sendTransform(t)

    """/robot_status=COARSE 才開啟偵測（手臂確定回到 Home、等目標）；其他值一律暫停。
    從暫停切到開啟時，清掉上一輪的追蹤結果和鎖定目標，重新開始。"""
    def status_callback(self, msg):
        active = (msg.data == 'COARSE')
        if active == self._active:
            return
        self._active = active
        if active:
            self.tracker = CoarseTomatoTracker()
            self._locked_key = None
            self._last_printed = None
            self.get_logger().info('手臂已在 Home，開啟粗定位偵測。')
        else:
            self.get_logger().info('手臂已接手（或不在等目標），粗定位暫停。')

    """快取最新的 CameraInfo，供反投影用的內參。"""
    def info_callback(self, msg):
        self.camera_info = msg

    """把深度影像轉成 cv2 array 並快取，轉換失敗時記錄錯誤。"""
    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'深度影像轉換失敗: {e}')

    """每幀：確認 camera_info/深度/TF 就緒後，跑偵測、疊圖、節流列印。"""
    def color_callback(self, msg):
        if not self._active:
            # 暫停中：不跑 YOLO、不追蹤、不發座標，只把原始影像顯示出來（視窗不會凍結）。
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception:
                return
            cv2.putText(cv_image, 'PAUSED (waiting for arm at Home)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self._show(cv_image)
            return

        if self.camera_info is None or self.latest_depth_img is None:
            return

        try:
            trans = self.tf_buffer.lookup_transform(
                BASE_FRAME, VEHICLE_CAMERA_OPTICAL_FRAME, rclpy.time.Time())
        except TransformException:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        ux, uy = self.camera_info.k[2], self.camera_info.k[5]
        if not (math.isfinite(fx) and math.isfinite(fy) and abs(fx) > 1e-6 and abs(fy) > 1e-6):
            self.get_logger().warn('camera_info 的 fx/fy 異常，這幀跳過偵測。')
            return

        results = self.detector.predict(cv_image, COARSE_YOLO_IMGSZ, COARSE_YOLO_CONF, YOLO_IOU)
        stamp = self.get_clock().now().to_msg()
        tomatoes = self.detector.detect(cv_image, results, fx, fy, ux, uy, trans,
                                         self.latest_depth_img, stamp=stamp)
        # 跨幀合併（見 coarse_tracker.py）：座標換成最近幾幀的中位數；id 每次呼叫都依「目前」
        # 到手臂基座的直線距離（base_x/y/z 向量長度）由近到遠重新編號，離基座最近的永遠是 ID 0，
        # 跟精定位 fine_target_selector.py 的 refresh_valid 同一套慣例——番茄互相接近時排名
        # 互換，id 會跟著換，只有內部的 key 才是跨幀不變的身份。
        tomatoes = self.tracker.update(tomatoes)
        self.latest_tomatoes = tomatoes
        self._maybe_start_selecting(tomatoes)
        self._publish_target(tomatoes, stamp)

        self._draw(cv_image, tomatoes)
        self._maybe_print()
        self._show(cv_image)

    """還沒鎖定目標、目前也沒有互動選取執行緒在跑、而且這一幀有番茄可以選，就開一條背景
    執行緒跑終端機互動選取（不能在 color_callback 這條 ROS callback 執行緒裡直接跑
    prompt_choose_id()——那是阻塞等使用者輸入的，會卡住整個節點收不到新的影像/深度）。
    daemon=True 的理由跟 fine_node.py 的 auto_pick_thread 一樣：互動選取可能卡在等
    終端輸入，Ctrl+C 只會送到主執行緒，非 daemon 的話這條執行緒會拖著 process 不讓退出。"""
    def _maybe_start_selecting(self, tomatoes):
        if not self._active or self._locked_key is not None or self._selecting or not tomatoes:
            return
        self._selecting = True
        threading.Thread(target=self._interactive_select_thread, daemon=True).start()

    """背景執行緒本體：跟 fine_node.py 共用同一套 TargetSelector.prompt_choose_id()
    （termios 接管終端機、打字期間每隔一段時間用 refresh_fn 即時刷新候選座標）。
    valid = {id: 番茄dict}——id 就是 coarse_tracker 依到基座距離重新排的顯示編號，
    本來就由近到遠排好，sort_key_fn 直接用 id 本身當排序依據即可，不用另外算。
    選中之後鎖定該番茄的 key（跨幀不變的內部身份，不是每次重排的 id）：prompt_choose_id
    回傳當下，valid 已經被 refresh_fn 原地更新過，valid[answer]['key'] 是最新的。"""
    def _interactive_select_thread(self):
        valid = {t['id']: t for t in self.tracker.confirmed()}
        answer = self.selector.prompt_choose_id(
            valid,
            refresh_fn=lambda v: ({t['id']: t for t in self.tracker.confirmed()}, {}),
            format_fn=self._format_choice_lines,
            sort_key_fn=lambda v: v,
            prompt_line_fn=lambda ids_str, buf:
                f"要靠近哪顆番茄？(可選 ID: {ids_str} / s=先不選 / r=重新偵測): {buf}")
        if isinstance(answer, int) and answer in valid and self._active:
            self._locked_key = valid[answer]['key']
            print(f"[粗定位] 已鎖定 ID:{answer}，送出座標給手臂，手臂接手後粗定位就會暫停。")
        self._selecting = False

    """互動選取畫面用：格式化候選清單成每一行文字，欄位跟 _draw_info_panel/_maybe_print
    同一套（到基座水平/3D 直線距離），純粹是換一個地方顯示同樣的資訊，不重算。"""
    def _format_choice_lines(self, valid, invalid_reasons):
        # ★ 2026-09-25：這行故意縮短、盡量少用中文——之前那版每行疊了水平距離+3D距離
        # 兩個中文欄位，行太長在某些終端機寬度下會被自動換行，跟 prompt_choose_id 用
        # 「每行佔幾個螢幕列」算好的游標移動量對不起來，導致畫面刷新時清錯範圍、
        # 新舊內容疊在一起（實測就是被這個逼出來的）。只留跟排序依據一致的 Dist
        # （到基座 3D 直線距離），不重複列水平距離。
        lines = []
        for vid in sorted(valid.keys()):
            t = valid[vid]
            dist_base = math.hypot(t['base_x'], t['base_y'], t['base_z'])
            lines.append(f"  [ID:{vid}] X={t['base_x']:+.3f} Y={t['base_y']:+.3f} Z={t['base_z']:+.3f} "
                         f"Dist={dist_base:.3f} conf={t['conf']:.2f}")
        return lines

    """把鎖定目標的即時座標發布給 arm_task_node（見 __init__ 的 target_pub 說明）。
    只在開啟中（COARSE）才會被呼叫：每幀都發，直到手臂收下、/robot_status 改成 BUSY、
    這裡跟著暫停為止——等於一次交接，不怕單一訊息漏掉。
    還沒選定（_locked_key 是 None）就不發，等使用者互動選取完成。已經選定的番茄這一幀
    如果跟丟了（key 不在 tomatoes 裡——連續 max_miss 幀沒偵測到，coarse_tracker 已經把
    整條 track 移除）就清掉鎖定、印警告，回到「還沒選定」狀態，下一幀有番茄可選時
    _maybe_start_selecting 會自動重新跳出互動選取，不用使用者自己重啟節點。"""
    def _publish_target(self, tomatoes, stamp):
        if self._locked_key is None:
            return
        target = next((t for t in tomatoes if t['key'] == self._locked_key), None)
        if target is None:
            self.get_logger().warn('鎖定的番茄跟丟了（超過容忍幀數沒偵測到），重新等待互動選取。')
            self._locked_key = None
            return
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = BASE_FRAME
        msg.point.x = target['base_x']
        msg.point.y = target['base_y']
        msg.point.z = target['base_z']
        self.target_pub.publish(msg)

    """在畫面上畫框、標 ID，並疊加左上角資訊面板（跟 fine_node.py 的
    Visualizer.draw_info_panel 同一種風格：半透明黑底、白字列出每顆的 base 座標）。
    （cv2.putText 不支援中文，這裡只用 ASCII。）"""
    def _draw(self, cv_image, tomatoes):
        YELLOW = (0, 255, 255)
        for t in tomatoes:
            locked = t['key'] == self._locked_key
            color = YELLOW if locked else (0, 200, 0)
            x1, y1, x2, y2 = (int(v) for v in t['bbox'])
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            label = f"ID{t['id']} LOCKED" if locked else f"ID{t['id']}"
            cv2.putText(cv_image, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        self._draw_info_panel(cv_image, tomatoes)
        if VEHICLE_CAMERA_EXTRINSIC_IS_PLACEHOLDER:
            cv2.putText(cv_image, 'EXTRINSIC = PLACEHOLDER (coords not accurate)', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    """左上角半透明資訊面板：列出這一幀看到的每顆番茄的 ID/base 座標/到基座距離/信心值。
    tomatoes 是 CoarseTomatoTracker.update() 的輸出，已經依「目前」到手臂基座的直線距離
    （base_x/y/z 向量長度，不是只有水平分量）由近到遠排好、id=0 是目前離基座最近的
    （跟 fine_target_selector.py 的 refresh_valid 同一套慣例），這裡不用再重排一次，
    直接照送進來的順序畫。DistBase 就是用來排序的那個值（3D 直線距離），Dist 則是原本
    就有的水平距離（忽略高度），兩個是不同用途的欄位，方便對照 id 排名對不對用 DistBase。"""
    def _draw_info_panel(self, img, tomatoes):
        WHITE, YELLOW = (255, 255, 255), (0, 255, 255)
        lines = [(f"{len(tomatoes)} tomatoes", YELLOW)]
        for t in tomatoes:
            conf = t.get('conf')
            conf_str = f" Conf={conf:.2f}" if conf is not None else ""
            horiz_dist = math.hypot(t['base_x'], t['base_y'])
            dist_base = math.hypot(t['base_x'], t['base_y'], t['base_z'])
            lines.append((f"[ID{t['id']}] X={t['base_x']:.3f} Y={t['base_y']:.3f} "
                          f"Z={t['base_z']:.3f} Dist={horiz_dist:.3f} "
                          f"DistBase={dist_base:.3f}{conf_str}", WHITE))

        font, scale, thick, line_h, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, 18, 8
        x0, y0 = 10, 40 if VEHICLE_CAMERA_EXTRINSIC_IS_PLACEHOLDER else 10
        max_w = max((cv2.getTextSize(ln, font, scale, thick)[0][0] for ln, _ in lines), default=0)
        panel_w, panel_h = max_w + pad * 2, line_h * len(lines) + pad * 2

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        y = y0 + pad + 12
        for ln, color in lines:
            cv2.putText(img, ln, (x0 + pad, y), font, scale, color, thick, cv2.LINE_AA)
            y += line_h

    """只在番茄清單有變化時才列印（base 座標，單位 m），不會每隔幾秒就刷一次：
    第一次一定印；之後只有「番茄新增/消失」或「任一顆位置跟上次印出來的相差超過
    COARSE_PRINT_CHANGE_M」才印。依據是 tracker 裡已確認的番茄（見 confirmed()），不是這一幀
    看到的，所以單幀漏偵測不會誤觸發。SCAN_PRINT_INTERVAL 仍是兩次列印之間的最短間隔。
    ★ snapshot 用 key（track 內部身份，終生不變）當比對依據，不能用 id——id 現在每次
    呼叫 confirmed() 都依當下深度重新排名，兩顆番茄深度排名互換時 id 對應的座標會跳掉，
    用 id 當 key 會被誤判成「有變化」狂重印，即使番茄根本沒動。"""
    """原地刷新同一個區塊（跟 fine_target_selector.py 的 prompt_choose_id 用同一套 ANSI
    游標控制手法：往上移動到上次印的第一行、清到畫面底、重印），不是每次變動都往下多
    印一段新的。番茄距離偵測邊界忽遠忽近、時有時無時（confidence 貼著門檻閃爍），
    改這個之前每次「有/沒有」切換都會整段重印一次，終端機被洗成一長串重複內容，
    改成原地刷新後畫面只會停在同一個位置更新數字，不會一直往下捲動。
    ★ self._selecting 是 True（互動選取執行緒的 prompt_choose_id 正在跑）時直接跳過：
    那條執行緒自己也在用同一套游標控制刷新同一個終端機，兩邊同時搶著動游標會互相
    干擾、畫面錯亂——互動選取進行中，候選清單本來就已經即時顯示在那邊的面板裡了，
    不需要這裡重複印一次。"""
    def _maybe_print(self):
        if self._selecting:
            return
        now = time.time()
        if now - self._last_print <= SCAN_PRINT_INTERVAL:
            return
        rows = self.tracker.confirmed()   # 已依目前深度排序、id=0 是目前最近的
        snapshot = {t['key']: (t['base_x'], t['base_y'], t['base_z']) for t in rows}
        if self._last_printed is not None and not self._print_changed(snapshot):
            return
        self._last_print = now
        self._last_printed = snapshot

        if not rows:
            lines = ['[粗定位] 目前沒有偵測到番茄']
        else:
            lines = [f'[粗定位] 偵測到 {len(rows)} 顆番茄（{BASE_FRAME} 座標，m，依到基座直線距離排序）：']
            for t in rows:
                horiz_dist = math.hypot(t['base_x'], t['base_y'])
                dist_base = math.hypot(t['base_x'], t['base_y'], t['base_z'])
                lines.append(f"  [ID:{t['id']}] x={t['base_x']:+.3f} y={t['base_y']:+.3f} z={t['base_z']:+.3f}  "
                             f"距基座(水平) {horiz_dist:.3f}  距基座(3D直線) {dist_base:.3f}  conf {t['conf']:.2f}")

        # ★ 2026-09-25：原本精算「螢幕列數」再移動游標，實測發現跟終端機實際換行狀況
        # 偶爾對不起來、清錯範圍疊字（跟互動選取那次撞到的是同一種問題）。改成每次都
        # 清整個畫面重畫，不用精算行數，不管內容多長都不會疊字，代價是刷新時畫面會閃一下。
        out = ["\033[2J\033[H", "\n".join(lines)]
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    """新清單 snapshot（{key: (x, y, z)}）跟上次印出來的比：track 組合不同，或任一顆位移超過門檻就算有變。"""
    def _print_changed(self, snapshot: dict) -> bool:
        if snapshot.keys() != self._last_printed.keys():
            return True
        return any(math.dist(snapshot[i], self._last_printed[i]) > COARSE_PRINT_CHANGE_M
                   for i in snapshot)

    """把 cv_image 放大 DISPLAY_SCALE 倍後顯示，只影響視窗大小。"""
    def _show(self, cv_image):
        if DISPLAY_SCALE != 1.0:
            cv_image = cv2.resize(cv_image, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                                   interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Coarse Tomato Detection', cv_image)
        cv2.waitKey(1)

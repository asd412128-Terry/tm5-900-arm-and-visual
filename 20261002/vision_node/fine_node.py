"""
============================================================================
 YOLOv11 番茄 / 果梗分割視覺節點 — 主節點
============================================================================
 流程：
   1. color_callback 每幀跑 YOLO 分割（委派給 ObjectDetector），算出
      番茄與果梗的世界座標 (含果梗 3D 方向向量)
   2. StemTracker 用滑動視窗做時間平滑，挑信心分數最高的一幀當代表
   3. 手臂回報 DONE 後，auto_pick_thread 委派 TargetSelector 列出候選、
      手動輸入 ID 選定目標
   4. 發布 /target_pose (位置 + 借用 orientation 欄位傳遞果梗 3D 方向向量)

 這支檔案只負責 ROS 訂閱/發布/callback 串接與跨模組協調，實際運算
 全部委派給 detector / skeleton / coordinates / stem_tracker /
 target_selector / mask_publisher / visualizer。
============================================================================
"""

import math
import os
import threading
import time
import cv2
import numpy as np
import yaml
import rclpy
import tf2_geometry_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from .config import (ARM_FLANGE_FRAME, CAMERA_EXTRINSIC_ROTATION_QUAT,
                      CAMERA_EXTRINSIC_TRANSLATION, CAMERA_INFO_TOPIC, CAMERA_OPTICAL_FRAME,
                      COLOR_TOPIC, DEPTH_TOPIC, DISPLAY_SCALE,
                      ENABLE_MANUAL_INTRINSIC_CALIB, MIN_STEM_DEPTH_PX, MODEL_PATH,
                      OCCLUDED_SCAN_GRACE, OCTOMAP_UPDATE_WAIT_SEC, SCAN_PRINT_INTERVAL,
                      VISION_MODE, WORLD_FRAME, YOLO_CONF, YOLO_IMGSZ, YOLO_IOU)
from .coordinates import CoordinateEstimator
from .fine_detector import ObjectDetector
from .fine_mask_publisher import TargetMaskBuilder
from .fine_skeleton import PedicelSkeletonizer
from .fine_stem_tracker import StemTracker
from .fine_target_selector import TargetSelector
from .fine_tomato_tracker import TomatoTracker
from .fine_visualizer import Visualizer

# 手動校正過的 RGB 相機內參檔（ost.yaml 格式），取代相機出廠發布的內參
# 由 ROS2 camera_calibration (cameracalibrator) 產生，與 checkerboard_pose_publisher.py
# 共用同一份檔案，確保手眼標定和這裡的 pixel backprojection 用的是同一組內參
CALIBRATION_FILE_PATH = os.path.expanduser('~/tm_ws/calibration/d435i_rgb_calib.yaml')


class VisionNode(Node):
    """初始化各個運算模組、TF、ROS 訂閱/發布與內部狀態。"""
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()

        self.get_logger().info(f'執行模式: {VISION_MODE}（VISION_MODE 環境變數切換），模型: {MODEL_PATH}')
        self.get_logger().info('正在載入 YOLOv11 果梗+番茄分割模型...')
        coord_estimator = CoordinateEstimator()
        self.coord = coord_estimator   # 供 StemTracker 平滑完像素/深度後，重新反投影用
        skeletonizer = PedicelSkeletonizer()
        self.detector = ObjectDetector(MODEL_PATH, coordinate_estimator=coord_estimator,
                                        skeletonizer=skeletonizer,
                                        min_stem_depth_px=MIN_STEM_DEPTH_PX)
        self.stem_tracker = StemTracker()
        self.tomato_tracker = TomatoTracker()
        self.target_selector = TargetSelector()
        self.mask_builder = TargetMaskBuilder()
        self.visualizer = Visualizer()

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.make_camera_tf()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.color_sub = self.create_subscription(Image, COLOR_TOPIC, self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.info_callback, 10)
        self.status_sub = self.create_subscription(String, '/robot_status', self.status_callback, 10)
        self.target_pub = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.no_target_pub = self.create_publisher(String, '/vision_status', 10)

        # ★ 點雲過濾已搬到獨立節點 cloud_filter_node.py。
        #   原因：實測證實 vision_node 這個 process 只要疊加多個 subscription
        #   （即使 callback 內容是空的），點雲的 publish() 就會被 MoveIt2 的
        #   PointCloudOctomapUpdater 靜默拒收，原因未明；拆成獨立、最小化的 process 後恢復正常。
        #   vision_node 這裡只需要在選定目標時，把合併後的膨脹遮罩發布成一張 Image，
        #   不再需要訂閱原始點雲、也不需要在這裡做逐點過濾。
        #
        #   2026-09-15：上面那個「疊加 subscription 導致點雲被拒收」的舊結論，從沒真的
        #   查出因果機制（「原因未明」），時間點也跟當時 real 模式 /clock 缺失（見
        #   PROGRESS.md / arm_node 那邊查到的根因）重疊，很可能是同一個 TF 時間戳空窗期
        #   問題被誤判成別的原因。/clock 修好後這裡重新加一個 subscription 測試，觀察點雲
        #   發布是否仍然正常；如果之後又發現點雲發不出去，優先懷疑是這裡新增的訂閱。
        self.mask_pub = self.create_publisher(Image, '/target_filter_mask', 10)
        self._octomap_done_event = threading.Event()
        self.octomap_done_sub = self.create_subscription(
            Bool, '/target_filter_done', self._octomap_done_callback, 10)
        # 2026-09-30：送新點雲前先清空 OctoMap（見 _clear_octomap_before_new_cloud）。
        self.clear_octomap_client = self.create_client(Empty, '/clear_octomap')

        # --- 狀態 ---
        self.task_completed = False
        self.camera_info = None
        self.calibrated_k = None
        self.calibrated_d = None
        self.latest_depth_img = None
        self.is_processing = False
        self._last_scan_print = 0.0
        self._occluded_scan_count = 0
        self._occluded_scan_grace = OCCLUDED_SCAN_GRACE
        self.latest_targets = []
        # 互動選取中的候選清單（build_valid_candidates/refresh_valid 用的那份），非 None
        # 時 CV2 視窗改用這份的編號畫框，跟終端機顯示同一套 ID，不要各自獨立編號。
        self._interactive_valid = None
        self.latest_tomatoes = []
        # _maybe_print_and_trigger_pick 算好的 (valid, invalid_reasons, all_occluded)，
        # 交給 auto_pick_thread 沿用同一份，不要重算一次——重算會讓 build_valid_candidates
        # 內建的「不能夾的」清單被印兩次，也可能因為兩次呼叫之間 targets 已經更新而
        # 算出不同編號，跟終端機剛印出來的對不起來。
        self._pending_valid = None

        # 啟動時直接讀入手動校正過的內參，之後 info_callback 收到 camera_info
        # 時會用這組值覆蓋掉相機出廠發布的 K，取代出廠值。只有實機需要，Isaac 模式跳過。
        if ENABLE_MANUAL_INTRINSIC_CALIB:
            self.load_calibration_file(CALIBRATION_FILE_PATH)
        else:
            self.get_logger().info(f'VISION_MODE={VISION_MODE}，不載入手動內參校正檔，直接用 camera_info topic 提供的內參。')

        self.get_logger().info('YOLOv11 視覺大腦啟動！手動選擇夾取模式開啟...')

    """cloud_filter_node.py 真的完成過濾+發布後才會收到這個，取代原本固定 sleep 的猜測。"""
    def _octomap_done_callback(self, msg: Bool):
        self._octomap_done_event.set()

    """送新點雲給 OctoMap 的前一刻才清空，清完馬上送（2026-09-30）。原本是手臂在「出發去
    精定位之前」清空，規劃時地圖是空的：植株不在地圖裡，而且清空後的空 octree 會讓
    MoveIt 碰撞檢查慢到每次規劃 7～16 秒。改到這裡之後，手臂規劃時地圖一直都有上一次
    掃描的植株，只有送新點雲前的一瞬間是空的（這時手臂不會規劃）。
    在背景執行緒（auto_pick_thread）裡呼叫：要等清空真的完成才送遮罩，不然清空可能
    晚於新點雲被處理，把新點雲一起清掉。"""
    def _clear_octomap_before_new_cloud(self):
        if not self.clear_octomap_client.service_is_ready():
            self.get_logger().warn('/clear_octomap 不在，跳過清空（新點雲會疊在舊地圖上）。')
            return
        future = self.clear_octomap_client.call_async(Empty.Request())
        deadline = time.time() + 3.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        if not future.done():
            self.get_logger().warn('/clear_octomap 3 秒內沒回應，直接送新點雲。')

    # -----------------------------------------------------------------
    # 基本 callback
    # -----------------------------------------------------------------
    """訂閱 /robot_status；收到 "DONE" 才允許開始/恢復偵測，其餘視為手臂移動中。"""
    def status_callback(self, msg):
        if msg.data == 'DONE':
            if not self.task_completed:
                self._occluded_scan_count = 0
            self.task_completed = True
        else:
            self.task_completed = False

    """廣播一次 link_6 → camera_optical_frame 的固定外參（相機掛載位置）。"""
    def make_camera_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = ARM_FLANGE_FRAME
        t.child_frame_id = CAMERA_OPTICAL_FRAME

        t.transform.translation.x = CAMERA_EXTRINSIC_TRANSLATION[0]
        t.transform.translation.y = CAMERA_EXTRINSIC_TRANSLATION[1]
        t.transform.translation.z = CAMERA_EXTRINSIC_TRANSLATION[2]

        t.transform.rotation.x = CAMERA_EXTRINSIC_ROTATION_QUAT[0]
        t.transform.rotation.y = CAMERA_EXTRINSIC_ROTATION_QUAT[1]
        t.transform.rotation.z = CAMERA_EXTRINSIC_ROTATION_QUAT[2]
        t.transform.rotation.w = CAMERA_EXTRINSIC_ROTATION_QUAT[3]
        self.tf_static_broadcaster.sendTransform(t)

    """從 ROS camera_calibration 產生的 ost.yaml 讀入手動校正的 K / D。"""
    def load_calibration_file(self, path: str):
        try:
            with open(path, 'r') as f:
                calib = yaml.safe_load(f)
            self.calibrated_k = calib['camera_matrix']['data']
            self.calibrated_d = calib['distortion_coefficients']['data']
            self.get_logger().info(f'已從校正檔載入內參：{path}\nK = {self.calibrated_k}')
        except Exception as e:
            self.calibrated_k = None
            self.calibrated_d = None
            self.get_logger().error(
                f'讀取校正檔失敗（{path}）：{e}\n'
                '將暫時退回使用 camera_info topic 提供的內參（未校正的出廠值）。')

    """快取最新的 CameraInfo，供反投影用的內參；若校正檔已載入，用校正值覆蓋出廠 K。"""
    def info_callback(self, msg):
        if self.calibrated_k is not None:
            msg.k = self.calibrated_k
            msg.d = self.calibrated_d
        self.camera_info = msg

    """把深度影像轉成 cv2 array 並快取，轉換失敗時記錄錯誤。"""
    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"深度影像轉換失敗: {e}")

    """一次性動作：算出目標果梗 + 番茄的合併膨脹遮罩，發布成一張 Image，
    交給獨立的 cloud_filter_node.py 去做實際的點雲過濾與發布給 OctoMap。"""
    def _publish_target_filter_mask(self, stem_obj: dict):
        combined, stem_px, tomato_px = self.mask_builder.build_combined_mask(stem_obj)
        if combined is None:
            self.get_logger().warn('目標果梗沒有 mask 資料，跳過本次遮罩發布。')
            return
        if tomato_px == 0:
            self.get_logger().warn('找不到目標番茄的 mask，本次只挖除果梗部分。')

        mask_msg = self.bridge.cv2_to_imgmsg(combined, encoding='mono8')
        mask_msg.header.stamp = self.get_clock().now().to_msg()
        self.mask_pub.publish(mask_msg)
        self.get_logger().info(
            f'[目標遮罩] 果梗遮罩 {stem_px} px, 番茄遮罩 {tomato_px} px，'
            f'已發布給 cloud_filter_node。')

    """遮擋換備用視角前：不排除任何目標，把『目前這個視角』看到的整個場景都拍一張塞進
    OctoMap。備用視角是繞著同一個目標點、固定半徑搖過去，這段路正好貼近造成遮擋的
    葉子/藤蔓，若不先拍下目前看得到的東西，手臂就是盲搖過去，見 arm_task_node.py
    的 _move_to_fine_alt。"""
    def _publish_environment_mask(self):
        if self.latest_depth_img is None:
            self.get_logger().warn('沒有深度圖可用，跳過遮擋換視角前的環境點雲快照。')
            return
        mask = np.zeros(self.latest_depth_img.shape[:2], dtype=np.uint8)
        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header.stamp = self.get_clock().now().to_msg()
        self.mask_pub.publish(mask_msg)
        self.get_logger().info('[遮擋換視角] 已發布目前視角的環境遮罩給 cloud_filter_node。')

    # -----------------------------------------------------------------
    # 主偵測迴圈
    # -----------------------------------------------------------------
    """主偵測迴圈：確認 camera_info/深度/TF 就緒後，跑 YOLO 偵測、追蹤、疊圖，並觸發夾取流程判斷。"""
    def color_callback(self, msg):  
        if self.camera_info is None or self.latest_depth_img is None:
            return

        try:
            trans = self.tf_buffer.lookup_transform(WORLD_FRAME, CAMERA_OPTICAL_FRAME, rclpy.time.Time())
        except TransformException:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        if not self.task_completed:
            self.latest_targets = []
            self.latest_tomatoes = []
            cv2.putText(cv_image, "手臂移動中，暫停偵測...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self._show(cv_image)
            return

        results = self.detector.predict(cv_image, YOLO_IMGSZ, YOLO_CONF, YOLO_IOU)
        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        ux_img, uy_img = self.camera_info.k[2], self.camera_info.k[5]

        if not (math.isfinite(fx) and math.isfinite(fy) and abs(fx) > 1e-6 and abs(fy) > 1e-6):
            self.get_logger().warn('camera_info 的 fx/fy 異常，這幀跳過偵測。')
            return

        stamp = self.get_clock().now().to_msg()
        detected_tomatoes, detected_objects = self.detector.detect(
            cv_image, results, fx, fy, ux_img, uy_img, trans, self.latest_depth_img, stamp=stamp)

        # 穩定每顆番茄的 occluded 判定 (連續多幀同一種結果才切換)，原地改寫
        # detected_tomatoes 裡每個 dict 的 'occluded' 欄位；detected_objects 裡的
        # 'paired_tomato' 是同一個物件參照，跟著一起拿到穩定後的值。
        self.tomato_tracker.update(detected_tomatoes)

        self.latest_tomatoes = detected_tomatoes
        if len(detected_objects) > 0:
            detected_objects = self.stem_tracker.update(detected_objects)
            # StemTracker 平滑的是像素座標+深度（源頭），不是 world_x/y/z、vx/vy/vz
            # 本身——這裡才把平滑後的源頭資料，用跟單幀 fallback 同一套換算，各自
            # 反投影一次，算出這一幀真正要用的抓取點世界座標跟方向向量。
            for obj in detected_objects:
                self._finalize_smoothed_stem(obj, fx, fy, ux_img, uy_img, trans, stamp)
            # StemTracker 視窗裡代表某根果梗的那一幀，可能是幾幀前留存的舊紀錄，
            # 它的 'paired_tomato' 是那時候留下的物件快照，不是這一幀 detected_tomatoes
            # 裡的同一個物件——重新指到這一幀真正的番茄物件，讓果梗畫框跟番茄畫框
            # 讀的是同一份 occluded 狀態，紅綠燈才會一起變，不會各跳各的。
            TargetSelector.resolve_live_pairing(detected_objects, detected_tomatoes)
            # 'paired_tomato' 導正完才能算「果實端點跟番茄中心」的距離（診斷用，見
            # _finalize_smoothed_stem 裡 calyx_world 的說明），calyx_world 反投影失敗
            # 或這幀沒配對到番茄就留 None，不列進面板。
            for obj in detected_objects:
                obj['calyx_tomato_dist'] = None
                nt = obj.get('paired_tomato')
                calyx_world = obj.get('calyx_world')
                if nt is not None and calyx_world is not None:
                    obj['calyx_tomato_dist'] = math.dist(
                        calyx_world, (nt['world_x'], nt['world_y'], nt['world_z']))
            detected_objects = sorted(detected_objects, key=lambda obj: obj['z_center'])
        self.latest_targets = detected_objects

        # 番茄框（紅/綠）畫在這裡；就算這幀沒偵測到任何果梗，只要有番茄還是要畫出來，
        # 不能因為 detected_objects 是空的就整個跳過。
        if len(detected_objects) > 0 or len(detected_tomatoes) > 0:
            self.visualizer.draw_tracked_overlay(cv_image, detected_objects, detected_tomatoes,
                                                  valid=self._interactive_valid)

        self._maybe_print_and_trigger_pick(detected_objects)

        self._show(cv_image)

    """StemTracker 平滑完的是像素座標+深度（源頭），這裡把平滑後的抓取點/calyx端/branch端
    (px,py,z) 各自反投影一次，原地覆寫 obj 的 world_x/y/z、vx/vy/vz——覆寫前這兩組欄位是
    『這一幀』的舊值，沒有意義。跟 fine_detector.py 算單幀 fallback 用同一套
    CoordinateEstimator 函式，兩邊算法保證一致。"""
    def _finalize_smoothed_stem(self, obj, fx, fy, ux_img, uy_img, trans, stamp):
        local_point = self.coord.backproject_to_local_point(
            obj['cx'], obj['cy'], obj['z_center'], fx, fy, ux_img, uy_img, stamp=stamp)
        world_point = tf2_geometry_msgs.do_transform_point(local_point, trans)
        if all(math.isfinite(v) for v in (world_point.point.x, world_point.point.y, world_point.point.z)):
            obj['world_x'] = world_point.point.x
            obj['world_y'] = world_point.point.y
            obj['world_z'] = world_point.point.z

        obj['vx'], obj['vy'], obj['vz'] = self.coord.pixel_depth_pair_to_unit_vector(
            obj['calyx_px'], obj['calyx_py'], obj['calyx_z'],
            obj['branch_px'], obj['branch_py'], obj['branch_z'],
            fx, fy, ux_img, uy_img, trans)

        # 診斷用：把「果實端點(calyx)」反投影成世界座標存起來，供 resolve_live_pairing()
        # 把 paired_tomato 導正成『這一幀』的番茄物件之後，再算跟番茄中心的距離——這裡先
        # 只算 calyx 的世界座標，不能在這裡就用 obj['paired_tomato']：這時候它還是
        # StemTracker 滑動視窗裡某個舊幀留存的番茄快照，跟這一幀的 calyx 世界座標對不上，
        # 算出來的距離會是垃圾值（實測量到 0.5m 這種明顯不合理的數字）。
        calyx_lp = self.coord.backproject_to_local_point(
            obj['calyx_px'], obj['calyx_py'], obj['calyx_z'], fx, fy, ux_img, uy_img, stamp=stamp)
        calyx_wp = tf2_geometry_msgs.do_transform_point(calyx_lp, trans)
        if all(math.isfinite(v) for v in (calyx_wp.point.x, calyx_wp.point.y, calyx_wp.point.z)):
            obj['calyx_world'] = (calyx_wp.point.x, calyx_wp.point.y, calyx_wp.point.z)
        else:
            obj['calyx_world'] = None

    """把 cv_image 放大 DISPLAY_SCALE 倍後顯示，只影響視窗大小，不影響偵測/座標計算。"""
    def _show(self, cv_image):
        if DISPLAY_SCALE != 1.0:
            cv_image = cv2.resize(cv_image, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                                   interpolation=cv2.INTER_LINEAR)
        cv2.imshow("YOLOv11 Realtime Vision", cv_image)
        cv2.waitKey(1)

    """節流列印這輪掃描結果，並在偵測到目標時觸發手動選取流程。"""
    def _maybe_print_and_trigger_pick(self, detected_objects):
        now = time.time()
        if self.is_processing or (now - self._last_scan_print) <= SCAN_PRINT_INTERVAL:
            return
        self._last_scan_print = now

        # 掃描階段就先算好 valid/invalid_reasons，用它們來編號、印終端機清單——
        # 跟畫面（_interactive_valid 也是同一份 build_valid_candidates 的產物）保證同一套
        # ID，使用者才不會照終端機編號選到畫面上完全不同的物理目標。
        valid, invalid_reasons, all_occluded = self.target_selector.build_valid_candidates(detected_objects)
        self.visualizer.print_scan_summary(detected_objects, self.latest_tomatoes, valid, invalid_reasons)

        # all_occluded 只是給終端機印字用的細節原因，不再拿來決定要不要試備用視角
        # （見 auto_pick_thread：只要這輪沒有能夾的目標，不管是完全沒偵測到、候選被判定
        # 遮擋、還是被其他原因刷掉，一律先試備用視角，符合「即時偵測到什麼狀況就反應
        # 什麼狀況」，不要因為原因分類不同就有些情況會被漏掉、系統看起來像沒反應）。
        self.is_processing = True
        self._pending_valid = (valid, invalid_reasons, all_occluded)
        # daemon=True：夾取流程可能卡在 prompt_choose_id() 等終端輸入，
        # Ctrl+C 產生的 KeyboardInterrupt 只會送到主執行緒；這條非 daemon
        # 的話，主執行緒中斷後整個 process 還是會被這條卡住的執行緒拖著
        # 不讓退出，要按第二次 Ctrl+C 或砍掉終端才會真的結束。
        threading.Thread(target=self.auto_pick_thread, daemon=True).start()

    # -----------------------------------------------------------------
    # 手動選取 / 發布目標
    # -----------------------------------------------------------------
    """夾取流程主體：列出候選、終端機互動選定目標、發布目標遮罩與 /target_pose，並阻塞等待手臂完成。"""
    def auto_pick_thread(self):
        # 沿用 _maybe_print_and_trigger_pick 剛算好、剛印在終端機上的那份，不要重算
        # （見 self._pending_valid 的說明）。目標清單在這個背景執行緒真的開始跑之前
        # 也可能已經被下一幀的 color_callback 更新成空的（race），但不要因此就直接
        # 回報 NO_TARGET 放棄——那等於繞過了下面的備用視角重試邏輯，這裡統一交給
        # 下面的 pickable 判斷處理（valid 是空的話 pickable 自然是 False，會走進
        # 同一套「先試備用視角」的分支）。
        valid, invalid_reasons, all_occluded = self._pending_valid
        pickable = any(vid not in invalid_reasons for vid in valid)
        if not pickable:
            # 2026-09-16：不管這輪「沒有能夾的目標」是因為完全沒偵測到、候選被判定
            # 遮擋、還是候選被其他原因刷掉（all_occluded 只是細節原因，不再拿來分支）
            # ——一律用同一套「連續 N 輪都這樣才真的換視角」邏輯即時反應，不要因為
            # 分類到不同原因就有些情況直接放棄回家、跳過備用視角，導致系統看起來
            # 像「用某些方式遮擋就不會反應」。只要偵測到「現在可以挑」的目標，
            # pickable 立刻變 True，下面就會馬上歸零重新開始，是即時的，不會卡住。
            self._occluded_scan_count += 1
            reason = '全部候選被判定遮擋' if all_occluded else '沒有能夾的目標'
            print(f"這輪{reason}（連續 {self._occluded_scan_count}/{self._occluded_scan_grace}）...")
            if self._occluded_scan_count >= self._occluded_scan_grace:
                print("已連續多輪確認看不到能挑的目標，先拍下目前視角的環境點雲，再通知手臂換視角重新掃描。")
                # 跟選定目標時同一套模式（見下面 _octomap_done_event 的用法）：
                # 先確定 cloud_filter_node 真的把這個視角的點雲塞進 OctoMap 了，
                # 才通知手臂開始搖過去，不然手臂可能在點雲還沒發完之前就先動了。
                self._octomap_done_event.clear()
                self._clear_octomap_before_new_cloud()
                self._publish_environment_mask()
                got_signal = self._octomap_done_event.wait(timeout=OCTOMAP_UPDATE_WAIT_SEC)
                if not got_signal:
                    self.get_logger().warn(
                        f'等待環境點雲完成訊號逾時（{OCTOMAP_UPDATE_WAIT_SEC}s 內沒收到），強制繼續。')
                self.no_target_pub.publish(String(data='OCCLUDED'))
                self._occluded_scan_count = 0
                self.task_completed = False
            # 還沒連續確認夠次數：先不通知手臂，維持 task_completed，讓偵測繼續跑，
            # 累積更多幀再判斷——避免剛到新視角、追蹤視窗還沒填滿就被誤判成全遮擋。
            self.is_processing = False
            return

        self._occluded_scan_count = 0
        miss_counts = []   # [(position, miss), ...]，用位置當識別鍵（見 refresh_valid 註解）
        # prompt_choose_id 內部用 valid.clear()/valid.update() 原地更新這個 dict（不是
        # 重新賦值），所以 self._interactive_valid 指到同一個物件，會自動跟著即時更新，
        # CV2 視窗讀到的永遠是當下這一份。
        self._interactive_valid = valid
        answer = self.target_selector.prompt_choose_id(
            valid,
            refresh_fn=lambda v: self.target_selector.refresh_valid(
                v, self.latest_targets, self.target_selector.max_reach_m,
                miss_counts=miss_counts),
            invalid_reasons=invalid_reasons)
        self._interactive_valid = None

        if answer == 's':
            print("略過這輪，不夾取，通知手臂回初始位置。")
            self.no_target_pub.publish(String(data='NO_TARGET'))
            self.task_completed = False
            self.is_processing = False
            return

        if answer == 'r':
            print("重新偵測中，稍後會重新列出候選...")
            self.is_processing = False
            return

        chosen_idx = answer
        target, vx, vy, vz, distance_to_base = valid[chosen_idx]
        print(f"已選擇 [ID:{chosen_idx}]，準備發送夾取指令...")

        # ★ 一次性動作：只在「決定要抓哪顆」的這一刻，用當下偵測到的番茄 mask
        #   建一份已避開目標的點雲塞給 OctoMap，不是持續過濾。
        # 2026-09-15：改成真的等 cloud_filter_node.py 發來的完成訊號，不再固定 sleep
        # 猜時間——這樣可以在點雲確實發布完之後盡快動作，不會又提早動（burst 還沒發完
        # 手臂就先動）也不會平白多等。OCTOMAP_UPDATE_WAIT_SEC 保留當逾時保底，訊號
        # 遲遲沒來時才強制繼續，避免真的卡死整條流程。
        self._octomap_done_event.clear()
        self._clear_octomap_before_new_cloud()
        self._publish_target_filter_mask(target)
        got_signal = self._octomap_done_event.wait(timeout=OCTOMAP_UPDATE_WAIT_SEC)
        if not got_signal:
            self.get_logger().warn(
                f'等待 cloud_filter_node 完成訊號逾時（{OCTOMAP_UPDATE_WAIT_SEC}s 內沒收到），強制繼續。')

        target_msg = PoseStamped()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = WORLD_FRAME
        target_msg.pose.position.x = target['world_x']
        target_msg.pose.position.y = target['world_y']
        target_msg.pose.position.z = target['world_z']

        # 借用 Orientation 欄位傳遞 3D 方向向量
        target_msg.pose.orientation.x = float(vx)
        target_msg.pose.orientation.y = float(vy)
        target_msg.pose.orientation.z = float(vz)
        target_msg.pose.orientation.w = 0.0

        self.target_pub.publish(target_msg)

        print("夾取指令已發出！等待手臂完成動作...")
        self.task_completed = False
        while not self.task_completed:
            time.sleep(1.0)

        print("\n手臂動作完成！重新啟動 YOLO 掃描...\n")
        self.is_processing = False

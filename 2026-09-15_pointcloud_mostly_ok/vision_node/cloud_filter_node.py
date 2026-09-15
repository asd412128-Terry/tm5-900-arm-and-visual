"""
獨立的點雲過濾節點。

只做一件事：訂閱對齊過彩色的深度圖 + vision_node 發來的目標遮罩，過濾後組成點雲發布給 OctoMap。
刻意跟 vision_node（YOLO 推論、多重 subscription、TF listener）完全分離成獨立 process，
因為實測證實 vision_node 那個 process 裡，只要疊加多個 subscription（即使 callback 內容是空的），
點雲的 publish() 就會被 MoveIt2 的 PointCloudOctomapUpdater 靜默拒收，原因未明。
這個節點只有兩個 subscription（深度圖 + 遮罩），維持跟已驗證成功的 minimal_relay_test.py
相近的最小化結構（2026-09-15 曾經為了「等 TF 就緒才發布」那個方向加過 TransformListener，
後來證實那個方向沒解決問題，已經拿掉，見下面改版記錄）。

跑法：
  VISION_MODE=isaac python3 cloud_filter_node.py --ros-args -p use_sim_time:=true

  use_sim_time:=true 在 Isaac Sim 下必帶：本節點發布過濾後點雲時會用自己的時鐘蓋掉
  header.stamp，若跟 TF 用的模擬時間對不上，MoveIt2 的 PointCloudOctomapUpdater 查
  transform 會失敗，整包點雲被靜默丟棄（RViz 上完全看不到東西，且不會報明顯錯誤）。

  相機內參依 VISION_MODE 環境變數自動切換：isaac 用下方 _ISAAC_INTRINSICS 的實測值，
  real 則跟 vision_node.py 讀同一份手動校正檔（見 CALIBRATION_FILE_PATH），校正檔換了
  只要改那個 yaml，這裡不用動。仍可用 -p fx:=... 等方式臨時覆寫。

  2026-09-02 除錯結論：只發一張的話，這一張的時間戳若剛好落在 /tf 廣播的空窗期
  （實測 /tf 平均只有 8Hz，抖動可達 190ms），MoveIt2 的 tf2_ros::MessageFilter 就會
  直接把這張靜默丟掉、不報錯，OctoMap 永遠收不到點；把 sensors_3d.yaml 的
  point_cloud_topic 暫時指到持續發布的原始 /camera/depth/points（~23Hz）則一定成功，
  證實問題在「只發一張、沒有重試機會」，不在 TF/QoS/is_dense 本身。修法：收到目標遮罩
  後，在 FILTER_BURST_DURATION_SEC 這段時間內，每收到一張新的深度圖就用同一份遮罩
  再過濾、再發布一次，形成短暫連發，比照原始點雲那樣多給 MoveIt2 幾次機會。

  2026-09-15 除錯結論：改版前訂閱的是 realsense driver 原始點雲 topic
  （/camera/camera/depth/color/points），它的 header.frame_id 是 camera_depth_optical_frame
  （深度鏡頭自己的座標系），但過濾時直接拿這些 XYZ 套用「彩色鏡頭」內參反投影回像素座標去
  跟遮罩（遮罩是根據彩色影像算的）比對——沒有先轉換深度→彩色鏡頭之間的實體 baseline
  （D435i 約 1.5cm），導致算出來的像素座標系統性偏移，遮罩挖不乾淨、目標點雲邊緣一直有
  殘影漏網。改成跟 vision_node.py 用同一份「已對齊彩色的深度圖」
  （/camera/camera/aligned_depth_to_color/image_raw，frame_id 已經是
  camera_color_optical_frame）自己組點雲，遮罩可以直接用像素座標 1:1 對照，不用再反投影
  比對，兩邊天生同一個座標系，不會有這個偏移問題。

  2026-09-15 改版（第一次）：拿掉「連發碰運氣」的 burst 機制，改成自己養一個
  tf2_ros.Buffer/TransformListener，主動問 can_transform() 確認 TF 就緒後才發布，
  且只發一次。**實機測試結果：cloud_filter_node 這邊確實有算出點雲、也呼叫了
  publish()（log 印出「深度圖有效點 296276 → 保留 259660 點，已發布給 OctoMap」），
  但 OctoMap 最後還是空的（GetPlanningScene 查到 octomap.data=[]）**——代表問題不在
  「TF 有沒有就緒」這個環節，卡在 MoveIt2 收到訊息之後、真正插入 octomap 之前的某個
  地方，原因還沒查出來，先放棄這個方向。

  2026-09-15 改版（第二次，目前版本）：退回「連發」，但吸取上一版的教訓——不是每收到
  一張新深度圖就重新過濾一次（那樣才會疊出「果梗變粗」），而是收到目標遮罩後**只算一次
  過濾結果**，把算好的同一份 PointCloud2 訊息在 FILTER_BURST_DURATION_SEC 秒內用
  REPUBLISH_INTERVAL_SEC 的週期重複發布同一份 bytes——內容完全相同，只是重複送達，
  藉此換取「多次機會被 MoveIt2 收進去」（跟最早 09-02 那版 burst 想解決的問題一樣），
  但因為每次都是同一批點、同一個時間戳，OctoMap 不會像疊多幀不同觀測那樣把細長物體
  疊粗。拿掉了 tf2_ros.Buffer/TransformListener（第一次改版加的，沒能解決問題，
  先移除減少複雜度跟訂閱數，回到檔頭一開始強調的最小化結構）。
"""
import os

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge

# 收到目標遮罩後，重複發布同一份過濾結果這麼久（見上面 2026-09-15 第二次改版說明）。
FILTER_BURST_DURATION_SEC = 2.0

# 連發的間隔——跟 sensors_3d.yaml 的 max_update_rate（目前 5.0，即 0.2s 一次）對齊，
# 抓比它快一點，確保這段時間內至少會落在它的處理視窗裡幾次。
REPUBLISH_INTERVAL_SEC = 0.1

# 深度值若明顯是 mm 為單位 (大於這個門檻) 就轉換成公尺；跟 vision_node/coordinates.py
# 的 DEPTH_MM_THRESHOLD 同一套判斷邏輯，保持一致。
DEPTH_MM_THRESHOLD = 10.0

# 2026-09-15：離群點濾除（outlier removal）。深度相機在物體邊緣/深度不連續處常見
# 「飛點雜訊」（flying pixels）——不對應任何真實表面的孤立雜訊點。遮罩邊界剛好就是
# 這種深度不連續的地方，過濾後常留下幾顆孤立的雜訊點飄在半空中，被 MoveIt2 當成獨立
# 障礙物卡住規劃。做法：對每個點查詢附近 OUTLIER_MIN_NEIGHBORS 個最近鄰的距離，
# 距離超過 OUTLIER_RADIUS_M 代表這個點附近沒有足夠鄰居支撐，視為孤立雜訊丟棄。
OUTLIER_RADIUS_M = 0.005       # 5mm 內要有足夠鄰居才算真實表面
OUTLIER_MIN_NEIGHBORS = 15     # 含自己在內，5mm 半徑內至少要有這麼多個點（2026-09-15：5 太寬鬆，
                                # 小撮聚在一起的雜訊團也濾不掉，調高逼真實表面才留得下來）

# 跟 vision_node/config.py 一樣，用 VISION_MODE 環境變數切換內參來源，
# 不用每次啟動都手動帶 -p fx:=... 等參數。
VISION_MODE = os.environ.get('VISION_MODE', 'real').strip().lower()
if VISION_MODE not in ('real', 'isaac'):
    VISION_MODE = 'isaac'

# 深度圖 topic 依 VISION_MODE 切換，跟 vision_node/config.py 的 DEPTH_TOPIC 同一份對照表：
#   real  → realsense-ros 已對齊彩色的深度圖（frame_id 是 camera_color_optical_frame）
#   isaac → Isaac Sim 的深度圖本來就跟彩色同一個視角，不用另外對齊
_DEPTH_TOPIC_BY_MODE = {
    'real':  '/camera/camera/aligned_depth_to_color/image_raw',
    'isaac': '/camera/depth/image_rect_raw',
}
DEPTH_TOPIC = _DEPTH_TOPIC_BY_MODE[VISION_MODE]

# k[0]=fx, k[4]=fy, k[2]=cx, k[5]=cy。
# 2026-09-02 用 `ros2 topic echo /camera/camera_info --once` 在 Isaac Sim 下實測。
_ISAAC_INTRINSICS = {'fx': 1108.5125019853992, 'fy': 1108.5125019853992, 'cx': 640.0, 'cy': 360.0}
_PLACEHOLDER_INTRINSICS = {'fx': 600.0, 'fy': 600.0, 'cx': 320.0, 'cy': 240.0}

# real 模式跟 vision_node.py 用同一份 ROS camera_calibration 產生的 ost.yaml 格式校正檔。
CALIBRATION_FILE_PATH = os.path.expanduser('~/tm_ws/calibration/d435i_rgb_calib.yaml')


def _load_real_intrinsics(path: str = CALIBRATION_FILE_PATH):
    """讀校正檔取 fx/fy/cx/cy；讀不到回傳 None，由呼叫端決定要不要退回佔位值。"""
    try:
        with open(path, 'r') as f:
            calib = yaml.safe_load(f)
        k = calib['camera_matrix']['data']
        return {'fx': k[0], 'fy': k[4], 'cx': k[2], 'cy': k[5]}
    except Exception:
        return None


class CloudFilterNode(Node):
    def __init__(self):
        super().__init__('cloud_filter_node')
        self.bridge = CvBridge()

        # 相機內參：isaac 用實測值；real 讀校正檔，讀失敗才退回佔位值並報錯。
        # 仍可用 -p fx:=... 等方式在啟動時覆寫。
        if VISION_MODE == 'isaac':
            intr = _ISAAC_INTRINSICS
        else:
            intr = _load_real_intrinsics()
            if intr is not None:
                self.get_logger().info(f'已從校正檔載入內參：{CALIBRATION_FILE_PATH}\nK = {intr}')
            else:
                intr = _PLACEHOLDER_INTRINSICS
                self.get_logger().error(
                    f'讀取校正檔失敗（{CALIBRATION_FILE_PATH}），改用佔位內參 {intr}，'
                    f'過濾區域會對不準目標！')
        self.declare_parameter('fx', intr['fx'])
        self.declare_parameter('fy', intr['fy'])
        self.declare_parameter('cx', intr['cx'])
        self.declare_parameter('cy', intr['cy'])

        self.sub_depth = self.create_subscription(
            Image, DEPTH_TOPIC, self.depth_callback, 10)
        self.sub_mask = self.create_subscription(
            Image, '/target_filter_mask', self.mask_callback, 10)

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub = self.create_publisher(PointCloud2, '/camera/depth/points_gated', pub_qos)
        # 2026-09-15：讓 vision_node.py 不用再靠固定 sleep 猜「octomap 應該處理完了」，
        # 改成真的等這個訊號——但要等整個連發窗口跑完才發，理由跟最早 burst 版一樣：
        # 相機掛在手臂上，訊號一發手臂就可能開始動，要確保連發期間視角沒變。
        self.pub_done = self.create_publisher(Bool, '/target_filter_done', 10)

        self._latest_depth_msg = None
        self._pending_mask = None
        self._cached_msg = None
        self._burst_start_time = None
        self._republish_timer = None
        self.get_logger().info(f'cloud_filter_node 啟動（訂閱深度圖 {DEPTH_TOPIC}），等待深度圖與目標遮罩...')

    def depth_callback(self, msg: Image):
        """存最新一幀；只有在「遮罩已到、但選定當下還沒收過任何深度圖」這個邊界情況，
        才拿這一幀來開始連發（見 mask_callback）。"""
        self._latest_depth_msg = msg
        if self._pending_mask is not None:
            mask = self._pending_mask
            self._pending_mask = None
            self._start_burst(msg, mask)

    def mask_callback(self, mask_msg: Image):
        """收到 vision_node 發來的目標遮罩（選定目標時發布一次）：
        用手上最新一幀深度圖過濾一次，組成點雲，之後在 FILTER_BURST_DURATION_SEC 秒內
        重複發布**同一份**結果（不重新過濾、不換幀），多給 MoveIt2 幾次機會收進去，
        同時避免『每幀重新過濾』造成的多幀疊圖變粗（見檔頭 2026-09-15 改版說明）。"""
        try:
            mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'遮罩轉換失敗: {e}')
            return

        depth_msg = self._latest_depth_msg
        if depth_msg is None:
            self.get_logger().warn('尚未收到任何深度圖，等 depth_callback 收到第一幀再開始連發。')
            self._pending_mask = mask
            return
        self._start_burst(depth_msg, mask)

    def _start_burst(self, depth_msg: Image, mask):
        """算一次過濾結果並快取，立刻發布第一次，然後開始週期性重發同一份訊息。"""
        msg = self._build_filtered_msg(depth_msg, mask)
        if msg is None:
            return
        self._cached_msg = msg
        self.pub.publish(msg)
        self._burst_start_time = self.get_clock().now()
        if self._republish_timer is not None:
            self._republish_timer.cancel()
        self._republish_timer = self.create_timer(REPUBLISH_INTERVAL_SEC, self._on_republish_tick)

    def _on_republish_tick(self):
        """週期性重發同一份快取好的訊息，直到 FILTER_BURST_DURATION_SEC 跑完才通知
        vision_node（見上面 pub_done 建立時的說明，不能提早發）。"""
        elapsed = (self.get_clock().now() - self._burst_start_time).nanoseconds / 1e9
        if elapsed >= FILTER_BURST_DURATION_SEC:
            if self._republish_timer is not None:
                self._republish_timer.cancel()
                self._republish_timer = None
            self._cached_msg = None
            self.pub_done.publish(Bool(data=True))
            return
        self.pub.publish(self._cached_msg)

    def _remove_outliers(self, pts: np.ndarray) -> np.ndarray:
        """丟掉附近沒有足夠鄰居支撐的孤立點（深度相機的飛點雜訊），
        見上面 OUTLIER_RADIUS_M/OUTLIER_MIN_NEIGHBORS 的說明。"""
        if pts.shape[0] <= OUTLIER_MIN_NEIGHBORS:
            return pts
        tree = cKDTree(pts)
        # k 個最近鄰的距離（含自己，所以查 OUTLIER_MIN_NEIGHBORS 個）；
        # 第 k 個鄰居的距離若超過半徑，代表半徑內鄰居不足 OUTLIER_MIN_NEIGHBORS 個。
        dists, _ = tree.query(pts, k=OUTLIER_MIN_NEIGHBORS, workers=-1)
        keep_mask = dists[:, -1] <= OUTLIER_RADIUS_M
        return pts[keep_mask]

    def _build_filtered_msg(self, depth_msg: Image, mask):
        """算一次過濾後的 PointCloud2；失敗回傳 None。呼叫端負責 publish（可能不只發一次，
        見 _start_burst/_on_republish_tick，同一份結果會被重複發送，這裡不重複計算）。"""
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value

        try:
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'深度影像轉換失敗: {e}')
            return None

        h, w = depth_img.shape[:2]
        if mask.shape[:2] != (h, w):
            self.get_logger().error(
                f'遮罩尺寸 {mask.shape[:2]} 跟深度圖尺寸 {(h, w)} 不一致，跳過這一幀。')
            return None

        z = depth_img.astype(np.float32)
        # 跟 vision_node/coordinates.py 的 depth_to_m 同一套判斷：明顯是 mm 就轉公尺。
        if np.nanmax(z) > DEPTH_MM_THRESHOLD:
            z = z / 1000.0

        total = int(np.count_nonzero(z > 0.0))

        # 深度圖跟遮罩本來就是同一個座標系（都是 camera_color_optical_frame 的像素網格），
        # 直接逐像素比對即可，不用像舊版那樣反投影回像素座標再比對。
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        valid = z > 0.0
        keep = valid & (mask == 0)
        kept_count = int(np.count_nonzero(keep))

        # 攤平成「只放保留下來的有效點」的未整理點雲（height=1），不要整張 h×w 網格都發出去
        # （網格大部分是 NaN）。2026-09-15 實測發現：MoveIt2 的 point_subsample 是「每隔 N 個
        # 索引取一點」，對整張大多是 NaN 的網格取樣，會取到一堆空白格，密度很稀疏；攤平成
        # 只含有效點的清單後，同一個 point_subsample 才會跟舊版（直接訂閱 realsense 原始點雲，
        # 本來就是攤平清單）密度一致，不用另外調 sensors_3d.yaml 的 point_subsample。
        pts = np.stack([x[keep], y[keep], z[keep]], axis=-1).astype(np.float32)
        pts = self._remove_outliers(pts)
        n = pts.shape[0]

        filtered_msg = PointCloud2()
        # 沿用深度圖原本的時間戳（拍攝當下相機蓋的），不要蓋成發布當下的「現在」。
        # 蓋成「現在」會跟 TF 廣播賽跑：MoveIt2 的 tf2_ros::MessageFilter 需要該精確時間戳
        # 對應的 TF 才會處理這包點雲，若那個時間點的 TF 還沒廣播出來就會被直接丟棄、不報錯，
        # 導致 OctoMap 永遠收不到點。原始時間戳早於「現在」，對應的 TF 一定已經廣播過。
        filtered_msg.header = depth_msg.header
        filtered_msg.height = 1
        filtered_msg.width = n
        filtered_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        filtered_msg.is_bigendian = False
        filtered_msg.point_step = 12
        filtered_msg.row_step = filtered_msg.point_step * n
        # 現在只放保留下來的有效點，沒有 NaN 了（舊版整張網格才會混雜 NaN，需要 is_dense=False
        # 如實反映；這版攤平後每個點都保證有限值），可以照實設 True。
        filtered_msg.is_dense = True
        filtered_msg.data = pts.tobytes()

        self.get_logger().info(
            f'[cloud_filter_node] 深度圖有效點 {total} → 保留 {kept_count} 點，'
            f'將在 {FILTER_BURST_DURATION_SEC}s 內重複發布同一份給 OctoMap。')
        return filtered_msg


def main(args=None):
    rclpy.init(args=args)
    node = CloudFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

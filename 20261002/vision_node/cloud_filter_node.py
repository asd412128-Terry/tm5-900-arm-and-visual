"""
獨立的點雲過濾節點。

只做一件事：訂閱原始點雲 + vision_node 發來的目標遮罩，過濾後發布給 OctoMap。
刻意跟 vision_node（YOLO 推論、多重 subscription、TF listener）完全分離成獨立 process，
因為實測證實 vision_node 那個 process 裡，只要疊加多個 subscription（即使 callback 內容是空的），
點雲的 publish() 就會被 MoveIt2 的 PointCloudOctomapUpdater 靜默拒收，原因未明。
這個節點只有兩個 subscription（原始點雲 + 遮罩），維持跟已驗證成功的 minimal_relay_test.py
相近的最小化結構。

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
  後，在 FILTER_BURST_DURATION_SEC 這段時間內，每收到一張新的原始點雲就用同一份遮罩
  再過濾、再發布一次，形成短暫連發，比照原始點雲那樣多給 MoveIt2 幾次機會。
"""
import os

import numpy as np
import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge

# 收到目標遮罩後，連續過濾+發布幾秒鐘（而不是只發一張），
# 避免唯一一張剛好落在 TF 空窗期就整輪落空。
FILTER_BURST_DURATION_SEC = 2.0

# 跟 vision_node/config.py 一樣，用 VISION_MODE 環境變數切換內參來源，
# 不用每次啟動都手動帶 -p fx:=... 等參數。
VISION_MODE = os.environ.get('VISION_MODE', 'real').strip().lower()
if VISION_MODE not in ('real', 'isaac'):
    VISION_MODE = 'isaac'

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

        self.sub_cloud = self.create_subscription(
            PointCloud2, '/camera/depth/points', self.cloud_callback, qos_profile_sensor_data)
        self.sub_mask = self.create_subscription(
            Image, '/target_filter_mask', self.mask_callback, 10)

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub = self.create_publisher(PointCloud2, '/camera/depth/points_gated', pub_qos)

        self._latest_cloud_msg = None
        self._active_mask = None
        self._active_mask_deadline = None
        self.get_logger().info('cloud_filter_node 啟動，等待點雲與目標遮罩...')

    def cloud_callback(self, msg: PointCloud2):
        """存最新一幀；若目前在「目標遮罩發布後的連發窗口」內，順便用同一份遮罩過濾+發布這一幀，
        讓 MoveIt2 有多次機會（比照原始點雲持續發布時的成功模式），不要只賭一張。"""
        self._latest_cloud_msg = msg
        if self._active_mask is None:
            return
        if self.get_clock().now() > self._active_mask_deadline:
            self._active_mask = None
            self._active_mask_deadline = None
            return
        self._filter_and_publish(msg, self._active_mask)

    def mask_callback(self, mask_msg: Image):
        """收到 vision_node 發來的目標遮罩（選定目標時發布一次）：
        立刻用最新一幀點雲過濾+發布一次，並開啟 FILTER_BURST_DURATION_SEC 秒的連發窗口，
        窗口內每收到新的原始點雲都會用這份遮罩再過濾+發布一次。"""
        try:
            mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'遮罩轉換失敗: {e}')
            return

        self._active_mask = mask
        self._active_mask_deadline = self.get_clock().now() + Duration(seconds=FILTER_BURST_DURATION_SEC)

        cloud_msg = self._latest_cloud_msg
        if cloud_msg is None:
            self.get_logger().warn('尚未收到任何點雲，等連發窗口內收到第一幀再過濾。')
            return
        self._filter_and_publish(cloud_msg, mask)

    def _filter_and_publish(self, cloud_msg: PointCloud2, mask):
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        h_mask, w_mask = mask.shape[:2]

        pts = np.frombuffer(bytearray(cloud_msg.data), dtype=np.float32).reshape(
            -1, cloud_msg.point_step // 4)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        valid_z = z > 0
        u = np.zeros_like(x, dtype=np.int32)
        v = np.zeros_like(y, dtype=np.int32)
        u[valid_z] = np.round(fx * x[valid_z] / z[valid_z] + cx).astype(np.int32)
        v[valid_z] = np.round(fy * y[valid_z] / z[valid_z] + cy).astype(np.int32)

        in_bounds = valid_z & (u >= 0) & (u < w_mask) & (v >= 0) & (v < h_mask)
        mask_hit = np.zeros_like(in_bounds)
        mask_hit[in_bounds] = mask[v[in_bounds], u[in_bounds]] > 0

        total = pts.shape[0]
        pts[mask_hit, 0] = float('nan')
        kept_count = int((~mask_hit).sum())

        filtered_msg = PointCloud2()
        # 沿用原始點雲的時間戳（拍攝當下 Isaac 蓋的），不要蓋成發布當下的「現在」。
        # 蓋成「現在」會跟 TF 廣播賽跑：MoveIt2 的 tf2_ros::MessageFilter 需要該精確時間戳
        # 對應的 TF 才會處理這包點雲，若那個時間點的 TF 還沒廣播出來就會被直接丟棄、不報錯，
        # 導致 OctoMap 永遠收不到點。原始時間戳早於「現在」，對應的 TF 一定已經廣播過。
        filtered_msg.header = cloud_msg.header
        filtered_msg.height = cloud_msg.height
        filtered_msg.width = cloud_msg.width
        filtered_msg.fields = cloud_msg.fields
        filtered_msg.is_bigendian = cloud_msg.is_bigendian
        filtered_msg.point_step = cloud_msg.point_step
        filtered_msg.row_step = cloud_msg.row_step
        # 上面把挖掉的點設成 NaN，這包資料不再保證無 NaN，is_dense 要如實設 False，
        # 否則下游 PCL-based 消費者（例如 MoveIt2 PointCloudOctomapUpdater）會信任這個
        # 旗標跳過 NaN 檢查，遇到未預期的 NaN 導致整包點雲處理失敗、OctoMap 建不出東西。
        filtered_msg.is_dense = False
        filtered_msg.data = pts.tobytes()

        self.pub.publish(filtered_msg)
        self.get_logger().info(
            f'[cloud_filter_node] 原始 {total} 點 → 保留 {kept_count} 點，已發布給 OctoMap。')


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

"""
獨立的點雲過濾節點。

只做一件事：訂閱原始點雲 + vision_node 發來的目標遮罩，過濾後發布給 OctoMap。
刻意跟 vision_node（YOLO 推論、多重 subscription、TF listener）完全分離成獨立 process，
因為實測證實 vision_node 那個 process 裡，只要疊加多個 subscription（即使 callback 內容是空的），
點雲的 publish() 就會被 MoveIt2 的 PointCloudOctomapUpdater 靜默拒收，原因未明。
這個節點只有兩個 subscription（原始點雲 + 遮罩），維持跟已驗證成功的 minimal_relay_test.py
相近的最小化結構。

跑法：
  python3 cloud_filter_node.py --ros-args -p use_sim_time:=true \
      -p fx:=<你的fx> -p fy:=<你的fy> -p cx:=<你的cx> -p cy:=<你的cy>

  相機內參預設值是佔位數字，請依實際 camera_info 內容用 -p 覆寫，
  或直接改下面 declare_parameter 的預設值。
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge


class CloudFilterNode(Node):
    def __init__(self):
        super().__init__('cloud_filter_node')
        self.bridge = CvBridge()

        # 相機內參：預設佔位值，請用 -p fx:=... 等方式在啟動時覆寫成正確數值
        # （跟 vision_node 訂閱的 /camera/camera_info 裡的 k[0], k[4], k[2], k[5] 一致）
        self.declare_parameter('fx', 600.0)
        self.declare_parameter('fy', 600.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)

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
        self.get_logger().info('cloud_filter_node 啟動，等待點雲與目標遮罩...')

    def cloud_callback(self, msg: PointCloud2):
        """只存最新一幀，不做任何處理。"""
        self._latest_cloud_msg = msg

    def mask_callback(self, mask_msg: Image):
        """收到 vision_node 發來的目標遮罩（選定目標時發布一次），
        用最新一幀點雲做過濾，並在這個 subscription callback 裡直接發布
        （跟 minimal_relay_test.py 驗證成功的結構一致：發布動作在訂閱 callback 內執行）。"""
        cloud_msg = self._latest_cloud_msg
        if cloud_msg is None:
            self.get_logger().warn('尚未收到任何點雲，跳過本次過濾。')
            return

        try:
            mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'遮罩轉換失敗: {e}')
            return

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
        filtered_msg.header = cloud_msg.header
        filtered_msg.header.stamp = self.get_clock().now().to_msg()
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

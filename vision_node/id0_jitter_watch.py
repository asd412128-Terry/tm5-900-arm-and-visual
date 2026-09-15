"""
============================================================================
 獨立診斷工具：即時印出 ID0（z_center 最近的果梗）raw vs 平滑後 落差
============================================================================
 完全獨立的節點，自己訂閱相機三路 topic、自己跑一份 YOLO + StemTracker，
 不 import、不修改 vision_node.py / stem_tracker.py 任何一行——只是重用
 detector.py / coordinates.py / skeleton.py / stem_tracker.py 這幾個本來就
 「純運算、不碰 ROS 訂閱發布」的模組（vision_node.py 自己也是這樣用它們）。

 ⚠️ 這會另外載入一份 YOLO 模型、對同一路相機影像再跑一次推論，等於在原本
 vision_node 之外多一份 GPU/CPU 負擔，兩個一起跑可能都會變慢。只是要看數字
 抖動幅度，看完建議關掉。

 跑法（需要 vision_node.py 或其他節點已經把 world→camera_optical_frame
 那條 TF 鏈接好，這支工具不會自己發布 link_6→camera_optical_frame 的靜態
 外參，純粹只查）：
     python3 -m vision_node.id0_jitter_watch                    # 實機
     VISION_MODE=isaac python3 -m vision_node.id0_jitter_watch  # Isaac Sim
============================================================================
"""
import math

import rclpy
import tf2_geometry_msgs
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from .config import (CAMERA_INFO_TOPIC, CAMERA_OPTICAL_FRAME, COLOR_TOPIC, DEPTH_TOPIC,
                      MIN_STEM_DEPTH_PX, MODEL_PATH, WORLD_FRAME, YOLO_CONF, YOLO_IMGSZ, YOLO_IOU)
from .coordinates import CoordinateEstimator
from .detector import ObjectDetector
from .skeleton import PedicelSkeletonizer
from .stem_tracker import StemTracker


class Id0JitterWatch(Node):

    def __init__(self):
        super().__init__('id0_jitter_watch')
        self.bridge = CvBridge()

        self.get_logger().info(f'載入 YOLO 模型：{MODEL_PATH}（跟 vision_node 各自獨立一份）')
        coord = CoordinateEstimator()
        self.coord = coord
        self.detector = ObjectDetector(MODEL_PATH, coordinate_estimator=coord,
                                        skeletonizer=PedicelSkeletonizer(),
                                        min_stem_depth_px=MIN_STEM_DEPTH_PX)
        self.stem_tracker = StemTracker()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_info = None
        self.latest_depth_img = None

        self.create_subscription(Image, COLOR_TOPIC, self.color_callback, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.info_callback, 10)
        self.get_logger().info('等待相機資料 + TF (world -> camera_optical_frame)...')

    def info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'深度影像轉換失敗: {e}')

    def _backproject_world(self, cx, cy, z, fx, fy, ux, uy, trans, stamp):
        local_point = self.coord.backproject_to_local_point(cx, cy, z, fx, fy, ux, uy, stamp=stamp)
        world_point = tf2_geometry_msgs.do_transform_point(local_point, trans)
        return world_point.point.x, world_point.point.y, world_point.point.z

    def color_callback(self, msg):
        if self.camera_info is None or self.latest_depth_img is None:
            return
        try:
            trans = self.tf_buffer.lookup_transform(WORLD_FRAME, CAMERA_OPTICAL_FRAME, rclpy.time.Time())
        except TransformException:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        ux, uy = self.camera_info.k[2], self.camera_info.k[5]
        if not (math.isfinite(fx) and math.isfinite(fy) and abs(fx) > 1e-6 and abs(fy) > 1e-6):
            return

        stamp = self.get_clock().now().to_msg()
        results = self.detector.predict(cv_image, YOLO_IMGSZ, YOLO_CONF, YOLO_IOU)
        _tomatoes, detected_objects = self.detector.detect(
            cv_image, results, fx, fy, ux, uy, trans, self.latest_depth_img, stamp=stamp)

        if not detected_objects:
            return

        detected_objects_by_depth = sorted(detected_objects, key=lambda o: o['z_center'])
        raw0 = detected_objects_by_depth[0]
        raw_world = self._backproject_world(raw0['cx'], raw0['cy'], raw0['z_center'], fx, fy, ux, uy, trans, stamp)

        smoothed_list = self.stem_tracker.update(detected_objects_by_depth)
        smoothed_list = sorted(smoothed_list, key=lambda o: o['z_center'])
        smoothed0 = smoothed_list[0]
        smoothed_world = self._backproject_world(smoothed0['cx'], smoothed0['cy'], smoothed0['z_center'],
                                                   fx, fy, ux, uy, trans, stamp)

        dz_mm = (raw0['z_center'] - smoothed0['z_center']) * 1000.0
        print(f"[ID0 raw]      cx={raw0['cx']:.1f} cy={raw0['cy']:.1f} z={raw0['z_center']:.4f}m "
              f"world=({raw_world[0]:.4f}, {raw_world[1]:.4f}, {raw_world[2]:.4f})")
        print(f"[ID0 smoothed] cx={smoothed0['cx']:.1f} cy={smoothed0['cy']:.1f} z={smoothed0['z_center']:.4f}m "
              f"world=({smoothed_world[0]:.4f}, {smoothed_world[1]:.4f}, {smoothed_world[2]:.4f})  "
              f"Δz={dz_mm:+.1f}mm")

        # 方向向量：raw 是這一幀當場算出的 vx/vy/vz（detector.py 算好、還沒被平滑覆寫）；
        # smoothed 要用平滑後的 calyx/branch 像素+深度重新反投影算一次——StemTracker
        # 回傳的 rec['vx/vy/vz'] 其實還是某一幀的舊值（見 stem_tracker.py 說明），
        # 不能直接拿來當「平滑後的向量」，這裡照 vision_node.py 同一套算法重算。
        raw_vec = (raw0['vx'], raw0['vy'], raw0['vz'])
        smoothed_vec = self.coord.pixel_depth_pair_to_unit_vector(
            smoothed0['calyx_px'], smoothed0['calyx_py'], smoothed0['calyx_z'],
            smoothed0['branch_px'], smoothed0['branch_py'], smoothed0['branch_z'],
            fx, fy, ux, uy, trans)
        dot = sum(a * b for a, b in zip(raw_vec, smoothed_vec))
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        print(f"[ID0 vector]   raw=({raw_vec[0]:+.3f}, {raw_vec[1]:+.3f}, {raw_vec[2]:+.3f})  "
              f"smoothed=({smoothed_vec[0]:+.3f}, {smoothed_vec[1]:+.3f}, {smoothed_vec[2]:+.3f})  "
              f"夾角={angle_deg:.1f}°")


def main(args=None):
    rclpy.init(args=args)
    node = Id0JitterWatch()
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

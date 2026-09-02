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
import yaml
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from .config import (ARM_FLANGE_FRAME, CAMERA_EXTRINSIC_ROTATION_QUAT,
                      CAMERA_EXTRINSIC_TRANSLATION, CAMERA_INFO_TOPIC, CAMERA_OPTICAL_FRAME,
                      COLOR_TOPIC, DEPTH_TOPIC, DISPLAY_SCALE, EMPTY_SCAN_GRACE,
                      ENABLE_MANUAL_INTRINSIC_CALIB, MIN_STEM_DEPTH_PX, MODEL_PATH,
                      OCTOMAP_UPDATE_WAIT_SEC, SCAN_PRINT_INTERVAL, VISION_MODE,
                      WORLD_FRAME, YOLO_CONF, YOLO_IMGSZ, YOLO_IOU)
from .coordinates import CoordinateEstimator
from .detector import ObjectDetector
from .mask_publisher import TargetMaskBuilder
from .skeleton import PedicelSkeletonizer
from .stem_tracker import StemTracker
from .target_selector import TargetSelector
from .tomato_tracker import TomatoTracker
from .visualizer import Visualizer

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
        self.mask_pub = self.create_publisher(Image, '/target_filter_mask', 10)

        # --- 狀態 ---
        self.task_completed = False
        self.camera_info = None
        self.calibrated_k = None
        self.calibrated_d = None
        self.latest_depth_img = None
        self.is_processing = False
        self._last_scan_print = 0.0
        self._empty_scan_count = 0
        self._empty_scan_grace = EMPTY_SCAN_GRACE
        self._no_target_sent = False
        self.latest_targets = []
        self.latest_tomatoes = []

        # 啟動時直接讀入手動校正過的內參，之後 info_callback 收到 camera_info
        # 時會用這組值覆蓋掉相機出廠發布的 K，取代出廠值。只有實機需要，Isaac 模式跳過。
        if ENABLE_MANUAL_INTRINSIC_CALIB:
            self.load_calibration_file(CALIBRATION_FILE_PATH)
        else:
            self.get_logger().info(f'VISION_MODE={VISION_MODE}，不載入手動內參校正檔，直接用 camera_info topic 提供的內參。')

        self.get_logger().info('YOLOv11 視覺大腦啟動！手動選擇夾取模式開啟...')

    # -----------------------------------------------------------------
    # 基本 callback
    # -----------------------------------------------------------------
    """訂閱 /robot_status；收到 "DONE" 才允許開始/恢復偵測，其餘視為手臂移動中。"""
    def status_callback(self, msg):
        if msg.data == 'DONE':
            if not self.task_completed:
                self._empty_scan_count = 0
                self._no_target_sent = False
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
        combined, stem_px, tomato_px = self.mask_builder.build_combined_mask(stem_obj, self.latest_tomatoes)
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
            # StemTracker 視窗裡代表某根果梗的那一幀，可能是幾幀前留存的舊紀錄，
            # 它的 'paired_tomato' 是那時候留下的物件快照，不是這一幀 detected_tomatoes
            # 裡的同一個物件——重新指到這一幀真正的番茄物件，讓果梗畫框跟番茄畫框
            # 讀的是同一份 occluded 狀態，紅綠燈才會一起變，不會各跳各的。
            TargetSelector.resolve_live_pairing(detected_objects, detected_tomatoes)
            detected_objects = sorted(detected_objects, key=lambda obj: obj['z_real'])
        self.latest_targets = detected_objects

        # 番茄框（紅/綠）畫在這裡；就算這幀沒偵測到任何果梗，只要有番茄還是要畫出來，
        # 不能因為 detected_objects 是空的就整個跳過。
        if len(detected_objects) > 0 or len(detected_tomatoes) > 0:
            self.visualizer.draw_tracked_overlay(cv_image, detected_objects, detected_tomatoes)

        self._maybe_print_and_trigger_pick(detected_objects)

        self._show(cv_image)

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

        self.visualizer.print_scan_summary(detected_objects, self.latest_tomatoes)

        if len(detected_objects) > 0:
            self._empty_scan_count = 0
            self.is_processing = True
            threading.Thread(target=self.auto_pick_thread).start()
        else:
            self._empty_scan_count += 1
            print(f"這幀沒有偵測到番茄(連續 {self._empty_scan_count}/{self._empty_scan_grace})...")
            if self._empty_scan_count >= self._empty_scan_grace and not self._no_target_sent:
                print("這輪掃描沒有偵測到任何番茄，通知手臂回初始位置。")
                self.no_target_pub.publish(String(data='NO_TARGET'))
                self._no_target_sent = True
                self.task_completed = False

    # -----------------------------------------------------------------
    # 手動選取 / 發布目標
    # -----------------------------------------------------------------
    """夾取流程主體：列出候選、終端機互動選定目標、發布目標遮罩與 /target_pose，並阻塞等待手臂完成。"""
    def auto_pick_thread(self):
        targets = self.latest_targets
        if not targets:
            print("進入夾取流程時目標清單已空，通知手臂回初始位置。")
            self.no_target_pub.publish(String(data='NO_TARGET'))
            self.task_completed = False
            self.is_processing = False
            return

        valid = self.target_selector.build_valid_candidates(targets)
        if not valid:
            print("這輪沒有可夾取的目標，通知手臂回初始位置。")
            self.no_target_pub.publish(String(data='NO_TARGET'))
            self.task_completed = False
            self.is_processing = False
            return

        answer = self.target_selector.prompt_choose_id(valid)

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
        self._publish_target_filter_mask(target)

        # MoveIt2 的 PointCloudOctomapUpdater 沒有對外的「處理完成」通知，
        # 用固定等待時間換取簡單可靠（配合 max_update_rate 抓一個夠用的餘裕）。
        time.sleep(OCTOMAP_UPDATE_WAIT_SEC)

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

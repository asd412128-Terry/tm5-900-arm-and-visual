# send axis_6 angle to arm.py
import rclpy
# 引入「幾何訊息」格式 「點座標」、「座標轉換」、「姿勢（位置+角度）」
from geometry_msgs.msg import PointStamped, TransformStamped, PoseStamped
# 引入「感測器訊息」格式 「影像」、「相機資訊」
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from rclpy.node import Node
# ROS的影像格式跟OpenCV影像格式不同，這是一個「翻譯橋樑」。
from cv_bridge import CvBridge

import cv2
import numpy as np
import math
# 多執行緒 一邊處理影像，一邊進行自動化調度
import threading
import time
import termios
import sys
# YOLO 函式庫
from ultralytics import YOLO
# 骨架化需要用到
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
# 載入處理 TF (Transform) 座標轉換的相關工具
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
# 專門幫幾何點位進行座標轉換的工具
import tf2_geometry_msgs

# ---------------------------------------------------------------------------
# 果梗骨架化相關函式 (從 stem_detection_point.py 搬過來)
# ---------------------------------------------------------------------------
def clean_pedicel_mask(mask: np.ndarray, min_area: int = 50):
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest_idx, cv2.CC_STAT_AREA] < min_area:
        return None
    return (labels == largest_idx).astype(np.uint8) * 255


def skeletonize_pedicel(mask: np.ndarray, erosion_kernel: int = 3, erosion_iter: int = 1,
                         min_area_for_erosion: int = 150):
    """回傳 (清理後mask, erosion後mask, 骨架bool陣列)"""
    clean = clean_pedicel_mask(mask)
    if clean is None:
        return None, None, None
    clean_area = int((clean > 0).sum())
    if clean_area < min_area_for_erosion:
        # mask本來就很小(果梗太細/太遠)，erosion 會直接把它吃光，這裡跳過
        eroded = clean
    else:
        kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)
        eroded = cv2.erode(clean, kernel, iterations=erosion_iter)
        if (eroded > 0).sum() == 0:
            # erosion 之後消失了，退回沒erosion的版本
            eroded = clean
    skeleton = skeletonize(eroded > 0)
    return clean, eroded, skeleton


def find_skeleton_endpoints(skeleton: np.ndarray) -> np.ndarray:
    """找骨架端點(只有1個鄰居的骨架pixel)，回傳 (N,2) 的 (y,x) 座標陣列"""
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    conv = convolve(skeleton.astype(int), kernel, mode="constant")
    endpoints = np.argwhere((conv == 11) & skeleton)
    return endpoints


def identify_endpoints_simple(endpoints: np.ndarray):
    """簡化版：假設果梗下方(y較大)是接近果實的一端(calyx)，上方是branch端"""
    ys = endpoints[:, 0]
    calyx_end = tuple(endpoints[int(np.argmax(ys))])
    branch_end = tuple(endpoints[int(np.argmin(ys))])
    return branch_end, calyx_end


def order_skeleton_path(skeleton: np.ndarray, start_point: tuple):
    """從 start_point 開始，沿著骨架走出一串有順序的 (y,x) 座標"""
    visited = set()
    path = []
    current = tuple(start_point)
    h, w = skeleton.shape
    while current is not None:
        path.append(current)
        visited.add(current)
        y, x = current
        next_pt = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] and (ny, nx) not in visited:
                    next_pt = (ny, nx)
                    break
            if next_pt:
                break
        current = next_pt
    return path


def find_grasp_point_pixel(ordered_path: list, grasp_ratio: float = 0.6):
    """
    取骨架路徑上的抓取點，回傳 ((y,x), index)。
    ordered_path[0] = calyx端(靠近番茄), ordered_path[-1] = branch端(遠離番茄)
    grasp_ratio: 0.0 = calyx端, 0.5 = 中點, 1.0 = branch端
    往上移(遠離番茄)就把 ratio 調大，例如 0.65 ~ 0.75
    """
    idx = int(len(ordered_path) * grasp_ratio)
    idx = min(idx, len(ordered_path) - 1)
    return ordered_path[idx], idx


def compute_growth_angle(ordered_path: list, grasp_idx: int, window: int = 10):
    """
    在抓取點附近取一段骨架，用前後各 window 個點算出 2D 生長方向角(度)。
    回傳角度定義：0° = 垂直向上，正值 = 順時鐘偏轉，範圍 -90° ~ +90°。
    """
    start = max(0, grasp_idx - window)
    end = min(len(ordered_path) - 1, grasp_idx + window)
    if start == end:
        return 0.0
    y1, x1 = ordered_path[start]
    y2, x2 = ordered_path[end]
    dx = x2 - x1
    dy = y2 - y1
    # atan2(dx, -dy): 以「垂直向上」為 0°，順時鐘為正
    angle_rad = math.atan2(dx, -(dy))
    return math.degrees(angle_rad)


def get_stem_grasp_point(mask: np.ndarray, grasp_ratio: float = 0.7):
    """
    輸入單一果梗的二值 mask，回傳 (cx, cy, angle_deg, ordered_path, grasp_idx)。
    angle_deg: 果梗生長方向角(度)，0°=垂直，正值=順時鐘偏轉(2D，左右歪)。
    ordered_path / grasp_idx: 骨架路徑與抓取點索引，之後拿去配深度算 3D 俯角。
    任何一步失敗就回傳 None。
    """
    clean, eroded, skeleton = skeletonize_pedicel(mask)
    if skeleton is None:
        return None
    endpoints = find_skeleton_endpoints(skeleton)
    if len(endpoints) < 2:
        return None
    branch_end, calyx_end = identify_endpoints_simple(endpoints)
    ordered_path = order_skeleton_path(skeleton, calyx_end)
    if len(ordered_path) == 0:
        return None
    grasp_point, grasp_idx = find_grasp_point_pixel(ordered_path, grasp_ratio)
    gy, gx = grasp_point
    angle_deg = compute_growth_angle(ordered_path, grasp_idx)
    return int(gx), int(gy), angle_deg, ordered_path, grasp_idx


def estimate_pedicel_direction(ordered_path, grasp_idx, depth_img, fx, fy, ux, uy, trans,
                               path_window: int = 12, end_span: int = 5, roi: int = 2,
                               min_pts_per_end: int = 3,
                               el_min_deg: float = 0.0, el_max_deg: float = 80.0):
    """
    「兩端三角形」法求果梗真正的 3D 走向，回傳 (azimuth_deg, elevation_deg)：
        azimuth  = 果梗在世界水平面上的方位角(atan2(dy,dx))；往前斜/往後斜就靠它區分
        elevation= 往下傾的仰角(0=水平, 90=直直往下)，夾在 [el_min, el_max]
    方向定義為 branch -> calyx(往果實、通常往下) = 夾爪要戳進去的方向。

    ★ 跟舊版差別：舊版只回一個純量 pitch，用 hypot 把水平距離取絕對值，
      結果「往前斜」和「往後斜」會算出同一個正值(分不出方向)、垂直果梗也被夾成 70°。
      現在把方向拆成 az(朝哪) + el(多陡) 兩個量，兩個問題都解掉。

    任一端有效點不足回 None(外層會退回「朝目標水平接近」)。
    ordered_path[0]=calyx端(靠果實), ordered_path[-1]=branch端(遠離果實)。
    反投影慣例與主流程 color_callback 完全相同。
    """
    n = len(ordered_path)
    if n < 3:
        return None
    H, W = depth_img.shape[:2]
    lo = max(0, grasp_idx - path_window)
    hi = min(n - 1, grasp_idx + path_window)

    def cluster_world_point(k_start, k_end):
        """把 [k_start, k_end] 這段骨架點逐一反投影，回傳世界座標的中位數點(不足則 None)。"""
        pts = []
        for k in range(k_start, k_end + 1):
            py, px = ordered_path[k]
            if not (0 <= px < W and 0 <= py < H):
                continue
            y0 = max(0, py - roi); y1 = min(H, py + roi + 1)
            x0 = max(0, px - roi); x1 = min(W, px + roi + 1)
            d = depth_img[y0:y1, x0:x1]
            d = d[d > 0]
            if d.size == 0:
                continue
            z = float(np.median(d))
            if z > 10.0:      # 單位是 mm 就換成 m(跟主流程同邏輯)
                z /= 1000.0
            if z <= 0.01:
                continue
            lp = PointStamped()
            lp.header.frame_id = 'camera_optical_frame'
            lp.point.x = -float((px - ux) * z / fx)
            lp.point.y = -float((py - uy) * z / fy)
            lp.point.z = float(z)
            wp = tf2_geometry_msgs.do_transform_point(lp, trans)
            pts.append([wp.point.x, wp.point.y, wp.point.z])
        if len(pts) < min_pts_per_end:
            return None
        return np.median(np.asarray(pts, dtype=float), axis=0)

    calyx_pt  = cluster_world_point(lo, min(hi, lo + end_span - 1))       # A：靠果實端
    branch_pt = cluster_world_point(max(lo, hi - end_span + 1), hi)       # B：靠枝條端
    if calyx_pt is None or branch_pt is None:
        return None

    # delta = B -> A(branch -> calyx，戳進去的方向)
    delta = calyx_pt - branch_pt
    dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
    horiz = math.hypot(dx, dy)
    if horiz < 1e-6 and abs(dz) < 1e-6:
        return None                       # A、B 幾乎重合，量不出方向

    az_deg = math.degrees(math.atan2(dy, dx))          # 方位角：往哪個水平方向
    el_deg = math.degrees(math.atan2(-dz, horiz))      # 仰角：往下多陡(dz 越負越陡)
    el_deg = max(el_min_deg, min(el_max_deg, el_deg))
    return az_deg, el_deg


def suppress_overlapping_masks(indices, mask_data_all, target_wh, overlap_thresh: float = 0.5):
    """
    同類 mask 去重：若兩個 mask 重疊比例過高(視為同一條梗被辨識成兩個)，只保留面積大的那個。
    重疊比例用 overlap coefficient = 交集 / 較小者面積，對「大框幾乎蓋住小框」特別敏感。

    indices      : 要比較的偵測索引清單(這裡是所有果梗的 index)
    mask_data_all: r.masks.data 轉出來的 numpy(每個 mask 是低解析度浮點圖)
    target_wh    : (w_img, h_img)，把 mask 縮回影像大小用
    overlap_thresh: 交集/較小面積 超過這個比例就當成重複(預設 0.5)

    回傳「要保留」的 index 集合(set)。
    """
    info = []
    for i in indices:
        m = cv2.resize(mask_data_all[i], target_wh, interpolation=cv2.INTER_NEAREST)
        mb = m > 0.5
        area = int(mb.sum())
        if area > 0:
            info.append((i, mb, area))

    # 面積大的先進來當「保留」，後面重疊到它的小的就被吃掉
    info.sort(key=lambda t: t[2], reverse=True)
    keep_idxs = []
    kept = []  # [(mask_bool, area), ...]
    for i, mb, area in info:
        is_dup = False
        for kmb, karea in kept:
            inter = int(np.logical_and(mb, kmb).sum())
            smaller = min(area, karea)
            if smaller > 0 and inter / smaller >= overlap_thresh:
                is_dup = True   # 跟已保留的某個高度重疊 -> 這個(較小的)丟掉
                break
        if not is_dup:
            keep_idxs.append(i)
            kept.append((mb, area))
    return set(keep_idxs)


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()                                      
        
        self.get_logger().info('正在載入 YOLOv11 果梗+番茄分割模型...')
        self.model = YOLO('/home/terry/Desktop/stem_isaac_train/runs/segment/tomato_stem/exp1-2/weights/best.pt')
        self.stem_class_id = 0
        self.tomato_class_id = 1
        # 準備一個「靜態廣播器」，用來告訴系統相機和手臂之間的固定位置關係。
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        # 呼叫自己寫的函數，發布座標架構
        # self.make_tf_bridge()  # 目前函式定義被註解掉了，先關閉呼叫
        self.make_camera_tf()                                         
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self) 
        # 接收彩色影像，收到後交給 color_callback 處理
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw', self.color_callback, 10)
        # 接收深度影像，收到後交給 depth_callback 處理
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        # 接收相機參數，收到後交給 info_callback 處理
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        # 接收手臂狀態，收到後交給 status_callback
        self.status_sub = self.create_subscription(String, '/robot_status', self.status_callback, 10)
        # 把算好的目標位置發送到 /target_pose 頻道
        self.target_pub = self.create_publisher(PoseStamped, '/target_pose', 10)
        
        self.task_completed = False 
        self.camera_info = None        
        self.latest_depth_img = None  
        self.is_processing = False   # 旗標：改為自動流程控制鎖
        self._last_scan_print = 0.0  # 節流：上次印偵測清單的時間
        self.latest_targets = []
        self.latest_tomatoes = []
        
        self.get_logger().info('YOLOv11 視覺大腦啟動！全自動掃描模式開啟...')

    def status_callback(self, msg):
        # 如果手臂說 (DONE) ，就把狀態改成 True。
        if msg.data == 'DONE':
            self.task_completed = True

    # 把人類角度轉成機器人角度的公式。
    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        return qx, qy, qz, qw
    '''
    # 定義「世界 (world)」跟「手臂基座 (base)」的關係
    def make_tf_bridge(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = 0.0, 0.0, 0.0
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)
    '''
    # 定義「手臂末端 (link_6)」跟「相機 (camera_optical_frame)」的相對位置
    def make_camera_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'link_6'
        t.child_frame_id = 'camera_optical_frame'
        
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.075
        t.transform.translation.z = 0.0521
        
        roll, pitch, yaw = 0.0, 0.0, 0.0
        qx, qy, qz, qw = self.euler_to_quaternion(math.radians(roll), math.radians(pitch), math.radians(yaw))
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = qx, qy, qz, qw
        self.tf_static_broadcaster.sendTransform(t)

    # 收到相機參數，就把它存到變數裡備用
    def info_callback(self, msg):
        self.camera_info = msg   
    # 把 ROS 傳來的深度訊息，翻譯成 OpenCV 的影像矩陣存起來
    def depth_callback(self, msg):
        try:
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"深度影像轉換失敗: {e}")

    def color_callback(self, msg):
        if self.camera_info is None or self.latest_depth_img is None: return 
        try:
            trans = self.tf_buffer.lookup_transform('world', 'camera_optical_frame', rclpy.time.Time()) 
        except TransformException: return 
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") 
        except Exception: return
        
        # ===== 單一模型推論，依 class id 分流番茄和果梗 =====
        results = self.model.predict(cv_image, imgsz=1024, conf=0.25, verbose=False)
        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        ux_img, uy_img = self.camera_info.k[2], self.camera_info.k[5]

        detected_tomatoes = []
        detected_objects = []

        for r in results:
            if r.boxes is None:
                continue
            cls_ids = r.boxes.cls.cpu().numpy()
            boxes_all = r.boxes.xyxy.cpu().numpy()
            has_masks = r.masks is not None
            if has_masks:
                mask_data_all = r.masks.data.cpu().numpy()
            h_img, w_img = cv_image.shape[:2]

            # ★ 果梗去重：同一條梗若被辨識成兩個(區域高度重疊)，只留面積大的那個
            stem_idxs = [j for j in range(len(cls_ids)) if int(cls_ids[j]) == self.stem_class_id]
            if has_masks and len(stem_idxs) > 1:
                keep_stem_idxs = suppress_overlapping_masks(stem_idxs, mask_data_all, (w_img, h_img))
            else:
                keep_stem_idxs = set(stem_idxs)

            for i in range(len(cls_ids)):
                b = boxes_all[i]
                cls_id = int(cls_ids[i])

                # ----- 番茄 (class 1)：用 bbox 中心 + 半徑補償 -----
                if cls_id == self.tomato_class_id:
                    cx_t = int((b[0] + b[2]) / 2)
                    cy_t = int((b[1] + b[3]) / 2)

                    if 0 <= cx_t < self.latest_depth_img.shape[1] and 0 <= cy_t < self.latest_depth_img.shape[0]:
                        window = 5
                        y_min = max(0, cy_t - window)
                        y_max = min(self.latest_depth_img.shape[0], cy_t + window + 1)
                        x_min = max(0, cx_t - window)
                        x_max = min(self.latest_depth_img.shape[1], cx_t + window + 1)
                        depth_roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
                        valid_depths = depth_roi[depth_roi > 0]

                        if len(valid_depths) > 0:
                            z_t = float(np.median(valid_depths))
                            if z_t > 10.0: z_t /= 1000.0

                            if z_t > 0.01:
                                w_pixel = b[2] - b[0]
                                h_pixel = b[3] - b[1]
                                avg_pixel_size = (w_pixel + h_pixel) / 2.0
                                f_avg = (fx + fy) / 2.0
                                tomato_radius = (avg_pixel_size * z_t) / f_avg / 2.0
                                z_center_t = z_t + tomato_radius

                                lp = PointStamped()
                                lp.header.frame_id = 'camera_optical_frame'
                                lp.header.stamp = self.get_clock().now().to_msg()
                                lp.point.x = -float((cx_t - ux_img) * z_center_t / fx)
                                lp.point.y = -float((cy_t - uy_img) * z_center_t / fy)
                                lp.point.z = float(z_center_t)
                                wp = tf2_geometry_msgs.do_transform_point(lp, trans)

                                detected_tomatoes.append({
                                    'cx': cx_t, 'cy': cy_t,
                                    'world_x': wp.point.x, 'world_y': wp.point.y, 'world_z': wp.point.z
                                })

                                cv2.rectangle(cv_image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 165, 255), 2)
                                cv2.circle(cv_image, (cx_t, cy_t), 5, (0, 165, 255), -1)
                                cv2.putText(cv_image, f"Tomato d={z_t:.3f}m",
                                            (int(b[0]), int(b[3]) + 15),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                                cv2.putText(cv_image, f"X={wp.point.x:.3f} Y={wp.point.y:.3f} Z={wp.point.z:.3f}",
                                            (int(b[0]), int(b[3]) + 35),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # ----- 果梗 (class 0)：用骨架抓取點 -----
                elif cls_id == self.stem_class_id and has_masks:
                    if i not in keep_stem_idxs:
                        continue   # 這個梗跟另一個更大的梗高度重疊，被去重丟掉
                    m_resized = cv2.resize(mask_data_all[i], (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (m_resized > 0.5).astype(np.uint8) * 255

                    grasp = get_stem_grasp_point(mask_bin)
                    if grasp is None:
                        continue
                    cx_pixel, cy_pixel, stem_angle, ordered_path, grasp_idx = grasp

                    # 用深度還原果梗真正的 3D 走向：方位角 az(往哪) + 仰角 el(多陡往下)
                    # 失敗(深度破洞/點太少)先設 None，等下面拿到 world 座標再退回「朝基座方位、水平」
                    pedicel_dir = estimate_pedicel_direction(
                        ordered_path, grasp_idx, self.latest_depth_img,
                        fx, fy, ux_img, uy_img, trans)

                    if 0 <= cx_pixel < self.latest_depth_img.shape[1] and 0 <= cy_pixel < self.latest_depth_img.shape[0]:
                        window = 5
                        y_min = max(0, cy_pixel - window)
                        y_max = min(self.latest_depth_img.shape[0], cy_pixel + window + 1)
                        x_min = max(0, cx_pixel - window)
                        x_max = min(self.latest_depth_img.shape[1], cx_pixel + window + 1)

                        depth_roi = self.latest_depth_img[y_min:y_max, x_min:x_max]
                        valid_depths = depth_roi[depth_roi > 0]

                        if len(valid_depths) > 0:
                            z_real = float(np.median(valid_depths))
                            if z_real > 10.0: z_real /= 1000.0

                            if z_real > 0.01:
                                z_center = z_real

                                local_point = PointStamped()
                                local_point.header.frame_id = 'camera_optical_frame'
                                local_point.header.stamp = self.get_clock().now().to_msg()
                                local_point.point.x = -float((cx_pixel - ux_img) * z_center / fx)
                                local_point.point.y = -float((cy_pixel - uy_img) * z_center / fy)
                                local_point.point.z = float(z_center)

                                world_point = tf2_geometry_msgs.do_transform_point(local_point, trans)

                                # 決定接近軸方向：估到就用果梗真實方向；估不到就退回
                                # 「朝基座方位、水平」(az=面對抓取點的方位, el=0)，等同舊的水平接近
                                if pedicel_dir is not None:
                                    az_deg, el_deg = pedicel_dir
                                else:
                                    az_deg = math.degrees(math.atan2(world_point.point.y,
                                                                     world_point.point.x))
                                    el_deg = 0.0

                                detected_objects.append({
                                    'bbox': b,
                                    'cx': cx_pixel, 'cy': cy_pixel,
                                    'z_real': z_real, 'z_center': z_center,
                                    'world_x': world_point.point.x,
                                    'world_y': world_point.point.y,
                                    'world_z': world_point.point.z,
                                    'angle': stem_angle,
                                    'az': az_deg,
                                    'el': el_deg
                                })

        self.latest_tomatoes = detected_tomatoes
        if len(detected_objects) > 0:
            # 依深度排序，最近的在最前面
            detected_objects = sorted(detected_objects, key=lambda obj: obj['z_real'])
            self.latest_targets = detected_objects 
            
            for idx, obj in enumerate(detected_objects):
                b = obj['bbox']
                cv2.rectangle(cv_image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.circle(cv_image, (obj['cx'], obj['cy']), 6, (255, 0, 0), -1)  # 藍色抓取點
                cv2.putText(cv_image, f"Stem ID:{idx} d={obj['z_real']:.3f}m ang={obj['angle']:.1f}° az={obj.get('az',0.0):.0f}° el={obj.get('el',0.0):.0f}°", (int(b[0]), int(b[1]) - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(cv_image, f"X={obj['world_x']:.3f} Y={obj['world_y']:.3f} Z={obj['world_z']:.3f}", 
                            (int(b[0]), int(b[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 偵測清單照印(節流，避免每格畫面洗版)——這段跟手臂歸位無關，隨時看得到偵測結果。
            # 只有「跳出來叫你輸入 ID 夾取」才需要等手臂回報歸位(task_completed)。
            now = time.time()
            if not self.is_processing and (now - self._last_scan_print) > 1.5:
                self._last_scan_print = now
                tomatoes = self.latest_tomatoes
                print("\n" + "="*60)
                print(f"偵測到 {len(detected_objects)} 個果梗 / {len(tomatoes)} 個番茄")
                print("-"*60)
                for idx, obj in enumerate(detected_objects):
                    print(f"  [ID:{idx}] Stem   X={obj['world_x']:.3f}, Y={obj['world_y']:.3f}, Z={obj['world_z']:.3f}, Angle={obj['angle']:.1f}°, Az={obj.get('az',0.0):.1f}°, El={obj.get('el',0.0):.1f}°")
                    # 找最近的番茄配對顯示
                    if len(tomatoes) > 0:
                        dists = [math.sqrt((obj['cx']-t['cx'])**2 + (obj['cy']-t['cy'])**2) for t in tomatoes]
                        nearest = tomatoes[int(np.argmin(dists))]
                        print(f"         Tomato X={nearest['world_x']:.3f}, Y={nearest['world_y']:.3f}, Z={nearest['world_z']:.3f}")
                print("="*60)
                # 手臂已歸位才真正啟動夾取(跳出輸入 ID)；還沒歸位就只顯示、先不夾。
                if self.task_completed:
                    self.is_processing = True
                    threading.Thread(target=self.auto_pick_thread).start()
                else:
                    print("(手臂尚未歸位，暫不夾取，持續掃描中…座標僅供參考)")

        cv2.imshow("YOLOv11 Realtime Vision", cv_image)
        cv2.waitKey(1)
        
    def auto_pick_thread(self):
        targets = self.latest_targets
        if not targets:
            self.is_processing = False
            return

        while True:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            user_input = input("請輸入要夾取的果梗 ID (輸入 r 重新掃描): ").strip().lower()
            
            # 按r, 結束這個函式，回到主程式重新辨識
            if user_input == 'r':
                self.is_processing = False
                return
            
            # 檢查輸入是不是數字，且數字有沒有在番茄清單的範圍內
            if user_input.isdigit() and 0 <= int(user_input) < len(targets):
                selected_id = int(user_input)
                target = targets[selected_id]
                
                distance_to_base = math.sqrt(target['world_x']**2 + target['world_y']**2 + target['world_z']**2)
                max_reach = 1.0

                if distance_to_base > max_reach:
                    print("\n" + "!"*50)
                    print(f" 目標距離基座 {distance_to_base:.3f} 公尺 ，已經超過安全工作範圍")
                    #print(f" 已經超過安全工作範圍 ({max_reach} 公尺)，強行夾取可能會導致 IK 運算失敗或手臂卡死。")
                    #print(" 請選擇其他較近的番茄，或輸入 r 重新掃描。")
                    print("!"*50 + "\n")
                    continue  # 跳過下面的發送指令，回到迴圈開頭讓使用者重新輸入

                print("\n" + "*"*40)
                print(f"目標鎖定！計算含厚度的中心深度 Z: {target['z_center']:.3f} 公尺")
                print(f"發送絕對座標: X={target['world_x']:.3f}, Y={target['world_y']:.3f}")
                print("*"*40)

                # 把三個角度打包成四元數送給手臂端(euler 慣例 'xyz'，手臂端用同慣例解回)：
                #   roll(x)      = 果梗左右歪 -> 夾爪繞接近軸的 roll，對準開口
                #   elevation(y) = 果梗往下傾多陡 -> 接近軸的仰角
                #   azimuth(z)   = 果梗水平方位 -> 接近軸的水平朝向(往前/往後就靠它區分)
                # 註：手臂端 J1 約束用的「面對目標方位」是它自己用位置算的，跟這裡的 azimuth 無關
                stem_angle_deg = target.get('angle', 0.0)
                el_deg = target.get('el', 0.0)
                az_deg = target.get('az', 0.0)
                qx, qy, qz, qw = self.euler_to_quaternion(
                    math.radians(stem_angle_deg), math.radians(el_deg), math.radians(az_deg))

                # 填寫要發送給手臂的消息包，包含目標位置和果梗角度
                target_msg = PoseStamped()
                target_msg.header.stamp = self.get_clock().now().to_msg()
                target_msg.header.frame_id = 'world'
                target_msg.pose.position.x = target['world_x']  
                target_msg.pose.position.y = target['world_y']
                target_msg.pose.position.z = target['world_z'] 
                target_msg.pose.orientation.x = float(qx)
                target_msg.pose.orientation.y = float(qy)
                target_msg.pose.orientation.z = float(qz)
                target_msg.pose.orientation.w = float(qw)
                    
                self.target_pub.publish(target_msg)
                
                print("夾取指令已發出！等待手臂完成動作...")
                self.task_completed = False 
                while not self.task_completed:
                    time.sleep(1.0) 
                
                print("\n手臂動作完成！重新啟動 YOLO 掃描...\n")
                self.is_processing = False 
                return
            else:
                print("無效的 ID。")

def main(args=None):
    rclpy.init(args=args)          
    node = VisionNode()            
    try: 
        rclpy.spin(node)          
    except KeyboardInterrupt: 
        pass 
    finally:
        node.destroy_node()        
        cv2.destroyAllWindows()    
        if rclpy.ok():
            rclpy.shutdown()           

if __name__ == '__main__':          
    main()
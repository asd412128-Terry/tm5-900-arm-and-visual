"""
============================================================================
 粗定位偵測：車載相機影像 → 番茄在 base 座標系下的 3D 位置
============================================================================
 只負責「辨識 + 座標換算」：跑 YOLO 取番茄類別、每顆用 mask 內深度中位數反投影，
 再用傳入的 tf transform 轉到 base。不碰 ROS 訂閱/發布，TF 由呼叫端 (coarse_node)
 查好傳進來，跟 fine_detector.py / coordinates.py 同一個分工。
 番茄中心深度的算法（表面深度 + 半徑修正）跟 fine_detector.py 的 _process_tomato_detection
 完全一致，兩顆相機量出來的番茄座標才是同一套定義。
============================================================================
"""

import math

import cv2
import numpy as np
import tf2_geometry_msgs
from ultralytics import YOLO

from .config import MIN_VALID_DEPTH_M, TOMATO_CLASS_ID, VEHICLE_CAMERA_OPTICAL_FRAME
from .coordinates import CoordinateEstimator

"""載入 YOLO 模型，把車載相機一幀影像跑成「番茄 list（含 base 座標）」。"""
class TomatoDetector:

    def __init__(self, model_path: str, coordinate_estimator: CoordinateEstimator = None,
                 tomato_class_id: int = TOMATO_CLASS_ID):
        self.model = YOLO(model_path)
        self.tomato_class_id = tomato_class_id
        self.coord = coordinate_estimator or CoordinateEstimator()

    """跑一次 YOLO 推論，回傳原始 results。"""
    def predict(self, cv_image, imgsz, conf, iou):
        return self.model.predict(cv_image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

    """跑完一次 YOLO 結果，回傳這一幀所有番茄的 list。每顆是 dict：
    cx/cy(bbox 中心像素)、bbox、depth(表面深度 m)、z_center(加番茄半徑後的中心深度 m)、
    base_x/base_y/base_z(base 座標系下的番茄中心 m)、conf、mask。
    沒有 mask 的偵測、深度量不到、座標算出 NaN 的番茄都直接略過。"""
    def detect(self, cv_image, results, fx, fy, ux, uy, trans, depth_img, stamp=None):
        tomatoes = []
        h_img, w_img = cv_image.shape[:2]

        for r in results:
            if r.boxes is None or r.masks is None:
                continue
            cls_ids = r.boxes.cls.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            masks = r.masks.data.cpu().numpy()

            for i in range(len(cls_ids)):
                if int(cls_ids[i]) != self.tomato_class_id:
                    continue
                tomato = self._locate_tomato(boxes[i], masks[i], float(confs[i]),
                                              fx, fy, ux, uy, trans, depth_img,
                                              w_img, h_img, stamp)
                if tomato is not None:
                    tomatoes.append(tomato)
        return tomatoes

    """單顆番茄：mask 深度中位數 → 加半徑修正 → 反投影 → 轉 base。任何一步失敗回傳 None。"""
    def _locate_tomato(self, box, raw_mask, conf, fx, fy, ux, uy, trans, depth_img,
                        w_img, h_img, stamp):
        m_resized = cv2.resize(raw_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mask_bin = (m_resized > 0.5).astype(np.uint8) * 255

        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)

        # 整個 mask 取深度中位數，不用 bbox 中心點開視窗：中心點可能剛好落在缺角/遮擋縫隙裡
        # （見 fine_detector.py 同一段的說明）。
        z_surface = self.coord.median_depth_in_mask(depth_img, mask_bin, 1)
        if z_surface is None:
            return None
        z_surface = self.coord.to_meters(z_surface)
        if z_surface <= MIN_VALID_DEPTH_M:
            return None

        avg_pixel_size = ((box[2] - box[0]) + (box[3] - box[1])) / 2.0
        f_avg = (fx + fy) / 2.0
        radius = (avg_pixel_size * z_surface) / f_avg / 2.0
        z_center = z_surface + radius

        local_pt = self.coord.backproject_to_local_point(
            cx, cy, z_center, fx, fy, ux, uy,
            frame_id=VEHICLE_CAMERA_OPTICAL_FRAME, stamp=stamp)
        base_pt = tf2_geometry_msgs.do_transform_point(local_pt, trans)
        x, y, z = base_pt.point.x, base_pt.point.y, base_pt.point.z
        if not all(math.isfinite(v) for v in (x, y, z)):
            return None

        return {
            'cx': cx, 'cy': cy, 'bbox': box,
            'depth': z_surface, 'z_center': z_center,
            'base_x': x, 'base_y': y, 'base_z': z,
            'conf': conf, 'mask': mask_bin,
        }

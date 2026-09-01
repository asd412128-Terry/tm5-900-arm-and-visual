"""
============================================================================
 物件偵測：YOLOv11 分割模型 → 番茄 / 果梗偵測結果
============================================================================
 負責「辨識」這一層：跑模型、篩選重疊偵測、把每個偵測框交給
 CoordinateEstimator / PedicelSkeletonizer 算出世界座標與抓取資訊，
 組成統一格式的 dict 回傳。不碰 ROS 訂閱/發布。
============================================================================
"""

import cv2
import math
import numpy as np
import tf2_geometry_msgs
from ultralytics import YOLO
from .config import (GRASP_RATIO_MAX, GRASP_RATIO_MIN, GRASP_TARGET_DIST_M,
                      MIN_VALID_DEPTH_M, OVERLAP_SUPPRESS_THRESH, STEM_CLASS_ID,
                      TOMATO_CLASS_ID, DEPTH_WINDOW, MIN_STEM_DEPTH_PX)
from .coordinates import CoordinateEstimator
from .occlusion import OcclusionChecker
from .skeleton import PedicelSkeletonizer

"""載入 YOLO 模型，把一幀影像跑成 (detected_tomatoes, detected_objects)。"""
class ObjectDetector:

    """載入 YOLO 模型，設定分類 ID 與座標/骨架化/遮擋判斷輔助模組。"""
    def __init__(self, model_path: str,
                 coordinate_estimator: CoordinateEstimator = None,
                 skeletonizer: PedicelSkeletonizer = None,
                 occlusion_checker: OcclusionChecker = None,
                 stem_class_id: int = STEM_CLASS_ID,
                 tomato_class_id: int = TOMATO_CLASS_ID,
                 min_stem_depth_px: int = MIN_STEM_DEPTH_PX):
        self.model = YOLO(model_path)
        self.stem_class_id = stem_class_id
        self.tomato_class_id = tomato_class_id
        self.min_stem_depth_px = min_stem_depth_px
        self.coord = coordinate_estimator or CoordinateEstimator()
        self.skeletonizer = skeletonizer or PedicelSkeletonizer()
        self.occlusion = occlusion_checker or OcclusionChecker()

    """多個果梗 mask 互相重疊超過門檻時，只保留面積較大的那個 (避免同一根果梗被偵測兩次)。"""
    @staticmethod
    def suppress_overlapping_masks(indices, mask_data_all, target_wh, overlap_thresh: float = OVERLAP_SUPPRESS_THRESH):
        info = []
        for i in indices:
            m = cv2.resize(mask_data_all[i], target_wh, interpolation=cv2.INTER_NEAREST)
            mb = m > 0.5
            area = int(mb.sum())
            if area > 0:
                info.append((i, mb, area))

        info.sort(key=lambda t: t[2], reverse=True)
        keep_idxs = []
        kept = []
        for i, mb, area in info:
            is_dup = False
            for kmb, karea in kept:
                smaller = min(area, karea)
                inter = int(np.logical_and(mb, kmb).sum())
                if smaller > 0 and inter / smaller >= overlap_thresh:
                    is_dup = True
                    break
            if not is_dup:
                keep_idxs.append(i)
                kept.append((mb, area))
        return set(keep_idxs)
    
    """跑一次 YOLO 推論，回傳原始 results（給呼叫端需要時使用）。"""
    def predict(self, cv_image, imgsz, conf):
        return self.model.predict(cv_image, imgsz=imgsz, conf=conf, verbose=False)

    """跑完一次 YOLO 結果，回傳 (detected_tomatoes, detected_objects)。不畫框（畫框交給 visualizer）。"""
    def detect(self, cv_image, results, fx, fy, ux_img, uy_img, trans, depth_img, stamp=None):
        detected_tomatoes = []
        detected_objects = []

        for r in results:
            if r.boxes is None:
                continue
            cls_ids = r.boxes.cls.cpu().numpy()
            boxes_all = r.boxes.xyxy.cpu().numpy()
            confs_all = r.boxes.conf.cpu().numpy()
            has_masks = r.masks is not None
            mask_data_all = r.masks.data.cpu().numpy() if has_masks else None
            h_img, w_img = cv_image.shape[:2]

            stem_idxs = [j for j in range(len(cls_ids)) if int(cls_ids[j]) == self.stem_class_id]
            if has_masks and len(stem_idxs) > 1:
                keep_stem_idxs = self.suppress_overlapping_masks(stem_idxs, mask_data_all, (w_img, h_img))
            else:
                keep_stem_idxs = set(stem_idxs)

            # 遮擋判斷需要「同一幀所有其他 instance 的 bbox」，先把番茄/果梗的 bbox 收集起來
            tomato_idxs = [j for j in range(len(cls_ids)) if int(cls_ids[j]) == self.tomato_class_id]
            tomato_boxes_all = [tuple(boxes_all[j]) for j in tomato_idxs]
            stem_boxes_kept = [tuple(boxes_all[j]) for j in keep_stem_idxs]

            for local_i, i in enumerate(tomato_idxs):
                other_boxes = [bx for k, bx in enumerate(tomato_boxes_all) if k != local_i] + stem_boxes_kept
                self._process_tomato_detection(boxes_all[i], fx, fy, ux_img, uy_img, trans, depth_img,
                                                detected_tomatoes,
                                                mask_data_all[i] if has_masks else None,
                                                w_img, h_img, stamp, other_boxes)

            if has_masks:
                for i in sorted(keep_stem_idxs):
                    self._process_stem_detection(boxes_all[i], i, mask_data_all, w_img, h_img,
                                                  fx, fy, ux_img, uy_img, trans, depth_img,
                                                  confs_all, detected_objects, stamp)

        return detected_tomatoes, detected_objects

    """單一番茄 (class 1) 偵測框 → 算世界座標 + 遮擋判斷、寫進 detected_tomatoes。
    需要有 mask 才能取深度 (raw_mask 沒給就跳過這個偵測)；mask 同時存進 dict 供選定為
    目標時的點雲過濾用。畫框交給 visualizer（需要本幀完整果梗清單才能判斷配對）。"""
    def _process_tomato_detection(self, b, fx, fy, ux_img, uy_img, trans, depth_img,
                                   detected_tomatoes, raw_mask=None, w_img=None, h_img=None, stamp=None,
                                   other_boxes=None):
        if raw_mask is None or w_img is None or h_img is None:
            return

        m_resized = cv2.resize(raw_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mask_bin_t = (m_resized > 0.5).astype(np.uint8) * 255

        metrics = self.occlusion.compute_shape_metrics(mask_bin_t)
        occluded, occlusion_reason, _ = self.occlusion.judge_occlusion(metrics, tuple(b), other_boxes or [])

        cx_t = int((b[0] + b[2]) / 2)
        cy_t = int((b[1] + b[3]) / 2)

        # 整個 mask 取深度中位數，不用 bbox 中心點開視窗——番茄被遮擋時常常是中間缺一塊
        # (mask 變成不規則形狀)，bbox 中心點很容易剛好落在缺角/遮擋縫隙裡，導致視窗內完全
        # 沒有 mask 像素、直接判定失敗，整顆番茄就這樣消失掉（跟果梗那邊踩到的是同一種坑）。
        z_t = self.coord.median_depth_in_mask(depth_img, mask_bin_t, 1)
        if z_t is None:
            return
        z_t = self.coord.to_meters(z_t)
        if z_t <= MIN_VALID_DEPTH_M:
            return

        w_pixel, h_pixel = b[2] - b[0], b[3] - b[1]
        avg_pixel_size = (w_pixel + h_pixel) / 2.0
        f_avg = (fx + fy) / 2.0
        tomato_radius = (avg_pixel_size * z_t) / f_avg / 2.0
        z_center_t = z_t + tomato_radius

        lp = self.coord.backproject_to_local_point(cx_t, cy_t, z_center_t, fx, fy, ux_img, uy_img, stamp=stamp)
        wp = tf2_geometry_msgs.do_transform_point(lp, trans)

        if not all(math.isfinite(v) for v in (wp.point.x, wp.point.y, wp.point.z)):
            return

        detected_tomatoes.append({
            'cx': cx_t, 'cy': cy_t,
            'bbox': b,
            'world_x': wp.point.x, 'world_y': wp.point.y, 'world_z': wp.point.z,
            'depth': z_t,   # 表面深度 (相機讀到的原始深度，公尺，尚未加上番茄半徑)
            'mask': mask_bin_t,
            'occluded': occluded,
            'occlusion_reason': occlusion_reason,
        })
        # 番茄框框顏色（要不要標紅/綠）要看它有沒有配對到果梗，這件事只有拿到本幀
        # 全部果梗清單後才能判斷，所以畫框改到 visualizer.draw_tracked_overlay 那邊做。

    """單一果梗 (class 0) 偵測框 → 骨架化找抓取點、算 3D 方向向量與世界座標，寫進 detected_objects。"""
    def _process_stem_detection(self, b, i, mask_data_all, w_img, h_img,
                                 fx, fy, ux_img, uy_img, trans, depth_img,
                                 confs_all, detected_objects, stamp=None):
        m_resized = cv2.resize(mask_data_all[i], (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mask_bin = (m_resized > 0.5).astype(np.uint8) * 255

        # 粗略深度：對果梗「骨架中點」附近的 mask 深度取中位數，只為了把 GRASP_TARGET_DIST_M
        # （實際公尺距離）換算成這一幀畫面裡對應的像素距離，交給骨架化去找抓取點。真正的抓取點
        # 深度在骨架化完成、拿到精確像素座標後才重新量一次（見下面的 z_real）。
        # ★ 不能用 bbox 中心點當錨點去抓——果梗常常是彎的，bbox 中心點很容易根本沒落在
        #   細長的 mask 上，導致這步空手而回、整根果梗在骨架化之前就被誤判丟掉。改用骨架
        #   路徑的中點，保證錨點落在果梗本體上。
        _, _, approx_skeleton = self.skeletonizer.skeletonize_pedicel(mask_bin)
        if approx_skeleton is None:
            return
        approx_path = self.skeletonizer.order_skeleton_path(approx_skeleton)
        if len(approx_path) < 2:
            return
        mid_y, mid_x = approx_path[len(approx_path) // 2]
        approx_z = self.coord.median_depth_in_mask(depth_img, mask_bin, self.min_stem_depth_px,
                                                     cx=mid_x, cy=mid_y)
        if approx_z is None:
            return
        approx_z = self.coord.to_meters(approx_z)
        if approx_z <= MIN_VALID_DEPTH_M:
            return
        target_px_dist = GRASP_TARGET_DIST_M * fx / approx_z

        grasp = self.skeletonizer.get_stem_grasp_point(mask_bin, target_px_dist,
                                                         GRASP_RATIO_MIN, GRASP_RATIO_MAX)
        if grasp is None:
            return
        cx_pixel, cy_pixel, stem_angle, ordered_path, grasp_idx = grasp
        tip_y, tip_x = ordered_path[0]   # debug：果實端端點像素座標，抓取點亂跳時比對用

        pedicel_dir = self.coord.estimate_pedicel_direction(
            ordered_path, grasp_idx, depth_img,
            fx, fy, ux_img, uy_img, trans,
            mask_bin=mask_bin)

        z_real = self.coord.median_depth_for_stem(depth_img, mask_bin, cx_pixel, cy_pixel,
                                                    DEPTH_WINDOW, self.min_stem_depth_px)
        if z_real is None:
            return
        z_real = self.coord.to_meters(z_real)
        if z_real <= MIN_VALID_DEPTH_M:
            return
        z_center = z_real

        local_point = self.coord.backproject_to_local_point(cx_pixel, cy_pixel, z_center, fx, fy, ux_img, uy_img,
                                                              stamp=stamp)
        world_point = tf2_geometry_msgs.do_transform_point(local_point, trans)

        if not all(math.isfinite(v) for v in (world_point.point.x, world_point.point.y, world_point.point.z)):
            return

        vx, vy, vz = pedicel_dir if pedicel_dir is not None else (0.0, 0.0, -1.0)

        detected_objects.append({
            'bbox': b,
            'cx': cx_pixel, 'cy': cy_pixel,
            'z_real': z_real, 'z_center': z_center,
            'world_x': world_point.point.x,
            'world_y': world_point.point.y,
            'world_z': world_point.point.z,
            'path_len': len(ordered_path),        # debug：骨架路徑點數，抓取點亂跳時比對用
            'tip_px': (tip_x, tip_y),              # debug：果實端端點像素座標
            'angle': stem_angle,
            'vx': vx, 'vy': vy, 'vz': vz,
            'conf': float(confs_all[i]),
            'mask': mask_bin,   # 果梗二值遮罩，供選定為目標時的點雲過濾用
        })

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
from .config import (GRASP_METHOD, GRASP_RATIO_MAX, GRASP_RATIO_MIN, GRASP_TARGET_DIST_M,
                      MIN_VALID_DEPTH_M, OVERLAP_SUPPRESS_THRESH, STEM_CLASS_ID,
                      TOMATO_CLASS_ID, DEPTH_WINDOW, MIN_STEM_DEPTH_PX, YOLO_IOU)
from .coordinates import CoordinateEstimator
from .occlusion import OcclusionChecker
from .skeleton import PedicelSkeletonizer
from .target_selector import TargetSelector

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
        self._prev_pairs = []   # 上一幀配對結果 [(stem_pos, tomato_pos), ...]，供配對做時間穩定用

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
    def predict(self, cv_image, imgsz, conf, iou=YOLO_IOU):
        return self.model.predict(cv_image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)

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
                # 配對跟判斷哪端是果實端(calyx)一起做：果梗兩個骨架端點分別跟每顆番茄的
                # 頂端接點算 3D 距離，全域由近到遠貪婪配對，贏的那一端就是果實端——
                # 不能每根果梗各自獨立找「最近番茄」，沒有唯一性保證，容易好幾根搶到
                # 同一顆番茄；細節見 TargetSelector.assign_stem_tomato_pairs。
                prepared = []
                for i in sorted(keep_stem_idxs):
                    p = self._prepare_stem_geometry(boxes_all[i], i, mask_data_all, w_img, h_img,
                                                      fx, fy, ux_img, uy_img, trans, depth_img, stamp)
                    if p is not None:
                        prepared.append(p)

                pairs, reverse_map = TargetSelector.assign_stem_tomato_pairs(
                    prepared, detected_tomatoes, prev_pairs=self._prev_pairs)

                new_prev_pairs = []
                for p in prepared:
                    anchor_t = pairs.get(id(p))
                    if anchor_t is None:
                        continue
                    reverse = reverse_map[id(p)]
                    fingerprint = tuple((a + b) / 2.0 for a, b in zip(p['end0_world'], p['end1_world']))
                    new_prev_pairs.append((fingerprint,
                                            (anchor_t['world_x'], anchor_t['world_y'], anchor_t['world_z'])))
                    self._finish_stem_detection(p, anchor_t, reverse, fx, fy, ux_img, uy_img, trans, depth_img,
                                                 confs_all, detected_objects, stamp)
                self._prev_pairs = new_prev_pairs

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

        # 果梗實際接在番茄「頂端」，不是幾何中心——配對用頂端接點距離比用中心準。
        # 深度不用這個像素本身量到的原始深度：bbox 頂端邊緣本來就是 mask 最容易缺角/
        # 雜訊最多的地方，直接在那裡取深度容易失敗或量到背景。改沿用跟中心點同一組
        # 半徑校正算出來的 z_center_t——頂端跟中心本來就是同一顆球面上的點，深度用
        # 同一個值只差在 x/y 的投影角度，比重新在雜訊邊緣量測穩。量不到才整顆退回中心。
        attach_world = None
        top_x, top_y = cx_t, int(b[1])
        lp_top = self.coord.backproject_to_local_point(top_x, top_y, z_center_t, fx, fy, ux_img, uy_img, stamp=stamp)
        wp_top = tf2_geometry_msgs.do_transform_point(lp_top, trans)
        if all(math.isfinite(v) for v in (wp_top.point.x, wp_top.point.y, wp_top.point.z)):
            attach_world = (wp_top.point.x, wp_top.point.y, wp_top.point.z)

        detected_tomatoes.append({
            'cx': cx_t, 'cy': cy_t,
            'bbox': b,
            'world_x': wp.point.x, 'world_y': wp.point.y, 'world_z': wp.point.z,
            'attach_world': attach_world,   # bbox 頂端中點的世界座標，供果梗配對用；失敗時為 None(配對退回中心)
            'depth': z_t,   # 表面深度 (相機讀到的原始深度，公尺，尚未加上番茄半徑)
            'mask': mask_bin_t,
            'occluded': occluded,
            'occlusion_reason': occlusion_reason,
        })
        # 番茄框框顏色（要不要標紅/綠）要看它有沒有配對到果梗，這件事只有拿到本幀
        # 全部果梗清單後才能判斷，所以畫框改到 visualizer.draw_tracked_overlay 那邊做。

    """單一像素點 + mask（用來在該點附近取深度中位數）→ 世界座標；任一步失敗回傳 None。"""
    def _pixel_to_world(self, px, py, mask_bin, depth_img, fx, fy, ux_img, uy_img, trans, stamp=None):
        z = self.coord.median_depth_in_mask(depth_img, mask_bin, self.min_stem_depth_px, cx=px, cy=py)
        if z is None:
            return None
        z = self.coord.to_meters(z)
        if z <= MIN_VALID_DEPTH_M:
            return None
        lp = self.coord.backproject_to_local_point(px, py, z, fx, fy, ux_img, uy_img, stamp=stamp)
        wp = tf2_geometry_msgs.do_transform_point(lp, trans)
        if not all(math.isfinite(v) for v in (wp.point.x, wp.point.y, wp.point.z)):
            return None
        return (wp.point.x, wp.point.y, wp.point.z)

    """第一階段：果梗 mask → 骨架化 → 兩個端點的世界座標，供 detect() 統一跟番茄配對
    （配對跟判斷哪端是果實端一起做，見 TargetSelector.assign_stem_tomato_pairs）。
    回傳 dict 或 None（骨架化/深度任一步失敗）。"""
    def _prepare_stem_geometry(self, b, i, mask_data_all, w_img, h_img,
                                fx, fy, ux_img, uy_img, trans, depth_img, stamp=None):
        m_resized = cv2.resize(mask_data_all[i], (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mask_bin = (m_resized > 0.5).astype(np.uint8) * 255

        _, _, approx_skeleton = self.skeletonizer.skeletonize_pedicel(mask_bin)
        if approx_skeleton is None:
            return None
        approx_path = self.skeletonizer.order_skeleton_path(approx_skeleton)
        if len(approx_path) < 2:
            return None

        end0_y, end0_x = approx_path[0]
        end1_y, end1_x = approx_path[-1]
        end0_world = self._pixel_to_world(end0_x, end0_y, mask_bin, depth_img, fx, fy, ux_img, uy_img, trans, stamp)
        end1_world = self._pixel_to_world(end1_x, end1_y, mask_bin, depth_img, fx, fy, ux_img, uy_img, trans, stamp)
        if end0_world is None or end1_world is None:
            return None

        return {
            'b': b, 'i': i, 'mask_bin': mask_bin,
            'end0_world': end0_world, 'end1_world': end1_world,
            'end0_px': (end0_x, end0_y), 'end1_px': (end1_x, end1_y),
        }

    """第二階段：配對階段已經決定好哪端是果實端(calyx)（reverse=True 代表 end1 是），
    這裡只管正式算抓取點、3D 方向向量與世界座標，寫進 detected_objects。
    不退回 y 座標猜方向——沒配對到番茄的果梗在 detect() 就已經被擋掉，不會走到這裡；
    猜錯方向的資料一旦混進 StemTracker 的平滑歷史，會把同一個 track 的加權平均污染掉。"""
    def _finish_stem_detection(self, prep, anchor_t, reverse, fx, fy, ux_img, uy_img, trans, depth_img,
                                confs_all, detected_objects, stamp=None):
        b, i, mask_bin = prep['b'], prep['i'], prep['mask_bin']

        # GRASP_METHOD='distance' 才需要換算 target_px_dist：calyx_px 是已經確定的 C 點，
        # 在它旁邊重新取深度換算（不是用第一階段的骨架中點深度——果梗對相機有夾角時
        # 中點深度可能跟 C 點深度差不少，換算會系統性偏）。'ratio' 模式不需要，直接跳過。
        target_px_dist = None
        if GRASP_METHOD == 'distance':
            calyx_px = prep['end1_px'] if reverse else prep['end0_px']
            calyx_z = self.coord.median_depth_in_mask(depth_img, mask_bin, self.min_stem_depth_px,
                                                        cx=calyx_px[0], cy=calyx_px[1])
            if calyx_z is None:
                return
            calyx_z = self.coord.to_meters(calyx_z)
            if calyx_z <= MIN_VALID_DEPTH_M:
                return
            target_px_dist = GRASP_TARGET_DIST_M * fx / calyx_z

        grasp = self.skeletonizer.get_stem_grasp_point(mask_bin, GRASP_RATIO_MIN, GRASP_RATIO_MAX,
                                                         target_px_dist=target_px_dist, reverse=reverse)
        if grasp is None:
            return
        cx_pixel, cy_pixel, stem_angle, ordered_path, grasp_idx = grasp
        tip_y, tip_x = ordered_path[0]      # 判定為果實端(calyx)的端點像素座標
        root_y, root_x = ordered_path[-1]   # 判定為枝條端(branch)的端點像素座標

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
            'tip_px': (tip_x, tip_y),              # 判定為果實端(calyx)的端點像素座標
            'root_px': (root_x, root_y),           # 判定為枝條端(branch)的端點像素座標
            'end0_px': prep['end0_px'],            # 兩個原始候選端點(未判斷方向前)，畫面比對用
            'end1_px': prep['end1_px'],
            'angle': stem_angle,
            'vx': vx, 'vy': vy, 'vz': vz,
            'conf': float(confs_all[i]),
            'mask': mask_bin,   # 果梗二值遮罩，供選定為目標時的點雲過濾用
            'paired_tomato': anchor_t,   # 配對只在這裡算一次，後面一律讀這個欄位，不重新配對
        })

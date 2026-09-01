"""
============================================================================
 座標計算：像素 + 深度 → 世界座標 / 果梗 3D 方向向量
============================================================================
 純運算，不碰 ROS 訂閱/發布。唯一跟外部世界的接觸點是傳入的
 tf transform（trans 參數），呼叫端（vision_node）負責查好 TF 再傳進來。
============================================================================
"""

import math
import numpy as np
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from .config import CAMERA_OPTICAL_FRAME, DEPTH_MM_THRESHOLD, MIN_VALID_DEPTH_M

"""把 (像素座標 + 深度) 轉成世界座標，並估算果梗的 3D 方向向量。"""
class CoordinateEstimator:

    """深度值若明顯是 mm 為單位 (大於 DEPTH_MM_THRESHOLD)，轉換成公尺；否則視為已是公尺。"""
    @staticmethod
    def to_meters(z: float) -> float:
        return z / 1000.0 if z > DEPTH_MM_THRESHOLD else z
    
    """用相機內參把單一像素 + 深度反投影成相機座標系下的 3D 點。"""
    @staticmethod
    def backproject_to_local_point(px, py, z, fx, fy, ux, uy,
                                    frame_id: str = CAMERA_OPTICAL_FRAME, stamp=None) -> PointStamped:
        lp = PointStamped()
        lp.header.frame_id = frame_id
        if stamp is not None:
            lp.header.stamp = stamp
        lp.point.x = float((px - ux) * z / fx)
        lp.point.y = float((py - uy) * z / fy)
        lp.point.z = float(z)
        return lp
    
    """在 (cx, cy) 附近 window×window 視窗內取有效深度的中位數 (原始單位，未轉公尺)。
    座標超出範圍或視窗內沒有有效深度時回傳 None。"""
    @staticmethod
    def median_depth_in_window(depth_img: np.ndarray, cx: int, cy: int, window: int):
        h, w = depth_img.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        y_min, y_max = max(0, cy - window), min(h, cy + window + 1)
        x_min, x_max = max(0, cx - window), min(w, cx + window + 1)
        valid = depth_img[y_min:y_max, x_min:x_max]
        valid = valid[valid > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid))
    
    """(cx, cy) 附近 11x11（window=5）ROI 內、且屬於 mask 的深度中位數（原始單位，未轉公尺）。
    不給 cx, cy 時，用「離 mask 重心最近的 mask 像素」當錨點（避免番茄缺角/果梗彎曲時，
    重心本身沒落在 mask 上）。有效像素數 < min_valid_px 或座標超出範圍時回傳 None。"""
    @staticmethod
    def median_depth_in_mask(depth_img: np.ndarray, mask_bin: np.ndarray, min_valid_px: int,
                              cx: int = None, cy: int = None, window: int = 5):
        if mask_bin.shape[:2] != depth_img.shape[:2]:
            return None
        if cx is None or cy is None:
            ys, xs = np.nonzero(mask_bin)
            if ys.size == 0:
                return None
            cy0, cx0 = float(np.mean(ys)), float(np.mean(xs))
            dist2 = (ys - cy0) ** 2 + (xs - cx0) ** 2
            nearest = int(np.argmin(dist2))
            cy, cx = int(ys[nearest]), int(xs[nearest])

        h, w = depth_img.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        y_min, y_max = max(0, cy - window), min(h, cy + window + 1)
        x_min, x_max = max(0, cx - window), min(w, cx + window + 1)
        depth_roi = depth_img[y_min:y_max, x_min:x_max]
        mask_roi = mask_bin[y_min:y_max, x_min:x_max]
        valid = depth_roi[(depth_roi > 0) & (mask_roi > 0)]
        if valid.size < min_valid_px:
            return None
        return float(np.median(valid))

    """果梗中心深度：優先用 mask 內的深度中位數；mask 內有效點數不足時退回不濾 mask 的 3x3 視窗。
    座標超出範圍或完全沒有有效深度時回傳 None。"""
    @staticmethod
    def median_depth_for_stem(depth_img: np.ndarray, mask_bin: np.ndarray, cx: int, cy: int,
                               window: int, min_valid_px: int):
        
        h, w = depth_img.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return None

        y_min, y_max = max(0, cy - window), min(h, cy + window + 1)
        x_min, x_max = max(0, cx - window), min(w, cx + window + 1)
        depth_roi = depth_img[y_min:y_max, x_min:x_max]

        valid_depths = None
        if mask_bin.shape[:2] == depth_img.shape[:2]:
            roi_mask = mask_bin[y_min:y_max, x_min:x_max]
            masked = depth_roi[(depth_roi > 0) & (roi_mask > 0)]
            if masked.size >= min_valid_px:
                valid_depths = masked

        if valid_depths is None:
            yb0, yb1 = max(0, cy - 1), min(h, cy + 2)
            xb0, xb1 = max(0, cx - 1), min(w, cx + 2)
            fb = depth_img[yb0:yb1, xb0:xb1]
            valid_depths = fb[fb > 0]

        if valid_depths.size == 0:
            return None
        return float(np.median(valid_depths))
    
    """計算果梗方向向量 (branch -> calyx)，回傳 3D 單位向量 (vx, vy, vz)。"""
    def estimate_pedicel_direction(self, ordered_path, grasp_idx, depth_img, fx, fy, ux, uy, trans,
                                    mask_bin=None,
                                    path_window: int = 20, end_span: int = 6, roi: int = 2,
                                    min_pts_per_end: int = 3):
        
        n = len(ordered_path)
        if n < 3:
            return None
        H, W = depth_img.shape[:2]
        use_mask = mask_bin is not None and mask_bin.shape[:2] == (H, W)
        lo = max(0, grasp_idx - path_window)
        hi = min(n - 1, grasp_idx + path_window)

        """把一段骨架路徑上的像素點轉成世界座標，回傳中位數代表點（有效點數不足回傳 None）。"""
        def cluster_world_point(k_start, k_end):
            pts = []
            for k in range(k_start, k_end + 1):
                py, px = ordered_path[k]
                if not (0 <= px < W and 0 <= py < H):
                    continue
                y0, y1 = max(0, py - roi), min(H, py + roi + 1)
                x0, x1 = max(0, px - roi), min(W, px + roi + 1)
                d = depth_img[y0:y1, x0:x1]
                if use_mask:
                    mroi = mask_bin[y0:y1, x0:x1]
                    d = d[(d > 0) & (mroi > 0)]
                else:
                    d = d[d > 0]
                if d.size == 0:
                    continue
                z = self.to_meters(float(np.median(d)))
                if z <= MIN_VALID_DEPTH_M:
                    continue
                lp = self.backproject_to_local_point(px, py, z, fx, fy, ux, uy)
                wp = tf2_geometry_msgs.do_transform_point(lp, trans)
                pts.append([wp.point.x, wp.point.y, wp.point.z])
            if len(pts) < min_pts_per_end:
                return None
            return np.median(np.asarray(pts, dtype=float), axis=0)

        calyx_pt = cluster_world_point(lo, min(hi, lo + end_span - 1))         # A：靠果實端
        branch_pt = cluster_world_point(max(lo, hi - end_span + 1), hi)       # B：靠枝條端
        if calyx_pt is None or branch_pt is None:
            return None
        if not (np.all(np.isfinite(calyx_pt)) and np.all(np.isfinite(branch_pt))):
            return None

        # delta = B -> A (branch -> calyx)
        delta = calyx_pt - branch_pt
        dx, dy, dz = float(delta[0]), float(delta[1]), float(delta[2])
        norm = math.hypot(dx, math.hypot(dy, dz))
        if norm < 1e-6:
            return None

        return dx / norm, dy / norm, dz / norm

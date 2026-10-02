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
from .config import (BACKPROJECT_XY_SIGN, CAMERA_OPTICAL_FRAME, DEPTH_MM_THRESHOLD,
                      MIN_VALID_DEPTH_M)

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
        lp.point.x = BACKPROJECT_XY_SIGN * float((px - ux) * z / fx)
        lp.point.y = BACKPROJECT_XY_SIGN * float((py - uy) * z / fy)
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

    """果梗中心深度：mask 內的深度中位數。mask 內有效點數不足 min_valid_px、座標超出範圍、
    或完全沒有有效深度時回傳 None，呼叫端要跳過這幀，不要用別的深度頂替。
    2026-09-09 拿掉原本「mask 內點數不夠就退回不濾 mask 的 3x3 視窗」那個 fallback——
    果梗細，mask 內有效點數本來就常常不夠，退回不濾 mask 的視窗等於直接採信附近任何
    深度值，很容易量到背景（果梗中間有縫隙、後面就是牆或桌子），比明顯太遠。改成跟
    median_depth_in_mask 一致：量不到就是 None，讓呼叫端這一幀直接跳過（點雲累加/
    StemTracker 平滑歷史都有處理跳過的情況），不要讓錯誤深度混進去。"""
    @staticmethod
    def median_depth_for_stem(depth_img: np.ndarray, mask_bin: np.ndarray, cx: int, cy: int,
                               window: int, min_valid_px: int):

        h, w = depth_img.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        if mask_bin.shape[:2] != depth_img.shape[:2]:
            return None

        y_min, y_max = max(0, cy - window), min(h, cy + window + 1)
        x_min, x_max = max(0, cx - window), min(w, cx + window + 1)
        depth_roi = depth_img[y_min:y_max, x_min:x_max]
        roi_mask = mask_bin[y_min:y_max, x_min:x_max]
        valid_depths = depth_roi[(depth_roi > 0) & (roi_mask > 0)]

        if valid_depths.size < min_valid_px:
            return None
        return float(np.median(valid_depths))
    
    """依「目標實際物理距離」從果實端 (ordered_path[0]) 沿骨架路徑找抓取點——路徑上每個
    點各自在小視窗內取中位數深度（不是整條路徑共用單一深度值假設同深度），反投影成相機
    座標系下的 3D 點，相鄰兩個「有效」點才算歐氏距離累加成真正的 3D 弧長。跟果梗對相機
    有夾角、深度沿路徑漸變的情況相容，不會因為用單一深度值系統性偏。
    某個路徑點附近深度量不到(mask 邊緣/雜訊)就跳過那個點、不中斷，用上一個有效點接續算
    下一段，避免單一雜訊點讓整段抓取失敗。累加到 target_dist_m 就提前回傳，不用走完整條
    路徑——骨架通常很短，實務上多半提前就回傳了。
    量不到 target_dist_m（走完整條路徑累積弧長還是不夠）時，退回 ratio_min~ratio_max
    之間的比例（依實際弧長佔目標距離的比例內插，邏輯跟 find_grasp_point_by_ratio 一致），
    確保抓取點一定落在路徑範圍內。
    回傳 (grasp_point, grasp_idx)，grasp_point 是 (y, x) 像素座標；任何一步失敗回傳 None。"""
    @classmethod
    def find_grasp_point_by_3d_distance(cls, ordered_path: list, mask_bin: np.ndarray,
                                         depth_img: np.ndarray, fx: float, fy: float,
                                         ux: float, uy: float, target_dist_m: float,
                                         ratio_min: float, ratio_max: float, window: int = 2):
        n = len(ordered_path)
        if n < 2:
            return None

        def point_3d(idx):
            py, px = ordered_path[idx]
            z = cls.median_depth_in_mask(depth_img, mask_bin, min_valid_px=1,
                                          cx=px, cy=py, window=window)
            if z is None:
                return None
            z = cls.to_meters(z)
            if z <= MIN_VALID_DEPTH_M:
                return None
            lp = cls.backproject_to_local_point(px, py, z, fx, fy, ux, uy)
            return np.array([lp.point.x, lp.point.y, lp.point.z])

        prev_pt = None
        cum = 0.0
        for k in range(n):
            cur_pt = point_3d(k)
            if cur_pt is None:
                continue
            if prev_pt is not None:
                cum += float(np.linalg.norm(cur_pt - prev_pt))
                if cum >= target_dist_m:
                    return ordered_path[k], k
            prev_pt = cur_pt

        coverage = cum / target_dist_m if target_dist_m > 0 else 0.0
        ratio = ratio_min + (ratio_max - ratio_min) * coverage
        idx = min(int(n * ratio), n - 1)
        return ordered_path[idx], idx

    """依「目標實際物理距離」估計抓取點，用頭尾兩端點(calyx、branch)的 3D 直線距離
    (弦長)取代逐點累加弧長——只需要量兩個點的深度，比 find_grasp_point_by_3d_distance
    快很多。把骨架路徑當直線處理：果梗彎曲時，弦長 < 實際弧長，用弦長算出來的
    「每像素代表多少實際距離」會被高估，換算出來的抓取點會比正確位置更靠近枝條端
    （系統性誤差，果梗越彎越短越明顯；果梗越直，弦長跟弧長越接近，誤差越小）。
    兩端點任一深度量不到、或弦長算出來趨近 0（理論上不會發生，保險起見）時回傳 None，
    沒有像 find_grasp_point_by_3d_distance 那樣的比例保底——這個方法本身不管路徑多長
    都能直接算比例，沒有「量不到目標距離」這種情況，只有「端點深度量測失敗」會失敗。
    回傳 (grasp_point, grasp_idx)，grasp_point 是 (y, x) 像素座標。"""
    @classmethod
    def find_grasp_point_by_chord_ratio(cls, ordered_path: list, mask_bin: np.ndarray,
                                         depth_img: np.ndarray, fx: float, fy: float,
                                         ux: float, uy: float, target_dist_m: float,
                                         window: int = 2):
        n = len(ordered_path)
        if n < 2:
            return None

        def point_3d(idx):
            py, px = ordered_path[idx]
            z = cls.median_depth_in_mask(depth_img, mask_bin, min_valid_px=1,
                                          cx=px, cy=py, window=window)
            if z is None:
                return None
            z = cls.to_meters(z)
            if z <= MIN_VALID_DEPTH_M:
                return None
            lp = cls.backproject_to_local_point(px, py, z, fx, fy, ux, uy)
            return np.array([lp.point.x, lp.point.y, lp.point.z])

        p_start = point_3d(0)
        p_end = point_3d(n - 1)
        if p_start is None or p_end is None:
            return None

        chord_len = float(np.linalg.norm(p_end - p_start))
        if chord_len <= 1e-9:
            return None

        idx = min(max(int(round((target_dist_m / chord_len) * n)), 0), n - 1)
        return ordered_path[idx], idx

    """計算果梗方向向量 (branch -> calyx)。回傳 dict：除了這一幀當場算出的單位向量
    (vx,vy,vz，供沒有追蹤歷史時的單幀 fallback用)，還有 calyx/branch 兩端『代表像素
    座標 + 深度中位數』(calyx_px/py/z、branch_px/py/z，深度是相機座標系公尺、還沒轉
    世界座標)——這兩組是給 StemTracker 拿去跨幀平滑用的源頭資料：先把像素/深度在
    時間上平均過，再反投影一次，而不是每幀各自反投影完才平均世界座標/向量（深度量測
    本身在特定位置會有雜訊/雙模態跳動，源頭平滑比較不會被單幀離群值拉走，細節見
    PROGRESS.md）。任何一端湊不到 min_pts_per_end 個有效深度點就回傳 None。"""
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

        """把一段骨架路徑上的像素點取深度，回傳代表像素座標 + 深度中位數 (px, py, z)
        （z 是相機座標系公尺，還沒反投影/轉世界座標）；有效點數不足回傳 None。"""
        def cluster_pixel_depth(k_start, k_end):
            pxs, pys, zs = [], [], []
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
                pxs.append(px)
                pys.append(py)
                zs.append(z)
            if len(pxs) < min_pts_per_end:
                return None
            return float(np.median(pxs)), float(np.median(pys)), float(np.median(zs))

        calyx = cluster_pixel_depth(lo, min(hi, lo + end_span - 1))         # A：靠果實端
        branch = cluster_pixel_depth(max(lo, hi - end_span + 1), hi)       # B：靠枝條端
        if calyx is None or branch is None:
            return None
        calyx_px, calyx_py, calyx_z = calyx
        branch_px, branch_py, branch_z = branch

        vx, vy, vz = self.pixel_depth_pair_to_unit_vector(
            calyx_px, calyx_py, calyx_z, branch_px, branch_py, branch_z, fx, fy, ux, uy, trans)

        return {
            'vx': vx, 'vy': vy, 'vz': vz,
            'calyx_px': calyx_px, 'calyx_py': calyx_py, 'calyx_z': calyx_z,
            'branch_px': branch_px, 'branch_py': branch_py, 'branch_z': branch_z,
        }

    """把 calyx/branch 兩端各自的『像素座標 + 相機座標系深度』反投影+轉世界座標，
    相減再單位化，回傳 (vx,vy,vz)（branch -> calyx 方向）；轉換失敗回傳 (0.0,0.0,-1.0)
    當保底。StemTracker 平滑完像素/深度之後，也是呼叫這個函式做『唯一一次』反投影，
    跟單幀 fallback 共用同一套換算，兩邊算法保證一致。"""
    def pixel_depth_pair_to_unit_vector(self, calyx_px, calyx_py, calyx_z,
                                         branch_px, branch_py, branch_z,
                                         fx, fy, ux, uy, trans):
        calyx_lp = self.backproject_to_local_point(calyx_px, calyx_py, calyx_z, fx, fy, ux, uy)
        calyx_wp = tf2_geometry_msgs.do_transform_point(calyx_lp, trans)
        branch_lp = self.backproject_to_local_point(branch_px, branch_py, branch_z, fx, fy, ux, uy)
        branch_wp = tf2_geometry_msgs.do_transform_point(branch_lp, trans)

        dx = calyx_wp.point.x - branch_wp.point.x
        dy = calyx_wp.point.y - branch_wp.point.y
        dz = calyx_wp.point.z - branch_wp.point.z
        if not all(math.isfinite(v) for v in (dx, dy, dz)):
            return 0.0, 0.0, -1.0
        norm = math.hypot(dx, math.hypot(dy, dz))
        if norm < 1e-6:
            return 0.0, 0.0, -1.0
        return dx / norm, dy / norm, dz / norm

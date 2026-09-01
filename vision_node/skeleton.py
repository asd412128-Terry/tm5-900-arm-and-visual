"""
============================================================================
 果梗骨架化：純 2D 像素運算，找出抓取點與生長角度
============================================================================
 只吃/吐 2D mask、像素座標，完全不碰深度或世界座標（那是 coordinates.py
 的工作），也不碰 ROS。
============================================================================
"""
import math
from collections import deque

import cv2
import numpy as np
from skimage.morphology import skeletonize

"""果梗 mask → 骨架化 → 抓取點像素座標 + 生長角度。"""
class PedicelSkeletonizer:
    
    @staticmethod
    def clean_pedicel_mask(mask: np.ndarray, min_area: int = 50):
        """只保留 mask 中面積最大的連通元件，過濾雜訊；面積太小視為無效回傳 None。"""
        mask = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels <= 1:
            return None
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[largest_idx, cv2.CC_STAT_AREA] < min_area:
            return None
        return (labels == largest_idx).astype(np.uint8) * 255

    @classmethod
    def skeletonize_pedicel(cls, mask: np.ndarray, erosion_kernel: int = 3, erosion_iter: int = 1,
                             min_area_for_erosion: int = 150):
        """果梗 mask 清理 → (視面積決定是否侵蝕瘦身) → 骨架化，回傳 (clean, eroded, skeleton)。"""
        clean = cls.clean_pedicel_mask(mask)
        if clean is None:
            return None, None, None
        clean_area = int((clean > 0).sum())
        if clean_area < min_area_for_erosion:
            eroded = clean
        else:
            kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)
            eroded = cv2.erode(clean, kernel, iterations=erosion_iter)
            if (eroded > 0).sum() == 0:
                eroded = clean
        skeleton = skeletonize(eroded > 0)
        return clean, eroded, skeleton
    
    """從 start 做 BFS，回傳 (骨架圖上離 start 最遠的點, {走訪過的點: 前驅點} )。"""
    @staticmethod
    def _bfs_farthest(skeleton: np.ndarray, start: tuple):
        h, w = skeleton.shape
        visited = {start: None}
        queue = deque([start])
        farthest = start
        while queue:
            cur = queue.popleft()
            farthest = cur
            y, x = cur
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] and (ny, nx) not in visited:
                        visited[(ny, nx)] = cur
                        queue.append((ny, nx))
        return farthest, visited

    """用兩次 BFS 找骨架圖的最長路徑（樹的直徑），回傳依 y 座標從 calyx 端
    （y 較大，靠果實）排到 branch 端（y 較小，靠枝條）的像素路徑。

    mask 侵蝕/骨架化後常見雜訊短分支（spur），若只沿單一鄰居往前走、
    遇到分岔點就走進短分支、在分支末端沒有下一格就直接停住，會讓抓取點
    卡在分岔附近甚至等於端點本身。兩次 BFS 找到的一定是全骨架「最長」
    的那條路徑，不會被短分支帶偏，也不會半路斷在分岔點上。"""
    @classmethod
    def order_skeleton_path(cls, skeleton: np.ndarray):
        skel_points = [tuple(p) for p in np.argwhere(skeleton)]
        if not skel_points:
            return []
        end_a, _ = cls._bfs_farthest(skeleton, skel_points[0])
        end_b, parents = cls._bfs_farthest(skeleton, end_a)
        path = []
        node = end_b
        while node is not None:
            path.append(node)
            node = parents[node]
        if path[0][0] < path[-1][0]:
            path.reverse()
        return path
    
    """依「目標像素距離」從果實端 (path[0]) 沿路徑找抓取點；這根果梗的總長度(像素弧長)
    不夠長、量不到目標距離時，退回 ratio_min~ratio_max 之間的比例（依總長度佔目標距離
    的比例內插：越接近量得到目標距離，比例越靠近 ratio_max），確保抓取點一定落在路徑範圍內。"""
    @staticmethod
    def find_grasp_point_by_distance(ordered_path: list, target_px_dist: float,
                                      ratio_min: float, ratio_max: float):
        n = len(ordered_path)
        if n < 2:
            return None

        cum = [0.0]
        for k in range(1, n):
            y1, x1 = ordered_path[k - 1]
            y2, x2 = ordered_path[k]
            cum.append(cum[-1] + math.hypot(x2 - x1, y2 - y1))
        total_len = cum[-1]

        if total_len >= target_px_dist:
            idx = next(k for k, d in enumerate(cum) if d >= target_px_dist)
        else:
            coverage = total_len / target_px_dist if target_px_dist > 0 else 0.0
            ratio = ratio_min + (ratio_max - ratio_min) * coverage
            idx = min(int(n * ratio), n - 1)
        return ordered_path[idx], idx
    
    """用抓取點前後一小段路徑估計果梗生長角度（度）。"""
    @staticmethod
    def compute_growth_angle(ordered_path: list, grasp_idx: int, window: int = 10):
        start = max(0, grasp_idx - window)
        end = min(len(ordered_path) - 1, grasp_idx + window)
        if start == end:
            return 0.0
        y1, x1 = ordered_path[start]
        y2, x2 = ordered_path[end]
        dx = x2 - x1
        dy = y2 - y1
        angle_rad = math.atan2(dx, -(dy))
        return math.degrees(angle_rad)
    
    """公開 API：果梗 mask + 目標像素距離 → (grasp_x, grasp_y, angle_deg, ordered_path, grasp_idx)。
    任何一步失敗都回傳 None。"""
    @classmethod
    def get_stem_grasp_point(cls, mask: np.ndarray, target_px_dist: float,
                              ratio_min: float, ratio_max: float):

        clean, eroded, skeleton = cls.skeletonize_pedicel(mask)
        if skeleton is None:
            return None
        ordered_path = cls.order_skeleton_path(skeleton)
        if len(ordered_path) < 2:
            return None
        result = cls.find_grasp_point_by_distance(ordered_path, target_px_dist, ratio_min, ratio_max)
        if result is None:
            return None
        grasp_point, grasp_idx = result
        gy, gx = grasp_point
        angle_deg = cls.compute_growth_angle(ordered_path, grasp_idx)
        return int(gx), int(gy), angle_deg, ordered_path, grasp_idx

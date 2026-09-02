"""
============================================================================
 番茄遮擋判斷：純 2D mask 形狀運算
============================================================================
 只吃番茄二值 mask，算 bbox 長寬比 / solidity 判斷形狀是否異常（被遮擋/缺角）。
 邏輯搬自 test_occlusion.py 的離線驗證腳本，門檻值沿用同一組。
 不碰深度/世界座標/ROS。
============================================================================
"""
import math
import cv2
import numpy as np
from .config import ASPECT_RATIO_HIGH, ASPECT_RATIO_LOW, SOLIDITY_THRESH

"""番茄 mask 形狀分析 + 遮擋判斷。"""
class OcclusionChecker:

    """輸入單顆番茄的二值 mask (H, W)，回傳 aspect_ratio / solidity / bbox / contour / ellipse。
    找不到輪廓時回傳 None。"""
    @staticmethod
    def compute_shape_metrics(mask_bin: np.ndarray):
        contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        mask_area = cv2.contourArea(contour)
        if mask_area <= 0:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        if w > 0 and h > 0:
            ellipse = ((x + w / 2.0, y + h / 2.0), (w, h), 0.0)
            ellipse_area = math.pi * (w / 2.0) * (h / 2.0)
            solidity = mask_area / ellipse_area if ellipse_area > 0 else 0
        else:
            ellipse = None
            solidity = 0

        return {
            "aspect_ratio": aspect_ratio,
            "solidity": solidity,
            "bbox_xywh": (x, y, w, h),
            "contour": contour,
            "ellipse": ellipse,
        }

    """box_a, box_b: (x1, y1, x2, y2)。回傳 box_b 對 box_a 的重疊比例 = 交集面積 / box_a 面積。"""
    @staticmethod
    def bbox_overlap_ratio(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
        if area_a <= 0:
            return 0.0
        return inter_area / area_a

    """判斷一顆番茄是否被遮擋。metrics 為 None（找不到輪廓）直接判定遮擋。
    other_boxes: 其他所有 instance 的 bbox（其他番茄 + 果梗，不含自己），
    目前只用來算 max_overlap 供顯示/調參參考，不參與判斷（跟 test_occlusion.py 一致）。
    回傳 (occluded: bool, reason: str, max_overlap: float)。"""
    @classmethod
    def judge_occlusion(cls, metrics, my_box, other_boxes):
        if metrics is None:
            return True, "no_contour", 0.0

        reasons = []
        ar = metrics["aspect_ratio"]
        sol = metrics["solidity"]

        if ar < ASPECT_RATIO_LOW or ar > ASPECT_RATIO_HIGH:
            reasons.append(f"aspect_ratio={ar:.2f}")
        if sol < SOLIDITY_THRESH:
            reasons.append(f"solidity={sol:.2f}")

        max_overlap = 0.0
        for ob in other_boxes:
            max_overlap = max(max_overlap, cls.bbox_overlap_ratio(my_box, ob))

        occluded = len(reasons) > 0
        reason_str = ";".join(reasons) if reasons else "clean"
        return occluded, reason_str, max_overlap

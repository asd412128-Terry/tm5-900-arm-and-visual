#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 目標遮罩計算：果梗 + 番茄 mask 合併膨脹
============================================================================
 只負責「算」出合併後的遮罩 array；不碰 ROS publish（design B：
 所有 publish 集中在 vision_node，這裡保持純運算方便單元測試）。
============================================================================
"""

import math

import cv2
import numpy as np

from .config import STEM_MASK_DILATE_PX, TARGET_MASK_DILATE_PX

"""算出目標果梗 + 番茄的合併膨脹遮罩，交給 vision_node 發布給 cloud_filter_node。
果梗一定挖（就是選定的目標本身）；番茄有配對到才一起挖，配對失敗就只挖果梗。"""
class TargetMaskBuilder:

    """設定果梗與番茄 mask 的膨脹核心大小。"""
    def __init__(self, stem_dilate_px: int = STEM_MASK_DILATE_PX,
                 tomato_dilate_px: int = TARGET_MASK_DILATE_PX):
        self.stem_dilate_px = stem_dilate_px
        self.tomato_dilate_px = tomato_dilate_px

    """用果梗中心點找最近的番茄，回傳它的 mask (可能是 None，代表這顆番茄沒有 mask 資料)。"""
    @staticmethod
    def find_matching_tomato_mask(stem_obj: dict, tomatoes: list):
        if not tomatoes:
            return None
        dists = [math.hypot(stem_obj['cx'] - t['cx'], stem_obj['cy'] - t['cy']) for t in tomatoes]
        nearest = tomatoes[int(np.argmin(dists))]
        return nearest.get('mask')

    """回傳 (combined_mask, stem_px_count, tomato_px_count) 或 (None, 0, 0)（果梗沒有 mask 資料時）。"""
    def build_combined_mask(self, stem_obj: dict, tomatoes: list):
        stem_mask = stem_obj.get('mask')
        if stem_mask is None:
            return None, 0, 0

        tomato_mask = self.find_matching_tomato_mask(stem_obj, tomatoes)

        stem_kernel = np.ones((self.stem_dilate_px, self.stem_dilate_px), np.uint8)
        dilated_stem = cv2.dilate(stem_mask, stem_kernel, iterations=1)
        combined = dilated_stem

        tomato_px = 0
        if tomato_mask is not None:
            tomato_kernel = np.ones((self.tomato_dilate_px, self.tomato_dilate_px), np.uint8)
            dilated_tomato = cv2.dilate(tomato_mask, tomato_kernel, iterations=1)
            tomato_px = int((dilated_tomato > 0).sum())
            combined = cv2.bitwise_or(dilated_stem, dilated_tomato)

        stem_px = int((dilated_stem > 0).sum())
        return combined, stem_px, tomato_px

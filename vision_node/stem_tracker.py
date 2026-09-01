"""
============================================================================
 時間平滑：滑動視窗挑信心分數最高的果梗偵測
============================================================================
"""

import math
from collections import deque
from .config import STEM_MATCH_DIST_PX, STEM_TRACK_WINDOW, STEM_TRACK_MAX_MISS

"""
結合「滑動視窗 + 信心分數挑選」的穩定邏輯，輸出含 3D 向量 (vx, vy, vz) 的格式。
"""
class StemTracker:
    
    RECORD_KEYS = ('bbox', 'cx', 'cy', 'world_x', 'world_y', 'world_z',
                   'z_real', 'z_center', 'angle', 'vx', 'vy', 'vz', 'conf', 'mask',
                   'path_len', 'tip_px')

    """設定配對距離、滑動視窗長度、track 消失門檻，初始化空的 track 清單。"""
    def __init__(self, match_dist_px: float = STEM_MATCH_DIST_PX,
                 window: int = STEM_TRACK_WINDOW, max_miss: int = STEM_TRACK_MAX_MISS):
        self.match_dist_px = match_dist_px
        self.window = window
        self.max_miss = max_miss
        self.tracks = []   # 每個 track: {'history': deque(整包 det dict), '_miss': int}
        
    """視窗裡挑 conf 最高的那一筆，整包直接回傳(不逐欄位混合)。"""
    @staticmethod
    def _best_record(history: deque) -> dict:
        
        return max(history, key=lambda d: d.get('conf', 0.0))

    """用像素距離把本幀偵測跟既有 track 配對、更新滑動視窗，回傳每個 track 目前的代表偵測
    （視窗內信心最高那筆）；清除連續配對失敗超過 max_miss 的 track。"""
    def update(self, detections: list) -> list:
        used_track = [False] * len(self.tracks)
        smoothed_out = []

        for det in detections:
            best_idx, best_dist = -1, self.match_dist_px
            for ti, tr in enumerate(self.tracks):
                if used_track[ti]:
                    continue
                # 配對用當前代表位置(最高信心那幀)
                ref = self._best_record(tr['history'])
                d = math.hypot(det['cx'] - ref['cx'], det['cy'] - ref['cy'])
                if d < best_dist:
                    best_dist, best_idx = d, ti

            if best_idx >= 0:
                tr = self.tracks[best_idx]
                tr['history'].append({k: det[k] for k in self.RECORD_KEYS if k in det})
                tr['_miss'] = 0
                used_track[best_idx] = True
                smoothed_out.append(dict(self._best_record(tr['history'])))
            else:
                # 新出現的果梗
                hist = deque(maxlen=self.window)
                hist.append({k: det[k] for k in self.RECORD_KEYS if k in det})
                self.tracks.append({'history': hist, '_miss': 0})
                used_track.append(True)
                smoothed_out.append(dict(hist[0]))

        # 清除消失的 track
        alive_tracks = []
        for ti, tr in enumerate(self.tracks):
            if not used_track[ti]:
                tr['_miss'] = tr.get('_miss', 0) + 1
                if tr['_miss'] > self.max_miss:
                    continue
            alive_tracks.append(tr)
        self.tracks = alive_tracks

        return smoothed_out

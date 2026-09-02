"""
============================================================================
 時間平滑：滑動視窗內取信心最高的一幀，穩定果梗偵測
============================================================================
"""

import math
from collections import deque
from .config import STEM_MATCH_DIST_PX, STEM_TRACK_WINDOW, STEM_TRACK_MAX_MISS

"""
用像素距離配對前後幀同一根果梗，滑動視窗內整幀取信心分數最高的一筆輸出。
"""
class StemTracker:

    RECORD_KEYS = ('bbox', 'cx', 'cy', 'world_x', 'world_y', 'world_z',
                   'z_real', 'z_center', 'angle', 'vx', 'vy', 'vz', 'conf', 'mask',
                   'path_len', 'tip_px', 'root_px', 'end0_px', 'end1_px', 'paired_tomato')

    """設定配對距離、滑動視窗長度、track 消失門檻，初始化空的 track 清單。"""
    def __init__(self, match_dist_px: float = STEM_MATCH_DIST_PX,
                 window: int = STEM_TRACK_WINDOW, max_miss: int = STEM_TRACK_MAX_MISS):
        self.match_dist_px = match_dist_px
        self.window = window
        self.max_miss = max_miss
        self.tracks = []   # 每個 track: {'history': deque(整包 det dict), '_miss': int}

    """視窗內取信心分數最高的一幀，整包原封不動輸出（不逐欄位混合平均）。
    ★ 不能對 cx/cy/world_x/y/z 逐欄位加權平均：果梗骨架通常是彎的，視窗內幾幀如果
    因為 mask 邊緣雜訊、深度取樣抖動讓抓取點沿骨架路徑跳到不同位置，對彎曲路徑上的
    座標做加權平均，算出來的點會落在骨架外面、不是任何一幀真正量到的位置——嚴重的話
    可以偏到接近另一端，即使每一幀單獨看方向判斷(calyx/branch)都是對的。整幀二選一
    保證輸出一定是某一幀真實量到、自洽的結果。"""
    @classmethod
    def _smoothed_record(cls, history: deque) -> dict:
        best = max(history, key=lambda d: d.get('conf', 0.0))
        return dict(best)

    """用像素距離把本幀偵測跟既有 track 配對、更新滑動視窗，回傳每個 track 目前的代表偵測
    （視窗內信心最高的一幀）；清除連續配對失敗超過 max_miss 的 track。"""
    def update(self, detections: list) -> list:
        used_track = [False] * len(self.tracks)
        smoothed_out = []

        for det in detections:
            best_idx, best_dist = -1, self.match_dist_px
            for ti, tr in enumerate(self.tracks):
                if used_track[ti]:
                    continue
                # 配對用當前代表位置(平滑後座標)
                ref = self._smoothed_record(tr['history'])
                d = math.hypot(det['cx'] - ref['cx'], det['cy'] - ref['cy'])
                if d < best_dist:
                    best_dist, best_idx = d, ti

            if best_idx >= 0:
                tr = self.tracks[best_idx]
                tr['history'].append({k: det[k] for k in self.RECORD_KEYS if k in det})
                tr['_miss'] = 0
                used_track[best_idx] = True
                rec = self._smoothed_record(tr['history'])
                smoothed_out.append(rec)
            else:
                # 新出現的果梗
                hist = deque(maxlen=self.window)
                hist.append({k: det[k] for k in self.RECORD_KEYS if k in det})
                self.tracks.append({'history': hist, '_miss': 0})
                used_track.append(True)
                smoothed_out.append(self._smoothed_record(hist))

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

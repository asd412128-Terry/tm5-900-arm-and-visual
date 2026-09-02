"""
============================================================================
 時間平滑：番茄「是否遮擋」要連續多幀都同一種判定才切換，避免單幀雜訊閃爍
============================================================================
"""

import math
from .config import TOMATO_MATCH_DIST_M, TOMATO_OCC_CONFIRM_FRAMES, TOMATO_TRACK_MAX_MISS

"""
occlusion.py 判斷一顆番茄是否遮擋，只看「這一幀」mask 的長寬比/solidity，完全沒有跨幀
記憶——分割雜訊讓形狀分數在門檻附近抖動時，occluded 就每幀真假亂跳，畫面紅綠燈跟著閃，
跟果梗那邊的配對閃爍是兩回事、互不相干。這裡用世界座標把前後幀同一顆番茄對起來，
遮擋判定要連續 confirm_frames 幀都是新結果才真的切換；直接把穩定後的結果寫回
tomato['occluded']，下游 (check_candidate / visualizer) 不用另外改。
"""
class TomatoTracker:

    """設定前後幀配對容忍距離、確認幀數、追蹤消失門檻，初始化空的 track 清單。"""
    def __init__(self, match_dist_m: float = TOMATO_MATCH_DIST_M,
                 confirm_frames: int = TOMATO_OCC_CONFIRM_FRAMES,
                 max_miss: int = TOMATO_TRACK_MAX_MISS):
        self.match_dist_m = match_dist_m
        self.confirm_frames = confirm_frames
        self.max_miss = max_miss
        self.tracks = []   # 每個 track: {'pos', 'stable_occluded', 'run_state', 'run_len', '_miss'}

    """用世界座標距離把本幀番茄跟既有 track 配對，把每顆番茄的 'occluded' 欄位原地改成
    穩定後的結果（要連續 confirm_frames 幀都是同一種判定才會真的切換）；清除連續配對
    失敗超過 max_miss 的 track。"""
    def update(self, tomatoes: list) -> None:
        used = [False] * len(self.tracks)

        for t in tomatoes:
            pos = (t['world_x'], t['world_y'], t['world_z'])
            raw = bool(t.get('occluded', False))

            best_idx, best_dist = -1, self.match_dist_m
            for ti, tr in enumerate(self.tracks):
                if used[ti]:
                    continue
                d = math.dist(pos, tr['pos'])
                if d < best_dist:
                    best_dist, best_idx = d, ti

            if best_idx >= 0:
                tr = self.tracks[best_idx]
                used[best_idx] = True
            else:
                # 新出現的番茄：直接用這一幀的判定當初始狀態，不用等 confirm
                tr = {'pos': pos, 'stable_occluded': raw, 'run_state': raw, 'run_len': 0, '_miss': 0}
                self.tracks.append(tr)
                used.append(True)

            tr['pos'] = pos
            tr['_miss'] = 0
            if raw == tr['run_state']:
                tr['run_len'] += 1
            else:
                tr['run_state'] = raw
                tr['run_len'] = 1
            if tr['run_len'] >= self.confirm_frames:
                tr['stable_occluded'] = tr['run_state']

            t['occluded'] = tr['stable_occluded']

        alive = []
        for ti, tr in enumerate(self.tracks):
            if not used[ti]:
                tr['_miss'] += 1
                if tr['_miss'] > self.max_miss:
                    continue
            alive.append(tr)
        self.tracks = alive

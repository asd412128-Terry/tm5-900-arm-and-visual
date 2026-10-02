"""
============================================================================
 粗定位時間平滑：跨幀配對同一顆番茄，座標取中位數，ID 固定不亂跳
============================================================================
 純運算，不碰 ROS。解決兩個問題：
   1. 座標抖動：同一顆番茄每幀量出來的 base 座標會有 mm 級雜訊，偶爾還有離群值（例如
      兩顆貼在一起的番茄被併框、深度在邊緣跳一下）→ 用最近 window 幀的「中位數」，
      比平均值不容易被單幀離群值拉走。（精定位的 fine_tomato_tracker 用算術平均，因為
      它只處理已經濾過的番茄；粗定位番茄堆在一起、併框機率高，所以改用中位數。）
   2. ID 亂跳：番茄之間的深度只差幾公分，逐幀依深度排序，前後幀的先後順序會對調。
      → 每顆番茄用 base 座標配對成一條 track，ID 綁在 track 上，配對到就保持不變。

 ID 規則（跟 fine_target_selector.py 的 refresh_valid() 2026-09-09 之後同一套慣例）：
   - track 累積 min_hits 次配對成功才「確認」，避免單幀雜訊當成新番茄。
   - 對外顯示的 id 每次呼叫 update()/confirmed() 都依「目前」到手臂基座的直線距離
     （base_x/y/z 向量長度，base 座標系原點就是手臂基座）由近到遠重新分配，0 永遠是
     目前離基座最近的——不是確認當下分配一次就固定不變。番茄之間距離接近時，排名
     互換會讓同一顆番茄的 id 跟著換，這是刻意的（要跟畫面/終端機當下的排序一致），
     跟精定位選取候選的行為一致。
   - track 本身的身份（跨幀配對、決定同一顆番茄）用內部 key（建立時分配、終生不變、
     絕不重複使用），不受 id 重新排序影響——上層（coarse_node.py）要判斷「同一顆番茄
     有沒有變化」要用這個 key 當比對依據，不能用 id（id 只是每次重新算的顯示排名）。
   - track 連續 max_miss 幀沒配到就移除。
 配對用 base 座標距離、一對一貪婪（距離近的先配）。match_dist_m 必須小於番茄之間的間距
 （實測貼在一起的番茄中心約差 4~5cm），又要大於單幀抖動（約幾 mm）。
============================================================================
"""

import math
import statistics
from collections import deque

from .config import (COARSE_TRACK_MATCH_DIST_M, COARSE_TRACK_MAX_MISS,
                      COARSE_TRACK_MIN_HITS, COARSE_TRACK_WINDOW)

_MEDIAN_KEYS = ('base_x', 'base_y', 'base_z', 'z_center', 'depth')

"""番茄 track 管理：update() 吃這一幀偵測到的番茄 list，回傳「已確認、且這一幀有看到」的
番茄 list（座標已換成中位數、加上固定的 'id'），依 id 由小到大排序。"""
class CoarseTomatoTracker:

    def __init__(self, match_dist_m: float = COARSE_TRACK_MATCH_DIST_M,
                 window: int = COARSE_TRACK_WINDOW,
                 max_miss: int = COARSE_TRACK_MAX_MISS,
                 min_hits: int = COARSE_TRACK_MIN_HITS):
        self.match_dist_m = match_dist_m
        self.window = window
        self.max_miss = max_miss
        self.min_hits = min_hits
        # 每個 track: {'key'（內部身份，建立時分配、終生不變）, 'history': deque[dict],
        #             'hits', 'miss', 'last': 最近一次配對到的偵測}
        self.tracks = []
        self._next_key = 0

    """track 目前的中位數值（key 是 _MEDIAN_KEYS 其中一個）。"""
    @staticmethod
    def _median(track: dict, key: str) -> float:
        return statistics.median(h[key] for h in track['history'])

    def _median_pos(self, track: dict):
        return (self._median(track, 'base_x'), self._median(track, 'base_y'),
                self._median(track, 'base_z'))

    """把一次偵測記進 track：進歷史視窗、更新最近偵測、命中次數 +1、miss 歸零。"""
    def _record(self, track: dict, det: dict) -> None:
        track['history'].append({k: det[k] for k in _MEDIAN_KEYS})
        track['last'] = det
        track['hits'] += 1
        track['miss'] = 0

    def update(self, tomatoes: list) -> list:
        # 1) 這一幀的偵測 vs 既有 track：用 base 座標距離一對一貪婪配對（近的先配）
        pairs = []
        for di, det in enumerate(tomatoes):
            p = (det['base_x'], det['base_y'], det['base_z'])
            for ti, tr in enumerate(self.tracks):
                d = math.dist(p, self._median_pos(tr))
                if d < self.match_dist_m:
                    pairs.append((d, di, ti))
        pairs.sort()

        det_used, tr_matched = set(), set()
        for _d, di, ti in pairs:
            if di in det_used or ti in tr_matched:
                continue
            det_used.add(di)
            tr_matched.add(ti)
            self._record(self.tracks[ti], tomatoes[di])

        # 2) 沒配到 track 的偵測 → 開新 track（key 建立當下就分配、終生不變、不重複使用）
        for di, det in enumerate(tomatoes):
            if di in det_used:
                continue
            tr = {'key': self._next_key, 'history': deque(maxlen=self.window),
                  'hits': 0, 'miss': 0, 'last': None}
            self._next_key += 1
            self._record(tr, det)
            self.tracks.append(tr)
            tr_matched.add(len(self.tracks) - 1)

        # 3) 沒配到偵測的 track：miss +1，超過 max_miss 就移除（ID 跟著釋出）
        for ti, tr in enumerate(self.tracks):
            if ti not in tr_matched:
                tr['miss'] += 1
        self.tracks = [tr for tr in self.tracks if tr['miss'] <= self.max_miss]

        # 4) 輸出：累積 hits 滿 min_hits（已確認）、而且這一幀有看到（miss == 0）的番茄；
        #    座標/深度換成視窗內中位數，其餘欄位（bbox、像素中心、mask、conf）沿用最近一次偵測。
        #    key 是內部身份（終生不變），id 留到最後依「目前」到基座距離重新排名再分配。
        out = []
        for tr in self.tracks:
            if tr['hits'] < self.min_hits or tr['miss'] > 0:
                continue
            t = dict(tr['last'])
            for k in _MEDIAN_KEYS:
                t[k] = self._median(tr, k)
            t['key'] = tr['key']
            t['hits'] = tr['hits']
            out.append(t)
        # 5) 依目前到基座距離由近到遠重新分配顯示用 id（0 = 目前離基座最近），跟 fine 的 refresh_valid 同一套慣例
        out.sort(key=lambda t: math.hypot(t['base_x'], t['base_y'], t['base_z']))
        for i, t in enumerate(out):
            t['id'] = i
        return out

    """目前所有已確認的番茄（不管這一幀有沒有看到），座標/深度是視窗內中位數，id 依目前到基座距離
    由近到遠重新分配（跟 update() 同一套慣例，0 = 目前離基座最近）。
    給終端機列印判斷「清單有沒有變」用：呼叫端要用 key（不變的內部身份）而不是 id
    （每次重新排名）當比對依據，否則兩顆番茄距離排名互換時會被誤判成「有變化」。
    track 只有真的新增、或連續 max_miss 幀都沒看到而被移除時，key 的集合才會改變，
    單幀漏偵測不會，所以不會因為信心貼著門檻的番茄偶爾閃一下就重印。"""
    def confirmed(self) -> list:
        out = []
        for tr in self.tracks:
            if tr['hits'] < self.min_hits:
                continue
            t = {k: self._median(tr, k) for k in _MEDIAN_KEYS}
            t['key'] = tr['key']
            t['conf'] = tr['last']['conf']
            out.append(t)
        out.sort(key=lambda t: math.hypot(t['base_x'], t['base_y'], t['base_z']))
        for i, t in enumerate(out):
            t['id'] = i
        return out

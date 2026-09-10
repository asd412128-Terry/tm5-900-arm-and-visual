"""
============================================================================
 時間平滑：滑動視窗內取信心最高的一幀，穩定果梗偵測
============================================================================
"""

import math
import time
from collections import deque
from .config import STEM_MATCH_DIST_PX, STEM_TRACK_WINDOW, STEM_TRACK_MAX_MISS

"""
用像素距離配對前後幀同一根果梗，滑動視窗內整幀取信心分數最高的一筆輸出。
"""
class StemTracker:

    RECORD_KEYS = ('bbox', 'cx', 'cy', 'world_x', 'world_y', 'world_z',
                   'z_real', 'z_center', 'angle', 'vx', 'vy', 'vz', 'conf', 'mask',
                   'path_len', 'grasp_idx', 'tip_px', 'root_px', 'end0_px', 'end1_px', 'paired_tomato',
                   # 平滑用的源頭資料（像素座標 + 相機座標系深度，還沒反投影），總共 6 個點：
                   # 番茄中心、骨架兩端點(tip/root，真正的路徑頭尾)、抓取點、算向量用的
                   # calyx/branch 兩簇。StemTracker 平滑這些，不是平滑 world_x/y/z、
                   # vx/vy/vz 本身。
                   'tomato_cx', 'tomato_cy', 'tomato_z',
                   'tip_x', 'tip_y', 'tip_z', 'root_x', 'root_y', 'root_z',
                   'calyx_px', 'calyx_py', 'calyx_z', 'branch_px', 'branch_py', 'branch_z')

    """設定配對距離、滑動視窗長度、track 消失門檻，初始化空的 track 清單。"""
    def __init__(self, match_dist_px: float = STEM_MATCH_DIST_PX,
                 window: int = STEM_TRACK_WINDOW, max_miss: int = STEM_TRACK_MAX_MISS):
        self.match_dist_px = match_dist_px
        self.window = window
        self.max_miss = max_miss
        self.tracks = []   # 每個 track: {'history': deque(整包 det dict), '_miss': int}

    """視窗內取信心分數最高的一幀當底，其餘欄位（bbox/mask/conf/像素端點……）原封不動
    沿用那一幀；world_x/y/z、vx/vy/vz 這兩組『算出來』的東西不直接平滑——2026-09-09
    實測證實：就算像素/索引完全沒變，深度相機在同一位置讀出來的深度值本身還是會跳
    （雙模態雜訊，不是連續小雜訊，細節見 PROGRESS.md），如果各自平滑「已經算好的
    world_x/y/z、vx/vy/vz」，本質上是在平均兩個可能天差地遠的最終結果，算出來的東西
    physically 可能兩個都不是。改成平滑**源頭**：番茄中心、骨架兩端點(tip/root)、
    抓取點、算向量用的 calyx/branch 兩簇，總共 6 個點，每個點各自(px,py,z)這組
    『像素座標+相機座標系深度』先在視窗內平均，最後才反投影一次算 world_x/y/z、
    vx/vy/vz（見 vision_node.py 呼叫 update() 之後那段，跟 CoordinateEstimator.
    pixel_depth_pair_to_unit_vector 共用同一套換算）。這裡回傳的 rec 裡
    world_x/y/z、vx/vy/vz 仍是『這一幀』的原始值（沒有意義，等著被覆寫），呼叫端
    務必在讀這兩組之前先做完反投影，不能直接用。"""
    # ★ 任何一幀是 NaN，sum() 整包會變 NaN、污染整個視窗的平均值（要等那幀被踢出視窗
    # 才會恢復）——跳過 NaN，只平均還有效的幀；全部都是 NaN 才回傳 default。
    @staticmethod
    def _safe_mean(history: deque, key: str, default: float) -> float:
        vals = [v for v in (d.get(key, default) for d in history) if math.isfinite(v)]
        return sum(vals) / len(vals) if vals else default

    # 平滑「一個點」的 (px, py, z) 三個分量，原地寫回 rec；6 個點都呼叫這同一個 def，
    # 不要 6 組各自複製一份平均邏輯。default 是這個點量不到時的退回值（通常是抓取點）。
    @classmethod
    def _smooth_point(cls, rec: dict, history: deque, best: dict,
                       px_key: str, py_key: str, z_key: str,
                       default_px: float, default_py: float, default_z: float):
        rec[px_key] = cls._safe_mean(history, px_key, best.get(px_key, default_px))
        rec[py_key] = cls._safe_mean(history, py_key, best.get(py_key, default_py))
        rec[z_key] = cls._safe_mean(history, z_key, best.get(z_key, default_z))

    @classmethod
    def _smoothed_record(cls, history: deque) -> dict:
        best = max(history, key=lambda d: d.get('conf', 0.0))
        rec = dict(best)

        # 抓取點先平滑，其他點量不到時都退回抓取點的值當保底
        cls._smooth_point(rec, history, best, 'cx', 'cy', 'z_center',
                           best.get('cx', 0.0), best.get('cy', 0.0), best.get('z_center', 0.0))
        gx, gy, gz = rec['cx'], rec['cy'], rec['z_center']

        cls._smooth_point(rec, history, best, 'tomato_cx', 'tomato_cy', 'tomato_z', gx, gy, gz)
        cls._smooth_point(rec, history, best, 'tip_x', 'tip_y', 'tip_z', gx, gy, gz)
        cls._smooth_point(rec, history, best, 'root_x', 'root_y', 'root_z', gx, gy, gz)
        cls._smooth_point(rec, history, best, 'calyx_px', 'calyx_py', 'calyx_z', gx, gy, gz)
        cls._smooth_point(rec, history, best, 'branch_px', 'branch_py', 'branch_z', gx, gy, gz)
        return rec

    # ★ 暫時診斷用（找完就可以刪）：記錄每個 track 每一幀原始（未平滑）量測值，
    # 用來看單幀雜訊實際幅度多大、判斷視窗要加到多長才夠。
    @staticmethod
    def _log_raw_debug(track_id, det):
        try:
            with open('/tmp/stem_raw_debug.log', 'a') as f:
                f.write(f"{time.time():.3f} track:{track_id} "
                        f"world=({det.get('world_x', float('nan')):.4f}, "
                        f"{det.get('world_y', float('nan')):.4f}, "
                        f"{det.get('world_z', float('nan')):.4f}) "
                        f"vec=({det.get('vx', float('nan')):.3f}, "
                        f"{det.get('vy', float('nan')):.3f}, "
                        f"{det.get('vz', float('nan')):.3f}) "
                        f"conf={det.get('conf', float('nan')):.3f} "
                        f"path_len={det.get('path_len')} grasp_idx={det.get('grasp_idx')} "
                        f"cx,cy=({det.get('cx')},{det.get('cy')}) "
                        f"tip_px={det.get('tip_px')} root_px={det.get('root_px')} "
                        f"end0_px={det.get('end0_px')} end1_px={det.get('end1_px')}\n")
        except OSError:
            pass

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
                self._log_raw_debug(id(tr), det)
            else:
                # 新出現的果梗
                hist = deque(maxlen=self.window)
                hist.append({k: det[k] for k in self.RECORD_KEYS if k in det})
                self.tracks.append({'history': hist, '_miss': 0})
                used_track.append(True)
                smoothed_out.append(self._smoothed_record(hist))
                self._log_raw_debug(id(self.tracks[-1]), det)

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

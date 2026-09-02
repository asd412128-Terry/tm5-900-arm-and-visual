"""
============================================================================
 目標選取：候選篩選 + 終端機互動
============================================================================
 純邏輯 + 終端機 I/O，不碰 ROS publish/subscribe。回傳篩選/選取結果，
 交給 vision_node 決定要發布什麼訊息。
============================================================================
"""
import math
import sys
import termios
from .config import MAX_REACH_M, PAIR_STICKY_DISCOUNT, PAIR_STICKY_MATCH_DIST_M

"""從 StemTracker 輸出的候選清單中，篩選可夾取目標並讓使用者手動選定。"""
class TargetSelector:
    

    """設定夾取安全工作半徑。"""
    def __init__(self, max_reach_m: float = MAX_REACH_M):
        self.max_reach_m = max_reach_m

    """把所有果梗跟番茄做「全域唯一」配對，配對跟判斷哪端是果實端(calyx)一起做：
    果梗實際接在番茄「頂端」，不是番茄的幾何中心，所以拿果梗兩個骨架端點
    (end0_world/end1_world) 分別去跟每顆番茄的頂端接點 (attach_world；沒有就退回
    番茄中心 world_x/y/z) 算 3D 世界座標距離，列出「(某端點, 某顆番茄)」的所有組合、
    由近到遠排序，依序貪婪配對——一顆番茄配走了就不能再被搶走，一根果梗也只會用
    其中一端配走一次；配對贏的那一端，就是果實端，不用再另外比較兩端誰近。
    回傳 (pairs, reverse)：
      pairs   = {id(stem_dict): tomato_dict}
      reverse = {id(stem_dict): bool}，True 代表 end1 是果實端、False 代表 end0 是
                （語意對應 PedicelSkeletonizer.get_stem_grasp_point 的 reverse 參數）。
    stems/tomatoes 任一為空則回傳 ({}, {})。
    只在 detector.py 偵測當下呼叫一次，結果存進該果梗物件的 'paired_tomato' 欄位跟著
    StemTracker 平滑一起帶走；check_candidate / visualizer 畫框跟資訊面板一律讀
    'paired_tomato'，不重新呼叫這個方法——避免用不同時間點/精度的座標各自重算，
    配出不一致的結果。
    prev_pairs：上一幀配對結果 [(stem_fingerprint, tomato_pos), ...]（detector.py
    逐幀維護、傳進來）。stem_fingerprint 用兩端點的中點代表「這根果梗上一幀在哪」——
    哪端是果實端可能因為量測雜訊改變，用中點才是不隨這個決定變動的穩定指紋。跟上一幀
    同一根果梗、同一顆番茄配對過的組合，距離打個折扣再排序——沒有這個折扣，果梗旁邊
    兩顆番茄距離幾乎相等時，深度/mask 雜訊會讓誰比較近的排序每幀互換，配對結果跟著
    在兩顆番茄間跳，畫面紅綠燈閃爍。折扣只影響排序，真的有更近的番茄還是會配走，
    不會卡死在錯誤配對上。"""
    @staticmethod
    def assign_stem_tomato_pairs(stems: list, tomatoes: list, prev_pairs: list = None):
        if not stems or not tomatoes:
            return {}, {}

        def _tomato_anchor(t):
            a = t.get('attach_world')
            return a if a is not None else (t['world_x'], t['world_y'], t['world_z'])

        def _sticky_tomato_pos(fingerprint):
            if not prev_pairs:
                return None
            for prev_fingerprint, prev_tomato_pos in prev_pairs:
                if math.dist(fingerprint, prev_fingerprint) < PAIR_STICKY_MATCH_DIST_M:
                    return prev_tomato_pos
            return None

        candidates = []   # (distance, stem, tomato, is_end1)
        for s in stems:
            fingerprint = tuple((a + b) / 2.0 for a, b in zip(s['end0_world'], s['end1_world']))
            sticky_tomato_pos = _sticky_tomato_pos(fingerprint)
            for t in tomatoes:
                t_pos = _tomato_anchor(t)
                sticky = (sticky_tomato_pos is not None and
                          math.dist(t_pos, sticky_tomato_pos) < PAIR_STICKY_MATCH_DIST_M)
                for is_end1, ep in ((False, s['end0_world']), (True, s['end1_world'])):
                    d = math.dist(ep, t_pos)
                    if sticky:
                        d *= PAIR_STICKY_DISCOUNT
                    candidates.append((d, s, t, is_end1))
        candidates.sort(key=lambda c: c[0])

        used_stem_ids, used_tomato_ids = set(), set()
        pairs, reverse = {}, {}
        for d, s, t, is_end1 in candidates:
            if id(s) in used_stem_ids or id(t) in used_tomato_ids:
                continue
            pairs[id(s)] = t
            reverse[id(s)] = is_end1
            used_stem_ids.add(id(s))
            used_tomato_ids.add(id(t))
        return pairs, reverse

    """把每根果梗記錄的 'paired_tomato'（可能是 StemTracker 視窗裡歷史某一幀留存的番茄
    物件，跟這一幀的 detected_tomatoes 不是同一個 Python 物件）重新指向『這一幀』
    detected_tomatoes 裡世界座標最近的那顆番茄，在 max_dist_m 內才算同一顆，in-place
    覆寫每根果梗 dict 的 'paired_tomato' 欄位；找不到夠近的就設成 None。
    ★ 這一步是果梗畫框（讀 paired_tomato.occluded）跟番茄畫框（讀番茄自己的 occluded）
    對齊到同一個物件的關鍵：不做這步，兩邊各自讀不同時間點留存的番茄快照，即使邏輯上
    是同一顆番茄，遮擋狀態也可能因為 TomatoTracker 逐幀更新的時間差而不同步，畫面上
    就會看到果梗跟它配對的番茄紅綠燈各跳各的、對不起來。必須在 StemTracker.update()
    之後、check_candidate / visualizer 使用之前呼叫一次。"""
    @staticmethod
    def resolve_live_pairing(stems: list, tomatoes: list, max_dist_m: float = PAIR_STICKY_MATCH_DIST_M) -> None:
        for s in stems:
            old = s.get('paired_tomato')
            if old is None:
                continue
            old_pos = (old['world_x'], old['world_y'], old['world_z'])
            match, match_d = None, max_dist_m
            for t in tomatoes:
                d = math.dist(old_pos, (t['world_x'], t['world_y'], t['world_z']))
                if d < match_d:
                    match, match_d = t, d
            s['paired_tomato'] = match

    """單一果梗候選是否可夾：位置座標有限、在安全工作半徑內、配對番茄沒被判定遮擋、
    方向向量非 NaN。回傳 (ok, reason, distance_to_base)；ok=False 時 reason 說明原因，
    ok=True 時 reason 是空字串。是 build_valid_candidates 跟 visualizer 畫框共用的唯一判斷來源，
    避免兩邊各寫一套、判斷標準跑掉。配對番茄直接讀 target['paired_tomato']（detector.py
    配對時存好、跟著 StemTracker 平滑一起帶過來），不在這裡重新配對——避免用不同時間點/
    精度的座標重算出不同的配對結果。"""
    @staticmethod
    def check_candidate(target: dict, max_reach_m: float):
        pos_ok = all(math.isfinite(target[k]) for k in ('world_x', 'world_y', 'world_z'))
        if not pos_ok:
            return False, "位置座標異常(inf/nan)", None

        distance_to_base = math.sqrt(target['world_x'] ** 2 + target['world_y'] ** 2 + target['world_z'] ** 2)
        if distance_to_base > max_reach_m:
            return False, f"距離基座 {distance_to_base:.3f} 公尺，超過安全工作範圍", distance_to_base

        nearest_tomato = target.get('paired_tomato')
        if nearest_tomato is None:
            return False, "沒有配對到番茄", distance_to_base
        if nearest_tomato.get('occluded'):
            return False, f"配對番茄被判定遮擋({nearest_tomato.get('occlusion_reason', '')})", distance_to_base

        vx, vy, vz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', -1.0)
        if any(math.isnan(v) for v in (vx, vy, vz)):
            return False, "向量估計失敗 (NaN)", distance_to_base

        return True, "", distance_to_base

    """依配對番茄的深度（離相機的距離；沒配對到番茄的退回果梗自己的 z_real）排序、
    用 check_candidate 過濾候選，列印候選清單，回傳 {idx: (target, vx, vy, vz, distance_to_base)}。"""
    def build_valid_candidates(self, targets: list) -> dict:
        def _depth_key(t):
            nt = t.get('paired_tomato')
            return nt['depth'] if nt is not None else t['z_real']
        candidates = sorted(range(len(targets)), key=lambda i: _depth_key(targets[i]))

        valid = {}
        print("\n" + "=" * 40)
        print("這輪可選候選：")
        for idx in candidates:
            target = targets[idx]

            ok, reason, distance_to_base = self.check_candidate(target, self.max_reach_m)
            if not ok:
                print(f"  [ID:{idx}] {reason}，跳過。")
                continue

            vx, vy, vz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', -1.0)
            print(f"  [ID:{idx}] 深度={target['z_real']:.3f}m | X={target['world_x']:.3f}, Y={target['world_y']:.3f}, "
                  f"Z={target['world_z']:.3f}（距基座 {distance_to_base:.3f} 公尺）"
                  f" | Vector:[{vx:.2f}, {vy:.2f}, {vz:.2f}]")

            valid[idx] = (target, vx, vy, vz, distance_to_base)
        print("=" * 40)
        return valid
    
    """終端機互動選取目標 ID。
    回傳選定的 idx（int），或 's'（這輪跳過）、'r'（重新偵測）。"""
    def prompt_choose_id(self, valid: dict):
        
        valid_ids_str = ", ".join(str(i) for i in sorted(valid.keys()))
        while True:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            answer = input(f"要夾取哪個 ID？(可選: {valid_ids_str} / "
                            f"s=這輪先不夾直接跳過 / r=重新偵測): ").strip().lower()

            if answer in ('s', 'r'):
                return answer

            if not answer.isdigit() or int(answer) not in valid:
                print(f"輸入無效，請輸入 {valid_ids_str} 其中一個，或輸入 s 跳過、r 重新偵測。")
                continue

            return int(answer)

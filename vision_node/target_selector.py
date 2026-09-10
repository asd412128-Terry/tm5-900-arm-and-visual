"""
============================================================================
 目標選取：候選篩選 + 終端機互動
============================================================================
 純邏輯 + 終端機 I/O，不碰 ROS publish/subscribe。回傳篩選/選取結果，
 交給 vision_node 決定要發布什麼訊息。
============================================================================
"""
import atexit
import math
import os
import select
import shutil
import sys
import termios
import time
import unicodedata
from .config import (CANDIDATE_REFRESH_INTERVAL_SEC, MAX_REACH_M, PAIR_STICKY_DISCOUNT,
                      PAIR_STICKY_MATCH_DIST_M, REFRESH_VALID_MAX_MISS)


"""中文等全形字元在終端機上佔 2 個字元寬，照 len() 算行寬會低估，導致遊標上移
行數算少、清畫面清不乾淨——ANSI 游標控制必須照『螢幕實際列數』算，不能照字串
元素個數算。"""
def _display_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1 for ch in s)


"""字串在寬度 width 的終端機裡實際會佔幾個螢幕列（自動換行也算進去）。"""
def _visual_rows(s: str, width: int) -> int:
    if width <= 0:
        return 1
    return max(1, -(-_display_width(s) // width))

"""從 StemTracker 輸出的候選清單中，篩選可夾取目標並讓使用者手動選定。"""
class TargetSelector:
    

    """設定夾取安全工作半徑。"""
    def __init__(self, max_reach_m: float = MAX_REACH_M):
        self.max_reach_m = max_reach_m

    """把所有果梗跟番茄做「全域唯一」配對，配對跟判斷哪端是果實端(calyx)一起做：
    拿果梗兩個骨架端點 (end0_world/end1_world) 分別去跟每顆番茄的中心 (world_x/y/z)
    算 3D 世界座標距離，列出「(某端點, 某顆番茄)」的所有組合、由近到遠排序，依序
    貪婪配對——一顆番茄配走了就不能再被搶走，一根果梗也只會用其中一端配走一次；
    配對贏的那一端，就是果實端，不用再另外比較兩端誰近。
    （原本用 bbox 頂端點配對，理論上比中心點準，但果實歪斜時頂端點不一定是實際接點，
    改用中心點更穩定。）
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
    def assign_stem_tomato_pairs(stems: list, tomatoes: list, prev_pairs: list = None,
                                  prev_reverse: list = None):
        if not stems or not tomatoes:
            return {}, {}

        def _tomato_anchor(t):
            return (t['world_x'], t['world_y'], t['world_z'])

        def _sticky_tomato_pos(fingerprint):
            if not prev_pairs:
                return None
            for prev_fingerprint, prev_tomato_pos in prev_pairs:
                if math.dist(fingerprint, prev_fingerprint) < PAIR_STICKY_MATCH_DIST_M:
                    return prev_tomato_pos
            return None

        # ★ 2026-09-09：哪端是果實端(is_end1/reverse)原本每幀從零重算，兩端到番茄
        # 距離接近時，端點深度雜訊會讓判定結果每幀亂翻，翻了整條骨架路徑順序就反過來，
        # 果梗一彎，算出來的方向向量會差很多、不只是正負號相反（實測：位置、信心都沒變，
        # 方向卻在 (0.15,0.24,-0.96) 和 (-0.98,-0.11,-0.15) 之間跳，就是這裡沒有時間
        # 穩定性造成的）。比照上面番茄配對已經在用的 sticky 折扣，同一招用在選端上：
        # 跟上一幀判定同一端，距離打折，沒有明確證據（新端真的近很多）就不要翻。
        def _sticky_is_end1(fingerprint):
            if not prev_reverse:
                return None
            for prev_fingerprint, prev_is_end1 in prev_reverse:
                if math.dist(fingerprint, prev_fingerprint) < PAIR_STICKY_MATCH_DIST_M:
                    return prev_is_end1
            return None

        candidates = []   # (distance, stem, tomato, is_end1)
        for s in stems:
            fingerprint = tuple((a + b) / 2.0 for a, b in zip(s['end0_world'], s['end1_world']))
            sticky_tomato_pos = _sticky_tomato_pos(fingerprint)
            sticky_is_end1 = _sticky_is_end1(fingerprint)
            for t in tomatoes:
                t_pos = _tomato_anchor(t)
                sticky = (sticky_tomato_pos is not None and
                          math.dist(t_pos, sticky_tomato_pos) < PAIR_STICKY_MATCH_DIST_M)
                for is_end1, ep in ((False, s['end0_world']), (True, s['end1_world'])):
                    d = math.dist(ep, t_pos)
                    if sticky:
                        d *= PAIR_STICKY_DISCOUNT
                    if sticky_is_end1 is not None and is_end1 == sticky_is_end1:
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

    """先用 check_candidate 把全部候選分成三堆：能夾、有配對到番茄但暫時被判定遮擋、
    其他原因不能夾（沒配對到/超出安全範圍/向量異常）。能夾的依配對番茄的深度（離相機
    的距離；沒配對到番茄的退回果梗自己的 z_center，StemTracker 平滑過的深度，不是
    單幀原始值 z_real）由近到遠排在前面，被遮擋的排在後面（同樣依深度排），兩堆合
    起來一起重新編號（0 起算，能夾的一定排在被遮擋的前面）。
    第三堆不佔用編號、只列原因參考用——遮擋是追蹤結果會隨時間變動的暫時狀態，讓它
    跟能夾的候選一起佔編號、一起進即時面板，遮擋解除時使用者馬上看得到、選得到；
    其他原因是結構性問題不會自己恢復，維持原本只列參考不佔編號的做法。
    回傳 (valid, invalid_reasons, all_occluded)：
      valid           = {新編號: (target, vx, vy, vz, distance_to_base)}
      invalid_reasons = {新編號: 原因}，只包含「有配對到但暫時不能夾」的那些（通常是遮擋）
      all_occluded    = True 表示這輪確實偵測到候選、但全部都因為「配對番茄被判定遮擋」
                         被排除、沒有其他原因的候選（跟「這輪根本沒偵測到任何候選」或
                         「候選被其他原因刷掉」要分開處理，只有全部都是遮擋時，才值得
                         換個視角重新掃描；其他情況換視角也沒用）。"""
    def build_valid_candidates(self, targets: list):
        def _depth_key(t):
            nt = t.get('paired_tomato')
            return nt['depth'] if nt is not None else t['z_center']

        # 可選的（pickable_list）跟暫時被遮擋的（occluded_list）分開排序、分開存，
        # 各自依深度近到遠排完，可選的接在前面、被遮擋的接在後面才編號——已失效/不能
        # 選的排最後，不要跟可選的混在一起用深度排序。
        pickable_list, occluded_list, rejected = [], [], []
        for target in targets:
            ok, reason, distance_to_base = self.check_candidate(target, self.max_reach_m)
            if not ok and not reason.startswith('配對番茄被判定遮擋'):
                rejected.append(reason)   # 除了「配對番茄被判定遮擋」以外的其他排除原因，只列原因，不佔編號
                continue
            entry = (target, ok, reason, distance_to_base)
            (pickable_list if ok else occluded_list).append(entry)

        pickable_list.sort(key=lambda e: _depth_key(e[0]))
        occluded_list.sort(key=lambda e: _depth_key(e[0]))

        valid = {}
        invalid_reasons = {}
        for vid, (target, ok, reason, distance_to_base) in enumerate(pickable_list + occluded_list):
            vx, vy, vz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', -1.0)
            valid[vid] = (target, vx, vy, vz, distance_to_base)
            if not ok:
                invalid_reasons[vid] = reason

        # 能夾的候選清單交給 prompt_choose_id() 的即時面板顯示（會持續更新），
        # 這裡不重印一次，避免同一份清單在終端機出現兩次。
        if rejected:
            print("\n" + "=" * 40)
            print("不能夾的（僅供參考，不佔編號）：")
            for reason in rejected:
                print(f"  - {reason}")
            print("=" * 40)

        pickable = any(vid not in invalid_reasons for vid in valid)
        all_occluded = (not pickable and bool(valid) and not rejected)
        return valid, invalid_reasons, all_occluded
    
    """把 valid 裡每個候選重新比對到『目前最新一幀』的座標：位置在 max_dist_m 內視為
    同一根果梗，就地更新座標/vx,vy,vz/distance_to_base；找不到夠近的、或更新後
    check_candidate 判定不合格，標成失效（原因寫進 invalid_reasons）。
    ★ 2026-09-09：編號改成每次呼叫都依『目前』深度（z_center）由近到遠重新分配
    （0 永遠是目前最近的），不再是整輪選取期間固定不變——雖然這代表使用者打字打到
    一半，深度更新時號碼理論上可能改指向別的物理目標，但這是使用者要的行為（要用
    「目前最近」的當下狀態選取，不是選取當下那一刻的固定順序）。miss_counts 追蹤的
    是「連續配不到幾次」，用來抓 target_selector.py 這邊比對用的內部索引（沿用上一輪
    的 old vid，不是新編號），重新編號後才搬到新編號下，確保容忍邏輯不會因為重編號
    就跟丟。
    ★ 「找不到夠近的」這個原因不會單幀立刻判失效：2026-09-09 實測證實，即使
    grasp_idx/取樣像素完全沒變，深度相機在同一位置讀出來的 3D 座標偶爾還是會跳掉
    （深度感測雜訊，不是演算法問題，細節見 PROGRESS.md）；YOLO 單幀漏偵測也是同一種
    「這一幀量測不能信、很快就恢復」的情況。超過 REFRESH_VALID_MAX_MISS 才真的判
    失效，避免面板一直閃。check_candidate 判定不合格（座標異常/超出範圍/沒配對到
    番茄/被遮擋/向量 NaN）是結構性問題、拿到的是「這一幀真的配對到」的量測，不受
    這個容忍影響，一律立刻判失效。
    回傳 (new_valid, invalid_reasons)。"""
    @staticmethod
    def refresh_valid(valid: dict, latest_targets: list, max_reach_m: float,
                       max_dist_m: float = PAIR_STICKY_MATCH_DIST_M,
                       miss_counts: list = None, max_miss: int = REFRESH_VALID_MAX_MISS):
        # ★ miss_counts 是 list [(position, miss), ...]，用『位置』當識別鍵，不能用
        # vid——vid 現在每次都依當下深度重新分配，兩個候選深度排名互換時，『編號 0』
        # 這次指的物理目標可能已經不是上次『編號 0』那個了，拿編號當鍵會把容忍次數
        # 算到錯的目標上（2026-09-09 修過一次用 vid 當鍵，是錯的，改成位置）。
        if miss_counts is None:
            miss_counts = []

        def _prev_miss(pos):
            for p, m in miss_counts:
                if math.dist(pos, p) < max_dist_m:
                    return m
            return 0

        # 先照舊編號逐一比對更新（沿用原本的比對/容忍邏輯），暫存結果，還不決定新編號
        updated = []   # [(target, vx, vy, vz, distance_to_base, reason_or_None, miss, pos)]
        for old_vid, (target, vx, vy, vz, distance_to_base) in valid.items():
            old_pos = (target['world_x'], target['world_y'], target['world_z'])
            match, match_d = None, max_dist_m
            for t in latest_targets:
                d = math.dist(old_pos, (t['world_x'], t['world_y'], t['world_z']))
                if d < match_d:
                    match, match_d = t, d

            if match is None:
                miss = _prev_miss(old_pos) + 1
                reason = "目標消失或移動過大" if miss > max_miss else None
                updated.append((target, vx, vy, vz, distance_to_base, reason, miss, old_pos))
                continue

            ok, reason, new_distance = TargetSelector.check_candidate(match, max_reach_m)
            new_vx = match.get('vx', 0.0)
            new_vy = match.get('vy', 0.0)
            new_vz = match.get('vz', -1.0)
            new_pos = (match['world_x'], match['world_y'], match['world_z'])
            updated.append((match, new_vx, new_vy, new_vz, new_distance, None if ok else reason, 0, new_pos))

        # 依「能不能選、目前深度」重新排序、重新編號：可選的排前面（依深度近到遠），
        # 已失效的一律排最後（同樣依深度近到遠）；miss_counts 改存這一輪每個候選的
        # 『位置』，下次呼叫用位置比對，不受編號重排影響
        updated.sort(key=lambda u: (u[5] is not None, u[0]['z_center']))
        new_valid = {}
        invalid_reasons = {}
        new_miss_list = []
        for new_vid, (target, vx, vy, vz, distance_to_base, reason, miss, pos) in enumerate(updated):
            new_valid[new_vid] = (target, vx, vy, vz, distance_to_base)
            new_miss_list.append((pos, miss))
            if reason:
                invalid_reasons[new_vid] = reason
        miss_counts[:] = new_miss_list
        return new_valid, invalid_reasons

    """把 valid 排版成即時面板要印的每一行文字（失效的候選附註原因），純格式化不列印。"""
    @staticmethod
    def _format_candidate_lines(valid: dict, invalid_reasons: dict) -> list:
        lines = []
        # 編號跟顯示順序都是 refresh_valid() 每次重繪時依「能不能選、目前深度」重新
        # 分配的（可選的排前面依深度近到遠，已失效的排最後），這裡再排一次是為了跟
        # build_valid_candidates 剛列出來、還沒跑過 refresh_valid 的第一次畫面一致。
        for vid in sorted(valid.keys(), key=lambda v: (v in invalid_reasons, valid[v][0]['z_center'])):
            target, vx, vy, vz, distance_to_base = valid[vid]
            reason = invalid_reasons.get(vid)
            if reason and reason.startswith('配對番茄被判定遮擋'):
                tag = "因遮擋無法抓取"
            elif reason:
                tag = f"已失效（{reason}）"
            else:
                tag = ""
            lines.append(
                f"  [ID:{vid}] 果梗 深度={target['z_center']:.3f}m | X={target['world_x']:.3f}, "
                f"Y={target['world_y']:.3f}, Z={target['world_z']:.3f}"
                f"（距基座 {distance_to_base:.3f} 公尺） | Vector:[{vx:.2f}, {vy:.2f}, {vz:.2f}]{tag}")
            nt = target.get('paired_tomato')
            if nt is not None:
                lines.append(
                    f"         番茄 深度={nt.get('depth', 0.0):.3f}m | X={nt['world_x']:.3f}, "
                    f"Y={nt['world_y']:.3f}, Z={nt['world_z']:.3f}")
        return lines

    """終端機互動選取目標 ID，等待輸入期間原地即時更新候選座標（不往下滾動畫面）。
    做法：把 stdin 切成 raw/cbreak 模式（關掉 ICANON/ECHO）自己逐字元讀取、自己回顯，
    這樣才能在『使用者打字打到一半』時，安全地用 ANSI 游標控制清掉畫面重繪——如果不
    自己接管輸入，直接對一般 input() 的畫面做游標操作，會把使用者已經打的字元從畫面上
    抹掉（字元其實還留在 tty 的輸入緩衝區，只是畫面跟緩衝區對不上，看起來像打字消失）。
    refresh_fn(valid) -> (new_valid, invalid_reasons)：沒有輸入的空檔，每隔
    refresh_interval 呼叫一次，取得最新座標重繪；refresh_fn 為 None 就不即時更新。
    invalid_reasons：呼叫端(build_valid_candidates)一開始就知道「暫時不能選」的
    候選（通常是遮擋），例如 {vid: '配對番茄被判定遮擋(...)'}，讓面板從第一次畫面
    就標註⚠，不用等第一次 refresh 才顯示；之後每次 refresh_fn 回傳的新版本會整個
    取代它，遮擋解除時該 vid 就不會再出現在裡面。
    回傳選定的 idx（int），或 's'（這輪跳過）、'r'（重新偵測）。"""
    def prompt_choose_id(self, valid: dict, refresh_fn=None,
                          refresh_interval: float = CANDIDATE_REFRESH_INTERVAL_SEC,
                          invalid_reasons: dict = None):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        termios.tcflush(sys.stdin, termios.TCIFLUSH)

        invalid_reasons = dict(invalid_reasons) if invalid_reasons else {}
        buf = ""
        error_line = None
        last_total_rows = 0
        last_refresh = time.time()

        def valid_ids_str():
            ids = sorted(valid.keys(), key=lambda v: valid[v][0]['z_center'])
            return ", ".join(str(i) for i in ids if i not in invalid_reasons)

        def redraw():
            nonlocal last_total_rows
            width = shutil.get_terminal_size((80, 24)).columns
            block = self._format_candidate_lines(valid, invalid_reasons)
            if error_line:
                block.append(error_line)
            block.append(f"要夾取哪個 ID？(可選: {valid_ids_str()} / "
                          f"s=這輪先不夾直接跳過 / r=重新偵測): {buf}")
            total_rows = sum(_visual_rows(ln, width) for ln in block)

            out = []
            if last_total_rows:
                up = last_total_rows - 1
                if up > 0:
                    out.append(f"\033[{up}A")
                out.append("\r\033[J")
            out.append("\r\n".join(block))
            sys.stdout.write("".join(out))
            sys.stdout.flush()
            last_total_rows = total_rows

        def restore_tty():
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        try:
            new_settings = termios.tcgetattr(fd)
            new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(fd, termios.TCSANOW, new_settings)
            # 這段可能跑在 daemon 執行緒裡：process 被 Ctrl+C 中斷結束時，這條
            # 執行緒會被直接放棄、下面的 finally 不會執行——註冊 atexit 當保險，
            # 確保 process 結束前終端機一定會被切回正常模式，不會卡在沒回顯的狀態。
            atexit.register(restore_tty)

            redraw()
            while True:
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    now = time.time()
                    if refresh_fn and now - last_refresh >= refresh_interval:
                        new_valid, new_invalid_reasons = refresh_fn(valid)
                        valid.clear()
                        valid.update(new_valid)
                        invalid_reasons.clear()
                        invalid_reasons.update(new_invalid_reasons)
                        last_refresh = now
                        redraw()
                    continue

                ch = os.read(fd, 1).decode(errors='ignore')

                if ch in ('\r', '\n'):
                    answer = buf.strip().lower()
                    buf = ""
                    error_line = None

                    if answer in ('s', 'r'):
                        return answer

                    if answer.isdigit() and int(answer) in valid:
                        vid = int(answer)
                        reason = invalid_reasons.get(vid)
                        if reason:
                            error_line = f"[ID:{vid}] 已失效（{reason}），請重新選擇。"
                            redraw()
                            continue
                        return vid

                    error_line = f"輸入無效，請輸入 {valid_ids_str()} 其中一個，或輸入 s 跳過、r 重新偵測。"
                    redraw()
                elif ch in ('\x7f', '\x08'):
                    buf = buf[:-1]
                    redraw()
                elif ch.isprintable():
                    buf += ch
                    redraw()
        finally:
            atexit.unregister(restore_tty)
            restore_tty()
            sys.stdout.write("\r\n")
            sys.stdout.flush()

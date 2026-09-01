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
from .config import MAX_REACH_M

"""從 StemTracker 輸出的候選清單中，篩選可夾取目標並讓使用者手動選定。"""
class TargetSelector:
    

    """設定夾取安全工作半徑。"""
    def __init__(self, max_reach_m: float = MAX_REACH_M):
        self.max_reach_m = max_reach_m

    """把所有果梗跟番茄做「全域唯一」配對：列出每一種果梗-番茄組合的 3D 世界座標距離
    （含深度，不是只看畫面上的 2D 像素距離），由近到遠排序、依序貪婪配對——一顆番茄
    配走了就不能再被別根果梗選走，避免兩根果梗剛好都離同一顆番茄近、結果都搶著配對
    到它，而它們真正對應的番茄反而配對失敗。回傳 {id(stem_dict): tomato_dict}，
    stems/tomatoes 任一為空則回傳空 dict。
    這是唯一的配對邏輯來源，check_candidate / visualizer 畫框跟資訊面板都呼叫這個，
    避免各處各寫一套配對、標準跑掉。"""
    @staticmethod
    def assign_stem_tomato_pairs(stems: list, tomatoes: list) -> dict:
        if not stems or not tomatoes:
            return {}
        pairs = []
        for s in stems:
            for t in tomatoes:
                d = math.sqrt((s['world_x'] - t['world_x']) ** 2 +
                              (s['world_y'] - t['world_y']) ** 2 +
                              (s['world_z'] - t['world_z']) ** 2)
                pairs.append((d, s, t))
        pairs.sort(key=lambda p: p[0])

        used_stem_ids, used_tomato_ids = set(), set()
        result = {}
        for d, s, t in pairs:
            if id(s) in used_stem_ids or id(t) in used_tomato_ids:
                continue
            result[id(s)] = t
            used_stem_ids.add(id(s))
            used_tomato_ids.add(id(t))
        return result

    """單一果梗候選是否可夾：位置座標有限、在安全工作半徑內、配對番茄沒被判定遮擋、
    方向向量非 NaN。回傳 (ok, reason, distance_to_base)；ok=False 時 reason 說明原因，
    ok=True 時 reason 是空字串。是 build_valid_candidates 跟 visualizer 畫框共用的唯一判斷來源，
    避免兩邊各寫一套、判斷標準跑掉。pairs 是 assign_stem_tomato_pairs 算好的全域配對表。"""
    @staticmethod
    def check_candidate(target: dict, pairs: dict, max_reach_m: float):
        pos_ok = all(math.isfinite(target[k]) for k in ('world_x', 'world_y', 'world_z'))
        if not pos_ok:
            return False, "位置座標異常(inf/nan)", None

        distance_to_base = math.sqrt(target['world_x'] ** 2 + target['world_y'] ** 2 + target['world_z'] ** 2)
        if distance_to_base > max_reach_m:
            return False, f"距離基座 {distance_to_base:.3f} 公尺，超過安全工作範圍", distance_to_base

        nearest_tomato = pairs.get(id(target))
        if nearest_tomato is None:
            return False, "沒有配對到番茄", distance_to_base
        if nearest_tomato.get('occluded'):
            return False, f"配對番茄被判定遮擋({nearest_tomato.get('occlusion_reason', '')})", distance_to_base

        vx, vy, vz = target.get('vx', 0.0), target.get('vy', 0.0), target.get('vz', -1.0)
        if any(math.isnan(v) for v in (vx, vy, vz)):
            return False, "向量估計失敗 (NaN)", distance_to_base

        return True, "", distance_to_base

    """依深度 (z_real，離相機的距離) 排序、用 check_candidate 過濾候選，列印候選清單，
    回傳 {idx: (target, vx, vy, vz, distance_to_base)}。"""
    def build_valid_candidates(self, targets: list, tomatoes: list = None) -> dict:
        tomatoes = tomatoes or []
        pairs = self.assign_stem_tomato_pairs(targets, tomatoes)

        candidates = sorted(range(len(targets)), key=lambda i: targets[i]['z_real'])

        valid = {}
        print("\n" + "=" * 40)
        print("這輪可選候選：")
        for idx in candidates:
            target = targets[idx]

            ok, reason, distance_to_base = self.check_candidate(target, pairs, self.max_reach_m)
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

"""
============================================================================
 畫面疊圖 / 終端機列印
============================================================================
 只負責「畫」跟「印」，不做任何偵測或狀態判斷邏輯。
============================================================================
"""

import cv2
from .config import ASPECT_RATIO_HIGH, ASPECT_RATIO_LOW, MAX_REACH_M, SOLIDITY_THRESH
from .target_selector import TargetSelector

"""在畫面上疊加偵測框/資訊面板，並在終端機列印掃描結果。"""
class Visualizer:

    """在畫面上畫出果梗框線/中心點（用 TargetSelector.check_candidate 跟實際篩選邏輯同一套
    標準判斷能不能夾：綠色=可以夾，紅色=不能夾），以及番茄框線（有配對到果梗+沒被遮擋才綠色），
    並疊加資訊面板。"""
    def draw_tracked_overlay(self, cv_image, detected_objects, detected_tomatoes):
        GREEN, RED = (0, 255, 0), (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for idx, obj in enumerate(detected_objects):
            if obj.get('paired_tomato') is None:
                continue  # 沒配對到番茄的果梗不畫，避免讓人誤以為可以選
            b = obj['bbox']
            ok, _, _ = TargetSelector.check_candidate(obj, MAX_REACH_M)
            color = GREEN if ok else RED
            cv2.rectangle(cv_image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
            cv2.circle(cv_image, (obj['cx'], obj['cy']), 6, color, -1)
            cv2.putText(cv_image, f"S{idx}", (int(b[0]), int(b[1]) - 6), font, 0.6, color, 2, cv2.LINE_AA)

        # 番茄標籤用「配對到的果梗編號」，讓同一對果梗/番茄的 S/T 數字一致方便對照；
        # 沒配對到的番茄才用自己在偵測清單裡的編號。
        # ★ obj['paired_tomato'] 在 vision_node.py 呼叫 StemTracker.update() 後，已經
        # 經過 TargetSelector.resolve_live_pairing() 重新指向這一幀 detected_tomatoes
        # 裡的同一個物件，這裡才能單純用 id() 比對——不能省略 resolve_live_pairing()，
        # 否則這裡的 nt 可能是舊幀留存的物件快照，跟 detected_tomatoes 對不上。
        tomato_label_num = {}
        for stem_idx, obj in enumerate(detected_objects):
            nt = obj.get('paired_tomato')
            if nt is not None:
                tomato_label_num[id(nt)] = stem_idx

        for idx, t in enumerate(detected_tomatoes):
            if id(t) not in tomato_label_num:
                continue  # 沒配對到果梗的番茄不畫，避免讓人誤以為可以選
            pickable = not t.get('occluded', False)
            color = GREEN if pickable else RED
            tb = t['bbox']
            label_num = tomato_label_num[id(t)]
            cv2.rectangle(cv_image, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])), color, 2)
            cv2.circle(cv_image, (t['cx'], t['cy']), 5, color, -1)
            cv2.putText(cv_image, f"T{label_num}", (int(tb[0]), int(tb[1]) - 6), font, 0.6, color, 2, cv2.LINE_AA)

        self.draw_info_panel(cv_image, detected_objects, detected_tomatoes)

    """分別判斷 aspect_ratio、solidity 是否超出 occlusion.py 判定遮擋的門檻（跟
    OcclusionChecker.judge_occlusion 同一組門檻，維持顯示跟實際判斷一致），各自
    獨立回傳，不合併成一個結果——面板才能讓兩個指標各自顯示自己的紅/綠，不會
    因為其中一個沒過就把兩個都標成紅色。超過門檻(異常)回傳 True。"""
    @staticmethod
    def _aspect_ratio_bad(aspect_ratio):
        return aspect_ratio < ASPECT_RATIO_LOW or aspect_ratio > ASPECT_RATIO_HIGH

    @staticmethod
    def _solidity_bad(solidity):
        return solidity < SOLIDITY_THRESH

    """在畫面左上角疊加半透明面板，列出每個果梗與配對番茄的世界座標、番茄的形狀指標
    （精簡版面）。每行可以有自己的顏色（YELLOW=標題，WHITE=座標，RED/GREEN=形狀指標
    是否超過 occlusion 判定門檻），用 (text, color) tuple 取代原本整行統一上色。"""
    def draw_info_panel(self, img, stems, tomatoes):
        WHITE, YELLOW, RED, GREEN = (255, 255, 255), (0, 255, 255), (0, 0, 255), (0, 255, 0)
        lines = [(f"{len(stems)} stems / {len(tomatoes)} tomatoes", YELLOW)]
        for idx, obj in enumerate(stems):
            lines.append((f"[{idx}]S X={obj['world_x']:.3f} Y={obj['world_y']:.3f} Z={obj['world_z']:.3f} D={obj['z_real']:.3f}", WHITE))
            nt = obj.get('paired_tomato')
            if nt is not None:
                lines.append((f"   T X={nt['world_x']:.3f} Y={nt['world_y']:.3f} Z={nt['world_z']:.3f} D={nt.get('depth', 0.0):.3f}", WHITE))
                ar, sol = nt.get('aspect_ratio'), nt.get('solidity')
                if ar is not None and sol is not None:
                    lines.append((f"   AR={ar:.2f}", RED if self._aspect_ratio_bad(ar) else GREEN))
                    lines.append((f"   Mask/All={sol:.2f}", RED if self._solidity_bad(sol) else GREEN))

        font, scale, thick, line_h, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1, 14, 6
        x0, y0 = 10, 10
        max_w = max((cv2.getTextSize(ln, font, scale, thick)[0][0] for ln, _ in lines), default=0)
        panel_w, panel_h = max_w + pad * 2, line_h * len(lines) + pad * 2

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

        y = y0 + pad + 9
        for ln, color in lines:
            cv2.putText(img, ln, (x0 + pad, y), font, scale, color, thick, cv2.LINE_AA)
            y += line_h

    """終端機列印本輪掃描到的果梗/番茄清單。"""
    def print_scan_summary(self, detected_objects, tomatoes):
        print("\n" + "=" * 60)
        print(f"偵測到 {len(detected_objects)} 個果梗 / {len(tomatoes)} 個番茄")
        print("-" * 60)
        for idx, obj in enumerate(detected_objects):
            print(f"  [ID:{idx}] Stem X={obj['world_x']:.3f}, Y={obj['world_y']:.3f}, Z={obj['world_z']:.3f} "
                  f"| Vec:[{obj.get('vx', 0):.2f}, {obj.get('vy', 0):.2f}, {obj.get('vz', -1):.2f}]")
            nearest = obj.get('paired_tomato')
            if nearest is not None:
                print(f"         Tomato X={nearest['world_x']:.3f}, Y={nearest['world_y']:.3f}, Z={nearest['world_z']:.3f}")
        print("=" * 60)

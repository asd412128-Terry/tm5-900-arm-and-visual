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
    並疊加資訊面板。
    ★ 2026-09-09：valid 是互動選取中的候選 dict（vision_node.py 的 self._interactive_valid，
    build_valid_candidates/refresh_valid 那份）。不是 None 時，**直接拿這份 dict 本身當畫面
    的資料來源**（不是拿 detected_objects 再用位置猜回去對應哪個 vid）——vid、座標、bbox
    全部照 valid 裡存的值畫，保證跟終端機顯示的是同一批物件、同一個編號，不會有兩邊對不
    起來的情況。valid 是 None（還沒進互動選取階段，例如剛開始掃描）才退回 detected_objects
    依偵測清單順序的流水編號，這只是暫時性的、跟終端機無關的顯示。"""
    def draw_tracked_overlay(self, cv_image, detected_objects, detected_tomatoes, valid=None):
        GREEN, RED = (0, 255, 0), (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        stem_labels = {}   # id(obj) -> 畫面上要顯示的編號字串，供番茄那邊對照用

        if valid is not None:
            source = [(str(vid), entry[0]) for vid, entry in valid.items()]
        else:
            source = [(str(idx), obj) for idx, obj in enumerate(detected_objects)
                      if obj.get('paired_tomato') is not None]  # 沒配對到番茄的果梗不畫

        for label, obj in source:
            stem_labels[id(obj)] = label
            b = obj['bbox']
            ok, _, _ = TargetSelector.check_candidate(obj, MAX_REACH_M)
            color = GREEN if ok else RED
            cv2.rectangle(cv_image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 2)
            cv2.circle(cv_image, (int(obj['cx']), int(obj['cy'])), 6, color, -1)
            cv2.putText(cv_image, f"ID{label}", (int(b[0]), int(b[1]) - 6), font, 0.6, color, 2, cv2.LINE_AA)

        if valid is not None:
            # valid 裡的 target 是最近一次 refresh_valid 留存的快照（可能是幾百毫秒前），
            # 跟這一幀的 detected_tomatoes 不是同一批物件、id() 對不起來——直接用每個
            # target 自己存的 paired_tomato 快照畫，不要拿去跟 detected_tomatoes 比對。
            for label, obj in source:
                nt = obj.get('paired_tomato')
                if nt is None:
                    continue
                pickable = not nt.get('occluded', False)
                color = GREEN if pickable else RED
                tb = nt['bbox']
                cv2.rectangle(cv_image, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])), color, 2)
                cv2.circle(cv_image, (int(nt['cx']), int(nt['cy'])), 5, color, -1)
                cv2.putText(cv_image, f"ID{label}", (int(tb[0]), int(tb[1]) - 6), font, 0.6, color, 2, cv2.LINE_AA)
        else:
            # 番茄標籤用「配對到的果梗編號」，讓同一對果梗/番茄的 S/T 數字一致方便對照；
            # ★ obj['paired_tomato'] 在 vision_node.py 呼叫 StemTracker.update() 後，已經
            # 經過 TargetSelector.resolve_live_pairing() 重新指向這一幀 detected_tomatoes
            # 裡的同一個物件，這裡才能單純用 id() 比對——這個分支跟上面的 stem_labels
            # 都是同一幀的 detected_objects/detected_tomatoes，id() 比對才會準。
            tomato_label_num = {}
            for obj in detected_objects:
                nt = obj.get('paired_tomato')
                if nt is not None and id(obj) in stem_labels:
                    tomato_label_num[id(nt)] = stem_labels[id(obj)]

            for t in detected_tomatoes:
                if id(t) not in tomato_label_num:
                    continue  # 沒配對到果梗（或果梗沒被畫出來）的番茄不畫，避免讓人誤以為可以選
                pickable = not t.get('occluded', False)
                color = GREEN if pickable else RED
                tb = t['bbox']
                label_num = tomato_label_num[id(t)]
                cv2.rectangle(cv_image, (int(tb[0]), int(tb[1])), (int(tb[2]), int(tb[3])), color, 2)
                cv2.circle(cv_image, (t['cx'], t['cy']), 5, color, -1)
                cv2.putText(cv_image, f"ID{label_num}", (int(tb[0]), int(tb[1]) - 6), font, 0.6, color, 2, cv2.LINE_AA)

        self.draw_info_panel(cv_image, source, detected_tomatoes)

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
    是否超過 occlusion 判定門檻），用 (text, color) tuple 取代原本整行統一上色。
    ★ stems 是 [(label, obj), ...]（跟 draw_tracked_overlay 的 source 同一份，label 是
    畫在框上的那個編號字串），不是單純的物件清單——這樣面板列出來的編號才會跟畫面上
    的框、跟終端機的 ID 三邊一致。"""
    def draw_info_panel(self, img, stems, tomatoes):
        WHITE, YELLOW, RED, GREEN = (255, 255, 255), (0, 255, 255), (0, 0, 255), (0, 255, 0)
        lines = [(f"{len(stems)} stems / {len(tomatoes)} tomatoes", YELLOW)]
        for label, obj in stems:
            lines.append((f"[ID{label}] X={obj['world_x']:.3f} Y={obj['world_y']:.3f} Z={obj['world_z']:.3f} D={obj['z_center']:.3f}", WHITE))
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

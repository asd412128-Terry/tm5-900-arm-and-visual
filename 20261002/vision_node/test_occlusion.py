import os
import glob
import signal
import math
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


def _force_exit(signum, frame):
    """Ctrl+C 強制關閉：就算卡在 plt.show() 的視窗事件迴圈裡也能立刻中斷，
    不等視窗回應，也不做一般的清理流程。"""
    print("\n收到 Ctrl+C，強制關閉程式...")
    plt.close('all')
    os._exit(0)


signal.signal(signal.SIGINT, _force_exit)

# ============ 模型與資料設定 (沿用 test_real1.py) ============
model = YOLO("/home/terry/Desktop/stem_isaac_train/runs/segment/tomato_stem/fake_tomato1024-2/weights/best.pt")
img_dir = "/home/terry/Desktop/stem_isaac_train/test_fake1"

CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.45

TOMATO_CLS = 1
STEM_CLS = 0  # 如果模型沒有這個 class,程式會自動跳過,不影響其他判斷

# ============ 遮擋判斷閾值 (先給預設值,自己跑完再調) ============
ASPECT_RATIO_LOW = 0.9    # bbox 長寬比下限,低於這個判定形狀異常
ASPECT_RATIO_HIGH = 1.3   # bbox 長寬比上限,高於這個判定形狀異常
SOLIDITY_THRESH = 0.85     # mask 面積 / 擬合橢圓面積,低於這個判定形狀跟橢圓差太多 (取代原本的 convex hull)
BBOX_OVERLAP_THRESH = 0.45  # 跟其他 instance 的 bbox 重疊比例,超過這個判定被壓到


def compute_shape_metrics(mask_bin):
    """
    輸入單顆番茄的二值 mask (H, W), 回傳 aspect_ratio 與 solidity。
    找不到輪廓時回傳 None。
    """
    contours, _ = cv2.findContours(mask_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 用最大的輪廓 (理論上一顆番茄合併後應該只有一片)
    contour = max(contours, key=cv2.contourArea)
    mask_area = cv2.contourArea(contour)
    if mask_area <= 0:
        return None

    # bbox 長寬比
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0

    # solidity: mask 面積 / bbox 內接橢圓面積 (橢圓固定水平、不跟著輪廓角度轉，直接用 bbox 的 w, h 當長短軸)
    if w > 0 and h > 0:
        ellipse = ((x + w / 2.0, y + h / 2.0), (w, h), 0.0)
        ellipse_area = math.pi * (w / 2.0) * (h / 2.0)
        solidity = mask_area / ellipse_area if ellipse_area > 0 else 0
    else:
        ellipse = None
        solidity = 0

    return {
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "bbox_xywh": (x, y, w, h),
        "contour": contour,
        "ellipse": ellipse,
    }


def bbox_overlap_ratio(box_a, box_b):
    """
    box_a, box_b: (x1, y1, x2, y2)
    回傳 box_b 對 box_a 的重疊比例 = 交集面積 / box_a 面積
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    if area_a <= 0:
        return 0.0
    return inter_area / area_a


def judge_occlusion(idx, tomato_boxes, tomato_metrics, other_boxes):
    """
    對第 idx 顆番茄判斷是否被遮擋。
    other_boxes: 所有「其他」instance 的 bbox 列表 (其他番茄 + stem, 不含自己)
    回傳 (occluded: bool, reason: str, max_overlap: float)
    """
    metrics = tomato_metrics[idx]
    reasons = []

    if metrics is None:
        return True, "no_contour", 0.0

    ar = metrics["aspect_ratio"]
    sol = metrics["solidity"]

    if ar < ASPECT_RATIO_LOW or ar > ASPECT_RATIO_HIGH:
        reasons.append(f"aspect_ratio={ar:.2f}")

    if sol < SOLIDITY_THRESH:
        reasons.append(f"solidity={sol:.2f}")

    my_box = tomato_boxes[idx]
    max_overlap = 0.0
    for ob in other_boxes:
        ov = bbox_overlap_ratio(my_box, ob)
        max_overlap = max(max_overlap, ov)

    # 第三個指標 (bbox overlap) 先拔掉，不納入遮擋判斷，只保留數值計算供顯示/調參參考
    # if max_overlap > BBOX_OVERLAP_THRESH:
    #     reasons.append(f"bbox_overlap={max_overlap:.2f}")

    occluded = len(reasons) > 0
    reason_str = ";".join(reasons) if reasons else "clean"
    return occluded, reason_str, max_overlap


# ============ 主流程 ============
image_paths = []
for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
    image_paths.extend(glob.glob(os.path.join(img_dir, ext)))

if not image_paths:
    print(f"⚠️ 在 {img_dir} 找不到任何圖片！")

for path in sorted(image_paths):
    results = model(path, imgsz=1024, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, agnostic_nms=True)

    if not results:
        print(f"⚠️ {os.path.basename(path)}: 圖片讀取失敗或沒有結果,跳過")
        continue

    r = results[0]

    if r.masks is None or r.boxes is None:
        print(f"{os.path.basename(path)}: 沒有偵測到任何 instance,跳過")
        continue

    cls_arr = r.boxes.cls.cpu().numpy()
    xyxy_arr = r.boxes.xyxy.cpu().numpy()
    masks_data = r.masks.data.cpu().numpy()  # (N, h, w), 注意這是 model 輸出解析度,不一定等於原圖

    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ {os.path.basename(path)}: cv2 無法讀取這張圖,跳過")
        continue
    img_h, img_w = img.shape[:2]

    # mask 要 resize 回原圖大小才能跟 bbox 對齊算輪廓
    tomato_idx_list = [i for i in range(len(cls_arr)) if int(cls_arr[i]) == TOMATO_CLS]
    stem_idx_list = [i for i in range(len(cls_arr)) if int(cls_arr[i]) == STEM_CLS]

    tomato_boxes = [tuple(xyxy_arr[i]) for i in tomato_idx_list]
    tomato_metrics = []
    for i in tomato_idx_list:
        m = masks_data[i]
        m_resized = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        tomato_metrics.append(compute_shape_metrics(m_resized))

    stem_boxes = [tuple(xyxy_arr[i]) for i in stem_idx_list]

    vis = img.copy()
    print(f"\n=== {os.path.basename(path)} ===")

    # 依 bbox 中心點 x 座標由右至左排序,產生顯示用的 id (#1 = 最右邊那顆)
    center_x_list = [(b[0] + b[2]) / 2.0 for b in tomato_boxes]
    order_right_to_left = sorted(range(len(tomato_boxes)), key=lambda j: center_x_list[j], reverse=True)
    display_id = {local_idx: rank + 1 for rank, local_idx in enumerate(order_right_to_left)}
    summary_entries = []

    for local_idx in range(len(tomato_idx_list)):
        # 其他 instance = 除了自己以外的所有番茄 + 所有 stem
        other_boxes = [b for j, b in enumerate(tomato_boxes) if j != local_idx] + stem_boxes

        occluded, reason, max_overlap = judge_occlusion(
            local_idx, tomato_boxes, tomato_metrics, other_boxes
        )

        x1, y1, x2, y2 = map(int, tomato_boxes[local_idx])
        color = (0, 0, 255) if occluded else (0, 255, 0)  # BGR: 紅=遮擋, 綠=乾淨
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # bbox 旁標 id,方便對照左上角清單是哪一顆
        id_label = f"#{display_id[local_idx]}"
        cv2.putText(vis, id_label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        metrics = tomato_metrics[local_idx]

        if metrics is not None:
            # mask 實際輪廓: 黃色實線 -> 缺角/凹陷處會直接看到往內縮
            cv2.drawContours(vis, [metrics["contour"]], -1, (0, 255, 255), 2)

            # 擬合橢圓: 藍色線 -> 輪廓跟橢圓之間的落差就是形狀偏離程度 (取代原本的 convex hull)
            if metrics["ellipse"] is not None:
                cv2.ellipse(vis, metrics["ellipse"], (255, 128, 0), 2)

        # 計算 l/w、mask/all 數值,並各自判斷是否合格(用來各自上色,不是看整體結果)
        if metrics is not None:
            ar_val = metrics["aspect_ratio"]
            sol_val = metrics["solidity"]
            ar_fail = ar_val < ASPECT_RATIO_LOW or ar_val > ASPECT_RATIO_HIGH
            sol_fail = sol_val < SOLIDITY_THRESH
            ar_str = f"{ar_val:.2f}"
            sol_str = f"{sol_val:.2f}"
        else:
            ar_fail = sol_fail = True
            ar_str = sol_str = "NA"

        summary_entries.append((display_id[local_idx], ar_str, sol_str, ar_fail, sol_fail, occluded))

        print(f"  tomato #{display_id[local_idx]} (raw_idx={local_idx}): occluded={occluded} | l/w={ar_str} mask/all={sol_str} "
              f"overlap={max_overlap:.2f} | reason=({reason})")

    # 左上角統一顯示所有番茄的判斷清單: id / l/w / mask/all / pass or fail
    # l/w 跟 mask/all 各自依「自己有沒有超標」上色,不是看整體 pass/fail 結果
    summary_font, summary_scale, summary_thick = cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
    line_h = 60
    GREEN, RED = (0, 255, 0), (0, 0, 255)
    seg_gap = 30
    for row, (tid, ar_str, sol_str, ar_fail, sol_fail, occluded) in enumerate(sorted(summary_entries, key=lambda e: e[0])):
        status = "fail" if occluded else "pass"
        status_color = RED if occluded else GREEN
        line_y = line_h + row * line_h
        cursor_x = 10
        segments = [
            (f"id={tid}", status_color),
            (f"l/w={ar_str}", RED if ar_fail else GREEN),
            (f"mask/all={sol_str}", RED if sol_fail else GREEN),
            (status, status_color),
        ]
        for seg_text, seg_color in segments:
            cv2.putText(vis, seg_text, (cursor_x, line_y), summary_font, summary_scale, seg_color, summary_thick)
            (seg_w, _), _ = cv2.getTextSize(seg_text, summary_font, summary_scale, summary_thick)
            cursor_x += seg_w + seg_gap

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(vis[:, :, ::-1])
    plt.title(os.path.basename(path))
    plt.axis("off")
    plt.draw()

    # 不用會整個霸占住的 plt.show()，改成每 0.1 秒交還控制權一次，
    # 這樣 Ctrl+C 才有機會被立刻接收到，不用等視窗自己有動靜才反應。
    # 視窗被關掉(叉掉)就自然跳出迴圈，進入下一張圖。
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)

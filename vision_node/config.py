"""
============================================================================
 視覺節點參數設定區
============================================================================
 平常只要改這裡。所有其他模組（detector / skeleton / coordinates /
 vision_node ...）一律從這裡 import 常數，不重複定義。
============================================================================
"""
import os

# --- 執行環境切換 (real / isaac) --------------------------------------------
# 用環境變數選，預設 real：
#   python3 -m vision_node.main                 → 實機 (real)
#   VISION_MODE=isaac python3 -m vision_node.main → Isaac Sim
# 舉凡「real 跟 isaac 這兩種環境本來就不一樣」的東西，都收斂到這裡用 VISION_MODE 切，
# 不要在其他檔案裡另外寫 if VISION_MODE == ... 的分支。
VISION_MODE = os.environ.get('VISION_MODE', 'real').strip().lower()
if VISION_MODE not in ('real', 'isaac'):
    VISION_MODE = 'isaac'

_MODEL_PATH_BY_MODE = {
    'real':  '/home/lab604/tm_ws/python_real/best.pt',
    'isaac': '/home/terry/Desktop/stem_isaac_train/runs/segment/tomato_stem/exp6/weights/best.pt',
}
_CAMERA_TOPICS_BY_MODE = {
    'real': {
        'color': '/camera/camera/color/image_raw',
        'depth': '/camera/camera/aligned_depth_to_color/image_raw',
        'camera_info': '/camera/camera/color/camera_info',
    },
    'isaac': {
        'color': '/camera/color/image_raw',
        'depth': '/camera/depth/image_rect_raw',
        'camera_info': '/camera/camera_info',
    },
}
# 相機外參 (link_6 -> camera_optical_frame)：
#   real  → easy_handeye2 手眼標定結果，標定日期 2026-08-27，14 組樣本，OpenCV/Tsai-Lenz，
#           搭配 ROS2 camera_calibration 重新校正過的 RGB 內參（見 d435i_rgb_calib.yaml）
#   isaac → USD 場景裡相機掛載位置是已知的固定值，不需要標定，沿用舊版 CAMERA_OFFSET_Y=0.12
_CAMERA_EXTRINSIC_BY_MODE = {
    #''' 原本的
    #'real': {
    #    'translation': (0.024829, 0.113278, -0.002086),                       # (x, y, z) 單位: m
    #    'rotation_quat': (-0.020266, -0.005718, -0.998664, 0.047190),         # (x, y, z, w)   
    #}
    #'''

    'real': {
        'translation': (0.031397, 0.125875, -0.020195),                       # (x, y, z) 單位: m
        'rotation_quat': (-0.006734, -0.048451, -0.998653, 0.017276),         # (x, y, z, w)
    },
    'isaac': {
        'translation': (0.0, 0.12, 0.0),
        'rotation_quat': (0.0, 0.0, 0.0, 1.0),
    },
}
# 反投影 x/y 正負號 (舊版 coordinates.py 寫死加負號)：
#   real  → 標定四元數本身就內含這個翻轉，不用再手動加負號
#   isaac → 單位旋轉沒有內含翻轉，沿用舊版手動負號
_BACKPROJECT_XY_SIGN_BY_MODE = {
    'real': 1.0,
    'isaac': -1.0,
}
BACKPROJECT_XY_SIGN = _BACKPROJECT_XY_SIGN_BY_MODE[VISION_MODE]
# 手動校正過的 RGB 內參檔 (d435i_rgb_calib.yaml) 只在實機需要：修正實體鏡頭的畸變。
# Isaac 模擬相機沒有鏡頭畸變，camera_info topic 發布的內參本來就是準的，不用另外覆蓋。
ENABLE_MANUAL_INTRINSIC_CALIB = (VISION_MODE == 'real')

# --- 模型與分類 ID ---------------------------------------------------------
MODEL_PATH = _MODEL_PATH_BY_MODE[VISION_MODE]
STEM_CLASS_ID = 0
TOMATO_CLASS_ID = 1

# --- 相機訂閱 topic ----------------------------------------------------------
COLOR_TOPIC = _CAMERA_TOPICS_BY_MODE[VISION_MODE]['color']
DEPTH_TOPIC = _CAMERA_TOPICS_BY_MODE[VISION_MODE]['depth']
CAMERA_INFO_TOPIC = _CAMERA_TOPICS_BY_MODE[VISION_MODE]['camera_info']

# --- YOLO 推論參數 ----------------------------------------------------------
YOLO_IMGSZ = 1024
YOLO_CONF = 0.75
YOLO_IOU = 0.45                      # NMS IoU 門檻，沿用 test_occlusion.py 調過的值（原本沒接進主流程，


# --- 顯示視窗 ----------------------------------------------------------
DISPLAY_SCALE = 1.0                  # cv2.imshow 顯示視窗的放大倍率，不影響偵測/座標計算

# --- 座標系名稱 --------------------------------------------------------------
WORLD_FRAME = 'world'
CAMERA_OPTICAL_FRAME = 'camera_optical_frame'
ARM_FLANGE_FRAME = 'link_6'          # 相機掛載的父座標系
CAMERA_EXTRINSIC_TRANSLATION = _CAMERA_EXTRINSIC_BY_MODE[VISION_MODE]['translation']
CAMERA_EXTRINSIC_ROTATION_QUAT = _CAMERA_EXTRINSIC_BY_MODE[VISION_MODE]['rotation_quat']

# --- 深度估計 -----------------------------------------------------------
DEPTH_WINDOW = 5                     # 番茄/果梗中心取深度的視窗半徑 (px)
MIN_STEM_DEPTH_PX = 5                # 果梗 mask 內深度點數低於這個值就退回 3x3 fallback
DEPTH_MM_THRESHOLD = 10.0            # 深度值大於這個數字視為單位是 mm，要 /1000 轉成公尺
MIN_VALID_DEPTH_M = 0.01             # 深度小於這個值視為無效

# --- 番茄遮擋判斷 (搬自 test_occlusion.py，門檻沿用同一組) -------------------------
#ASPECT_RATIO_LOW = 0.9               # isaac_bbox 長寬比下限
#ASPECT_RATIO_HIGH = 1.5              # isaac_bbox 長寬比上限
ASPECT_RATIO_LOW = 0.9                # real_bbox 長寬比下限
ASPECT_RATIO_HIGH = 1.3               # real_bbox 長寬比上限
SOLIDITY_THRESH = 0.85                # mask 面積 / 擬合橢圓面積，低於這個判定形狀跟橢圓差太多

# --- 果梗骨架化 / 抓取點 -----------------------------------------------------
# 'ratio'：固定沿骨架路徑走 (GRASP_RATIO_MIN+GRASP_RATIO_MAX)/2 比例(從calyx算起)，
#          不管果梗多長都保證落在 C 端附近同一個相對位置；適合果梗普遍偏短(量測約在
#          2cm 以內)的情境，用固定物理距離很容易逼近甚至超過整根果梗長度。
# 'distance'：固定物理距離 GRASP_TARGET_DIST_M，太短量不到才退回比例保底；適合果梗
#          長度差異大、且長果梗夠長時的情境。目前實測這批果梗普遍偏短，先用 'ratio'。
GRASP_METHOD = 'distance'
GRASP_RATIO_MIN = 0.4
GRASP_RATIO_MAX = 0.5
GRASP_TARGET_DIST_M = 0.015           # 只有 GRASP_METHOD='distance' 時才用，抓取點目標離calyx的實際距離(m)
OVERLAP_SUPPRESS_THRESH = 0.5        # 果梗 mask 互相重疊比例超過此值視為重複偵測

# --- StemTracker 滑動視窗 ------------------------------------------------
STEM_MATCH_DIST_PX = 40.0            # 前後幀配對同一根果梗的最大像素距離
STEM_TRACK_WINDOW = 7                # 滑動視窗長度 (幀數)
STEM_TRACK_MAX_MISS = 5              # 連續幾幀沒配對到就判定 track 消失

# --- 配對 / 遮擋 時間穩定 (修紅綠燈閃爍) -------------------------------------
# assign_stem_tomato_pairs、occlusion.py 都只吃「這一幀」的量測，深度/mask 邊緣雜訊
# 會讓配對對象或遮擋判定在門檻附近來回跳，畫面紅綠燈跟著閃。果梗跟番茄各自獨立閃
# （框顏色算法本來就分開），所以分開穩定，互不相干：
#   PAIR_STICKY_*  穩定「這根果梗配到哪顆番茄」(target_selector.py)
#   TOMATO_*       穩定「這顆番茄是否判定遮擋」(tomato_tracker.py)
PAIR_STICKY_MATCH_DIST_M = 0.03      # 判定「前一幀同一根果梗/同一顆番茄」的最大位移容忍(公尺)
PAIR_STICKY_DISCOUNT = 0.7           # 前一幀配對過的番茄，距離打這個折扣再排序，
                                      # 避免在幾乎等距的候選番茄之間，因量測雜訊每幀跳配
TOMATO_MATCH_DIST_M = 0.03           # 番茄前後幀配對容忍距離(公尺)，用於穩定遮擋判斷
TOMATO_OCC_CONFIRM_FRAMES = 3        # 遮擋判定要連續幾幀改變才真的切換，單幀雜訊不算數
TOMATO_TRACK_MAX_MISS = 5            # 番茄追蹤連續幾幀沒配對到就視為消失，清掉暫存狀態

# --- 掃描 / 選取流程 -------------------------------------------------------
EMPTY_SCAN_GRACE = 2                 # 連續幾次空掃描才回報 NO_TARGET
SCAN_PRINT_INTERVAL = 1.5            # 終端機列印候選清單的節流間隔 (s)
MAX_REACH_M = 1.5                    # 距離基座超過此值的候選直接排除

# --- 點雲閘門 / 目標過濾 -----------------------------------------------
# ★ 原本這個 gate + 轉發是寫在手臂端 (arm_car_vector_z.py) 的 TM5MTaskNode，
#   現在搬過來這裡，理由：mask / depth / camera_info / TF 全部都已經在這支程式手上，
#   不用再把這些資料跨節點丟來丟去。
# ★ 掃描期間刻意不轉發點雲，OctoMap 保持空白（呼應 _enter_scanning 的全域清空），
#   直到選定目標的那一刻，才發布目標遮罩，交給獨立的 cloud_filter_node.py 一次性建圖。
TARGET_MASK_DILATE_PX = 15           # 目標番茄 mask 膨脹核心大小 (px)，先給保守值，實測後再調
STEM_MASK_DILATE_PX = 9              # 目標果梗 mask 膨脹核心大小 (px)，果梗較細先給比番茄小的值，實測後再調
OCTOMAP_UPDATE_WAIT_SEC = 1.5        # 發布過濾點雲後，等 MoveIt2 的 octomap updater 處理完再繼續

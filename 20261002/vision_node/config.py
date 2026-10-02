"""
============================================================================
 視覺節點參數設定區
============================================================================
 平常只要改這裡。所有其他模組（fine_* 精定位、coarse_* 粗定位、coordinates
 ...）一律從這裡 import 常數，不重複定義。
 檔名約定：fine_* = 精定位（手臂相機）、coarse_* = 粗定位（車載相機）、
 沒前綴 = 兩邊共用（config / coordinates）；cloud_filter_node 是獨立節點，名稱不動。
============================================================================
"""
import os

# --- 執行環境切換 (real / isaac) --------------------------------------------
# 用環境變數選，預設 real：
#   python3 -m vision_node.fine_main                 → 實機 (real)
#   VISION_MODE=isaac python3 -m vision_node.fine_main → Isaac Sim
# 舉凡「real 跟 isaac 這兩種環境本來就不一樣」的東西，都收斂到這裡用 VISION_MODE 切，
# 不要在其他檔案裡另外寫 if VISION_MODE == ... 的分支。
VISION_MODE = os.environ.get('VISION_MODE', 'real').strip().lower()
if VISION_MODE not in ('real', 'isaac'):
    VISION_MODE = 'isaac'

# --- 場景切換 (car / lab)（只影響 isaac 模式下的車載相機外參，real 還沒有這個區分）--
# 跟 arm_node/config.py 的 MODE 對應同一組場景，但這裡是獨立的常數（vision_node 不 import
# arm_node），改場景時兩邊都要記得改：這裡改 SCENE_MODE，arm_node/config.py 改 MODE。
SCENE_MODE = 'car'   # 'car' 或 'lab' ← 只改這一行切換車載相機外參

_MODEL_PATH_BY_MODE = {
    'real':  '/home/lab604/tm_ws/python_real/yolo_model/1024/best.pt',
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
YOLO_IOU = 0.45                      # NMS IoU 門檻，沿用離線遮擋測試腳本（已刪除）調過的值（原本沒接進主流程，


# --- 顯示視窗 ----------------------------------------------------------
DISPLAY_SCALE = 1.0                  # cv2.imshow 顯示視窗的放大倍率，不影響偵測/座標計算

# --- 座標系名稱 --------------------------------------------------------------
WORLD_FRAME = 'world'
CAMERA_OPTICAL_FRAME = 'camera_optical_frame'
ARM_FLANGE_FRAME = 'link_6'          # 相機掛載的父座標系
CAMERA_EXTRINSIC_TRANSLATION = _CAMERA_EXTRINSIC_BY_MODE[VISION_MODE]['translation']
CAMERA_EXTRINSIC_ROTATION_QUAT = _CAMERA_EXTRINSIC_BY_MODE[VISION_MODE]['rotation_quat']

# --- 車載相機（粗定位）-------------------------------------------------------
# ★ 2026-09-21：isaac 的 topic 跟外參已填入真值；real 還是【佔位值】（topic 跟外參都要
#   等實機相機確定再填）。real 填入真值之前，算出來的 base 座標是不準的，只能驗證流程跑得通
#   （coarse_node 啟動時會印警告，畫面上也會標示）。
# 外參 (VEHICLE_CAMERA_PARENT_FRAME -> VEHICLE_CAMERA_OPTICAL_FRAME)：
#   isaac → 從 USD 讀相機對父節點的位移/旋轉；real → 之後標定。格式跟上面手臂相機一樣。
# 反投影 x/y 正負號沿用上面的 BACKPROJECT_XY_SIGN（coordinates.py 共用同一個常數），
# 所以填 isaac 外參時，旋轉要跟手臂相機同一套慣例（單位旋轉 + 手動負號）。
BASE_FRAME = 'base'                  # 粗定位結果要轉到的座標系（必須跟 arm_node/config.py 的 BASE_FRAME 一致）
VEHICLE_CAMERA_PARENT_FRAME = BASE_FRAME
VEHICLE_CAMERA_OPTICAL_FRAME = 'vehicle_camera_optical_frame'
_VEHICLE_CAMERA_TOPICS_BY_MODE = {
    'real': {
        'color': '/vehicle_camera/color/image_raw',            # TODO: 佔位值
        'depth': '/vehicle_camera/depth/image_rect_raw',       # TODO: 佔位值
        'camera_info': '/vehicle_camera/camera_info',          # TODO: 佔位值
    },
    'isaac': {   # 已對照場景 ActionGraph 的 ROS2CameraHelper 設定（frameId=fixed_cam_optical_frame）
        'color': '/vehicle_camera/color/image_raw',
        'depth': '/vehicle_camera/depth/image_rect_raw',
        'camera_info': '/vehicle_camera/camera_info',
    },
}
_VEHICLE_CAMERA_EXTRINSIC_REAL = {
    'translation': (0.0, 0.0, 0.0),                        # TODO: 佔位值 (x, y, z) 單位: m
    'rotation_quat': (0.0, 0.0, 0.0, 1.0),                 # TODO: 佔位值 (x, y, z, w)
}
# isaac 模式下車載相機外參依 SCENE_MODE（car/lab）分開存，兩個場景的相機掛法/位置不一樣。
# 兩組都是直接在 Isaac Sim 的 Script Editor 裡查 base -> Camera_OmniVision_OV9782_Color 的
# 解析後 world transform（用 omni.usd.get_context().get_stage() + UsdGeom.XformCache，
# 不是手動疊歐拉角/資產內部偏移算出來的——後者容易算錯，直接讀解析後的結果最準）。
# 旋轉慣例：BACKPROJECT_XY_SIGN=-1 的 isaac 模式，TF 旋轉 = 相機 prim 旋轉 · Ry(180°)
# （USD 相機預設看 -Z，Ry(180°) 轉成 ROS optical frame 慣例的 +Z；用手臂相機 Came_link
# 的 USD 旋轉 Ry(180°) → TF 單位旋轉 驗證過，這裡沿用同一套公式）。
_VEHICLE_CAMERA_EXTRINSIC_BY_SCENE = {
    # 2026-09-21 從場景 9_lab_big tomato_v1.usd 查出：base -> Camera_OmniVision_OV9782_Color
    # 相機實際視線：水平偏航 -25.6°、俯仰 -6.6°（略朝下）。
    'lab': {
        'translation': (0.007584, 0.484331, 0.512971),         # (x, y, z) 單位: m
        'rotation_quat': (0.629232, 0.401418, 0.351337, 0.565236),   # (x, y, z, w)
    },
    # 2026-09-25 從場景 new_car_arm_v2.usd 查出：base -> Camera_OmniVision_OV9782_Color
    # 量到的解析後 pose：translate=(0.442003, -0.008527, 0.514699)，
    # quat(w,x,y,z)=(0.223509, 0.223509, 0.670853, 0.670852)，套上面 Ry(180°) 慣例算出下面
    # 這組 TF 旋轉。相機實際視線：水平偏航 -126.9°、俯仰 ≈0°（水平安裝，沒有明顯朝下/朝上）。
    'car': {
        'translation': (0.442003, -0.008527, 0.514699),        # (x, y, z) 單位: m
        'rotation_quat': (-0.670853, 0.223509, 0.223510, -0.670853),   # (x, y, z, w)
    },
}
if VISION_MODE == 'isaac':
    _VEHICLE_CAMERA_EXTRINSIC = _VEHICLE_CAMERA_EXTRINSIC_BY_SCENE[SCENE_MODE]
else:
    _VEHICLE_CAMERA_EXTRINSIC = _VEHICLE_CAMERA_EXTRINSIC_REAL
_VEHICLE_CAMERA_EXTRINSIC_BY_MODE = {'real': _VEHICLE_CAMERA_EXTRINSIC_REAL, 'isaac': _VEHICLE_CAMERA_EXTRINSIC}
# 只有 real 還是佔位值（isaac 已經填好真值），佔位值會讓 coarse_node 啟動時印警告、畫面標示
VEHICLE_CAMERA_EXTRINSIC_IS_PLACEHOLDER = (VISION_MODE == 'real')
VEHICLE_COLOR_TOPIC = _VEHICLE_CAMERA_TOPICS_BY_MODE[VISION_MODE]['color']
VEHICLE_DEPTH_TOPIC = _VEHICLE_CAMERA_TOPICS_BY_MODE[VISION_MODE]['depth']
VEHICLE_CAMERA_INFO_TOPIC = _VEHICLE_CAMERA_TOPICS_BY_MODE[VISION_MODE]['camera_info']
VEHICLE_CAMERA_EXTRINSIC_TRANSLATION = _VEHICLE_CAMERA_EXTRINSIC_BY_MODE[VISION_MODE]['translation']
VEHICLE_CAMERA_EXTRINSIC_ROTATION_QUAT = _VEHICLE_CAMERA_EXTRINSIC_BY_MODE[VISION_MODE]['rotation_quat']

# 粗定位自己的 YOLO 設定，跟精定位的 MODEL_PATH / YOLO_CONF 互相獨立、各調各的。
# 粗定位相機離番茄遠、番茄在畫面裡小，信心值通常比精定位近拍低，用同一個 0.75 很容易
# 整批被濾掉。數值依實測調過（見下面 COARSE_YOLO_IMGSZ 的說明）。
# 模型目前先指向跟精定位同一顆（單張照片測試有辨識出來），之後要換粗定位專用模型，
# 只要改 COARSE_MODEL_PATH。iou 仍共用 YOLO_IOU（imgsz 已獨立成 COARSE_YOLO_IMGSZ）。
COARSE_MODEL_PATH = MODEL_PATH
# 2026-09-21 實測（車載相機 1280x720、番茄距離約 0.8m、4 顆貼在一起）：
#   imgsz=1024 → 被遮住的那顆併進旁邊的框，4 顆只認出 3 顆（框寬 33/40/54px，54 是併框）
#   imgsz=1280 → 4 顆各一個框（寬 34/36/39/40px），信心 0.91/0.90/0.89/0.30
#   iou 調高(0.7)會讓同一顆重複框，所以 iou 維持共用的 0.45 不動。

# 1280 = 車載相機原圖寬度，不縮圖；比 1024 慢一些。第 4 顆被擋掉大半、信心只有 0.30，
# 所以信心門檻降到 0.25（乾淨場景實測沒有多餘誤判，之後遇到帶葉子的場景再重新確認）。
COARSE_YOLO_IMGSZ = 1280
COARSE_YOLO_CONF = 0.4

# 粗定位多幀合併（coarse_tracker.py）：跨幀配對同一顆番茄、座標取中位數、ID 固定。
# 車是停下來才做粗定位，番茄不會動，所以不用像精定位那樣防移動。
COARSE_TRACK_MATCH_DIST_M = 0.03     # 前後幀配成同一顆的最大 base 座標距離 (m)。要小於番茄之間
                                      # 的間距（貼在一起的約差 4~5cm，否則會配錯到旁邊那顆），
                                      # 又要大於單幀抖動（實測約幾 mm）
COARSE_TRACK_WINDOW = 10             # 中位數視窗長度（最近幾幀）
COARSE_TRACK_MIN_HITS = 5            # 累積配對幾次才確認、發 ID（過濾單幀雜訊）
COARSE_TRACK_MAX_MISS = 30           # 連續幾幀沒配到就移除、釋出 ID（信心貼著門檻的番茄偶爾會
                                      # 漏幀，給比較寬的容忍度，避免 ID 因此重發）
COARSE_PRINT_CHANGE_M = 0.01         # 終端機只在「番茄清單有變」時才印：新增/消失，或任一顆位置
                                      # 跟上次印出來的相差超過這個距離 (m)；沒變就不印，不會一直刷

# --- 深度估計 -----------------------------------------------------------
DEPTH_WINDOW = 5                     # 番茄/果梗中心取深度的視窗半徑 (px)
MIN_STEM_DEPTH_PX = 5                # 果梗 mask 內深度點數低於這個值就退回 3x3 fallback
DEPTH_MM_THRESHOLD = 10.0            # 深度值大於這個數字視為單位是 mm，要 /1000 轉成公尺
MIN_VALID_DEPTH_M = 0.01             # 深度小於這個值視為無效

# --- 番茄遮擋判斷 (搬自離線遮擋測試腳本，門檻沿用同一組) -------------------------
ASPECT_RATIO_LOW = 0.9               # real & isaac_bbox 長寬比下限
ASPECT_RATIO_HIGH = 1.5              # real_bbox 長寬比上限
#ASPECT_RATIO_HIGH = 1.3             # real_bbox 長寬比上限
SOLIDITY_THRESH = 0.9               # mask 面積 / 擬合橢圓面積，低於這個判定形狀跟橢圓差太多

# --- 果梗骨架化 / 抓取點 -----------------------------------------------------
# 'ratio'：固定沿骨架路徑走 (GRASP_RATIO_MIN+GRASP_RATIO_MAX)/2 比例(從calyx算起)，
#          不管果梗多長都保證落在 C 端附近同一個相對位置；適合果梗普遍偏短(量測約在
#          2cm 以內)的情境，用固定物理距離很容易逼近甚至超過整根果梗長度。
# 'distance'：固定物理距離 GRASP_TARGET_DIST_M，太短量不到才退回比例保底；適合果梗
#          長度差異大、且長果梗夠長時的情境。目前實測這批果梗普遍偏短，先用 'ratio'。
GRASP_METHOD = 'ratio'
GRASP_RATIO_MIN = 0.4
GRASP_RATIO_MAX = 0.5
GRASP_TARGET_DIST_M = 0.015           # 只有 GRASP_METHOD='distance' 時才用，抓取點目標離calyx的實際距離(m)

# GRASP_METHOD='distance' 時，怎麼找那個目標距離的點（只有這個模式才會讀）：
# 'accumulate'：沿骨架逐點量深度、反投影成 3D 座標，相鄰點累加真實弧長，找到累加至
#               GRASP_TARGET_DIST_M 的點——不管果梗彎不彎都精確，但要逐點量深度，
#               運算量較大（有做提前停止優化，見 coordinates.py find_grasp_point_by_3d_distance）。
# 'chord_ratio'：只量頭尾兩個端點的深度，用端點 3D 直線距離(弦長)算比例、乘上路徑總
#               點數直接定位——只要量 2 個點，快很多，但把骨架路徑當直線處理：果梗
#               彎曲時弦長 < 實際弧長，算出來的抓取點會比正確位置更靠近枝條端（系統性
#               誤差隨彎曲程度增加，果梗越直越短誤差越小）。
GRASP_DISTANCE_MODE = 'accumulate'
OVERLAP_SUPPRESS_THRESH = 0.5        # 果梗 mask 互相重疊比例超過此值視為重複偵測

# --- StemTracker 滑動視窗 ------------------------------------------------
STEM_MATCH_DIST_PX = 40.0            # 前後幀配對同一根果梗的最大像素距離
STEM_TRACK_WINDOW = 15                # 滑動視窗長度 (幀數)
STEM_TRACK_MAX_MISS = 5              # 連續幾幀沒配對到就判定 track 消失

# --- 配對 / 遮擋 時間穩定 (修紅綠燈閃爍) -------------------------------------
# assign_stem_tomato_pairs、fine_occlusion.py 都只吃「這一幀」的量測，深度/mask 邊緣雜訊
# 會讓配對對象或遮擋判定在門檻附近來回跳，畫面紅綠燈跟著閃。果梗跟番茄各自獨立閃
# （框顏色算法本來就分開），所以分開穩定，互不相干：
#   PAIR_STICKY_*  穩定「這根果梗配到哪顆番茄」(fine_target_selector.py)
#   TOMATO_*       穩定「這顆番茄是否判定遮擋」(fine_tomato_tracker.py)
PAIR_STICKY_MATCH_DIST_M = 0.03      # 判定「前一幀同一根果梗/同一顆番茄」的最大位移容忍(公尺)
PAIR_STICKY_DISCOUNT = 0.7           # 前一幀配對過的番茄，距離打這個折扣再排序，
                                      # 避免在幾乎等距的候選番茄之間，因量測雜訊每幀跳配
MAX_STEM_TOMATO_PAIR_DIST_M = 0.05   # 實測校準過的值：關掉門檻(inf)測試時量到正確配對
                                      # ~0.041m、錯誤硬湊的配對 ~0.54~0.58m，取中間值。
                                      # 果梗端點(果實端)到番茄中心點的距離上限(公尺)，
                                      # 超過就算是目前最近的候選也不配對，避免孤立果梗
                                      # 硬配一顆明顯不是它的番茄
TOMATO_MATCH_DIST_M = 0.03           # 番茄前後幀配對容忍距離(公尺)，用於穩定遮擋判斷
TOMATO_OCC_CONFIRM_FRAMES = 3        # 遮擋判定要連續幾幀改變才真的切換，單幀雜訊不算數
TOMATO_TRACK_MAX_MISS = 5            # 番茄追蹤連續幾幀沒配對到就視為消失，清掉暫存狀態
TOMATO_POS_SMOOTH_WINDOW = 7         # 番茄中心座標跨幀平均的視窗長度(幀數)，跟 STEM_TRACK_WINDOW 同量級，做法統一

# --- 掃描 / 選取流程 -------------------------------------------------------
# 2026-09-16：拿掉原本獨立的 EMPTY_SCAN_GRACE（連續幾次空掃描就直接回報 NO_TARGET）
# ——完全沒偵測到任何番茄，可能是真的沒有，也可能是被完全遮擋（例如故意拿東西整個
# 擋住鏡頭），單看偵測結果沒辦法分辨，卻直接跳過備用視角重試邏輯放棄回家，導致真的
# 完全遮擋時永遠不會觸發換視角。改成兩種情況共用下面這個 OCCLUDED_SCAN_GRACE，統一
# 都先試過備用視角、真的都試完還是看不到才回家（見 fine_node.py 的
# _maybe_print_and_trigger_pick）。
OCCLUDED_SCAN_GRACE = 3              # 連續幾輪判定「看不到能挑的目標」（候選全部遮擋，
                                      # 或整幀完全沒偵測到）才真的回報 OCCLUDED、換視角；
                                      # 剛到新視角時追蹤視窗還沒填滿，避免只看一輪就誤判
SCAN_PRINT_INTERVAL = 1.5            # 終端機列印候選清單的節流間隔 (s)
MAX_REACH_M = 1.5                    # 距離基座超過此值的候選直接排除
CANDIDATE_REFRESH_INTERVAL_SEC = 1.0 # 等待使用者輸入 ID 期間，即時面板重繪的節流間隔 (s)
# refresh_valid() 判定候選「消失或移動過大」原本單幀沒配到就立刻標失效——單幀深度雜訊
# (2026-09-09 實測：同一像素、同一 grasp_idx，深度相機讀出來的 3D 座標還是會跳，見
# PROGRESS.md) 或 YOLO 單幀漏偵測都會誤觸發，讓即時面板一直閃「已失效」。連續失敗超過
# 這個次數才真的判失效，容忍偶發的單幀壞讀值。
REFRESH_VALID_MAX_MISS = 2

# --- 點雲閘門 / 目標過濾 -----------------------------------------------

# ★ 原本這個 gate + 轉發是寫在手臂端 (arm_car_vector_z.py) 的 TM5MTaskNode，
#   現在搬過來這裡，理由：mask / depth / camera_info / TF 全部都已經在這支程式手上，
#   不用再把這些資料跨節點丟來丟去。
# ★ 掃描期間刻意不轉發點雲；選定目標的那一刻，fine_node 先清空 OctoMap、再發布目標遮罩，
#   交給獨立的 cloud_filter_node.py 一次性建圖（2026-09-30 起清空改在這裡，不在手臂出發前）。
# ★ 2026-09-30 測試開關：選定目標時，點雲要挖掉哪些東西再送給 OctoMap。
#   True  = 挖掉「目標果梗 + 配對到的番茄」（原本的做法）
#   False = 只挖目標果梗，番茄留在點雲裡當障礙物（測試中：要留意夾爪下探時會不會被判定撞到番茄）
FILTER_CARVE_TOMATO = False
TARGET_MASK_DILATE_PX = 15           # 目標番茄 mask 膨脹核心大小 (px)，先給保守值，實測後再調
STEM_MASK_DILATE_PX = 31             # 目標果梗 mask 膨脹核心大小 (px)，2026-09-15：cv2.dilate 實際擴張約 N/2，9→15 只多擴張 3px 沒感覺，跳大做決定性測試（實際擴張約 15px）
OCTOMAP_UPDATE_WAIT_SEC = 2.5        # 發布過濾點雲後，等 cloud_filter_node.py 完成訊號的逾時保底；
                                      # 2026-09-15 改版：cloud_filter_node.py 正常路徑一等到 TF 就緒
                                      # （通常一兩幀內）就馬上發訊號，不用再等固定時間；這裡的 2.5s
                                      # 只是保底，要大於它自己的 TF_WAIT_TIMEOUT_SEC（2.0s）逾時保底，
                                      # 否則對方都還沒判定逾時、這邊就先自己放棄等待了

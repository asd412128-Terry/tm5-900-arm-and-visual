"""
============================================================================
 arm_node.config — 參數設定
============================================================================
 只要改環境，改最上面的 MODE 就好，不用動下面任何一行。
 MODE = 'car' → 差速車場景（Isaac Sim / 實車測試用，無固定障礙物）
 MODE = 'lab' → 實驗室桌面場景（有固定桌面/隔板/籃子等障礙物）
============================================================================
"""
import math
import os

# --- 執行環境切換 (real / isaac) --------------------------------------------
# 用環境變數選，預設 real：
#   python3 -m arm_node.main                → 實機 (real)
#   ARM_MODE=isaac python3 -m arm_node.main → Isaac Sim
# 跟下面的 MODE (car/lab，場景障礙物) 是兩個獨立的開關，互不影響。
ARM_MODE = os.environ.get('ARM_MODE', 'real').strip().lower()
if ARM_MODE not in ('real', 'isaac'):
    ARM_MODE = 'isaac'

# 2026-09-22：正常操作一定要 True——手臂回到 Home 後只會乖乖等 coarse_node 真的發來
# /coarse_target_point 才前往精定位，不會拿 FAKE_COARSE_TOMATO_POSE 隨便動（見
# arm_task_node.py 的 _start_new_round）。
# 想單獨測精定位（不開 coarse_node、不用等）才設 REQUIRE_COARSE_TARGET=0，這樣手臂
# 回到 Home 後會立刻用 FAKE_COARSE_TOMATO_POSE 頂著跑，行為退回今天加 coarse_node
# 整合之前那樣：
#   REQUIRE_COARSE_TARGET=0 ARM_MODE=isaac python3 -m arm_node.main
REQUIRE_COARSE_TARGET = os.environ.get('REQUIRE_COARSE_TARGET', '1').strip() not in ('0', 'false', 'False')

MODE = 'car'   # 'car' 或 'lab' ← 只改這一行切換環境

# ===========================================================================
# 依 MODE 切換的參數
# ===========================================================================
if MODE == 'car':
    ENABLE_CAR_BODY = True
    OBSTACLES = []

    POSE_HOME_DEG = [-90.0, -15.0, 65.0, -50.0, 90.0, 0.0]
    #POSE_HOME_DEG =[-90.0, -7.0, 125.0, -118.0, 90.0, 0.0]
    POSE_FINE_DEG = [-90.0, -7.0, 125.0, -118.0, 90.0, 0.0]

    # 假粗定位番茄座標（車用相機還沒接上，先頂著測 -30cm 管線；見下方共用參數區塊
    # COARSE_TO_FINE_RETREAT_M 的說明）。2026-09-03 用 /compute_fk 對 POSE_FINE_DEG
    # 實測、往前推 30cm 算出來。
    FAKE_COARSE_TOMATO_POSE = (-0.1223, -0.7242, 0.4838, 0.707107, 0.0, 0.0, 0.707107, math.radians(-90.0))
    # ★ 2026-09-25：手臂相機接近番茄的水平方位角，用固定值，不是「從基座畫一條線到
    # 番茄」算出來的角度——car 模式下，車子每次要開始摘採前都會先導航對齊、平行到
    # 植株排列方向，同一趟只固定往車身側面伸出去摘，不會因為番茄在排列中的哪個位置
    # 就跟著換角度。這個值是從上面 FAKE_COARSE_TOMATO_POSE 的姿態反推出來的
    # （已知在 real 上驗證過水平且垂直，J5=90°），供 arm_task_node.py 的
    # _camera_facing_pose 用，取代原本用 atan2(番茄y, 番茄x) 算方位角的做法。
    FIXED_APPROACH_AZIMUTH_DEG = -90.0

    VG_FINGER_SIZE = [0.005, 0.005, 0.05]   # 單根手指（虛擬夾爪碰撞體）[X, Y, Z]，對應 car 真實夾爪形狀

elif MODE == 'lab':
    ENABLE_CAR_BODY = False
    OBSTACLES = [
        {'id': 'table',           'type': 'cube',     'pos': [0.75, -0.1325, 0.03],     'size': [0.7, 1.205, 0.03]},
        #{'id': 'front_partition', 'type': 'cube',     'pos': [1.125, -0.1325, 0.29575], 'size': [0.05, 1.205, 0.5015]},
        #{'id': 'side_partition',  'type': 'cube',     'pos': [0.499, 0.495, 0.29575],   'size': [1.202, 0.05, 0.5015]},
        #{'id': 'computer',        'type': 'cube',     'pos': [0.7, -0.635, 0.25],       'size': [0.46, 0.18, 0.41]},
        #{'id': 'wall',            'type': 'cube',     'pos': [-0.5, 0.0, 0.5],          'size': [0.06, 2.0, 1.5]},
        #{'id': 'basket',          'type': 'cylinder', 'pos': [0.55, -0.63, 0.5063],     'size': [0.1, 0.075]},  # [高, 半徑]
    ]

    #POSE_HOME_DEG = [0.0, -15.0, 65.0, -50.0, 90.0, 0.0]
    POSE_HOME_DEG = [0.0, 0.0, 135.0, -135.0, 90.0, 0.0]
    POSE_FINE_DEG = [0.0, 0.0, 135.0, -135.0, 90.0, 0.0]

    # 假粗定位番茄座標（車用相機還沒接上，先頂著測 -30cm 管線；見下方共用參數區塊
    # COARSE_TO_FINE_RETREAT_M 的說明）。2026-09-03 用 /compute_fk 對 POSE_FINE_DEG
    # 實測、往前推 30cm 算出來。
    FAKE_COARSE_TOMATO_POSE = (0.7041, -0.1223, 0.3892, 0.5, 0.5, 0.5, 0.5, math.radians(0.0))
    # 同上 car 區塊的說明：固定接近方位角，從 FAKE_COARSE_TOMATO_POSE 的姿態反推出來。
    FIXED_APPROACH_AZIMUTH_DEG = 0.0
    VG_FINGER_SIZE = [0.005, 0.005, 0.05]    # isaac根手指（虛擬夾爪碰撞體）[X, Y, Z]，對應 isaac 真實夾爪形狀
    #VG_FINGER_SIZE = [0.005, 0.02, 0.075]   # real單根手指（虛擬夾爪碰撞體）[X, Y, Z]，對應 real 真實夾爪形狀

else:
    raise ValueError(f'未知 MODE: {MODE!r}，只能是 "car" 或 "lab"')


# ===========================================================================
# 共用參數（不分環境）
# ===========================================================================

# --- MoveIt 基本設定 -------------------------------------------------------
ARM_GROUP    = 'tmr_arm'
BASE_FRAME   = 'base'
EEF_LINK     = 'flange'
# 必須跟 vision_node/config.py 的 CAMERA_OPTICAL_FRAME 同一個字串——arm_task_node.py
# 清完 octomap 後補發的那包空點雲，frame_id 要用這個已知有接到 base 的 frame，
# 不然 MoveIt2 的 tf2_ros::MessageFilter 查不到 transform，一樣會被靜默丟棄，
# 白戳一場（見 cloud_filter_node.py 當初 sim_camera 斷鏈那次的教訓）。
CAMERA_OPTICAL_FRAME = 'camera_optical_frame'
ARM_FLANGE_FRAME = 'link_6'   # 手臂相機掛載的父座標系，跟 vision_node/config.py 同一個字串
# ★ 2026-09-22：跟 vision_node/config.py 的 CAMERA_EXTRINSIC_TRANSLATION/ROTATION_QUAT
# 逐值複製過來的（同一組手眼標定結果，依 ARM_MODE 對應 vision 端的 VISION_MODE）。
# 原本 MathUtils.camera_facing_flange_pose() 要用的相機外參(cam_t/cam_z)是查
# ARM_FLANGE_FRAME → CAMERA_OPTICAL_FRAME 這條 TF 取得，但這條 TF 是 fine_node.py
# 廣播的——如果只開 arm_task_node、沒開 fine_node.py，TF 查不到，會靜默退回「相機=
# 法蘭中心」的簡化假設（cam_t=0），算出來的目標點變成讓「法蘭」對準番茄、不是「相機」
# 對準番茄，實測證實這樣是錯的。這組外參本來就是寫死的已知常數、不是即時算出來的，
# 直接複製一份在這裡用，不用查 TF、不用依賴 fine_node.py 有沒有開。
# 兩邊要同步改：這裡改了記得也去確認 vision_node/config.py 的值有沒有跟著更新過。
_ARM_CAMERA_EXTRINSIC_BY_MODE = {
    'real': {
        'translation': (0.031397, 0.125875, -0.020195),
        'rotation_quat': (-0.006734, -0.048451, -0.998653, 0.017276),
    },
    'isaac': {
        'translation': (0.0, 0.12, 0.0),
        'rotation_quat': (0.0, 0.0, 0.0, 1.0),
    },
}
CAMERA_EXTRINSIC_TRANSLATION = _ARM_CAMERA_EXTRINSIC_BY_MODE[ARM_MODE]['translation']
CAMERA_EXTRINSIC_ROTATION_QUAT = _ARM_CAMERA_EXTRINSIC_BY_MODE[ARM_MODE]['rotation_quat']
PIPELINE_ID  = 'ompl'
PLANNER_ID   = 'RRTstarkConfigDefault'
ARM_JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

# --- 夾爪幾何與開合量 -------------------------------------------------------
GRIPPER_LENGTH  = 0.16      # isaac_法蘭面到夾爪咬合中心的距離 (m)

#GRIPPER_LENGTH  = 0.175      # real_法蘭面到夾爪咬合中心的距離 (m)
APPROACH_DIST   = 0.10      # 預備點 A 沿接近軸再往後退多少 (m)
GRIPPER_PREOPEN = 0.010     # 出發前先張開
GRIPPER_GRASP   = 0.015     # 到位後夾緊
GRIPPER_RELEASE = 0.0       # 放開 / 收合

# 實體夾爪：透過 TM 手臂的數位 IO 控制（/set_io, tm_msgs/srv/SetIO）
GRIPPER_IO_MODULE      = 1
GRIPPER_IO_TYPE        = 1
GRIPPER_IO_PIN         = 0
GRIPPER_IO_OPEN_STATE  = 0.0   # 開
GRIPPER_IO_CLOSE_STATE = 1.0   # 閉

# 虛擬夾爪碰撞體（VG_FINGER_SIZE 依 MODE 決定，見上方，對應各自場景的真實夾爪形狀）
VG_FINGER_Z      = 0.062                  # 手指中心沿「該手指 link 自己的 z 軸」的位置
VG_FINGER_OFF_X  = 0.012                  # 兩指往中間收的局部 X 偏移量
VG_TOUCH_LINKS   = ['left_finger_link', 'right_finger_link', 'gripper_base_link']

# --- 車體 (掛在 base 底下，隨基座移動) --------------------------------------
ENABLE_VIRTUAL_GRIPPER = True
CAR_BODY_SIZE    = [1.0, 0.6, 0.5]        # 長 x 寬 x 高 (m)
CAR_BODY_OFFSET  = [0.0, 0.0, -0.25]      # 方塊中心相對 base 原點；頂面貼齊 z=0
CAR_TOUCH_LINKS  = ['base', 'link_1']

# --- 預設關節姿態 (單位：度)（POSE_HOME_DEG / POSE_FINE_DEG 依 MODE 決定）----
POSE_BASKET_DEG = [-42.0, 29.0, 31.0, -15.0, 90.0, 0.0]

# --- 精定位目標（座標+姿態版）------------------------------------------------
# _move_to_fine 現在走 go_to_pose（座標+姿態，OMPL 解 IK），不是 go_to_joints。
# 流程：粗定位（車用相機）算出番茄 base 座標 (x,y,z) → arm_task_node.py 用
# MathUtils.camera_facing_flange_pose 算出「手臂相機面對番茄、光軸通過番茄中心、
# 離番茄 COARSE_TO_FINE_RETREAT_M 公尺」時法蘭該去的位置+姿態，交給 go_to_pose。
# ★ 2026-09-22 改用這個算法之前是 retreat_along_local_z（沿番茄姿態的局部 Z 軸
# 直接退 30cm，法蘭本身退到那個位置）；手臂相機不在法蘭軸線上（側向偏移約 13cm、
# 光軸跟法蘭 Z 差約 5.6°），舊算法會讓「法蘭」對準番茄、但「相機」偏離番茄。新算法
# 反過來先讓相機對準，再回推法蘭該在哪，細節見 camera_facing_flange_pose 的說明。
# 這段運算是每次精定位都會真的執行的程式碼，不是預先算好寫死結果。
#
# 車用相機還沒接上，先用假座標頂著測整條管線：FAKE_COARSE_TOMATO_POSE（依 MODE 分
# 別定義在上面 car/lab 區塊）只有前三個值 (x,y,z) 會被使用，假裝是「粗定位算出來的
# 番茄座標」；後面的四元數+yaw 是舊算法（retreat_along_local_z）留下的欄位，
# 現在的算法用不到，格式保留只是不想動到 tuple 長度（呼叫端用 [:3] 取前三個）。
# ★ 這組假座標原本是拿舊算法反推、驗證退回 POSE_FINE_DEG（已知在 real 上測過可達）
# 這個性質——改用 camera_facing_flange_pose 之後這個驗證性質不再成立（新算出來的
# 法蘭位置會因為相機偏移補償而跟 POSE_FINE_DEG 不同），新算法算出來的姿態目前
# 【還沒實機/Isaac 驗證過可達性跟避碰】，上機測試時第一件事要確認這點。
# 之後接上真的車用相機，把各 MODE 的 FAKE_COARSE_TOMATO_POSE 前三個值換成相機
# 即時算出來的番茄座標就好，arm_task_node.py 不用再改。
# 格式：(x, y, z, qx, qy, qz, qw, yaw)——後五個欄位目前未使用，見上方說明。
COARSE_TO_FINE_RETREAT_M = 0.30

# --- 遮擋備用視角 --------------------------------------------------------------
# 這輪候選番茄「全部」被判定遮擋時（vision_node 發 OCCLUDED），依序換到這些視角
# 重新掃描，都試過還是不行才真的回 Home。
# 跟主流程共用同一個番茄目標點（FAKE_COARSE_TOMATO_POSE 的 x,y,z），把水平接近
# 方向繞世界 Z 軸(垂直軸)偏移這些角度，一樣透過 MathUtils.camera_facing_flange_pose
# 算「手臂相機對準番茄」的法蘭姿態（該函式的 azimuth_offset_deg 參數）——距離跟
# 「相機有沒有面對目標」都保證不變，只換方位角，準心保證還對著同一個點。
ALT_VIEW_AZIMUTH_OFFSETS_DEG = [50.0, -50.0]

# 點雲轉發 / 過濾已搬到視覺端 (vision_node)，本模組不再直接碰點雲。

# --- 速度 / 規劃參數 ---------------------------------------------------------
JOINT_VEL, JOINT_ACC = 0.2, 0.2    # 關節空間移動
POSE_VEL,  POSE_ACC  = 0.2, 0.2    # OMPL 位姿移動
CART_VEL,  CART_ACC  = 0.15, 0.15    # 笛卡爾直線

PLAN_TIME_JOINT = 5.0
PLAN_TIME_POSE  = 5.0
PLAN_ATTEMPTS   = 15

CART_MAX_STEP     = 0.01    # 直線路徑每 1 cm 取一點
CART_MIN_FRACTION = 0.95    # 直線完成度低於此值就判定失敗

POS_TOLERANCE = 0.001               # 位置約束球半徑 (m)
ORI_TOLERANCE = 0.05                # 姿態約束各軸容差 (rad)

# ★ 2026-09-30：J1 約束中心改回 yaw，統一用「基座 → 這一步法蘭要到的點」的方位角
# （見 controller.py 的 jc1），容忍度回到 ±40°。實測（假座標番茄、09-29 遠的那顆，
# 主視角/±50° 備用視角/預備點 A）需要的 J1 跟這個方位角差 1～27°，都在 ±40° 內。
# 09-23～09-30 的「出發前當下角度 ±90°」版本：
# J1_TOLERANCE  = math.radians(90.0)
J1_TOLERANCE  = math.radians(40.0)  # J1 可偏離「基座→要到的點」方位角幾度

# go_to_pose (OMPL) 規劃時，同一個末端姿態常有好幾組手肘上/下的關節解，鎖 joint_3
# 在這個中心角度附近，避免規劃跳到手肘翻到另一側的分支。中心先抓 90 度 (試驗值，
# 依 POSE_HOME/FINE/BASKET_DEG 現有的 joint_3 都是正值 31~135 度推測)，joint_3
# 硬體限位是 ±155 度，容差再大也會被限位收斂，實測後再依實際效果調整。
ELBOW_UP_CENTER    = math.radians(90.0)
ELBOW_UP_TOLERANCE = math.radians(90.0)

# joint_6（腕部滾轉）約束：J6 硬體限位 ±270 度，同一個末端姿態在角度上每隔 360 度
# 就有一個等效解，容差沒鎖住的話 OMPL 可能選到「多轉一整圈」才能到的那個等效解
# （例如選到 +170 度而不是 -190 度以外的等效角）。鎖在 POSE_HOME/FINE/BASKET_DEG
# 現有姿態都採用的 0 度附近，容差抓 ±90 度，比涵蓋所有等效解所需的 ±180 度更緊，
# 排除掉需要多轉一圈才能到的那個分支，防止腕部整個轉一圈。
J6_CENTER    = math.radians(0.0)
J6_TOLERANCE = math.radians(90.0)

# joint_5（腕部彎曲）約束：鎖在 POSE_HOME/FINE/BASKET_DEG 都採用的 90 度附近 ±90 度，
# 也就是 [0°, 180°]，不跨過 0°（腕部奇異點，J4/J6 共軸）也不跨正負號，擋住腕部翻轉。
# ★ 2026-10-01：中心從「出發前 joint_5 當下的角度」改成固定 J5_CENTER，容忍度從
# ±180° 收到 ±90°。當下角度版本的問題：/compute_ik 實測（番茄 (-0.109,-0.688,0.602)）
# 主視角/+50°/−50° 備用視角需要的 J5 = 87°/173°/−1°，從 +50° 視角（J5≈174°）出發
# 去 −50° 要轉 175°，當下角度 ±90° 一定無解。固定中心的代價：−50° 視角需要的
# J5≈−1° 仍在 [0°, 180°] 外面，會規劃失敗——但那個姿態本來就貼著奇異點，擋掉也好。
# 原本的當下角度版本（見 controller.py 的 jc5）：
# J5_MAX_STEP = math.radians(180.0)
J5_CENTER    = math.radians(90.0)
J5_TOLERANCE = math.radians(90.0)

# --- 任務流程 ----------------------------------------------------------------
GO_TO_BASKET        = False   # True = 夾完先去籃子放；False = 直接回 Home
RETURN_HOME_MAX_RETRIES = 3   # 任何一步失敗後，退回初始姿態最多重試幾次才放棄、轉 IDLE 請人工檢查
ENABLE_ALT_VIEW = True   # False = 全部候選被遮擋時直接回初始位置，不切換備用視角重掃
PAUSE_AT_APPROACH   = 0.3     # 抵達點 A 後停頓 (s)
PAUSE_AFTER_GRASP   = 1.0     # 夾緊後停頓
PAUSE_AFTER_RELEASE = 0.3     # 放開後停頓
PAUSE_BEFORE_IDLE   = 0.5     # 回 Home 後等手臂穩定
PAUSE_BEFORE_SCAN   = 1.0     # 抵達精定位後、開始偵測前停頓

# --- 到達確認（2026-09-29）-------------------------------------------------------
# MoveIt 回報成功之後，停頓完再用實際量到的姿態確認一次「真的到了」，才開啟對應的
# 視覺偵測：回 Home 確認到了才開粗定位（/robot_status=COARSE），到精定位觀測點確認
# 到了才開精定位（/robot_status=DONE）。沒到就當作這一步失敗，走原本的失敗處理。
HOME_ARRIVAL_TOL_DEG     = 2.0    # Home：每個關節跟 POSE_HOME_DEG 的差距上限（度）
FINE_ARRIVAL_POS_TOL_M   = 0.01   # 精定位觀測點：法蘭實際位置跟目標位置的差距上限（m）
FINE_ARRIVAL_ORI_TOL_DEG = 5.0    # 精定位觀測點：法蘭實際姿態跟目標姿態的夾角上限（度）

# --- 出發前可達性檢查（2026-09-29）-----------------------------------------------
# 往預備點 A 出發前，先用 /compute_ik 確認預備點 A、夾取點到得了；到不了直接報錯、
# 放棄這顆、留在觀測點重新掃描，不用等 OMPL 規劃 5 秒失敗才知道（見 arm_task_node
# 的 _process_target）。
REACH_IK_TIMEOUT_SEC = 0.3        # 每個 IK 種子最多算多久（s），會依序試：當下姿態、POSE_FINE、Home

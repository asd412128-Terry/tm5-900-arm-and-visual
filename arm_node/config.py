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

MODE = 'lab'   # 'car' 或 'lab' ← 只改這一行切換環境


# ===========================================================================
# 依 MODE 切換的參數
# ===========================================================================
if MODE == 'car':
    ENABLE_CAR_BODY = True
    OBSTACLES = []

    #POSE_HOME_DEG = [-90.0, -15.0, 65.0, -50.0, 90.0, 0.0]
    POSE_HOME_DEG =[-90.0, -7.0, 125.0, -118.0, 90.0, 0.0]
    POSE_FINE_DEG = [-90.0, -7.0, 125.0, -118.0, 90.0, 0.0]

    VG_FINGER_EXT_SIZE = [0.005, 0.005, 0.01]

elif MODE == 'lab':
    ENABLE_CAR_BODY = False
    OBSTACLES = [
        {'id': 'table',           'type': 'cube',     'pos': [0.75, -0.1325, 0.03],     'size': [0.7, 1.205, 0.03]},
        {'id': 'front_partition', 'type': 'cube',     'pos': [1.125, -0.1325, 0.29575], 'size': [0.05, 1.205, 0.5015]},
        {'id': 'side_partition',  'type': 'cube',     'pos': [0.499, 0.495, 0.29575],   'size': [1.202, 0.05, 0.5015]},
        {'id': 'computer',        'type': 'cube',     'pos': [0.7, -0.635, 0.25],       'size': [0.46, 0.18, 0.41]},
        {'id': 'wall',            'type': 'cube',     'pos': [-0.5, 0.0, 0.5],          'size': [0.06, 2.0, 1.5]},
        {'id': 'basket',          'type': 'cylinder', 'pos': [0.55, -0.63, 0.5063],     'size': [0.1, 0.075]},  # [高, 半徑]
    ]

    #POSE_HOME_DEG = [0.0, -15.0, 65.0, -50.0, 90.0, 0.0]
    POSE_HOME_DEG = [0.0, 0.0, 135.0, -135.0, 90.0, 0.0]
    POSE_FINE_DEG = [0.0, 0.0, 135.0, -135.0, 90.0, 0.0]

    VG_FINGER_EXT_SIZE = [0.005, 0.005, 0.015]

else:
    raise ValueError(f'未知 MODE: {MODE!r}，只能是 "car" 或 "lab"')


# ===========================================================================
# 共用參數（不分環境）
# ===========================================================================

# --- MoveIt 基本設定 -------------------------------------------------------
ARM_GROUP    = 'tmr_arm'
BASE_FRAME   = 'base'
EEF_LINK     = 'flange'
PIPELINE_ID  = 'ompl'
PLANNER_ID   = 'RRTstarkConfigDefault'
ARM_JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

# --- 夾爪幾何與開合量 -------------------------------------------------------
GRIPPER_LENGTH  = 0.175      # isaac_法蘭面到夾爪咬合中心的距離 (m)
#GRIPPER_LENGTH  = 0.17      # real_法蘭面到夾爪咬合中心的距離 (m)
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

# 虛擬夾爪碰撞體
VG_FINGER_SIZE   = [0.005, 0.005, 0.05]   # 單根手指本體 [X, Y, Z]
VG_FINGER_Z      = 0.062                  # 手指本體中心沿「該手指 link 自己的 z 軸」的位置
VG_FINGER_OFF_X  = 0.012                  # 兩指往中間收的局部 X 偏移量

# 手指延伸段（VG_FINGER_EXT_SIZE 依 MODE 決定，見上方）
VG_FINGER_EXT_Z  = (VG_FINGER_Z + VG_FINGER_SIZE[2] / 2.0
                     + VG_FINGER_EXT_SIZE[2] / 2.0)
VG_TOUCH_LINKS   = ['left_finger_link', 'right_finger_link', 'gripper_base_link']

# --- 車體 (掛在 base 底下，隨基座移動) --------------------------------------
ENABLE_VIRTUAL_GRIPPER = True
CAR_BODY_SIZE    = [1.0, 0.6, 0.5]        # 長 x 寬 x 高 (m)
CAR_BODY_OFFSET  = [0.0, 0.0, -0.25]      # 方塊中心相對 base 原點；頂面貼齊 z=0
CAR_TOUCH_LINKS  = ['base', 'link_1']

# --- 預設關節姿態 (單位：度)（POSE_HOME_DEG / POSE_FINE_DEG 依 MODE 決定）----
POSE_BASKET_DEG = [-42.0, 29.0, 31.0, -15.0, 90.0, 0.0]

# 點雲轉發 / 過濾已搬到視覺端 (vision_node)，本模組不再直接碰點雲。

# --- 速度 / 規劃參數 ---------------------------------------------------------
JOINT_VEL, JOINT_ACC = 0.2, 0.2    # 關節空間移動
POSE_VEL,  POSE_ACC  = 0.2, 0.2    # OMPL 位姿移動
CART_VEL,  CART_ACC  = 0.15, 0.15    # 笛卡爾直線

PLAN_TIME_JOINT = 1.5
PLAN_TIME_POSE  = 5.0
PLAN_ATTEMPTS   = 15

CART_MAX_STEP     = 0.01    # 直線路徑每 1 cm 取一點
CART_MIN_FRACTION = 0.95    # 直線完成度低於此值就判定失敗

POS_TOLERANCE = 0.001               # 位置約束球半徑 (m)
ORI_TOLERANCE = 0.05                # 姿態約束各軸容差 (rad)
J1_TOLERANCE  = math.radians(40.0)  # J1 面對目標的彈性範圍

# go_to_pose (OMPL) 規劃時，同一個末端姿態常有好幾組手肘上/下的關節解，鎖 joint_3
# 在這個中心角度附近，避免規劃跳到手肘翻到另一側的分支。中心先抓 90 度 (試驗值，
# 依 POSE_HOME/FINE/BASKET_DEG 現有的 joint_3 都是正值 31~135 度推測)，joint_3
# 硬體限位是 ±155 度，容差再大也會被限位收斂，實測後再依實際效果調整。
ELBOW_UP_CENTER    = math.radians(90.0)
ELBOW_UP_TOLERANCE = math.radians(90.0)

# --- 任務流程 ----------------------------------------------------------------
GO_TO_BASKET        = False   # True = 夾完先去籃子放；False = 直接回 Home
PAUSE_AT_APPROACH   = 1.0     # 抵達點 A 後停頓 (s)
PAUSE_AFTER_GRASP   = 1.0     # 夾緊後停頓
PAUSE_AFTER_RELEASE = 1.0     # 放開後停頓
PAUSE_BEFORE_IDLE   = 1.0     # 回 Home 後等手臂穩定
PAUSE_BEFORE_SCAN   = 1.0     # 抵達精定位後、開始偵測前停頓

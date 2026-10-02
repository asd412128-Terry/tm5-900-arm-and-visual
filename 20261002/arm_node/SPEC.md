# arm_node 技術規格

TM5M 機械臂任務節點 — ROS 2 Humble / rclpy / MoveIt2 / Python 3.10
位置：`/home/terry/tm_ws/python_isaac/arm_node`
整理依據：2026-08-25 程式碼狀態（實測可 import、可啟動）

## 00 概觀

`arm_node` 是溫室番茄採收機械臂的「手」端：訂閱視覺端（`vision_node`）算好的抓取點世界座標與果梗 3D 方向，用向量幾何組出一個「沿果梗方向夾持、垂直果梗切入」的正交夾爪姿態，透過 MoveIt2 規劃並執行「接近 → 下探 → 夾取 → 退回 → （可選）放籃子 → 回家」的任務狀態機。啟動時同時把靜態障礙物、車體、虛擬夾爪碰撞體寫進 Planning Scene，讓規劃器知道環境長什麼樣子。

整包程式把「算」跟「串接」分開：`math_utils`（純向量幾何）與 `scene_builder`（純場景建構）不碰任務流程；唯一負責 ROS 訂閱/發布、狀態機推進的是 `arm_task_node.py`；`controller.py` 是介於兩者之間的 MoveIt 動作/服務 wrapper。

## 01 系統情境

`arm_node` 不是獨立運作的節點，假設以下都已在同一個 ROS graph 上運行：

- **MoveIt2 `move_group`** — 提供 `move_action`、`execute_trajectory` 兩個 action，`compute_cartesian_path`、`apply_planning_scene`、`check_state_validity` 三個 service，以及 `/clear_octomap` service（通常由 `PointCloudOctomapUpdater` 提供）
- **`vision_node`**（同層目錄）— 發布 `/target_pose`、`/vision_status`，訂閱 `/robot_status`
- **關節狀態來源**（Isaac Sim 或真實驅動）— 發布 `/joint_states`，供 TF 查詢法蘭面目前朝向

流向：`arm_node` 發布 `/robot_status`（DONE/BUSY）告知 vision 何時可以掃描 → vision 選定目標後發布 `/target_pose` 或 `/vision_status=NO_TARGET` → `arm_node` 算姿態、驅動 MoveIt2 規劃執行 → 動作完成後回到「等待下一輪」。

### MODE 切換

`config.py` 最上面 `MODE = 'car'` 或 `'lab'` 決定一整組跟環境相關的參數（障礙物清單、是否掛車體、Home/精定位關節角、虛擬夾爪延伸段大小），改這一行即可切換，其餘程式碼不用動。

## 02 模組架構

| 檔案 | 職責 |
|---|---|
| `main.py` | 進入點：`from .arm_task_node import main` 後直接呼叫，環境由 `config.MODE` 決定 |
| `arm_task_node.py` | `TM5MTaskNode`：任務狀態機主節點，ROS 訂閱/發布、TF 查詢、狀態流程推進，協調 `controller` / `math_utils` |
| `controller.py` | `TM5MController`：封裝 MoveIt2 action/service（joint/pose/cartesian 目標、Planning Scene 套用、夾爪指令、關節狀態偽發布），不含任務流程邏輯 |
| `scene_builder.py` | `SceneBuilder`：把靜態障礙物、車體、虛擬夾爪寫進 MoveIt Planning Scene |
| `math_utils.py` | `MathUtils.calculate_grasp_and_approach`：純函式，由番茄座標＋果梗方向向量推出正交夾爪姿態、抓取點、預備點，不碰 ROS/TF/MoveIt |
| `config.py` | 依 `MODE` 切換的環境參數 ＋ 共用參數（MoveIt 設定、夾爪幾何、速度、任務流程時序），其餘模組一律從這裡 import |

## 03 ROS 2 介面

| 方向 | 名稱 | 型別 | 說明 |
|---|---|---|---|
| SUB | `/target_pose` | `geometry_msgs/PoseStamped` | `position` 為抓取點世界座標；**借用** `orientation.x/y/z` 傳遞果梗 3D 方向單位向量（沿用 vision 端慣例），非法向量（norm≈0）就放棄這顆重新掃描 |
| SUB | `/vision_status` | `std_msgs/String` | 值 `"NO_TARGET"` 觸發本輪結束、回初始位置 |
| SUB | `/joint_states` | `sensor_msgs/JointState` | 累積成 `_joint_pos_map`，供列印目前角度與 `current_joints` 屬性使用 |
| PUB | `/robot_status` | `std_msgs/String` | `"DONE"`（`scanning=True`）／`"BUSY"`，`0.5s` 定時器重發，讓 vision 知道何時可以送目標 |
| PUB | `/gripper_command` | `sensor_msgs/JointState` | 夾爪開合指令，給下游夾爪驅動/模擬器執行 |
| PUB | `/joint_states` | `sensor_msgs/JointState` | 節點**自己也發布**左右手指關節位置（`0.5s` 定時 + 每次 `control_gripper` 呼叫時），補齊 robot_state 給 MoveIt／TF 使用 |
| Action client | `move_action` | `moveit_msgs/MoveGroup` | 關節空間／姿態空間規劃＋執行 |
| Action client | `execute_trajectory` | `moveit_msgs/ExecuteTrajectory` | 執行直線（Cartesian）軌跡 |
| Service client | `compute_cartesian_path` | `moveit_msgs/GetCartesianPath` | 算下探/退回的直線路徑 |
| Service client | `apply_planning_scene` | `moveit_msgs/ApplyPlanningScene` | 寫入障礙物 / 掛載車體與虛擬夾爪碰撞體 |
| Service client | `check_state_validity` | `moveit_msgs/GetStateValidity` | 診斷用（見 08 已知限制，目前沒有呼叫點） |
| Service client | `/clear_octomap` | `std_srvs/Empty` | 每次「要去精定位」前清空一次 OctoMap |

節點名稱：`tm5m_task_node`（`rclpy.node.Node('tm5m_task_node')`）

## 04 運作週期

以 `current_step` 字串驅動的簡單狀態機，每一步動作完成都經過同一個 `on_action_completed(success)` callback 分派下一步。`MultiThreadedExecutor` + `ReentrantCallbackGroup`：一邊動作、一邊仍能收 `/target_pose` 等 topic。

```
啟動：等 MoveIt Server ready → 放開夾爪 + 載入 Planning Scene → 回 Home (INIT)
  │
INIT 完成 → 清 OctoMap → 前往精定位 (TO_FINE)
  │
TO_FINE 完成 → 停頓 PAUSE_BEFORE_SCAN → 開始掃描
  (scanning=True，/robot_status 開始回報 DONE)
  │
  ├─ 收到 /target_pose  ──► _process_target：算 grasp/approach 姿態
  │                          → 夾爪 PREOPEN → 前往預備點 A (APPROACH)
  │
  └─ 收到 /vision_status=NO_TARGET ──► 回 Home (RETURN)

APPROACH 到位 → 停頓 PAUSE_AT_APPROACH → 直線下探到 Goal (DESCEND)
DESCEND 到位  → 夾爪閉合(GRASP) → 停頓 PAUSE_AFTER_GRASP → 直線退回點 A (LIFT)
LIFT 到位     → GO_TO_BASKET？
                ├─ True  → 前往籃子 (BASKET) → 放開夾爪 → 停頓 → 回精定位 (TO_FINE)
                └─ False → 直接回精定位 (TO_FINE)，重新掃描下一顆

RETURN(Home) 完成 → 停頓 PAUSE_BEFORE_IDLE → 放開夾爪 → 回精定位 (TO_FINE)
```

失敗處理：任何一步 `on_action_completed(False)` 都會嘗試「放開夾爪＋退回初始姿態重來」；若連退回初始姿態都失敗，就整個停在 `IDLE`（`_reset_to_idle`）並回報 `DONE`，不再自動重試，需人工檢查。

## 05 座標系與姿態計算（`math_utils.py`）

1. **`base_yaw = atan2(pos.y, pos.x)`** — 依目標世界座標算出的基座水平朝向，之後同時當作 J1 關節約束的中心角（見下）。
2. **參考方向 `h`**：優先查 `world → link_6`（`EEF_LINK`）目前的 TF（實際上是用當下 `/joint_states` 做 FK 查出來），取旋轉矩陣第三欄當 `h`；查不到或四元數非法就退回純水平假設 `[cos(base_yaw), sin(base_yaw), 0]`。
3. **接近軸 `z_axis`**：把 `h` 投影到垂直果梗方向 `stem_dir` 的平面上；若 `h` 幾乎平行 `stem_dir`（退化），改拿世界 `-Z` 當參考再投影一次。
4. **`y_axis = -stem_dir`**（對齊夾爪開合方向，指向枝條端），**`x_axis = y_axis × z_axis`** 維持右手定則，組成 3×3 旋轉矩陣轉四元數。
5. **抓取點/預備點**：`grasp = tomato_pos − GRIPPER_LENGTH · z_axis`，`approach = grasp − APPROACH_DIST · z_axis`。
6. `base_yaw` 額外被塞進 `MoveGroup` 目標的 J1 `JointConstraint`（容差 `J1_TOLERANCE`），限制姿態解算時基座朝向不會轉到離目標太遠的角度，但 `z_axis` 本身仍是自由投影、不受鉸鏈限制。

`calculate_grasp_and_approach` 完全是純函式（輸入來自呼叫端），方便單獨測試，不牽涉任何 ROS API。

## 06 場景建構（`scene_builder.py`）

`build_all()` 依序做三件事：

- **障礙物**（`config.OBSTACLES`）：world collision object（`is_diff=True` 累加），`car` 模式空清單、`lab` 模式有桌面/隔板/電腦/牆/籃子共 6 個
- **車體**（`ENABLE_CAR_BODY`，僅 `car` 模式開）：掛在 `base` 下的長方體，`touch_links=[base, link_1]`，隨基座移動
- **虛擬夾爪**（`ENABLE_VIRTUAL_GRIPPER`，兩模式皆開）：左右手指各兩段碰撞體（手指本體固定尺寸 + 延伸段依 `MODE` 給不同大小），分開掛在 `left_finger_link`／`right_finger_link` 上，會隨夾爪開合一起動

## 07 設定參數（`config.py`）

### 依 MODE 切換

| 參數 | `car` | `lab` | 意義 |
|---|---|---|---|
| `ENABLE_CAR_BODY` | `True` | `False` | 是否掛車體碰撞體 |
| `OBSTACLES` | `[]` | 6 個固定障礙物 | 桌面/隔板/電腦/牆/籃子等 |
| `POSE_HOME_DEG` | `[-90,-15,65,-50,90,0]` | `[0,-15,65,-50,90,0]` | 初始關節角（度） |
| `POSE_FINE_DEG` | `[-90,-7,125,-118,90,0]` | `[0,-10,135,-125,90,0]` | 精定位關節角（度） |
| `VG_FINGER_EXT_SIZE` | `[0.005,0.005,0.01]` | `[0.005,0.005,0.015]` | 虛擬夾爪手指延伸段尺寸 |

### 共用參數

| 群組 | 參數 | 預設值 | 意義 |
|---|---|---|---|
| MoveIt | `ARM_GROUP` / `BASE_FRAME` / `EEF_LINK` | `tmr_arm` / `base` / `flange` | Planning group 與座標系 |
| | `PIPELINE_ID` / `PLANNER_ID` | `ompl` / `RRTstarkConfigDefault` | 規劃管線與演算法 |
| 夾爪幾何 | `GRIPPER_LENGTH` | 0.16 m | 法蘭面到夾爪咬合中心距離 |
| | `APPROACH_DIST` | 0.10 m | 預備點沿接近軸再退多少 |
| | `GRIPPER_PREOPEN` / `GRIPPER_GRASP` / `GRIPPER_RELEASE` | 0.010 / 0.015 / 0.0 | 三種開合寬度 |
| 虛擬夾爪碰撞體 | `VG_FINGER_SIZE` / `VG_FINGER_Z` / `VG_FINGER_OFF_X` | 見程式 | 手指本體碰撞盒尺寸與位置 |
| 任務流程 | `GO_TO_BASKET` | `False` | 夾完是否先去籃子再回精定位 |
| | `PAUSE_AT_APPROACH` / `PAUSE_AFTER_GRASP` / `PAUSE_AFTER_RELEASE` / `PAUSE_BEFORE_IDLE` / `PAUSE_BEFORE_SCAN` | 2.0 / 1.0 / 1.0 / 1.5 / 1.0 s | 各步驟間的固定停頓 |
| 速度/規劃 | `JOINT_VEL,ACC` / `POSE_VEL,ACC` / `CART_VEL,ACC` | 0.2 / 0.2 / 0.1 | 三種移動的速度縮放 |
| | `PLAN_TIME_JOINT` / `PLAN_TIME_POSE` / `PLAN_ATTEMPTS` | 1.5 / 5.0 / 15 | 規劃時限與嘗試次數 |
| | `CART_MAX_STEP` / `CART_MIN_FRACTION` | 0.01 m / 0.95 | 直線路徑取點間距／完成度門檻 |
| 約束容差 | `POS_TOLERANCE` / `ORI_TOLERANCE` / `J1_TOLERANCE` | 0.001 m / 0.05 rad / 40° | 位置球半徑／姿態容差／J1 彈性範圍 |

## 08 已知限制

- **不是標準 colcon package。** 沒有 `package.xml` / `setup.py`，以 Python module 方式執行（跟 `vision_node` 相同慣例）。
- **MODE 切換是改原始碼常數，不是 launch 參數。** `config.py` 頂端 `MODE='car'|'lab'`，換場景要進檔案改一行再重跑，沒有 CLI/環境變數 override。
- **`check_state_validity` 是孤兒功能。** `controller.py` 建了對應 service client 並實作了 `check_current_state_validity()` 診斷方法，但 `arm_task_node.py` 目前沒有任何地方呼叫它——純粹是手動除錯用的備用工具，沒接進正常流程。
- **`GO_TO_BASKET=False`（目前預設）時，連續夾取不是「放開＋歸位」的乾淨動作。** `LIFT` 完成後直接呼叫 `_move_to_fine()` 前往精定位重新掃描，夾爪並未在此釋放；要等到下一輪 `_process_target` 一開始呼叫 `GRIPPER_PREOPEN` 才會鬆開，也就是說上一顆番茄實際上是在「開始接近下一個目標」的當下、於精定位姿態附近被放開，而不是真正送進籃子或退回初始位置——這是目前程式碼的行為，不是刻意設計的丟棄動作，使用前要注意。
- **兩層自我發布 `/joint_states`。** 節點自己用 `0.5s` 定時器＋每次 `control_gripper()` 偽造左右手指的 `/joint_states`（並非讀真實回授）。如果下游夾爪硬體/模擬器自己也發布手指的 joint state，兩邊會互相覆蓋打架。
- **姿態計算有雙層退化但只保護了一層。** `calculate_grasp_and_approach` 在 `h` 與 `stem_dir` 近乎平行時，退回世界 `-Z` 當參考再投影一次，但沒有再檢查這第二次投影是否也接近零向量（即果梗本身也接近垂直向下、與退避參考同方向的極端情況）；機率很低，目前程式碼沒有針對這個雙重退化加保護，遇到時可能得到不穩定或非法的旋轉矩陣。
- **外部 SIGTERM 中斷時 `rclpy.shutdown()` 可能丟例外。** 本次測試用 `timeout` 送 SIGTERM 中斷正在等待 MoveIt Server 的節點，觀察到 `finally` 區塊呼叫 `rclpy.shutdown()` 時噴 `RCLError: failed to shutdown: rcl_shutdown already called`；正常用 Ctrl+C（SIGINT，被 `except KeyboardInterrupt` 接住）不會遇到，僅在非互動式強制關閉（例如某些 launch 系統送 SIGTERM）時才會看到這個無害但沒被特別處理的例外。

## 09 啟動方式與驗證

不是標準 colcon package，以 Python module 方式執行：

```bash
cd /home/terry/tm_ws/python_isaac
python3 -m arm_node.main
```

**本機驗證（2026-08-25）**：ROS 2 humble 已 source，`rclpy` / `tf2_ros` / `moveit_msgs` / `geometry_msgs` / `sensor_msgs` / `shape_msgs` / `std_srvs` / `scipy` / `numpy` 皆可正常 `import`；`arm_node` 各模組（`config` / `math_utils` / `scene_builder` / `controller` / `arm_task_node`）逐一 import 無誤。實際執行 `python3 -m arm_node.main` 後節點正常啟動，印出「大腦節點啟動！等待 MoveIt Server 連線...」並持續等待——這台機器沒有跑 MoveIt2 `move_group`，卡在等待 Server 是**預期行為**，代表程式本身（含 import 鏈、rclpy 生命週期）沒有問題，缺的是下游服務。

啟動前需要就緒的前置節點/服務：

- **MoveIt2 `move_group`**：提供 `move_action` / `execute_trajectory` action 與 `compute_cartesian_path` / `apply_planning_scene` / `check_state_validity` service，且機器人描述（URDF）要包含 `base` / `flange` / `link_1` / `link_6` / `left_finger_link` / `right_finger_link` 等 frame
- **`/clear_octomap` service**：通常隨 MoveIt2 `PointCloudOctomapUpdater` 一起起來；沒有的話只會印 warn 並跳過清空，不會讓節點掛掉
- **關節狀態來源**：需有節點持續發布 `/joint_states`，供查詢法蘭面目前朝向（`_get_current_flange_rotation`）使用；查不到時會自動退回水平參考，不會崩潰
- **`vision_node`**：需訂閱 `/robot_status`，並發布 `/target_pose` 與 `/vision_status`

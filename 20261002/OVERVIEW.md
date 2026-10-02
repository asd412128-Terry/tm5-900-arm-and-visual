# 番茄採收手臂 — 專案總記錄

最後更新：2026-09-29

> **這份記什麼**：整個專案的框架、每個階段用了什麼方法、為什麼這樣設計、目前狀態和待辦。
> **不記**每天的過程（試了什麼、除錯經過）——那些寫在 `STATUS.md`（每日紀錄）。
>
> **什麼時候更新**：只在某項工作「確定完成」或「確定定案、不再改」時才更新；還在實驗、
> 還可能改回去的東西先不寫進來。
>
> 範圍：車子導航目前**不在範圍內**，假設車子已經停好、和植株行平行。

圖例：✅ 完成且驗證過　🟡 做了但還沒驗證　🔴 有已知問題　⬜ 還沒開始

---

## 一、系統架構

ROS 2 Humble + MoveIt2 + Python 3.10。`arm_node` 和 `vision_node` 都不是 colcon package，
用 `python3 -m` 執行。

```
 車用相機 ──► coarse_node ──/coarse_target_point──► arm_task_node ──MoveIt2──► 手臂
                                                     │  ▲
                                          /robot_status│  │/target_pose、/vision_status
                                                     ▼  │
 手臂相機 ──► fine_node ──/target_filter_mask──► cloud_filter_node ──點雲──► MoveIt2 OctoMap
```

| Process | 啟動方式 | 職責 |
|---|---|---|
| MoveIt2 `move_group` | `ros2 launch tm5-900_moveit_config demo.launch.py` | 規劃、執行、Planning Scene、OctoMap |
| `arm_task_node` | `ARM_MODE=isaac python3 -m arm_node.main` | 任務狀態機、算姿態、驅動 MoveIt |
| `coarse_node` | `VISION_MODE=isaac python3 -m vision_node.coarse_main` | 車用相機粗定位 |
| `fine_node` | `VISION_MODE=isaac python3 -m vision_node.fine_main` | 手臂相機精定位、遮擋判斷、選目標 |
| `cloud_filter_node` | `VISION_MODE=isaac python3 cloud_filter_node.py --ros-args -p use_sim_time:=true` | 過濾點雲給 OctoMap |

### 環境切換（四種組合，兩個開關互相獨立）

| 開關 | 位置 | 值 | 管什麼 |
|---|---|---|---|
| `ARM_MODE` / `VISION_MODE` | 環境變數 | `real` / `isaac` | 相機 topic、內參、外參、夾爪 IO |
| `MODE`（arm）/ `SCENE_MODE`（vision） | `arm_node/config.py`、`vision_node/config.py` 的常數 | `car` / `lab` | 障礙物、車體、Home/精定位姿態、車用相機外參 |

兩邊的 config 不會互相 import，**改一邊要手動同步另一邊**；場景相關參數要四種組合都填。

## 二、整體流程與各階段方法

```
② 車用相機粗定位 → ③ 手臂移到精定位姿態 → ④ 手臂相機精定位
→ ⑤ 選定目標、建 OctoMap → ⑥ 接近 → 下探 → 夾取 → 退回 → ⑦（放籃子）→ 回到 ③
      遮擋時：④ → 換備用視角 ±50° → 重掃 → 都不行就回 Home
```

| # | 階段 | 方法 | 主要檔案 | Isaac | 實機 |
|---|---|---|---|---|---|
| ② | 車用相機粗定位 | YOLO 分割取番茄 → mask 內深度中位數 + 番茄半徑修正 → 反投影 → TF 轉 base 座標。跨幀用 base 座標一對一貪婪配對成 track，座標取最近 10 幀**中位數**（番茄常併框，中位數比平均抗離群值），累積 5 次才確認。終端機選定後持續發布該顆座標到 `/coarse_target_point` | `coarse_node.py`、`coarse_detector.py`、`coarse_tracker.py` | 🟡 | 🔴 topic 和外參是佔位值 |
| ③ | 移到精定位姿態 | `camera_facing_flange_pose`：先決定「相機」位置（離番茄 `COARSE_TO_FINE_RETREAT_M`=0.30m、光軸通過番茄中心、水平接近方位角固定 `FIXED_APPROACH_AZIMUTH_DEG`），再用相機外參回推法蘭位姿；交給 `go_to_pose`（OMPL 解 IK + 關節約束） | `arm_task_node.py`、`math_utils.py` | 🔴 常規劃失敗 | 🟡 |
| ④ | 手臂相機精定位 | YOLOv11 同時分割番茄/果梗 → 果梗 mask 重疊抑制 → 骨架化 → 兩次 BFS 找最長路徑（樹的直徑，天生忽略雜訊短分支）→ 依 `GRASP_RATIO`=0.7 取抓取點 → 抓取點前後骨架段取 3D 點算果梗方向。`StemTracker` 滑動視窗取信心最高的一筆 | `fine_node.py`、`fine_detector.py`、`fine_skeleton.py`、`coordinates.py`、`fine_stem_tracker.py` | ✅ | ✅ |
| ④' | 遮擋判斷 + 備用視角 | 番茄 mask 形狀：bbox 長寬比（0.9～1.5）+ solidity（mask 面積 / 擬合橢圓面積 ≥ 0.89）。連續 3 輪沒有能挑的目標 → 發 `OCCLUDED` → 手臂繞番茄把方位角偏 ±50° 重掃 → 都不行才回 Home | `fine_occlusion.py`、`ALT_VIEW_AZIMUTH_OFFSETS_DEG` | 🟡 | 🟡 |
| ⑤ | 選定目標 | 依到基座距離排序、排除超出 `MAX_REACH_M` 的候選，**終端機手動輸入 ID** | `fine_target_selector.py` | ✅ | ✅ |
| ⑤' | 點雲過濾 → OctoMap | 掃描期間不送點雲（OctoMap 保持空白，避免候選目標本身變障礙物）。選定那一刻用「目標果梗 + 最近番茄」膨脹遮罩挖掉對齊深度圖中的目標，只算一次，2 秒內每 0.1s 重複發送同一份；`is_dense=False` | `cloud_filter_node.py`、`fine_mask_publisher.py`、`sensors_3d.yaml` | 🔴 octomap 一直是空的 | 🟡 |
| ⑥ | 接近、下探、夾取、退回 | 夾爪姿態：夾爪 y 軸 = −果梗方向，接近軸 = 法蘭目前朝向投影到垂直果梗的平面（退化時改用 −Z），組正交矩陣。抓取點 = 番茄 − `GRIPPER_LENGTH`·z，預備點 A 再退 `APPROACH_DIST`=0.10m。到 A 用 `go_to_pose`，下探/退回用笛卡兒直線 | `math_utils.calculate_grasp_and_approach`、`controller.py` | ✅ | 🟡 |
| ⑥' | 夾爪 | Isaac：發 `/gripper_command` + 自己補發手指 `/joint_states`；實機：TM 數位 IO（`/set_io`） | `controller.control_gripper` | ✅ | 🟡 |
| ⑦ | 放籃子 / 回 Home | `GO_TO_BASKET` 決定夾完先去籃子或直接回精定位；回 Home 用關節目標 | `arm_task_node.py` | 🔴 | 🔴 |
| — | 失敗復原 | 任何一步失敗 → 放開夾爪、退回 Home，重試 3 次，還不行就轉 IDLE 等人工 | `on_action_completed` | ✅ | ✅ |

## 三、MoveIt 規劃設定

| 項目 | 目前做法 |
|---|---|
| 規劃器 | OMPL `RRTstarkConfigDefault`，位姿 5s、關節 5s，位姿 15 次嘗試 |
| 位姿目標 | 位置球 1mm + 姿態 ±0.05rad |
| 關節約束（位姿目標用，`check_reachable` 共用） | J1：基座 → 這一步法蘭要到的點的方位角 ±40°；J3：不加約束（實驗中）；J5：固定 90° ±90°（[0°, 180°]）；J6：固定 0° ±90°（防止相機上下顛倒）。關節目標（`go_to_joints`）不套這組 |
| Planning Scene | car：掛在 `base` 的車體方塊；兩模式都有虛擬夾爪手指方塊；lab：桌面等固定障礙物 |
| OctoMap | 解析度 2mm、`max_range` 0.8m、訂閱 `/camera/depth/points_gated`；每次去精定位前清空 |

## 四、關鍵設計決策（為什麼這樣做）

- **點雲過濾必須是獨立 process**：同一個 process 疊加多個 subscription 時，點雲會被 `PointCloudOctomapUpdater` 靜默拒收，原因不明。
- **點雲要連發，不能只發一張**：只發一張時，時間戳落在 `/tf` 空窗期會被 MoveIt 的 MessageFilter 靜默丟掉。
- **`/target_pose` 借用 orientation 傳果梗方向**：`x/y/z` 是方向單位向量、`w`=0，不是四元數。
- **讓相機對準番茄，而不是讓法蘭對準**：手臂相機側向偏移約 13cm、光軸差約 5.6°，法蘭對準會讓相機偏離番茄。
- **相機外參寫死在 arm config**：不查 `fine_node` 廣播的 TF，只開 arm_node 時也算得對。
- **接近方位角用固定值**：車子會先對齊植株行，同一趟固定往側面伸出去摘，不隨番茄位置飄。
- **J1 約束以「這一步要到的點」的方位角為中心**：以前用果梗點方位角，不跟著備用視角移動，−50° 視角會超出 ±40°；改用要到的點之後實測差 1～27°。
- **J3 暫時不加約束**：精定位姿態合法 IK 解太少，拿掉看規劃失敗會不會減少；代價是可能解到手肘翻到另一側。
- **J5 約束固定在 90°**：當下角度版本在備用視角之間 J5 要轉到 175°（實算），必然無解；固定 [0°, 180°] 不跨 0°，擋住腕部翻轉和腕部奇異點。
- **J6 約束固定在 0°**：改成當下角度會讓 J6 一路飄，相機上下顛倒（實測證實）。

## 五、待辦與里程碑

| 里程碑 | 目標 | 包含的工作 | 狀態 |
|---|---|---|---|
| **M1：Isaac 上能穩定摘一顆** | 從 ② 到 ⑦ 在 Isaac 連續跑 10 輪不中斷 | ① 位姿移動改成先 `/compute_ik`（帶種子 + 約束）解出關節角，再 `go_to_joints`<br>② 關節移動改用 RRTConnect<br>③ 確認 octomap 真的有建起來、修好 `camera_pcl_frame` TF 斷鏈<br>④ 解決放下番茄的問題（`GO_TO_BASKET=False` 時番茄不會被正確放下） | 🔴 進行中 |
| **M2：Isaac 上能全自動** | 不用人手動輸入 | 粗定位、精定位的目標選取改成自動（例如挑最近且到得了的）；移除 `cv2.imshow` / `input()` 的依賴 | ⬜ |
| **M3：實機移植** | 同一套流程在實機跑通 | 填好車用相機 topic 和外參（`vision_node/config.py:116-128`）；驗證精定位姿態到得了；驗證 octomap；確認實機夾爪參數；四種組合參數填齊 | ⬜ |
| **維護** | — | 更新兩份 SPEC（停在 08-25）；果梗殘影、果梗變粗；夾爪碰撞 mesh 簡化（8.4 萬面）；`ompl_planning.yaml` 的 `termination_condition` 要留或還原 | — |

## 六、已知限制

- `GO_TO_BASKET=False` 時，上一顆番茄要到下一輪開始接近時才鬆開，不是真正放下。
- 目標選取是阻塞式終端機互動，需要 tty；`cv2.imshow` 需要顯示環境。
- `arm_task_node` 自己補發手指的 `/joint_states`，下游若也發會互相覆蓋。
- 模型權重路徑寫死在本機。
- 果梗方向在深度稀疏時退回 `(0, 0, -1)`，是猜測值。
- 精定位姿態加上關節約束後，合法 IK 解很少且集中在 J3 接近硬體限位處（M1 ① 要處理）。
- 備用視角 ±50° 太大：J5 要 ≈173°（貼硬體限位）/ ≈−1°（貼腕部奇異點），−50° 視角在 J5 約束下會規劃失敗。要縮小 `ALT_VIEW_AZIMUTH_OFFSETS_DEG`。

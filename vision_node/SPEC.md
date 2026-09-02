# vision_node 技術規格

番茄／果梗 YOLOv11 分割視覺節點 — ROS 2 Humble / rclpy / Python 3.10 / Ultralytics YOLOv11-seg
位置：`/home/terry/tm_ws/python_isaac/vision_node`
整理依據：2026-08-25 程式碼狀態

## 00 概觀

`vision_node` 是溫室番茄採收機械臂的視覺大腦：訂閱 RGB-D 相機影像，用 YOLOv11 分割模型同時辨識「番茄」與「果梗」，把果梗骨架化找出抓取點與 3D 生長方向，反投影成世界座標後，交給操作者在終端機手動選定一顆目標，再發布抓取姿態給手臂端執行。

整包程式把「算」跟「串接」分開：`detector` / `coordinates` / `skeleton` / `stem_tracker` / `target_selector` / `mask_publisher` 都是不碰 ROS 訂閱/發布的純運算模組，唯一負責訂閱、發布、跨模組協調的是 `vision_node.py` 這支主節點。

## 01 系統情境

`vision_node` 不是獨立運作的節點，假設以下都已在同一個 ROS graph 上運行：

- 相機來源（Isaac Sim 或相機驅動）— 發布 `/camera/color/image_raw`、`/camera/depth/image_rect_raw`、`/camera/camera_info`
- 手臂控制節點（`arm_node`）— 發布 `/robot_status`，訂閱 `/target_pose`、`/vision_status`
- `cloud_filter_node.py`（`vision_node/` 目錄下的獨立節點，跟 `vision_node.py` 分開啟動成不同 process）— 訂閱 `/target_filter_mask`，做一次性點雲過濾，交給 MoveIt2 OctoMap 更新避障地圖

流向：相機三路訊號 + 手臂狀態 → `vision_node` → 抓取指令（`/target_pose`）與空目標通知（`/vision_status`）回手臂 → 選定目標當下發布 `/target_filter_mask` → `cloud_filter_node.py` → MoveIt2 OctoMap。

### 座標系（TF）

節點啟動時用 `StaticTransformBroadcaster` 廣播一次 `link_6`（`ARM_FLANGE_FRAME`）→ `camera_optical_frame` 的固定外參（平移 `(0, CAMERA_OFFSET_Y, 0)`，旋轉為單位四元數），並持續查詢 `world → camera_optical_frame` 供每幀反投影使用；查不到 TF 的那幀直接跳過偵測。

## 02 模組架構

除了 `vision_node.py`，其餘模組都不 import ROS 發布/訂閱機制，方便單獨測試。

| 檔案 | 職責 |
|---|---|
| `main.py` | 進入點：`rclpy.init` → 建立 `VisionNode` → `spin` → 收尾銷毀節點與 OpenCV 視窗 |
| `vision_node.py` | 主節點：ROS 訂閱/發布、TF 廣播與查詢、callback 串接，協調以下所有模組 |
| `config.py` | 全域常數集中定義（模型路徑、YOLO 參數、frame 名稱、深度/追蹤/選取/遮罩門檻），其餘模組一律從這裡 import，不重複定義 |
| `detector.py` | `ObjectDetector`：載入 YOLO 模型、跑推論、依 class 分流番茄/果梗、果梗重疊抑制，組成統一格式的偵測結果 |
| `coordinates.py` | `CoordinateEstimator`：像素+深度反投影成世界座標、深度中位數估算、果梗 3D 方向向量估計 |
| `skeleton.py` | `PedicelSkeletonizer`：果梗 mask 清理 → 骨架化 → 兩次 BFS 找最長路徑（樹的直徑）排序骨架 → 依比例取抓取點與生長角度。純 2D 像素運算 |
| `stem_tracker.py` | `StemTracker`：滑動視窗時間平滑，用像素距離配對前後幀同一根果梗，視窗內挑信心分數最高的一筆整包輸出 |
| `target_selector.py` | `TargetSelector`：依距基座距離排序候選、過濾超出工作範圍或座標異常的候選，終端機互動讓使用者輸入要夾取的 ID |
| `mask_publisher.py` | `TargetMaskBuilder`：合併「目標果梗」與「最近番茄」的膨脹遮罩，純運算，不做 publish |
| `visualizer.py` | `Visualizer`：cv2 疊圖（框線/ID/資訊面板）與終端機列印掃描結果，不含任何判斷邏輯 |
| `cloud_filter_node.py` | 獨立節點 `CloudFilterNode`：跟 `vision_node.py` 分開的 process，訂閱原始點雲 + `/target_filter_mask`，反投影過濾後發布 `/camera/depth/points_gated`；不被 `vision_node.py` import，兩者只透過 ROS topic 溝通 |

## 03 ROS 2 介面

| 方向 | Topic | 型別 | 說明 |
|---|---|---|---|
| SUB | `/camera/color/image_raw` | `sensor_msgs/Image` | RGB 影格，觸發主偵測迴圈 `color_callback` |
| SUB | `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | 對齊深度圖，`passthrough` 編碼直接轉 cv2 array |
| SUB | `/camera/camera_info` | `sensor_msgs/CameraInfo` | 相機內參 fx/fy/cx/cy（`k[0],k[4],k[2],k[5]`），用於反投影 |
| SUB | `/robot_status` | `std_msgs/String` | 資料為 `"DONE"` 才會開始/恢復偵測，其餘值視為手臂移動中 |
| PUB | `/target_pose` | `geometry_msgs/PoseStamped` | `position` 為世界座標抓取點；**借用** `orientation.x/y/z` 傳遞果梗 3D 方向單位向量，`orientation.w` 固定 `0.0`（非真正四元數） |
| PUB | `/vision_status` | `std_msgs/String` | 值 `"NO_TARGET"`：本輪無可夾取目標，通知手臂回初始位置 |
| PUB | `/target_filter_mask` | `sensor_msgs/Image` (mono8) | 選定目標的那一刻才發布一次；果梗+番茄合併膨脹遮罩，交給 `cloud_filter_node.py` |

節點名稱：`vision_node`（`rclpy.node.Node('vision_node')`）

## 04 運作週期

節點以「等待手臂完成 → 掃描 → 手動選定 → 發布指令 → 再次等待」為一個循環，由 `task_completed` / `is_processing` 兩個旗標把「感知」與「決策」串起來；實際選取/發布是 `auto_pick_thread` 開的獨立執行緒，避免終端機 `input()` 卡住 `rclpy.spin`。

```
① 等待 /robot_status = DONE          (status_callback → task_completed = True)
   │
② 掃描：YOLO 偵測 + StemTracker 平滑   (color_callback，每幀執行)
   │
③ 本輪有候選目標？ ── 否，連續空掃描 ≥ EMPTY_SCAN_GRACE ──┐
   │ 是                                                    │
④ 終端機輸入 ID／s／r  (prompt_choose_id，阻塞)             │
   │  │            └─ r：重新偵測 → 回到 ②                 │
   │  └─ s：跳過本輪 ───────────────────────────────────────┤
   │ 輸入有效 ID                                             │
⑤ 發布 /target_filter_mask，等待 OCTOMAP_UPDATE_WAIT_SEC     │
   │                                                         ▼
⑥ 發布 /target_pose（含果梗方向向量）；task_completed=False   發布 /vision_status = NO_TARGET
   │                                                         task_completed = False
⑦ 阻塞等待手臂完成動作（task_completed → True）                │
   │                                                         │
   └──────────────────────── 回到 ① ◄───────────────────────┘
```

## 05 偵測與座標流程

每一幀（`color_callback`）在確認 `camera_info`、深度圖、TF 都齊備後才會往下跑。

### 番茄分支（class 1）

取 bbox 中心點，在 `DEPTH_WINDOW` 視窗內取深度中位數；用「bbox 平均邊長換算的像素半徑」估出番茄半徑，把深度往前推一個半徑（`z_center = z + radius`），近似取到番茄「表面中心」而非最靠近相機的邊緣，再反投影＋TF 轉世界座標。有 mask 的話一併存二值遮罩，供之後選定目標時的點雲過濾配對使用。

### 果梗分支（class 0）

多個果梗 mask 先做重疊抑制（`suppress_overlapping_masks`）：兩個 mask 交集面積佔較小者比例超過 `OVERLAP_SUPPRESS_THRESH` 就視為重複偵測，只留面積較大的一個。

單一果梗 mask 走 `PedicelSkeletonizer`：取最大連通元件 → 視面積決定是否先侵蝕瘦身 → 骨架化 → 兩次 BFS 找骨架圖上的最長路徑（樹的直徑：任取一點 BFS 找最遠點 A，再從 A BFS 找最遠點 B，A—B 就是整根骨架最長的路徑），依 y 座標把路徑統一成從 calyx 端（y 較大，靠果實）排到 branch 端（y 較小，靠枝條）→ 依 `GRASP_RATIO`（預設 0.7，偏向 calyx 端）取抓取點像素座標與生長角度。

> 早期版本是「找兩個端點（依 y 最大/最小挑）→ 從 calyx 端沿單一鄰居往前走、不回溯」，遇到 mask 侵蝕/骨架化後常見的雜訊短分支（spur）時，端點誤判或走訪在分岔點提早斷掉，會讓抓取點卡在分岔附近甚至等於果實端端點本身。改成兩次 BFS 找最長路徑後，天生會忽略任何比主幹短的分支，不受這個問題影響。

抓取點附近的深度優先用 mask 內有效點數的中位數，不足 `MIN_STEM_DEPTH_PX` 才退回不濾 mask 的 3×3 視窗。3D 方向向量則是在抓取點前後各取一段骨架路徑（`end_span` 個點），分別聚合成 branch 端、calyx 端兩個世界座標點，取 `calyx − branch` 並單位化，得到「由枝條指向果實」的方向；任一端有效深度點數不足 `min_pts_per_end` 就回傳 `None`，由呼叫端退回 `(0, 0, -1)`。

### 時間平滑（StemTracker）

每幀輸出的果梗清單，用像素距離（`STEM_MATCH_DIST_PX` 內視為同一根）跟既有 track 配對；配對成功就把整包偵測結果塞進該 track 的滑動視窗（長度 `STEM_TRACK_WINDOW`），輸出時整包取視窗內信心分數最高的一筆（不做逐欄位混合平均）。連續 `STEM_TRACK_MAX_MISS` 幀配對不到的 track 視為消失並移除。

## 06 目標遮罩交接

點雲過濾原本寫在手臂端的任務節點裡，後來搬進 `vision_node`，理由是 mask／depth／camera_info／TF 這時全部都已經在手上，不用再跨節點傳遞；但實際「逐點過濾點雲」被拆到獨立的 `cloud_filter_node.py` 進行 —— `vision_node` 只在選定目標的那一刻，把果梗（膨脹 `STEM_MASK_DILATE_PX` px）與最近番茄（膨脹 `TARGET_MASK_DILATE_PX` px，找不到配對就只挖果梗）合併成一張二值遮罩，發布一次給 `/target_filter_mask`。掃描期間刻意不轉發任何點雲，OctoMap 保持空白，直到這一刻才建圖，避免掃描中把候選目標本身也一起變成障礙物。

> `cloud_filter_node.py` 把命中遮罩的點設成 NaN 來挖除後，`filtered_msg.is_dense` 早期版本直接複製原始點雲的 `is_dense`（原始點雲通常是 `True`）。這會讓宣稱「保證無 NaN」的訊息其實混了 NaN，MoveIt2 `PointCloudOctomapUpdater`（底層用 PCL 解析）信任這個旗標跳過 NaN 檢查，導致整包點雲處理失敗、OctoMap 完全建不出 voxel，且沒有任何錯誤 log（2026-08-25 用 `get_planning_scene` service 查證實 `resolution=0.0, data=[]`，即使近距離有效點數高達十幾萬個）。改成過濾後一律設 `is_dense = False`，如實反映資料含 NaN。

## 07 設定參數（`config.py`）

| 群組 | 參數 | 預設值 | 意義 |
|---|---|---|---|
| 模型與分類 | `MODEL_PATH` | `/home/terry/Desktop/stem_isaac_train/runs/segment/tomato_stem/exp3/weights/best.pt` | YOLOv11 分割模型權重，寫死本機路徑 |
| | `STEM_CLASS_ID` | 0 | 果梗類別 ID |
| | `TOMATO_CLASS_ID` | 1 | 番茄類別 ID |
| YOLO 推論 | `YOLO_IMGSZ` | 1024 | 推論輸入邊長 |
| | `YOLO_CONF` | 0.5 | 置信度門檻 |
| 座標系 | `WORLD_FRAME` | `world` | 世界座標系名稱 |
| | `CAMERA_OPTICAL_FRAME` | `camera_optical_frame` | 相機光學座標系 |
| | `ARM_FLANGE_FRAME` | `link_6` | 相機掛載的父座標系 |
| | `CAMERA_OFFSET_Y` | 0.12 m | 相機相對 link_6 的 Y 偏移 |
| 深度估計 | `DEPTH_WINDOW` | 5 px | 番茄/果梗中心取深度的視窗半徑 |
| | `MIN_STEM_DEPTH_PX` | 5 | 果梗 mask 內有效深度點數低於此值即退回 3×3 fallback |
| | `DEPTH_MM_THRESHOLD` | 10.0 | 深度值大於此數視為單位是 mm，需 /1000 轉公尺 |
| | `MIN_VALID_DEPTH_M` | 0.01 m | 深度小於此值視為無效 |
| 骨架化/抓取點 | `GRASP_RATIO` | 0.7 | 抓取點沿骨架路徑的位置比例（0=branch 端，1=calyx 端） |
| | `OVERLAP_SUPPRESS_THRESH` | 0.5 | 果梗 mask 重疊比例超過此值視為重複偵測 |
| StemTracker | `STEM_MATCH_DIST_PX` | 40.0 px | 前後幀配對同一根果梗的最大像素距離 |
| | `STEM_TRACK_WINDOW` | 7 | 滑動視窗長度（幀數） |
| | `STEM_TRACK_MAX_MISS` | 5 | 連續幾幀沒配對到就判定 track 消失 |
| 掃描/選取 | `EMPTY_SCAN_GRACE` | 2 | 連續幾次空掃描才回報 NO_TARGET |
| | `SCAN_PRINT_INTERVAL` | 1.5 s | 終端機列印候選清單的節流間隔 |
| | `MAX_REACH_M` | 1.0 m | 距離基座超過此值的候選直接排除 |
| 目標遮罩 | `TARGET_MASK_DILATE_PX` | 15 px | 目標番茄 mask 膨脹核心大小 |
| | `STEM_MASK_DILATE_PX` | 9 px | 目標果梗 mask 膨脹核心大小（較細，先給比番茄小的值） |
| | `OCTOMAP_UPDATE_WAIT_SEC` | 1.5 s | 發布過濾點雲後，等 MoveIt2 octomap updater 處理完再繼續 |

## 08 已知限制

- **模型路徑寫死在本機。** `MODEL_PATH` 指向 `/home/terry/Desktop/...`，換機器部署要記得改，沒有相對路徑或環境變數 fallback。
- **點雲過濾必須是獨立 process。** 實測發現 `vision_node` 只要疊加多個 subscription（即使 callback 是空的），點雲 `publish()` 就會被 MoveIt2 的 `PointCloudOctomapUpdater` 靜默拒收，原因未明；因此拆成最小化的 `cloud_filter_node.py`，不要把它併回主節點。
- **目標選取是阻塞式終端機互動，不是全自動。** `prompt_choose_id` 用 `input()`，且 `termios.tcflush(sys.stdin, ...)` 假設 stdin 是互動式 tty；用 launch file 背景執行或非 tty 環境下這段會出問題。
- **`cv2.imshow` 需要顯示環境。** 無頭（headless）機器上跑會直接噴錯，目前沒有可關閉視覺化的開關。
- **OctoMap 沒有「處理完成」事件。** `OCTOMAP_UPDATE_WAIT_SEC`（1.5 秒）是固定等待時間換取簡單可靠，不是真正的同步機制；如果 MoveIt2 端處理變慢，這個等待可能不夠。
- **`/target_pose` 的 orientation 欄位是借用的。** `x/y/z` 放的是果梗 3D 方向單位向量、`w` 固定 `0.0`，不是合法四元數；下游手臂端解析這個 topic 時必須知道這個慣例，否則會誤當成真的姿態旋轉。
- **果梗方向估計在深度稀疏時會退回預設值。** `estimate_pedicel_direction` 任一端有效深度點數不足就回傳 `None`，呼叫端退回 `(0, 0, -1)`（垂直向下），不會讓整筆偵測失敗，但下游要留意這是一個「猜測值」而非真實估計。

## 09 啟動方式

不是標準 colcon package（沒有 `package.xml` / `setup.py`），以 Python module 方式執行：

```bash
cd /home/terry/tm_ws/python_isaac
python3 -m vision_node.main
```

`cloud_filter_node.py` 是分開啟動的獨立 process（理由見 08 已知限制），另開一個終端機執行：

```bash
cd /home/terry/tm_ws/python_isaac/vision_node
python3 cloud_filter_node.py --ros-args -p use_sim_time:=true \
    -p fx:=<你的fx> -p fy:=<你的fy> -p cx:=<你的cx> -p cy:=<你的cy>
```

> 早期版本 `cloud_filter_node.py` 放在平行目錄 `python_isaac/point/`，跟 `minimal_relay_test.py` 等驗證用的實驗腳本混在一起；但它其實是這個套件在正式運作時會用到的節點，不是實驗程式，2026-08-25 搬進 `vision_node/` 跟其他模組放在一起管理，行為本身沒有變動，只是搬了檔案位置。

啟動前需要就緒的前置節點：

- **相機來源**：發布 `/camera/color/image_raw`、`/camera/depth/image_rect_raw`、`/camera/camera_info`（Isaac Sim 或相機驅動）
- **TF 樹**：`world → link_6` 要能查到；`link_6 → camera_optical_frame` 由本節點自己廣播
- **手臂狀態**：需有節點發布 `/robot_status`，值為 `DONE` 時本節點才開始掃描
- **下游訂閱**：需有節點訂閱 `/target_pose`（執行夾取）與 `/target_filter_mask`（`cloud_filter_node.py`，過濾點雲）

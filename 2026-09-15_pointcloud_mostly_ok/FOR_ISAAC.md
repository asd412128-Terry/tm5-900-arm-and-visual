# 給 Isaac 機器那邊的摘要（2026-09-15，來自 real 機器上的改動）

## cloud_filter_node.py 今天改了什麼

1. **過濾邏輯本身沒動**——深度圖轉點雲、遮罩比對、離群點濾除
   （`OUTLIER_MIN_NEIGHBORS=15`, `OUTLIER_RADIUS_M=5mm`）都沒改。

2. **發布機制大改**：原本收到目標遮罩後，2 秒內每收到一張新深度圖就重新過濾、
   重新發布一次（舊 burst，會疊出「果梗變粗」）。現在改成**只用一幀深度圖過濾一次**，
   把算好的同一份 PointCloud2 訊息在 `FILTER_BURST_DURATION_SEC`（2.0s）內，每
   `REPUBLISH_INTERVAL_SEC`（0.1s）重複發送**同一份 bytes**——不重新計算、不換幀。

3. **中間繞了一個彎路，real 機器上已經證偽，Isaac 不用再試一次：**
   試過拿掉連發、改成自己養一個 `tf2_ros.Buffer`/`TransformListener`，主動問
   `can_transform()` 確認 TF 就緒後才發布唯一一次。**實機測試結果：log 確實印出
   「已發布給 OctoMap」，代表 TF 就緒判定通過、也真的呼叫了 publish()，但用
   `GetPlanningScene`（`components=32`）查 octomap，`data=[]`，完全是空的。**
   代表問題根本不在「TF 有沒有就緒」，卡在 MoveIt2 收到訊息之後的某個環節，
   原因還沒查出來。**這個方向已經放棄，程式碼裡的 `tf2_ros.Buffer`/
   `TransformListener` 也已經拿掉**，回到只有兩個 subscription（深度圖+遮罩）
   的最小化結構。

4. **isaac 分支這次完全沒有實機驗證過**，只有讀程式碼推理：
   - `VISION_MODE=isaac` 對應的深度 topic（`/camera/depth/image_rect_raw`）跟
     `_ISAAC_INTRINSICS` 都沒改動。
   - 但「只算一次、重複發送同一份」這個新機制**沒在 isaac 上測過**，理論上
     應該跟 real 模式行為一致，但沒實測驗證。

5. `sensors_3d.yaml`（`tm5-900_moveit_config`）的 `point_subsample` 從 15 調到
   10——這是 real 機器上 real 機器的 moveit config 檔，**不在 python_real repo
   裡**，如果 Isaac 用的是不同的 moveit config（例如 tm5-900 isaac 版），這個
   改動不會自動套用，要另外確認 Isaac 那邊的 `sensors_3d.yaml` 要不要同步調整。

## 上 Isaac 機器測試時，優先驗證這件事

選定目標之後，用這個指令直接查 octomap 有沒有真的建出來，**不要只看 RViz**：

```
ros2 service call /get_planning_scene moveit_msgs/srv/GetPlanningScene \
  "{components: {components: 32}}"
```

看 `octomap.data` 是不是空的（`data=[]` 代表沒建出來，components 一定要用
32，8 是 `WORLD_OBJECT_NAMES` 不是 octomap，之前查錯過一次）。

如果 Isaac 上也是空的，代表這不是 real 機器特有的問題，是這個新機制本身有
共通的 bug，值得優先查；如果 Isaac 上正常，代表可能是 real 機器這邊某個
環境特有的因素（例如 real 相機的 depth topic 特性、或 real 那邊的 TF 廣播
狀況跟 isaac 模擬時鐘不同）。

## 還沒解決的問題（跟今天改動無關，isaac/real 共通）

- 果梗選定後殘影沒清乾淨。
- 果梗看起來比實際粗（理論上這版新機制應該改善，但要先確認 octomap 能建出來
  才能驗證）。

詳細診斷過程可以看 real 機器上 auto-memory 的 `project_cloud_filter_rewrite.md`
（如果 Isaac 那邊的 Claude 沒有這份記憶，可以請它去 real 機器的
`~/.claude/projects/-home-lab604-tm-ws-python-real/memory/` 找）。

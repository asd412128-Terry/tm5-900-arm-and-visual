# 2026-09-15 備份說明

點雲大部分 OK，可能還需要改。

## 今天做的事
- `vision_node/cloud_filter_node.py`：修掉深度/彩色鏡頭沒對齊的系統性偏移、點雲密度異常稀疏、`OCTOMAP_UPDATE_WAIT_SEC` 固定 sleep 猜時間這三個問題（已驗證正常）。
- 試過「拿掉連發、改成主動等 TF 就緒才發一次」——**實機測試證偽**：TF 確實就緒、也真的呼叫了 publish，但 OctoMap 最後還是空的，真正原因還沒查出來。已放棄這個方向。
- 目前版本：退回連發，但只算一次過濾結果、在 2 秒內重複發送同一份訊息（不重新過濾、不換幀），避免舊版「每幀重新過濾」造成的果梗變粗。**這個版本還沒實機驗證過**，下次上機第一件事是用 `ros2 service call /get_planning_scene ... {components: {components: 32}}` 確認 `octomap.data` 是否非空。
- 順便調過 `sensors_3d.yaml` 的 `point_subsample`（15→1→5→10，目前留在 10）——證實這個參數只影響密度，不是造成 OctoMap 全空的原因。

## 還沒解決
- 果梗選定後殘影沒清乾淨。
- 果梗看起來比實際粗（理論上這版應該改善，但要等 OctoMap 能正常建圖才能驗證）。
- OctoMap 現在到底能不能正常建出來，還沒有實機確認過。

詳細診斷過程見 auto-memory：`project_cloud_filter_rewrite.md`。

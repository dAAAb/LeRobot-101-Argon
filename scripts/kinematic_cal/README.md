# kinematic_cal

腕部相機＋固定棋盤的運動學校準工具（關節零位、手眼、桌面平面）。完整說明、結果與待辦見 [docs/KINEMATIC_CALIBRATION_2026-10-01.md](../../docs/KINEMATIC_CALIBRATION_2026-10-01.md)。

- 只有 `kcal_search.py` 與 `kcal_run.py` 會移動手臂，而且只寫五個身體關節的 Goal_Position。
- 新拍攝的資料寫到 `.runtime/kinematic-cal/`（git 忽略）；已發布的資料在 `config/calibration/kinematic-sessions/`。
- 執行前先停掉 Astra 服務（串列埠獨占），並確認已安裝 `ffmpeg`。

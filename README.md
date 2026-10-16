# LeRobot 101 Argon

> **用 Google Search 般自然簡潔的介面，指揮 LeRobot SO-101 機械臂做出人類單手自然動作與桌面簡單任務。**  
> 結合 **LeRobot-101-Astra** 運動學與安全底座、**AIRI MotionGPT** 人體單手 3D 動作重映射，以及 **Google Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`) / Gemini 4 Argon** 具身空間推理。

---

## 核心特色

1. **Google Search 天然簡潔介面 ([`app/page.tsx`](app/page.tsx))**
   - 支援文字、語音（`SpeechRecognition`）、參考照片上傳與即時腕部相機點選。
   - 首頁與搜尋結果頁直接整合 **「單手自然動作 (MotionGPT)」** 快速膠囊列與 **「Gemini Robotics ER 2 空間推理」** 雙模分頁。

2. **人體單手動作重映射：AIRI MotionGPT → LeRobot SO-101 ([`server/motiongpt_arm.py`](server/motiongpt_arm.py), [`components/MotionRetargetCard.tsx`](components/MotionRetargetCard.tsx))**
   - 整合自 `airi/course/ntu-vh2026/motiongpt` 的 `HumanML3D` 22 關節 20 FPS 動作引擎（支援 Apple Silicon 原生 MLX / PyTorch 與內建經驗證的 MLX 實測動作片段：`right-wave`, `stretch`, `boxing`, `dance`, `bow` 及合成自然手勢 `cheer`, `nod`, `beckon`）。
   - **單手空間重映射（Single-Arm Spatial Retargeting）**：
     - 即時建立人體軀幹正交基底 $(\hat{\mathbf{e}}_{\text{left}}, \hat{\mathbf{e}}_{\text{up}}, \hat{\mathbf{e}}_{\text{front}})$，自動偵測慣用活動單手或依指令指定**右手 / 左手**（若指定相反側手會自動執行矢狀面鏡像對稱）。
     - 抽取單手肩 $\to$ 肘 $\to$ 腕球面方位角、仰角、肘彎曲角與軀幹俯仰／扭轉角，平滑映射至 SO-101 五軸（`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`）。
     - **3D 桌面安全淨空投影**：每一帧皆通過 URDF 連桿 STL 網格與標定傾斜桌面的 3D 淨空檢查（保證運動連桿離桌 $\ge 55\text{ mm}$）。
   - 前端提供**雙 3D 畫布即時預覽**（左側 22 關節人體骨架並高亮選中單手、右側 SO-101 五軸軌跡與桌面安全線），支援時間軸拖曳與一鍵實體手臂執行。

3. **Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`) 具身空間推理 ([`server/er2_brain.py`](server/er2_brain.py), [`components/ER2SpatialCard.tsx`](components/ER2SpatialCard.tsx))**
   - 使用官方 `google-genai` SDK 串接 `gemini-robotics-er-2-preview`，輸出 `0–1000` 正規化 2D 點選座標 `point: [y, x]`、邊界框 `box_2d: [ymin, xmin, ymax, xmax]` 與物件底部接觸腳點（`bottom_left`, `bottom_right`, `top_center`）。
   - 支援切換 `thinking_level` (`none` / `low` / `high`)，並自動將底部接觸點投影至三位置棋盤格標定之 3D 傾斜桌面（[`config/calibration/kinematic-model-20261002.json`](config/calibration/kinematic-model-20261002.json)，中位誤差 2.8 mm），直接算出桌面物件的公制 3D 座標 $(X, Y, Z)$、半徑與高度。

4. **完整知識庫與安全護欄 ([`docs/ARGON_KNOWLEDGE_BASE.md`](docs/ARGON_KNOWLEDGE_BASE.md))**
   - 完整保留 `LeRobot-101-Astra` 的單一馬達擁有者執行緒、永不寫入馬達 EEPROM、軟體緊急停止門閂（`StopLatch`）、夾爪低功率保護與多視角主動觀察知識庫。

---

## 快速啟動

```bash
# 1. (選用) 設定 Gemini API Key 以啟用 Gemini Robotics ER 2 即時空間點選
export GEMINI_API_KEY="your-api-key"

# 2. (選用) 若需下載完整 1.33 GB 本地 MotionGPT 權重以即時生成任意新英文 Prompt
python3 motiongpt/download-model.py

# 3. 啟動後端 (port 8102) 與前端 (port 3100)
./start.sh
```

---

## API 端點一覽

| 端點 | 說明 |
| :--- | :--- |
| `GET /api/status` | 回傳機械臂、腕部相機、`er2` (`gemini-robotics-er-2-preview`) 與 `motiongpt` 狀態 |
| `GET /api/motiongpt/presets` | 取得內建單手自然動作清單（揮手、伸懶腰、打拳、跳舞、鞠躬、歡呼、點頭、招手） |
| `POST /api/motiongpt/plan` | 將自然語言指令與指定單手 (`auto` / `right` / `left`) 轉換為 HumanML3D 22 關節與 SO-101 5 軸重映射軌跡（純計算、不驅動馬達） |
| `POST /api/motiongpt/execute` | 在實體 LeRobot SO-101 手臂上平滑執行通過 3D 桌面淨空驗證的單手重映射軌跡 |
| `POST /api/er2/spatial` | 呼叫 `gemini-robotics-er-2-preview` 標定腕部影像中的 `0–1000` 點選、邊界框與 3D 桌面公制座標 |
| `POST /api/search` / `POST /api/command` | Google Search 風格物件搜尋、動作建議與多步任務執行 |
| `POST /api/kinematic-grasp/plan` & `/execute` | 基於三位置棋盤格標定與 3D 桌面射線交點的圓柱物件抓取與放回 |

---

## 測試驗證

```bash
PYTHONPATH=.:tests python3 -m unittest test_argon_motiongpt_and_er2 test_search_intent -v
```

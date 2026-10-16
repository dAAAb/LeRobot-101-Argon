# LeRobot 101 Argon — 核心知識庫 (Knowledge Base)

本文件整合自三個核心參考來源，作為 **LeRobot 101 Argon** 的架構與空間運動控制知識庫：

1. **LeRobot-101-Astra**：Google Search 風格自然介面、Feetech STS3215 安全馬達驅動、三位置棋盤格運動學校準（2.8 mm 中位誤差）、傾斜桌面 3D 射線交點圓柱定位與連續軌跡控制。
2. **AIRI (`ntu-vh2026/motiongpt` & `stage-ui-three`)**：本地 MotionGPT（Apple Silicon MLX / PyTorch MPS / CPU）生成 `HumanML3D` 22 關節 3D 人體動作，並將**人體單手（Single-Arm）動作空間重映射（Retargeting）**至 LeRobot SO-101 五軸機械臂。
3. **Google AI Studio — Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`) & Gemini 4 Argon**：具身空間推理（Embodied Reasoning）、`0–1000` 正規化 2D 點選（`point: [y, x]`）與邊界框（`box_2d: [ymin, xmin, ymax, xmax]`）、多步任務規劃與視覺閉環驗證。

---

## 1. 硬體與運動學校準知識（繼承自 LeRobot-101-Astra）

### 1.1 硬體規格與安全鐵律
- **機械臂**：LeRobot SO-101 Follower（5 個本體旋轉關節 `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll` + 1 個夾爪關節 `gripper`），採用 Feetech STS3215 串列伺服馬達（4096 ticks / $360^\circ$）。
- **感測器**：僅使用安裝於手腕夾爪上方的 `icspring camera`（1920×1080 腕部相機），不使用外部深度相機。
- **安全不變量**：
  - **絕不寫入馬達 EEPROM**：所有移動僅寫入 RAM `Goal_Position`，保護校準與馬達壽命。
  - **軟體停止門閂（`StopLatch`）**：使用者按下「停止」時立即切斷運動迴圈，讀取當前 `Present_Position` 並維持原位。
  - **夾爪低功率保護（`GripperPowerManager`）**：接觸與保持階段限制 PWM（`150`），防止堵轉過熱。
  - **傾斜桌面 3D 碰撞防護（`LocalCartesianModel`）**：所有關節軌跡在執行前皆通過 URDF 連桿 STL 網格與標定桌面平面的 3D 淨空檢查（預設安全高度 $\ge 50\text{ mm}$）。

### 1.2 三位置棋盤格運動學標定 (`config/calibration/kinematic-model-20261002.json`)
- **關鍵發現**：單純依賴 CAD URDF 進行開環抓取會因關節零點偏差（約 $1.5^\circ\text{–}3^\circ$）與桌面傾斜（約 $1.06^\circ$）產生 $23\text{ mm}$ 以上的系統誤差。
- **軟體關節零點修正 (`server/kinematic_offsets.py`)**：
  - `shoulder_lift`: $-1.881^\circ$
  - `elbow_flex`: $+2.805^\circ$
  - `wrist_flex`: $-1.375^\circ$
- **精度指標**：標定後在桌面工作區達到 **2.8 mm 中位 3D 定位誤差**（最大 4.5 mm）。
- **macOS OpenCV 注意事項**：在設定或載入 `cv2` 前必須設定 `OPENCV_OPENCL_RUNTIME=disabled`，避免 macOS 背景執行緒呼叫 `findChessboardCornersSB` 時發生 OpenCL segfault。

---

## 2. 人體單手動作重映射：MotionGPT HumanML3D → LeRobot SO-101

### 2.1 MotionGPT 輸出格式 (`humanml3d-22`)
- **輸入限制**：MotionGPT 的 T5 tokenizer 僅支援英文詞彙（中文會退化為 `<unk>`），因此自然語言搜尋框收到中文動作指令（如「揮揮手」、「右手打拳」、「伸個懶腰」、「鞠躬打招呼」）時，會先轉譯為簡潔英文 `motionPrompt`（如 `"A person waves hello with their right hand."`）。
- **3D 骨架座標系**：右手座標系，`+Y` 向上、`+Z` 為初始正面朝向、`+X` 為解剖學左側（人物左手方向），帧率 `20 FPS`，每帧包含 22 個 3D 關節點：
  - 軀幹：`0: pelvis`, `3: spine1`, `6: spine2`, `9: spine3`, `12: neck`, `15: head`
  - 左手鏈：`13: left_collar` $\to$ `16: left_shoulder` $\to$ `18: left_elbow` $\to$ `20: left_wrist`
  - 右手鏈：`14: right_collar` $\to$ `17: right_shoulder` $\to$ `19: right_elbow` $\to$ `21: right_wrist`

### 2.2 單手動作抽取與重映射數學模型 (`server/motiongpt_arm.py`)
為了將人類單手（預設為活動量較大的慣用手，或依指令指定左手／右手）的自然動作套用到單臂 5-DOF 的 LeRobot SO-101 上，Argon 實作了**軀幹正交基底投影 + 球面運動學重映射 + 3D 桌面淨空投影**：

1. **即時人體軀幹正交基底 $(\hat{\mathbf{e}}_{\text{left}}, \hat{\mathbf{e}}_{\text{up}}, \hat{\mathbf{e}}_{\text{front}})$**：
   - 由雙肩向量 $\mathbf{s} = \mathbf{p}_{16} - \mathbf{p}_{17}$（左肩減右肩）與脊柱向量 $\mathbf{v} = \mathbf{p}_{12} - \mathbf{p}_{0}$（頸部減骨盆）建立每帧的局部軀幹座標系：
     $$\hat{\mathbf{e}}_{\text{up}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}, \quad \hat{\mathbf{e}}_{\text{front}} = \text{normalize}(\hat{\mathbf{e}}_{\text{left, raw}} \times \hat{\mathbf{e}}_{\text{up}}), \quad \hat{\mathbf{e}}_{\text{left}} = \hat{\mathbf{e}}_{\text{up}} \times \hat{\mathbf{e}}_{\text{front}}$$
   - 同時追蹤軀幹相對初始帧的俯仰角 $\theta_{\text{torso\_pitch}}$（例如「鞠躬 `bow`」時脊柱前傾）與水平扭轉角 $\psi_{\text{torso\_yaw}}$（例如「跳舞 `dance`」時轉身），使全身性語義動作也能自然反映在機械臂姿態上。

2. **主動單手自動選擇 (`select_active_human_arm`)**：
   - 若指令明確指定「左手 / left」或「右手 / right」，直接採用該側手臂鏈。
   - 否則計算左右手腕與手肘在軀幹座標系下的總路徑長度與高度變化量，自動選取動作幅度較大的主力單手（預設同分時優先選右手 `right`）。

3. **人類單手鏈 $\to$ SO-101 五軸關節映射**：
   - 定義軀幹座標系下的上臂單位向量 $\mathbf{u} = \text{normalize}(\mathbf{p}_{\text{elbow}} - \mathbf{p}_{\text{shoulder}})$ 與前臂單位向量 $\mathbf{f} = \text{normalize}(\mathbf{p}_{\text{wrist}} - \mathbf{p}_{\text{elbow}})$，以及肩到腕的伸展向量 $\mathbf{w} = \mathbf{p}_{\text{wrist}} - \mathbf{p}_{\text{shoulder}}$。
   - 將分量轉換為手臂側向 $y_{\text{arm}}$、前向 $z_{\text{arm}}$、向上 $x_{\text{up}}$：
     - **`shoulder_pan`（底座水平旋轉）**：由手腕前向與側向方位角 $\text{atan2}(y_{\text{arm}}, z_{\text{arm}})$ 加上軀幹水平扭轉角 $\psi_{\text{torso\_yaw}}$ 驅動。
     - **`shoulder_lift`（肩部俯仰）**：由上臂仰角 $\arcsin(u_{\text{up}})$ 與軀幹前傾角 $\theta_{\text{torso\_pitch}}$ 共同驅動（人類抬手時 SO-101 `shoulder_lift` 往負角度抬升；人類鞠躬時溫和前傾）。
     - **`elbow_flex`（肘部彎曲）**：由人類上臂與前臂夾角 $\phi_{\text{elbow}} = \arccos(\text{clamp}(\mathbf{u} \cdot \mathbf{f}, -1, 1))$ 映射至 SO-101 安全高淨空工作區間。
     - **`wrist_flex`（手腕俯仰）**：由前臂相對上臂的垂直仰角變化與點頭／揮手節奏驅動，維持視覺與姿態自然流暢。
     - **`wrist_roll`（手腕翻轉）**：由前臂橫向擺動角速度與手腕側向偏移驅動，使「揮手 (`right-wave`)」、「打拳 (`boxing`)」、「跳舞 (`dance`)」具備鮮明的末端表現力。

4. **3D 桌面淨空約束與相位平滑 (`Table Clearance Projection`)**：
   - 每一帧重映射出的關節角 $\mathbf{q}_t$ 皆輸入 `LocalCartesianModel.pose_clearance()` 計算完整 URDF 連桿網格與傾斜桌面的最小距離 $d_{\text{table}}(\mathbf{q}_t)$。
   - 若 $d_{\text{table}}(\mathbf{q}_t) < 55\text{ mm}$，演算法自動沿梯度方向微調抬升 `shoulder_lift` 與收攏 `elbow_flex`，確保整段單手動作 100% 位於安全空間內，並在頭尾加上平滑過渡包絡線（Fade-in / Fade-out Ramp）。

---

## 3. Gemini Robotics ER 2 (`gemini-robotics-er-2-preview`) 具身推理整合

### 3.1 官方 SDK 與模型規範
- 使用官方 **`google-genai`** Python SDK（`from google import genai`），支援：
  - **`gemini-robotics-er-2-preview`**：專為具身空間推理、2D 點選、邊界框定位、軌跡規劃與多步任務編排設計。
  - **`gemini-3.8-pro-preview` / Gemini 4 Argon**：高階空間推理與複雜場景理解。
- **空間座標格式 (`0–1000` 正規化整數)**：
  - **Pointing (`point`)**：`[y, x]`，範圍 `0..1000`，代表目標物件抓取點、底部接觸腳點（`bottom_left`, `bottom_right`）或放置目標中心。
  - **Bounding Box (`box_2d`)**：`[ymin, xmin, ymax, xmax]`，範圍 `0..1000`。
- **雙引擎無縫切換 (`server/er2_brain.py` + `server/brain.py`)**：
  - 當環境變數提供 `GEMINI_API_KEY` 時，可直接透過 `GeminiRoboticsER2Brain` 執行低延遲空間標註、物件搜尋、單手動作指令解析與多步技能規劃（`thinking_level="low"` 用於即時定位，`thinking_level="high"` 用於多步任務規劃）。
  - 同時完整保留原有的 Codex CLI 視覺推理引擎作為雙模後援，並可在 UI 設定面板即時查看目前啟用的推理核心。

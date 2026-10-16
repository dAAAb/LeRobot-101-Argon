# 選配 MotionGPT：文字真的生成新的動作

這個本機模組使用官方 **MotionGPT-base** 權重，由文字經 FLAN-T5 產生動作 token，再經 VQ 解碼器還原成 HumanML3D 的 22 個關節座標。它不是依關鍵字挑選預先寫好的跳舞動畫；原本 AIRI 的 `wave`、`dance` 等短動作仍是另一條較快、可預期的程序動畫路徑。

我們選擇有完整小型推論路徑的 MotionGPT-base 作為教學實驗。它不是最新的 MotionGPT3，也不保證理解每種舞名、中文敘述或複雜的左右組合。先用一句簡單英文描述身體動作，生成後觀看結果；「成功取得有限座標」和「動作符合語意」要分開評測。

**AIRI Local 0.4.1 的設定頁可輸入中文。** 頁面先使用「意識」目前的本機語言模型轉成簡短英文，顯示實際描述，再交給 MotionGPT。若只測英文，不需要這個轉換步驟。沒有本機對話模型時會提示設定，不會暗中使用雲端服務。

中文前處理有實際原因：這版 tokenizer 把「跳高！」的中文字轉成 `<unk>`，而非理解後再決定跳法。[原始 tokenizer 與跳躍測試](../results/motiongpt/jump-language/README.zh-TW.md)保留輸入、輸出與高度數值。

## 安裝與授權

安裝器內的 **MotionGPT Base（VRM）** 預設不勾選。第一次選用需從官方來源下載，八個必要檔案合計約 **1.336 GB**，另需 Python、PyTorch 等執行環境。模型檔不放進我們公開的 Full/Lite 安裝包，也不轉存到 fork 的 release。下載完成後，推論不需要網路。

官方程式碼是 MIT；官方權重卡目前只標 `license: cc`，沒有寫出明確的 Creative Commons 版本及再散布條件。因此不能把權重宣稱為 MIT，或視為已取得公開重新打包授權。這個功能採用使用者自行從官方固定版本下載的方式。公開商用或再散布前應向上游確認授權。

- [官方 MotionGPT 程式與說明](https://github.com/OpenMotionLab/MotionGPT)
- [官方 MotionGPT-base 模型卡](https://huggingface.co/OpenMotionLab/MotionGPT-base)
- [較新的 MotionGPT3](https://github.com/OpenMotionLab/MotionGPT3)

`model-manifest.json` 固定每個檔案的來源 commit、大小與 SHA-256；引擎每次載入前重新驗證。模型以 `torch.load(weights_only=True)` 載入，沒有執行下載的 pickle 任意程式。只使用文字模型、VQ 解碼器及兩個正規化陣列；不需 SMPL 授權資產、訓練資料、評測模型或雲端 API。

開發環境重現（在本資料夾執行，需 Python 3.12 與 uv）：

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python -B download-model.py
AIRI_MOTIONGPT_MODEL_DIR="$PWD/.models" PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B server.py
```

`download-model.py --verify-only` 只驗證、不下載。下載器不接受任意模型 URL，且保留至少 8 GiB 可用磁碟。`requirements-lock.txt` 記錄這次測試的全部 Python 套件版本；正式 App 另外把 Python 本身搬成可攜式環境，不能直接搬開發 `.venv` 的外部 symlink。

## 三種真正的運算後端：CPU、MPS、MLX

CPU 與 MPS 使用 PyTorch；MPS 把算子送到 Apple Metal GPU。新增的 **MLX 是原生 MLX／Metal 推論**：T5 encoder、逐步 decoder、self/cross-attention KV cache、VQ Conv1d 解碼與關節還原都用 `mlx.core` 執行。它不是把 PyTorch CPU 包上一個 MLX 名稱。第一次載入用 PyTorch 讀取已核對雜湊的 checkpoint、轉成 safetensors，之後生成不用 PyTorch。

MLX 轉換檔只寫在使用者的 `payload/motiongpt/.mlx-cache/`，約 **1.034 GB（986 MiB）**，不寫入已簽章 App、不重新下載模型，也不重新散布轉換權重。第一次轉換及 GPU 暖機需要額外時間。轉換程式保留 8 GiB 磁碟空間；檔案與來源 checkpoint 的雜湊保存在本機 `conversion.json`。

2026-10-08，Apple M4 Max／128 GB，PyTorch 2.8.0、MLX 0.32.3、float32、單一請求、seed 42。下表是各提示第 **2 次**生成，不含模型載入（CPU／MPS 約 3 秒、已有轉換快取的 MLX 約 1 秒）。

| 提示 | CPU | MPS GPU | MLX GPU | CPU／MPS／MLX 動作 token 數 |
| --- | ---: | ---: | ---: | --- |
| 右手揮手 | 0.271 秒 | 0.587 秒 | **0.076 秒** | 17／17／16 |
| 向前鞠躬 | 0.303 秒 | 0.868 秒 | **0.096 秒** | 21／26／21 |
| 開心跳舞 | 0.609 秒 | 1.562 秒 | **0.182 秒** | 49／48／48 |
| 雙臂向頭頂伸展 | 0.325 秒 | 0.747 秒 | **0.098 秒** | 23／22／23 |
| 雙臂拳擊組合 | 0.481 秒 | 1.578 秒 | **0.181 秒** | 39／49／48 |

這是這台機器的兩次取樣，並非所有 Mac 的結論。不同框架的 RNG 即使 seed 相同也可能生成不同 token、片長和動作，不能把每列當成完全等量運算。第一段 MLX 揮手在 GPU 暖機時約 0.896 秒，第二次約 0.076 秒。這也說明「換成 GPU」與「改成適合 GPU 的原生運算和 cache」是兩件事。

因此新版服務 `auto` 優先選可用的 MLX，載入失敗才回 CPU，並透過 `/health` 的 `auto_fallback_reason` 說明原因。也能明確選 `cpu`、`mps`、`mlx`；指定某 GPU 失敗會回報錯誤，保留原本可用的後端，不假裝切換成功。

為了確認速度沒有靠錯誤計算換來，我們另外用**完全相同的輸入、16 步 decoder token 與 VQ token**比較 PyTorch CPU 和 MLX GPU。最大絕對誤差：encoder `1.8e-7`、logits `6.1e-5`、VQ features `1.14e-5`、最後關節 `4.8e-7` 公尺；16 步 logits 的 argmax 全相同。這項比對也在沒有安裝 PyTorch 的 MLX runtime 成功執行，另在正式可攜式 runtime 重跑通過。

原始三後端各五提示、兩次取樣共 30 段座標與時間位於 `../results/motiongpt/backends/`；數值比對為 `mlx-parity.json`、`mlx-parity-portable.json`。初版 CPU／MPS 的舊紀錄保留，讓同學看見試錯過程。不要只看「模型說它在跳舞」：應直接看關節序列、VRM 播放、左右手、朝向及返回姿態。

## 原始骨架的動作品質檢查

我們把真實 CPU 輸出的 22 個關節畫成固定正面與側面圖，沒有混入程序動畫。橘色是角色右肢，青色是左肢。

| 提示 | 結果與限制 |
| --- | --- |
| 右手揮手 | [3.4 秒骨架 GIF](../results/motiongpt/right-wave-cpu-skeleton.gif)：右手高舉並左右往返，左手維持低位，最後放下。 |
| 前彎鞠躬後站直 | [4.2 秒骨架 GIF](../results/motiongpt/bow-cpu-skeleton.gif)：確實前彎後回正，但多了未要求的抬左腳。 |
| 開心跳舞 | [9.8 秒骨架 GIF](../results/motiongpt/dance-cpu-skeleton.gif)：雙臂、步伐與轉身都有變化，原始 root 也有位移，並非原地循環。 |
| 雙臂向頭頂伸展 | [關鍵格圖](../results/motiongpt/stretch-cpu-keyframes.png)：92 格中有 57 格雙腕同時高於頭部，雙臂抬高、伸展，再放下，符合這次提示。 |
| 雙臂拳擊組合 | [關鍵格圖](../results/motiongpt/boxing-cpu-keyframes.png)：看見兩臂交替伸出、收回與步伐，外觀符合拳擊類動作。但 22 關節沒有手指，無法確認握拳或專業拳擊技術。 |

伸展與拳擊的提示沒有依賴既有 `wave`／`dance` 動作清單。這兩段各是一次 seed 42 的輸出，不能據此宣稱任意新動作都能正確生成。

[完整骨架檢查紀錄](../results/motiongpt/skeleton-preview-audit.md)列出座標方向、偏差與雜湊來源。這份檢查只驗證模型的原始座標與投影圖，**不代表原生 App 或 VRM 播放已完成驗收**。

## API 與 VRM 接法

服務只監聽 `127.0.0.1:17905`，`GET /health` 回傳模型是否已載入、實際 device。`POST /v1/motions/generate`：

```json
{"prompt":"A person waves their right hand to greet someone.","seed":42}
```

`POST /v1/runtime` 以 `{"device":"mlx"}` 切換實際後端，接受 `auto`／`cpu`／`mps`／`mlx`。重新載入期間回報 busy，生成與另一個切換請求得到 409；載入成功才更換引擎。`GET /health` 的 `selected_device` 是使用者選項，`device` 才是正在運作的引擎。

輸出 `format: humanml3d-22`、`fps: 20`、`joints[frame][joint][xyz]`。提示上限 512 字元；seed 為 0 到 2147483647 的整數；不接受額外任意模型、URL、時長參數。一次僅生成一段，忙碌回傳 409；模型生成失敗回傳錯誤，不偷偷替換成程序動畫。座標必須有限且在 ±100 公尺內；輸出最多 196 格（9.8 秒），超過會標示 `truncated`，保留 `generated_frames` 原長度。

HumanML3D 使用右手座標：+Y 向上、初始面向 +Z，**角色自己的左側為 +X**。22 個關節名稱直接附在回應，避免把觀看者的左邊當成角色的左手。AIRI 的 VRM 播放器再依 VRM 0.x／1.0 的骨架座標轉換，並保留表情及語音嘴型。0.4.1 保留相對首幀的上下位移，按來源與角色腿長比例縮放，限制為下降 0.9 腿長、上升 1.5 腿長。水平走位仍清除，取消或結束會回復原位。沒有腳底接觸修正、物理平衡、手指動作或 Live2D 重定向；若來源首幀已蹲下，它仍以該幀為高度基準。

HTTP 請求限制 8 KiB；Host 限 localhost／127.0.0.1，網頁來源只允許本機 AIRI 的 17900／5174 連線。無 Origin 的本機 CLI 可以呼叫。這不是對外網路服務。

## 驗證

```sh
.venv/bin/python -B -m unittest discover -p 'test_*.py' -v
.venv/bin/python -B benchmark.py --devices cpu mps \
  --cases right-wave bow dance --output ../results/motiongpt
.venv/bin/python -B validate_mlx.py --reference
.venv/bin/python -B validate_mlx.py --compare
.venv/bin/python -B benchmark_backends.py --device mlx \
  --output ../results/motiongpt/backends --repeats 2
```

測試涵蓋根部旋轉及座標還原、API 型別與邊界、拒絕外站來源、同時生成衝突、超量請求，以及失敗時不回傳假動作。骨架數值與 VRM 重定向測試不等於人類看過動畫後的品質評分；課堂上仍應比較提示與播放結果。

本機相容層與來源節錄位於 `engine.py`、`vendor/`；原始 MIT 聲明及固定來源 commit 保留於 `vendor/LICENSE.MotionGPT` 與 `vendor/README.md`。

MLX T5 參考 [Apple 官方 mlx-examples](https://github.com/ml-explore/mlx-examples/blob/796f5b53cab69a3d48a44233ce21aae889e94a08/t5/t5.py)，保留 MIT 聲明，補齊本模型需要的 padding mask、GELU-new、cross-attention cache。權重轉換和 GPU 引擎分别在 `mlx_convert.py`、`mlx_model.py`、`mlx_engine.py`。

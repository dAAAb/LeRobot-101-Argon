# 兩內夾墊與開口的離線校準

目前提供的是實測資料的生成／驗證工具，**尚未取得這台手臂的合格夾墊量測，也沒有產生或啟用真實校準 profile**。相機棋盤校準不等於夾墊校準；馬達能張合，也不等於知道兩內夾面中心在何處。

工具不連接相機或馬達、不改 `config/robot.json`，也不使用 D435。輸入必須先由實際量測建立；不能把 CAD 座標、兩個影像像素，或已知物件尺寸填成實測夾墊座標。

## 要量什麼

以 `gripper_link` 為座標系，單位公尺，記錄實際馬達讀值 `raw`，以及**同一接近截面**上固定內表面與活動內表面的兩個交點。這是 `matched_contact_cross_section`：不能直接拿兩塊夾墊任意質心的距離當開口。旋轉開合的活動指會改變適合接近的截面，因此每個開度都記錄自己的單位向量 `approach_axis`。

`approach_axis` 指向接近物件的方向；它應與兩接觸交點的連線近乎垂直。工具不會偷偷投影交點或旋轉該方向來消除前後差異。抓取中心取兩交點的中點，開口取兩點間距，僅代表這個截面的空隙，不能證明整條插入路徑有空間。

可接受的方法：

- `mechanical-survey`：用量具或可量測的治具取得三維座標，清楚記錄如何把量具基準對到 `gripper_link`。只量一個開口寬度不足以建立兩個三維點。
- `calibrated-wrist-3d`：使用已獨立驗證的腕部相機校準、可觀測的參考幾何與實際量測。必須指定原校準 fingerprint，執行時提供同一份有效的 `--wrist-profile`。只是移動整隻手臂，不能讓與相機剛性連動的固定夾墊產生相對視差；這種影像不能自行變成三維量測。

每筆量測需保留原始證據及其 SHA256、獨立 acquisition ID，並聲明三維點的最大位置不確定度與方向不確定度。證據可以是原始照片、量測表或三維重建記錄；共同的座標登錄文件另外放在 `registration_evidence`，不能用它代替每次的獨立量測。

## 最小資料集

至少 **3 個訓練開度 + 2 個獨立保留開度**。保留開度須位於訓練開度之間，跨至少兩個插值區間；不可把訓練照片複製或重新命名當保留資料。開口範圍至少跨 5 mm；應涵蓋實際預計張開、接觸及合攏後的範圍。超出範圍的開度不外插。

這是最小數值流程，不是全面精度保證。增加訓練點可描述非線性，但必須另取新的保留量測；不要把驗證失敗的保留點拿去擬合後仍稱它們是獨立驗證。

輸入 JSON 的結構如下；省略號代表必須補上的**實測資料**，這份結構示意不能直接產生 profile：

```json
{
  "schema_version": 1,
  "source": "wrist-only",
  "measurement_origin": "physical-measurement",
  "measurement_method": "mechanical-survey",
  "coordinate_frame": "gripper_link",
  "landmark_definition": "matched_contact_cross_section",
  "units": "m",
  "geometry_revision": "目前模型與馬達校準的64位SHA256",
  "jaw_mount_revision": "這次實際夾指／夾墊安裝版本",
  "registration_description": "具體說明如何量出XYZ與接近方向，以及如何對到gripper_link",
  "registration_evidence": [{"path": "evidence/registration.pdf", "sha256": "檔案SHA256"}],
  "measurements": [{
    "measurement_id": "獨立量測編號",
    "acquisition_id": "獨立取得證據的編號",
    "split": "training",
    "raw": 0,
    "fixed_pad_m": ["實測x", "實測y", "實測z"],
    "moving_pad_m": ["實測x", "實測y", "實測z"],
    "approach_axis": ["實測單位向量x", "y", "z"],
    "position_uncertainty_m": "實測方法支持的正值",
    "axis_uncertainty_deg": "實測方法支持的正值",
    "evidence": [{"path": "evidence/measurement-01.jpg", "sha256": "檔案SHA256"}]
  }]
}
```

`split` 只能是 `training` 或 `heldout`。證據路徑相對於輸入 JSON 所在資料夾，不接受資料夾外的路徑。所有數值欄位實際上都須填數字；示意中的字串與 `raw: 0` 不是預設值。腕部三維法額外提供 `wrist_calibration_fingerprint`。

## 執行與結果

在已安裝專案 Python 依賴的環境執行：

```sh
python scripts/calibrate_jaws.py measurements/jaws.json --output .runtime/jaw-calibration/run-01
```

若採腕部三維量測，加上 `--wrist-profile config/calibration/實際使用的-profile.json`。工具會核對目前本機模型／馬達設定的 geometry fingerprint；純讀取模型，不會開啟串口。

通過時產生 `profile.json`、`report.json`，exit code 為 0。輸出資料夾必須是新的，避免覆蓋先前證據。失敗時只寫 report，exit code 為 2，不產生可啟用 profile。

訓練點定義分段線性插值；保留量測不參與擬合。報告列出兩夾墊、中點、開口、接近方向及閉合方向的殘差。預設驗收包含量測不確定度：兩點與中點各 ≤3 mm、開口 ≤4 mm、方向 ≤3°。位置範圍含訓練和保留量測的不確定度，不把零殘差當零誤差。

`profile_fingerprint` 由生產程式的 `JawCalibration` 計算；輸入量測、模型版本、安裝版本與證據雜湊一起保留。新 profile 相容既有 `jaw_calibration_profile` 載入流程，但工具**不會自動選用**。

檔案雜湊只確認來源未被換掉，不能證明照片中的量測做對了。這是已聲明實測資料的數值驗證，仍需工程審核原始證據與座標登錄，並用獨立實機檢查確認適用範圍。它不證明抓牢、力量、外物避障或實際抓取成功。

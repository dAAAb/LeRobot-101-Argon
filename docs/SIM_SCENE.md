# 工程用 MuJoCo / 3D 場景

`scripts/export_sim_scene.py` 把已保存的腕部觀察紀錄轉成可重現的 3D 快照。輸入包含六軸實測 encoder；工具不開相機、不接串口、不呼叫控制器，也不執行模型推論或馬達指令。

匯出內容：

- `scene.glb`：完整 URDF 視覺網格（含固定爪與活動爪）、白色列印件、深色馬達、模型桌面與腕部相機；可直接由瀏覽器 Three.js GLTFLoader 載入。橘色夾爪全轉動包絡保留為可選疊圖，預設隱藏。
- `scene.xml`、`meshes/`：可搬移的 MuJoCo 模型，`measured_pose` keyframe 保存實測姿勢。沒有 actuator。
- `scene.json`：明確分開實測角度、CAD 推算的 link / camera pose、桌面先驗、物件位置的不確定度與候選視角。
- `verification.json`：MuJoCo FK 與目前 `geometry.Kinematics` 的獨立數值比較，包含每個已解析 link 和相機前向軸。

GLB 座標為公尺、機器人底座框架、**Z 向上**。Three.js 頁面設定 `camera.up.set(0, 0, 1)`；桌面 grid 位於 XY 平面，每格 5 cm。目前頁面可拖曳旋轉、縮放，並以箭頭和連線顯示相機方向及候選觀察位置；這些操作只改工程視圖。沒有自動移動到候選視角的按鈕。MuJoCo 桌面檢視器需保留整份輸出資料夾（含 `meshes/`），載入 `scene.xml` 後選 `measured_pose` keyframe 才是紀錄當時姿勢，不能把 XML 零位當成實測姿勢。

本機工程頁面是 [http://localhost:3100/engineering](http://localhost:3100/engineering)，由唯讀 API 載入 `.runtime/sim-scene/current/` 的場景、腕部 JPEG、深度與核對結果。「重新載入快照」只讀取已匯出的檔案，不取像、不移動手臂。

綠色相機箭頭沿用 CAD 安裝先驗；若深度紀錄含當次影像更新的相機先驗，另外用橘色箭頭顯示，仍不宣稱完成標定。頁面中的腕部照片附上拍攝時間，並標示為非即時影像。若深度資料的照片 ID 或相機 session 不符，頁面隱藏該份距離及點雲。

## 執行

MuJoCo 與 trimesh 是工程選用套件，不加入產品執行依賴。本機已在忽略的 `.runtime/mujoco-venv` 安裝 MuJoCo 3.14.0 / trimesh 5.1.0，並透過該環境的 `.pth` 沿用現有 LeRobot OS Python 數值套件；安裝沒有更動 OS 環境。新環境可另建 venv 並安裝專案原有 requirements 加 `mujoco==3.14.0 trimesh==5.1.0`。

```sh
.runtime/mujoco-venv/bin/python scripts/export_sim_scene.py \
  --frame .runtime/active-depth-current.json \
  --output .runtime/sim-scene/new-snapshot

# 已有符合該張照片的腕部深度估算時，再加入 --depth。
.runtime/mujoco-venv/bin/python scripts/export_sim_scene.py \
  --frame .runtime/active-depth-current.json \
  --depth .runtime/active-depth-result.json \
  --output .runtime/sim-scene/new-snapshot-with-depth

.runtime/mujoco-venv/bin/python -m unittest discover -s tests -p test_sim_scene.py
```

輸出資料夾必須尚不存在；錯誤的匯出不會覆蓋既有場景。工程 API 如需要固定 `/current` URL，應在成功產生新資料夾後切換指向。

頁面先載入 `scene.json`，再把該份紀錄的 `frame.id` 帶在 GLB、JPEG、深度與核對資料的 `?frame_id=...` 請求中。API 若發現 `/current` 已換成其他照片，會回覆 409，頁面提示重新載入，不混用新模型和舊照片。下載連結也攜帶相同 ID。除取得初始 `scene.json` 外，沒有 frame ID 的檔案請求會回覆 400。

每次 API 請求會固定開啟當時的資料夾 inode，再讀取該資料夾內的 manifest 和檔案；即使發佈流程在途中重新命名 `/current` 或切換其符號連結，回應也只會使用同一份資料夾。請持續以完成的整包資料夾替換 `/current`，不要逐檔原地覆寫。API 只提供白名單檔案，拒絕檔案符號連結，並限制 GLB 64 MiB、JPEG 16 MiB、其他檔案 1 MiB。

CLI 會解碼驗證 `frame.path` 的實際 JPEG，核對有提供的影像尺寸，並保存原始位元組為 `wrist.jpg`，不重新壓縮或改拿最新畫面替代。不存在、損毀或尺寸不符的影像會使匯出失敗，不會另行拍照。沒有影像路徑的歷史紀錄仍可匯出純幾何場景。整份場景、影像與選用深度會先在暫存資料夾完成，再一次發佈到輸出位置。

Python 介面為 `build_scene(frame, *, targets=None, candidates=None, project_root=None)` 與 `export_scene(destination, frame, ...)`。前者只建構 JSON，後者另匯出網格和 MJCF 並驗證 FK。`verify_mujoco(scene_path, raw, ...)` 可用任意明確的 encoder 快照重新比較 FK，不會執行 `mj_step`。

若 Python 呼叫端也需要照片和深度，使用 `scripts.export_sim_scene.export_bundle(destination, frame, *, targets=None, candidates=None, depth=None)`；這是 CLI 使用的完整打包介面。單獨呼叫 `server.sim_scene.export_scene` 只匯出模型與場景資訊。

## 物件和深度

`--depth` 接受 `active_depth.estimate_active_depth` 的已保存結果，只允許 `source="wrist-only"`、`current_frame_id` 與 frame ID 相同、`provenance.camera_generation` 與場景相機相同。資料必須是有限數值；深度點最多 64 個。綁定錯誤或不存在的相機來源會拒絕匯出，避免把舊視角的距離顯示在新快照旁。

必須提供估距實際使用的最後一張 frame 紀錄與其影像。即使估距後又拍到姿勢未變的新照片，新的 frame ID 也不能替代原 ID；`stationary_refresh` 是另外一次影像核對，不會讓 CLI 放寬這項綁定。不要手改 ID 來通過檢查；應改用相符的原始 frame，或以新 frame 重算深度。

深度結果的原始可見表面點和 **相機軸向誤差區間** 原樣保存為 `depth.json`，供工程頁面疊圖。軸向估算誤差不等於各方向相同的球形誤差，因此這條流程不會自動生成球形 target、完整圓筒或抓取碰撞體；物件 ROI 與整個場景的未指派表面也保留各自 scope。

預設沒有物件位置。看見白色圓筒、知道它的名稱或曾量過直徑，都不會自動得到它相對手臂的深度。可選 `--targets` 接受最多 16 筆明確證據：

```json
[
  {
    "id": "surface-1",
    "label": "白色圓筒的可見表面",
    "source": "wrist_multiview_estimate",
    "evidence_id": "實際估算紀錄的識別碼",
    "coordinate_frame": "robot_base_model",
    "reference": "visible_surface",
    "position_m": [0.30, 0.02, 0.12],
    "uncertainty_radius_m": 0.025
  }
]
```

以上數字僅說明格式，**不是目前圓筒的量測**。位置省略時保留文字紀錄，不畫 3D 物件。有位置時以點和半透明誤差球顯示，仍然不是完整物件碰撞體；`uncertainty_radius_m` 是輸入證據所聲明的誤差界限，匯出工具不會替它提供正確性保證。

來源可為 `wrist_multiview_estimate`、`manual_measurement` 或 `engineering_observer`。D435 工程資料必須先轉換到機器人底座座標，並提供 `registration_evidence_id`，否則拒絕顯示有位置的點。所有匯出資料都標成 `engineering_only=true` / `production_eligible=false`，不會流入正式辨識或動作判斷。

## 目前驗證與限制

2026-09-24 使用已保存的六軸實測姿勢成功匯出：MuJoCo 與現有 FK 所有已解析 link 的最大位置差約 `1.3e-16 m`。這驗證的是 **URDF 轉換一致性**，不是機器人實際位置有如此精準。原有 `raw=2047` 的 CAD 零位、腕部相機安裝外參仍是先驗。

活動夾爪使用原始 STL 網格，但 CAD 零位與實際開口尚未校準。為了完整呈現機構，工程匯出將實測 gripper encoder 在馬達 min/max 內的行程比例，線性映到 URDF 的角度範圍；這只是外觀先驗，不是實測夾爪角度。raw 已含馬達內 homing offset，不重複套用 offset。`robot.jaw.display_pose` 明列映射、行程界限、示意角度與 `visual_only=true`，`robot.jaw.raw` 則保留實測值；五軸身體的已解析 FK 與示意夾爪分開。

MuJoCo 把此活動爪保存為固定的視覺快照，沒有 gripper joint、actuator 或活動爪碰撞幾何。控制器的完整 360° 夾爪桌面防護完全不變，不能用畫面中的開口取代它。GLB 節點 `moving_jaw_so101_v1_link_0` 是實心活動爪；`jaw_unknown_full_sweep` 是可選不確定範圍，標記 `engineering_role=jaw_uncertainty_envelope`。白色列印件與深色馬達是本機外觀覆寫，不修改原始 URDF、幾何或運動學。

相機外觀及安裝位置仍沿用 CAD 先驗，不代表照片中實際支架已完成建模。相機預覽使用顯示用 45° 垂直視角，並標記 `intrinsics_calibrated=false`。桌面高度沿用控制器的 `z=0` 先驗；0.8 m 顯示寬度不是桌面實際尺寸。

MuJoCo 模型的慣性沿用 URDF，但沒有完成接觸、摩擦、柔順、馬達力矩或抓取成功率校準；網格碰撞的凸化也可能與真實外形不同。因此可以用它檢查姿態、視角與模型轉換，不能把模擬不碰撞直接當成實機通行許可。深度仍需真實影像配對、已知相機位移與不確定度驗證。

官方參考：[MuJoCo 模型與 URDF/MJCF 說明](https://mujoco.readthedocs.io/en/stable/modeling.html)、[Python 與模型檢視器](https://mujoco.readthedocs.io/en/stable/python.html)、[相機座標與 XML 定義](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)。先前 LeRobot OS 已有離線 Three.js / URDF 瀏覽器；本工具沿用同一份 follower URDF，增加獨立 MuJoCo FK 驗證和明確的深度來源紀錄。

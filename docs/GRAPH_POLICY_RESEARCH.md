# Graph-as-Policy：實機感知、空間狀態與技能圖研究

研究日期：2026-09-23。範圍為官方文件及公開原始碼，沒有安裝或執行 GaP、連接硬體、修改產品或使用 D435 資料。使用者提供的 X 連結未能讀取，不據此判定影片細節。

**建議採用有量測來源的空間狀態、明確技能圖及完成證據；不直接移植完整 GaP GPU 執行環境。** GaP 公開實機範例的 3D 位置主要來自已校正 RGB-D 與機器人運動學。把座標寫成 JSON 是傳遞量測結果的方式，本身不會產生深度。

## 實際範例用了什麼

| 層次 | 公開實作 | 對本專案的意義 |
|---|---|---|
| Franka 搬 Jello 盒 | Panda、Robotiq、ZED、DINO/SAM3 GPU 工具及 VLM；robots_realtime bridge 執行動作 | 有深度與另一套控制器，不能視為 SO-101 單 RGB 相機即插即用 |
| UR cable 範例 | UR 關節回讀 + 腕部 ZED；僅觀測，不註冊移動/夾爪工具 | 公開 GaP connector 本身不能完成插線 |
| UR5e USB-C 完整動作 | 另一個 ROS 專案讀取視覺 JSON，再做位姿控制、力控制及插入驗證 | 成功來自感知、專用控制器與回饋整合，不只是語言模型 |

來源：[Franka 官方範例](https://graph-robots.github.io/graph-as-policy/examples/real-franka-pick-place.html)、[UR 官方範例](https://graph-robots.github.io/graph-as-policy/examples/cable-ur.html)、[USB-C 原始碼](https://github.com/TobiasRecker/usb_c_insertion/tree/874c4be425146f6c9a2e969e7a23228e42e3b12f/usb_c_insertion)。官方 connector 文件也只列出 `franka` 與 `ur_zed`，沒有 SO-101 adapter。[Connector 文件](https://graph-robots.github.io/graph-as-policy/real-robots/connectors.html)

## 3D 數值從哪裡來

感知程式先讓 Grounding-DINO 找 2D 候選，由 VLM 區分目標，SAM3 產生 mask；再把 mask、**公尺制深度**、相機內參及 camera-to-world 位姿交給幾何工具，產生 `N×3` 點雲。接著去除雜點並擬合 OBB。VLM 負責「是哪一個」，深度與幾何運算負責「在哪裡、尺寸多少」。目前 canonical skill 還包含候選裁切的兩兩比較、近照驗證及多相機幾何融合；這些不能誤算成只有單張 RGB 的能力。[感知 skill 原始碼](https://github.com/graph-robots/open-robot-skills/blob/752dfbd96968d4d08379f8111765efbf6457800b/skills/perceiving-objects/SKILL.md)、[UR 感知程式](https://github.com/graph-robots/graph-as-policy/blob/c8b5515df80d3029ca5dc7f7f1ec1c327b70bbea/examples/cable_ur/graph/scripts/locate_white_tape/perceive_dino_vlm.py)

實際反投影為 `X=(u-cx)Z/fx`、`Y=(v-cy)Z/fy`，再以 camera-to-world transform 轉成世界座標。`Z` 必須已有來源；此函式不從 RGB 猜深度。UR 的相機位姿由實際關節角、URDF FK 與手眼校正矩陣組合。該實例另有相機安裝方向修正，因此其座標轉換不可直接複製到我們的腕部支架。[反投影實作](https://github.com/graph-robots/open-robot-skills/blob/752dfbd96968d4d08379f8111765efbf6457800b/tools/geometry/_impl.py#L915)、[UR 相機位姿](https://github.com/graph-robots/graph-as-policy/blob/c8b5515df80d3029ca5dc7f7f1ec1c327b70bbea/gap/envs/ur_zed_env.py#L232)

資料型別是普通 TypedDict 與 numpy array，graph 透過 `$ref` 引用欄位，不必每一步讓 LLM 重新敘述全部空間：

| 型別 | 實際欄位與單位 |
|---|---|
| `CameraFrame` | `rgb`、`depth`（m）、`intrinsics`（3×3）、`pose`（camera-to-world） |
| `PointCloud` | `points`（N×3），可附 colors |
| `Se3Pose` | `position:{x,y,z}` + `rotation:{w,x,y,z}` |
| `OrientedBoundingBox` | `center`、`extent`、`orientation`；extent 是 **半尺寸** |
| `JointState` | 關節角 positions（rad），可附 names |

核心型別的 quaternion 為 **wxyz**。UR 報告另輸出 `quaternion_xyzw`，接線時必須明確轉換；不能只看到四個數字就當成同一格式。[型別定義](https://github.com/graph-robots/graph-as-policy/blob/c8b5515df80d3029ca5dc7f7f1ec1c327b70bbea/gap-core/src/gap_core/types.py)

UR 的 report 再對目標點雲做 RANSAC 平面擬合，輸出 `/tmp/white_tape_location.json`：`position`、`camera_frame_position`、`bounding_box`、`plane.normal/d/n_inliers/n_points`、`orientation.quaternion_xyzw/euler_xyz_deg`。這是能被文字模型讀取的**數值化空間摘要**，背後仍保留量測點雲。USB-C 專案的 loader 讀這份結構，並因桌面擺放假設主要使用 yaw，並非通用任意姿態理解。[報告程式](https://github.com/graph-robots/graph-as-policy/blob/c8b5515df80d3029ca5dc7f7f1ec1c327b70bbea/examples/cable_ur/graph/scripts/report_location.py#L111)、[USB-C loader](https://github.com/TobiasRecker/usb_c_insertion/blob/874c4be425146f6c9a2e969e7a23228e42e3b12f/usb_c_insertion/scripts/vision_pose_loader.py#L23)

## 技能圖與「完成」要分開看

Franka workflow 明列 `perceive_container → perceive_target → grasp → transport` 與成功/失敗分支。抓取使用 OBB 產生 top-down pose，依序張爪、到高處、對齊、下降、合爪；此範例使用 direct IK，沒有 motion planner。原始圖中 `grasped` 是合爪節點後的 noop 出口，`placed` 也是放開及退回後的流程出口，不能單靠名字當成實際成功證據。`perceive_target` 的 `on_error: table_clean` 還把感知例外導向清空完成；我們應區分「確認無物件」與「相機/模型/幾何失敗」。[實際 workflow](https://github.com/graph-robots/graph-as-policy/blob/c8b5515df80d3029ca5dc7f7f1ec1c327b70bbea/examples/real_franka_pick_place/graph/workflow.json)

官方文件明說該 Franka 範例的 placement verification script 沒有接進圖。另一套 GaP checkpoints 讀取**模擬器特權 world state**；實機 connector 沒有 `world_snapshot` 時會略過，並不因此提供真機完成保證。API/CLI 預設 `warn` 也不會因 failed checkpoint 中止。[Franka 驗證限制](https://graph-robots.github.io/graph-as-policy/examples/real-franka-pick-place.html#how-the-graph-works)、[Checkpoint 文件](https://graph-robots.github.io/graph-as-policy/running/checkpoints.html#ground-truth-is-sim-only)

USB-C 配套程式則有真正的動作回饋：插入控制器依 tool-frame 接觸力調整速度，並檢查插入深度/力條件；另一個 verifier 用小幅 Y/Z 位移後的反力判斷 retention。這些依賴其力回饋與專用控制器，不能把 SO-101 馬達 load/current 直接視作等價的六軸力矩感測。[插入控制](https://github.com/TobiasRecker/usb_c_insertion/blob/874c4be425146f6c9a2e969e7a23228e42e3b12f/usb_c_insertion/scripts/insertion_controller.py)、[插入後驗證](https://github.com/TobiasRecker/usb_c_insertion/blob/874c4be425146f6c9a2e969e7a23228e42e3b12f/usb_c_insertion/scripts/post_insertion_verifier.py)

另需注意，library 的 `geometry.build_world_config` 可把 RGB-D 做成障礙 mesh/slab，但它的 FK 機身點排除具有 Franka 專用假設；無可重建幾何時會回傳空 world。**缺少障礙模型不能解讀為場景沒有障礙。** [碰撞場景建構原始碼](https://github.com/graph-robots/open-robot-skills/blob/752dfbd96968d4d08379f8111765efbf6457800b/tools/geometry/tools.py#L782)

## 建議移植到 SO-101 的部分

以下是本專案的設計建議，不是已完成的 GaP 整合。

1. **先建立有來源的 state JSON。** 為物件保留穩定 ID、label、各 frame 的 mask/bbox、量測時間、實際 joint ticks、相機內參/手眼校正版本、座標系、metric position/尺寸、誤差、遮蔽及資料來源。未建立公尺尺度時填 `null/unknown`，不要把視覺語言模型信心當成座標精度。保留底層影像與重投影殘差供追查。
2. **腕部單目用主動多視角建立尺度。** 用 FK 與實際回讀關節角估計各視角相機位姿，追蹤同一物件對應點，檢查足夠基線/視差後三角化。移動量應由預期視差及重建不確定性選擇，不是只增加幅度；純旋轉或極小平移通常不能提供可靠深度。先校正並驗證 K、鏡頭畸變、camera-to-wrist 與 FK，再信任座標。遮蔽面仍標未知。
3. **把抓/摸當補充量測。** 用經校正的夾爪開口、位置回差、電流/load 趨勢與再次觀察，增加物件寬度/接觸狀態證據。先分開「疑似接觸」「抓穩」「抬離桌面」三種狀態；未驗證的電流不能直接換算接觸力或推出物體材質。
4. **重用我們的受限 controller，顯式記錄政策圖。** `observe → localize → assess → execute → verify` 每個節點記錄輸入 evidence IDs、選用技能、實際前後狀態、完成條件及路由原因。Astra 決定語意與計畫，Jev 只排序已合法候選；LLM 不取得任意馬達寫入介面。失敗/未知走可見的停止或補觀察分支。
5. **真機 checkpoint 必須有真機證據。** 位移讀 settled encoder/FK；抓取結合夾爪回饋、物件相對夾爪追蹤及抬升後仍持有的影像。graph 走到 done 與物理驗證分開儲存；未實作驗證就顯示未確認，不能套用模擬 oracle 的名稱。
6. **D435 僅供工程評估。** 可離線比較單腕方案估計誤差、觀察碰撞/持物結果；影像、depth、座標、推理摘要都不進終端使用者的 state、Astra/Jev 決策或控制迴路。測試須在 D435 不可用時仍能正常執行產品流程。

第一個可交付里程碑是「單腕 evidence → 帶 unknown/來源的 state → 技能節點 trace → 有實測結果的 receipt」。之後再把經驗證的多視角 metric pose 接到抓取；新增 framework 或生成漂亮 3D 場景都不能替代此量測鏈。

## 查核版本

- 官方文件：[GaP 首頁](https://graph-robots.github.io/graph-as-policy/docs/index.html)。
- `graph-robots/graph-as-policy`：`c8b5515df80d3029ca5dc7f7f1ec1c327b70bbea`。
- `graph-robots/open-robot-skills`：`752dfbd96968d4d08379f8111765efbf6457800b`。
- `TobiasRecker/usb_c_insertion`：`874c4be425146f6c9a2e969e7a23228e42e3b12f`。

版本由公開 GitHub tree API 查核，特定原始碼使用上述 commit 固定連結；GaP 網頁文件與主分支仍可能更新。

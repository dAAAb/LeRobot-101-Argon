# 案例與來源

## Jay Chooi / RoboCurve

[作者測試報告](https://openai.robocurve.org/gpt-6-astra/) 使用雙臂 I2RT YAM、俯視與兩台腕部相機，以及機器人本體狀態。Astra 給末端位姿，再透過 IK 與低速控制器執行。簡單方塊入碗表現與精密插入表現差距很大，不能據此聲稱 SO-101 單腕相機有同樣成功率。

[原始貼文](https://x.com/chooi_jeq/status/2096064315115839904) · [作者說明 EE→IK](https://x.com/chooi_jeq/status/2096064324729233759) · [開源控制評測工具](https://github.com/robocurve/inspect-robots)

## Sam Altman 轉貼的 SO-101 繪圖

[原作者示範](https://x.com/cdngdev/status/2097339677128982873) 是 SO-101 繪製金門大橋。[操作說明](https://x.com/cdngdev/status/2097339677745516710) 包含人給的定位參考、背景監看及逐輪回饋；[SO-101 型號確認](https://x.com/cdngdev/status/2097497391251456122)。這證明 Astra 能結合本機工具控制手臂，不等於接上任意相機就能可靠完成通用抓放。

## Lingxiao 的重建展示

[原始案例](https://x.com/Lingxiao234/status/2096992059731443923) 主要是從 RGB 影像與紀錄動作重建場景／模擬，不是與本專案相同的即時抓取。[作者也說明](https://x.com/Lingxiao234/status/2096992132527702382) 其中物件動力學未完全重現。Blender/Meshy 可用於建模，但不能取代相機與手臂的幾何驗證。

## 本地 URDF

`assets/so101_follower_cam.urdf` 沿用使用者的 `/Users/neo/LeRobot-101-OS` 專案。檔案保留 onshape-to-robot 與 Onshape 來源資訊；目前只使用關節與連桿變換，未複製完整 STL，也不宣稱已完成完整碰撞檢查。若未來公開散布，需先核對上游 CAD／模型授權。

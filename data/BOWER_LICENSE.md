# Bower Waste Annotations 資料與授權

- 資料集：BowerApp/bower-waste-annotations
- 官方頁面：<https://huggingface.co/datasets/BowerApp/bower-waste-annotations>
- 提供者：Bower（Sugi Group AB），與 Google.org 合作建立
- 授權：MIT

官方資料卡說明，資料包含 1,440 張由消費者手機拍攝的垃圾與日常物品圖片，以及經人工複核的 bounding box、材質與物件類型。官方只提供一個 validation split；本專案為模型開發，以穩定的 image ID 雜湊重新切分為 train、validation、test，並在 manifest 保留來源與轉換資訊。

本專案的轉換：

- 所有有效 bounding box 轉成單類 `waste` YOLO 標籤。
- Bounding box 加入少量邊界後裁切，依來源 material 保守映射至 `glass / metal / paper / plastic / unknown`。
- Composite、Hazardous waste、Electronic waste、Organic waste、Textiles、Wood、Rubber & Leather、Other、空白材質等不強行歸入可回收四類，統一映射為 `unknown`。
- 僅在與所有標註框及安全邊界完全不相交時，才取一個框外背景作為無垃圾負樣本；仍需人工抽查。

使用或發布衍生成果時，應標示 Bower（Sugi Group AB）、Google.org、資料集網址與 MIT 授權。資料集不能取代使用者實際手機、房間、桌面與光線的本地獨立測試資料。

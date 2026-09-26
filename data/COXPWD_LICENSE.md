# CoxPWD 2025 資料與授權

- 資料集：DSLR and Smartphone Camera-Based Plastic Waste Detection Dataset from Cox’s Bazar Sea Beach, Bangladesh
- 版本：3
- DOI：<https://doi.org/10.17632/bdzg4tjy63.3>
- 授權：CC BY 4.0
- 官方頁面：<https://data.mendeley.com/datasets/bdzg4tjy63/3>

本專案只在使用者執行 `scripts/import_coxpwd.py` 時下載或匯入資料，不將原始圖片提交至版本控制。匯入程式會將原本 14 種塑膠物件的 bounding box 合併成單一 `waste` 偵測類別，並在 manifest 保留原始類別、來源、版本、授權及檔案雜湊。

公開資料、衍生資料或模型時，需標示原作者、資料集名稱、DOI、CC BY 4.0 授權連結，並說明本專案將 COCO 14 類標註轉成單類 YOLO bounding box。這份資料只涵蓋塑膠，不應聲稱它能代表玻璃、金屬、紙類或一般室內垃圾。

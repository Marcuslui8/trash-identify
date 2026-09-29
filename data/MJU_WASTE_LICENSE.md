# MJU-Waste 來源與授權

- 官方專案：<https://github.com/realwecan/mju-waste>
- 論文：Tao Wang et al., “A Multi-Level Approach to Waste Object Segmentation”, Sensors 2020, 20(14), 3816。
- 官方下載：<https://drive.google.com/file/d/1o101UBJGeeMPpI-DSYI6oh-tLk9AHXMny/view>
- 官方 repository 標示 MIT License；下載的圖片與標註仍應保留原作者與論文引用資訊。

本專案不把 MJU-Waste 圖片放入 Git repository。使用者需自行從官方來源下載並接受其授權條件，再以 `scripts/import_mju_waste.py` 轉換。轉換後所有偵測框統一為單一 `waste` 類別，材質仍由第二階段 CNN 判斷。

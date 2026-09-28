# Mendeley Illegal Dumping Dataset 授權與使用說明

本專案只使用下列資料集的 `Real/Negative` 與 `Real/Critical_negative` 圖片，作為 YOLO training split 的無垃圾負樣本：

- 名稱：An Image Dataset for Illegal Dumping Detection in Urban Areas of Bangladesh
- 作者：Ruhul Amin、Syed Naimul Islam、Sakib Hossain、Al Habib、Sumaiya Tabassum、Mehedi Hasan
- DOI：<https://doi.org/10.17632/rkh7nz2rcp.1>
- 資料頁：<https://data.mendeley.com/datasets/rkh7nz2rcp/1>
- 授權：[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## 專案內的處理方式

- 僅供學校研究與非商業用途，不可直接沿用到商業產品。
- 只匯入原始 `Negative` 與 `Critical_negative`，不使用訓練用擴增副本。
- 逐檔以官方 SHA-256 去除完全重複圖片並驗證下載內容。
- 重新編碼為 JPEG、移除 EXIF，長邊縮至最多 1600 px。
- 全部放入 YOLO `train`，以空 `.txt` 表示沒有垃圾物件；不放入 validation 或 test。
- 來源頁說明列出 288 張 Negative 與 200 張 Critical Clean，但 2026-09-28 公開 API 實際列出 278 與 200 個檔案；其中另有 50 個完全重複檔，因此匯入 428 張唯一圖片。

這些街景負樣本可降低誤報，但拍攝地點與使用者的桌面、房間、垃圾桶及手機鏡頭不同，不能取代本地手機獨立驗收集。

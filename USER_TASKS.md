# 使用者需要準備與執行的工作

這份文件只列出目前無法由程式自行完成、需要由專題人員準備或確認的內容。建議依照順序進行。

## 1. 確認虛擬環境與 GPU

```powershell
cd "D:\master proj\trash-identify"
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

第一個指令應顯示 `.venv\Scripts\python.exe`。這台電腦有 RTX 3050 4GB；如果 CUDA 顯示 `False`，請依 PyTorch 官方 Windows/Pip/CUDA 選擇器安裝 GPU 版 PyTorch：<https://docs.pytorch.org/get-started/locally/>。

安裝專案功能：

```powershell
python -m pip install -e ".[dev,training,detection]"
```

## 2. 訓練目前的 CNN 材質分類器

RealWaste 已匯入並完成切分，可以直接進行分類器 smoke test：

```powershell
python -m waste_system.training.train `
  --annotations data/annotations/realwaste_split.csv `
  --epochs 1 `
  --freeze-epochs 1 `
  --batch-size 16 `
  --device auto
```

確認不會發生缺檔、CUDA out of memory 或資料讀取錯誤後，正式訓練：

```powershell
python -m waste_system.training.train `
  --annotations data/annotations/realwaste_split.csv `
  --epochs 20 `
  --freeze-epochs 5 `
  --batch-size 16 `
  --device auto
```

若 GPU 記憶體不足，將 `--batch-size 16` 改成 `8`。完成後評估及匯出：

```powershell
python -m waste_system.training.evaluate --device auto
python -m waste_system.training.export
```

必須人工檢查 `reports/confusion_matrix.png` 及 `reports/misclassified.csv`。不要只看 Accuracy；重點是 Unknown 被錯判為可回收類別，以及 Glass/Plastic 互相混淆的情況。

## 3. 拍攝 YOLO 多物件資料

RealWaste 只有分類標籤，不能直接訓練多物件偵測。本專案已加入 TACO 匯入工具。2026-09-19 已成功匯入全部 1,500 張實景垃圾圖片及 4,784 個框，可直接作為偵測基礎：

```powershell
python scripts/import_taco.py
python scripts/validate_detection_dataset.py data/detection
python scripts/preview_detection_labels.py data/detection --count 30
```

若日後官方逐圖網址失效，可從 [TACO 官方 Zenodo 封存](https://zenodo.org/records/3354286) 下載並解壓，找到直接包含 `batch_1`、`batch_2` 等資料夾的目錄後執行：

```powershell
python scripts/import_taco.py --local-image-root "D:\path\to\extracted\TACO\data"
python scripts/validate_detection_dataset.py data/detection
```

請人工開啟 `reports/detection_preview/train`，確認至少 30 張框的位置正確。TACO 原始 segmentation metadata 保留在 `data/taco/annotations_source.json`，目前 YOLO 階段只使用 bounding box。

TACO 的拍攝環境與你的手機實際使用場景不完全相同，因此仍需要另外用手機拍攝本地資料：

- 加入 TACO 後，第一輪至少再拍 200～300 張；正式版本建議累積 500 張以上本地圖片。
- 每張包含 0～8 件垃圾，空畫面也要保留一部分作為負樣本。
- 拍攝不同桌面、地板、垃圾桶、室內外及光線。
- 加入遮擋、重疊、遠近大小、旋轉、髒污與壓扁物品。
- 加入手、鞋子、餐具等非垃圾物件，降低誤判。
- 不要全部在相同背景或相同角度拍攝。

避免將含有人臉、門牌、文件或其他個人資訊的影像放入資料集。

### 加入 CoxPWD 手機／戶外塑膠資料（建議）

CoxPWD 2025 v3 有 4,195 張 640×640 圖片與 14 種塑膠垃圾 bounding box，可補強手機、戶外、小物件、遮擋與光線變化。它只能補強塑膠偵測，不能取代玻璃、金屬、紙類與本地負樣本。

以下指令會下載約 382 MB 的官方 ZIP、核對 SHA-256，保留官方 train/valid/test，並把 14 個來源類別轉成目前偵測器使用的單一 `waste` 類別：

```powershell
python scripts/import_coxpwd.py --download
python scripts/validate_detection_dataset.py data/detection
python scripts/preview_detection_labels.py data/detection --count 30
```

若網路中斷，再執行相同的 `--download` 指令會從 `.part` 暫存檔續傳；不要手動刪除該檔案。完整 ZIP 下載後才會開始匯入。

若已自行從 [Mendeley Data](https://data.mendeley.com/datasets/bdzg4tjy63/3) 下載 ZIP：

```powershell
python scripts/import_coxpwd.py --archive "D:\path\to\Plastic Waste Detection on Cox’s Bazar Sea Beach.coco.zip"
```

匯入後必須抽查框的位置，並保留資料集名稱、原作者、DOI、CC BY 4.0 與轉換說明。詳見 `data/COXPWD_LICENSE.md`。不要把 14 個類別硬映射成玻璃／金屬／紙類；它們全都是塑膠物件的外觀子類。

### TACO 授權人工確認

TACO 標註採 CC BY 4.0，但圖片可能採不同公開授權。匯入後請檢查 `data/annotations/taco_import.csv` 的 `source_license`：

- `ODBL (c) OpenLitterMap & Contributors`：保留來源與 ODbL 歸屬。
- `CC`：仍應保留原始 URL 與作者／來源資訊。
- `UNSPECIFIED`：正式散布圖片、模型商業化或發布衍生資料前必須逐一確認，不得自行假定可商用。

詳細說明見 `data/TACO_LICENSE.md`。

## 4. 標註 Bounding Box

使用 CVAT、Label Studio、Roboflow 或其他 YOLO 格式標註工具。偵測類別只設一個：

```text
waste
```

規則：

- 每一件可分離垃圾各畫一個框。
- 框住完整物品，避免留下過多背景。
- 被遮擋但仍可辨識的物品仍要標註。
- 不標註垃圾桶、桌面、手與其他非垃圾物件。
- 完全無法辨識或小到只剩數個像素的物品可以忽略。
- AI 預標註只能節省時間，仍需人工逐框確認。

YOLO 標籤格式：

```text
class_id x_center y_center width height
```

本專案只有 class `0`：

```text
0 0.5125 0.4778 0.3250 0.6111
```

座標與寬高皆為 0～1 的相對值。

## 5. 放置偵測資料

```text
data/detection/
├── images/
│   ├── train/
│   ├── validation/
│   └── test/
└── labels/
    ├── train/
    ├── validation/
    └── test/
```

同一場景連拍、同一件垃圾的不同角度必須放在同一個 split，不能分散到 train 和 test。建議比例為 70%／15%／15%。圖片與標籤必須同名，例如：

```text
images/train/scene_001.jpg
labels/train/scene_001.txt
```

完成後驗證：

```powershell
python scripts/validate_detection_dataset.py data/detection
```

只有顯示 `Detection dataset is valid` 才開始訓練。

TACO 匯入器以整個 `batch_N` 為群組切分，避免同批背景跨 train/test；自行拍攝的連拍照片也應採相同原則。

## 6. 訓練 YOLO 偵測器

本專案使用輕量 nano 模型作為手機／即時推論起點：

```powershell
python -m waste_system.training.train_detector `
  --data configs/detection.yaml `
  --epochs 80 `
  --batch-size 8 `
  --device 0
```

訓練完成後，先查看驗證結果與錯誤圖片，再將最佳權重複製為：

```text
models/waste_detector.pt
```

若日後將專案閉源、商業化或嵌入產品，使用 Ultralytics 前需要重新審查授權：<https://www.ultralytics.com/license>。目前 RealWaste 也採 CC BY-NC-SA 4.0，本專案現階段應視為學術、非商業用途。

## 7. 啟動即時手機掃描介面

CNN 與 YOLO 權重都準備完成後：

```powershell
Copy-Item .env.example .env
python -m uvicorn waste_system.api.main:app `
  --host 0.0.0.0 `
  --port 8000 `
  --env-file .env
```

電腦上開啟：

```text
http://127.0.0.1:8000/scanner
```

按「開啟即時鏡頭」並允許相機權限後，系統會持續辨識並在畫面上顯示 bounding box；按「停止鏡頭」會停止攝影機與推論。若瀏覽器不支援即時鏡頭，仍可使用下方的單張照片選擇功能。

同一個 Wi-Fi 的手機可使用電腦區域網路 IP，例如：

```text
http://192.168.1.20:8000/scanner
```

手機瀏覽器的即時鏡頭通常要求 HTTPS 安全來源。`http://127.0.0.1` 只適用於電腦本機；手機使用區域網路 HTTP 網址時可能只能上傳照片而不能持續開鏡頭。

首次設定時，以電腦目前的 Wi-Fi IPv4 建立區域網路憑證：

```powershell
cd "D:\master proj\trash-identify"
powershell -ExecutionPolicy Bypass -File scripts\setup_local_https.ps1 `
  -IpAddress 10.10.40.2
```

只把 `certs/trash-identify-ca.crt` 傳到自己的手機並安裝／信任，不可分享 `.key` 私鑰。若原本 HTTP API 正在運行，也可由手機開啟 `http://10.10.40.2:8000/local-ca.crt` 下載公開憑證。之後執行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_local_https.ps1
```

手機和電腦保持同一網路，再開啟 `https://10.10.40.2:8443/scanner`。指令中的 IP 必須使用 `ipconfig` 目前顯示的 Wi-Fi IPv4。IP 改變時重新執行設定腳本並重啟服務；只要保留原本的 CA 檔案，手機不必重裝 CA。若校園 Wi-Fi 阻擋裝置互連，改用手機熱點。此 CA 只用於開發，完成測試後可從手機憑證設定移除。

## 完成條件

- `/health` 的 `classifier_available` 與 `detector_available` 都是 `true`。
- CNN 測試集 Macro F1、各類 Recall 與混淆矩陣已人工確認。
- YOLO 已使用獨立 test split 評估 mAP、Precision、Recall。
- 手機能開啟即時鏡頭、持續取得多個 bounding box，也能使用單張照片備援。
- 每個 bounding box 都能取得五分類與回收說明。
- 模糊或過暗照片會要求重新拍攝。

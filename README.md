# 智慧垃圾辨識系統

這是一套可由電腦或手機瀏覽器使用的多物件垃圾辨識系統。目前流程為：

```text
手機鏡頭／上傳照片
        ↓
YOLO 偵測每件垃圾的位置
        ↓
OpenCV 逐框裁切與影像品質檢查
        ↓
EfficientNet-B0 CNN 判斷材質
        ↓
信心門檻與安全決策
        ↓
中文分類結果與回收建議
```

支援五種輸出：

- `glass`：玻璃
- `metal`：鐵鋁
- `paper`：紙類
- `plastic`：塑膠
- `unknown`：無法分類

YOLO 只負責找出垃圾；CNN 負責材質分類。回收建議目前由安全文字模板產生，尚未連接外部 LLM，因此不需要 API Key，也不會把手機照片傳到第三方服務。

## 目前完成狀態

- YOLO 多物件偵測模型：`models/waste_detector.pt`
- EfficientNet-B0 五分類模型：`models/waste_classifier.pt`
- 單張照片與即時手機鏡頭辨識
- 圖片上的 bounding box、中文類別及信心值
- FastAPI 健康檢查、單物件與多物件 API
- 區域網路 HTTPS，讓手機瀏覽器取得相機權限
- RealWaste、TACO 與 CoxPWD 資料匯入工具
- 低信心、模糊、過暗與無法分類保護

尚未完成的部分：

- 真正的外部 LLM／RAG 回收規則查詢
- 金屬、磁性與重量感測器硬體
- 使用者帳號、雲端部署與辨識紀錄資料庫
- CoxPWD 完整下載及重新訓練後的模型比較

## 快速開始：電腦使用

需求：Windows、Python 3.11 以上。

第一次安裝：

```powershell
cd "D:\master proj\trash-identify"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,training,detection]"
Copy-Item .env.example .env
```

一般啟動：

```powershell
cd "D:\master proj\trash-identify"
.\.venv\Scripts\Activate.ps1
python -m uvicorn waste_system.api.main:app `
  --host 127.0.0.1 `
  --port 8000 `
  --env-file .env
```

開啟：

- 掃描頁：<http://127.0.0.1:8000/scanner>
- 健康檢查：<http://127.0.0.1:8000/health>
- API 文件：<http://127.0.0.1:8000/docs>

`/health` 應顯示 `classifier_available`、`detector_available` 與 `multi_object_ready` 全部為 `true`。

## 手機即時鏡頭

手機與電腦必須連接同一個 Wi-Fi。手機瀏覽器通常要求 HTTPS 才允許即時相機，因此不能使用電腦本機的 `127.0.0.1:8000`。

### 1. 查詢電腦目前 IP

```powershell
ipconfig
```

找到 `Wireless LAN adapter Wi-Fi` 下方的 `IPv4 Address`。以下以 `<電腦IPv4>` 表示，請換成實際數字。

### 2. 第一次建立 HTTPS 憑證

```powershell
cd "D:\master proj\trash-identify"
powershell -ExecutionPolicy Bypass -File scripts\setup_local_https.ps1 `
  -IpAddress <電腦IPv4>
```

只將下列公開 CA 憑證傳到自己的手機並安裝／信任：

```text
certs/trash-identify-ca.crt
```

絕對不要分享或上傳 `certs/trash-identify-ca.key` 與 `certs/server.key`。

若需要由手機下載公開 CA，可暫時在專案目錄啟動 HTTP 下載頁：

```powershell
New-Item -ItemType Directory -Force .tmp\ca-download | Out-Null
Copy-Item certs\trash-identify-ca.crt .tmp\ca-download\trash-identify-ca.crt -Force
python -m http.server 8081 --bind 0.0.0.0 --directory .tmp\ca-download
```

手機開啟：

```text
http://<電腦IPv4>:8081/trash-identify-ca.crt
```

安裝完成後可按 `Ctrl+C` 關閉臨時下載頁。

### 3. 啟動手機 HTTPS 掃描服務

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_local_https.ps1
```

保持終端機開啟，手機進入：

```text
https://<電腦IPv4>:8443/scanner
```

允許相機權限後，按「開啟即時鏡頭」。系統會持續辨識，也保留單張照片上傳模式。

如果 IP 改變，只需以新 IP 重新執行 `setup_local_https.ps1` 並重啟服務。只要原本 CA 沒有刪除，手機通常不必重新安裝 CA。

如果手機完全無法連線：

- 確認手機與電腦在相同 Wi-Fi。
- 確認服務顯示 `Uvicorn running on https://0.0.0.0:8443`。
- Windows 防火牆詢問時，允許 Python 在私人網路通訊。
- 校園或公共 Wi-Fi 可能隔離裝置；可改用手機熱點，讓電腦連上熱點後重新取得 IP 與憑證。

## API

### 多物件辨識

主要端點：`POST /v1/predict-multiple`

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/predict-multiple `
  -Form @{ file = Get-Item ".\sample.jpg"; region = "Taiwan" }
```

回應包含每件垃圾的 bounding box、YOLO 偵測信心、CNN 材質結果及回收說明。

### 單物件分類與決策測試

- `POST /v1/predict`：分類已裁切好的單件垃圾圖片，不執行 YOLO。
- `POST /v1/decide`：使用五類機率與可選感測器資料測試決策引擎。

## 資料集

### RealWaste：CNN 材質分類

RealWaste 共匯入 4,752 張圖片：

| 專案類別 | 圖片數 |
|---|---:|
| glass | 420 |
| metal | 790 |
| paper | 961 |
| plastic | 921 |
| unknown | 1,660 |

```powershell
python scripts/import_realwaste.py "C:\Users\ASUS\Downloads\realwaste.zip"
python scripts/split_dataset.py
```

授權與來源見 [data/REALWASTE_LICENSE.md](./data/REALWASTE_LICENSE.md)。RealWaste 採 CC BY-NC-SA 4.0，目前專案應視為學術及非商業用途。

### TACO：YOLO 垃圾定位

TACO 已匯入 1,500 張實景圖片及 4,784 個 bounding box。60 個來源類別統一轉為單一 `waste` 類別，避免用外觀名稱猜測材質。

```powershell
python scripts/import_taco.py
python scripts/validate_detection_dataset.py data/detection
python scripts/preview_detection_labels.py data/detection --count 30
```

授權注意事項見 [data/TACO_LICENSE.md](./data/TACO_LICENSE.md)。

### CoxPWD：塑膠與手機戶外場景補強

CoxPWD 2025 v3 包含 4,195 張 640×640 圖片及 14 種塑膠垃圾 bounding box。匯入器會核對官方 SHA-256、支援中斷續傳，並將所有框轉成單一 `waste` 類別。

```powershell
python scripts/import_coxpwd.py --download
python scripts/validate_detection_dataset.py data/detection
python scripts/preview_detection_labels.py data/detection --count 30
```

若下載中斷，重新執行同一條 `--download` 指令即可從 `.part` 暫存檔續傳。CoxPWD 只能補強塑膠與戶外場景，不能取代玻璃、金屬、紙類、本地室內照片或負樣本。授權與引用見 [data/COXPWD_LICENSE.md](./data/COXPWD_LICENSE.md)。

## 模型訓練

### CNN 五分類

```powershell
python -m waste_system.training.train `
  --annotations data/annotations/realwaste_split.csv `
  --epochs 20 `
  --freeze-epochs 5 `
  --batch-size 16 `
  --device auto

python -m waste_system.training.evaluate --device auto
python -m waste_system.training.export
```

輸出模型為 `models/waste_classifier.pt`。

### YOLO 垃圾偵測

```powershell
python -m waste_system.training.train_detector `
  --data configs/detection.yaml `
  --epochs 80 `
  --batch-size 8 `
  --device 0
```

訓練完成後，將最佳權重複製為 `models/waste_detector.pt`。訓練前必須驗證資料並人工抽查 bounding box；同一物品或連拍場景不可分散到不同 split。

## 測試

```powershell
python -m pytest
python scripts/validate_detection_dataset.py data/detection
```

CNN 評估不能只看 Accuracy，還要檢查 Macro F1、各類 Recall、混淆矩陣，以及 `unknown` 被誤判成可回收物的比例。

## 專案結構

```text
trash-identify/
├── configs/                 # 系統與 YOLO 設定
├── data/                    # 標註、偵測資料與授權說明
├── models/                  # 正式 CNN 與 YOLO 權重
├── scripts/                 # 資料匯入、檢查與 HTTPS 工具
├── src/waste_system/
│   ├── api/                 # FastAPI 與手機掃描頁
│   ├── inference/           # OpenCV、YOLO、CNN、決策流程
│   ├── llm/                 # 目前為安全回收建議模板
│   └── training/            # CNN 與 YOLO 訓練、評估及匯出
├── tests/
├── USER_TASKS.md
├── update.md
└── README.md
```

## 已知限制

- 目前 YOLO 只判斷垃圾位置，材質由第二階段 CNN 判斷。
- 透明玻璃與透明塑膠只靠 RGB 圖片仍容易混淆。
- 鐵與鋁不一定能由外觀穩定區分；未來可加入磁性感測器。
- 紙杯、鋁箔包等複合材質應保守歸為 `unknown`。
- 公開資料集和實際手機場景有差異，仍需補拍本地圖片與負樣本。
- 自建 CA 只適合區域網路開發；公開部署應使用正式網域、受信任 HTTPS、登入與存取限制。
- 若未來商業化，需要重新審查資料集與 Ultralytics 授權。

需要人工完成的拍攝、標註、訓練及驗收工作集中在 [USER_TASKS.md](./USER_TASKS.md)，每次專案變更記錄在 [update.md](./update.md)。

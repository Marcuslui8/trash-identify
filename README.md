# 智慧垃圾辨識系統

版本：`0.8.0`

本系統可由電腦或同一區域網路內的手機瀏覽器使用，支援單張照片及即時鏡頭的多物件垃圾辨識。

```text
手機鏡頭／上傳照片
        ↓
YOLO26s 偵測垃圾位置
        ↓
OpenCV 逐框裁切與影像品質檢查
        ↓
EfficientNet-B0 CNN 判斷材質
        ↓
信心門檻與安全決策
        ↓
中文分類結果與回收建議
```

輸出類別：

- `glass`：玻璃
- `metal`：鐵鋁
- `paper`：紙類
- `plastic`：塑膠
- `unknown`：無法分類

YOLO 負責找出每件垃圾的位置，CNN 再判斷材質。回收說明由本機安全文字模板產生，不需要 API Key，照片也不會傳送給外部 LLM 服務。

## 系統現況

- 正式偵測模型：`models/waste_detector.pt`（YOLO26s，640 px）
- 正式分類模型：`models/waste_classifier.pt`（EfficientNet-B0）
- 支援照片上傳、手機即時後鏡頭、多個 bounding box 與中文結果
- 提供 `/health`、單物件及多物件辨識 API
- 提供區域網路 HTTPS 與本機手機資料蒐集頁
- 已整合 RealWaste、TACO、CoxPWD、Bower 與 Mendeley 資料
- 具備模糊、過暗、低信心、重複照片及無法分類保護

正式 YOLO26s 在獨立 test split 的結果：

| 指標 | 結果 |
|---|---:|
| Precision | 0.781 |
| Recall | 0.586 |
| mAP50 | 0.662 |
| mAP50-95 | 0.445 |
| RTX 3050 平均推論 | 約 9.48 ms／張 |

## 安裝

需求：Windows、Python 3.11 以上。

```powershell
cd "D:\master proj\trash-identify"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,training,detection]"
Copy-Item .env.example .env
```

## 電腦啟動

```powershell
cd "D:\master proj\trash-identify"
.\.venv\Scripts\Activate.ps1
python -m uvicorn waste_system.api.main:app `
  --host 127.0.0.1 `
  --port 8000 `
  --env-file .env
```

- 掃描頁：<http://127.0.0.1:8000/scanner>
- 健康檢查：<http://127.0.0.1:8000/health>
- API 文件：<http://127.0.0.1:8000/docs>

`/health` 的 `classifier_available`、`detector_available` 與 `multi_object_ready` 應全部為 `true`。

## 手機即時鏡頭

手機和電腦必須連接同一個 Wi-Fi。手機瀏覽器需要可信任的 HTTPS 才能持續使用相機。

### 第一次設定，或電腦 IP 改變時

用 `ipconfig` 查出 Wi-Fi 的 IPv4，以下以 `<電腦IPv4>` 表示：

```powershell
cd "D:\master proj\trash-identify"
powershell -ExecutionPolicy Bypass -File scripts\setup_local_https.ps1 `
  -IpAddress <電腦IPv4>
```

只把公開憑證 `certs/trash-identify-ca.crt` 傳到自己的手機並安裝／信任。不可分享或上傳 `certs/trash-identify-ca.key`、`certs/server.key`。

若手機需要直接下載公開 CA，可暫時執行：

```powershell
New-Item -ItemType Directory -Force .tmp\ca-download | Out-Null
Copy-Item certs\trash-identify-ca.crt .tmp\ca-download\trash-identify-ca.crt -Force
python -m http.server 8081 --bind 0.0.0.0 --directory .tmp\ca-download
```

手機開啟 `http://<電腦IPv4>:8081/trash-identify-ca.crt`，安裝完成後按 `Ctrl+C` 關閉臨時下載頁。

### 啟動 HTTPS 服務

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_local_https.ps1
```

腳本會顯示目前可用網址。若本系統已經佔用 8443，腳本會回報服務已運行，不會重複啟動。

- 手機掃描：`https://<電腦IPv4>:8443/scanner`
- 手機蒐集：`https://<電腦IPv4>:8443/collector`
- 健康檢查：`https://<電腦IPv4>:8443/health`

保持終端機開啟，在手機允許相機權限後按「開啟即時鏡頭」。若校園或公共 Wi-Fi 隔離裝置，可讓電腦連接手機熱點，再以新的 IPv4 重新產生伺服器憑證。

## 手機資料蒐集器

`/collector` 支援單張拍攝及每 3 秒連拍，資料只保存在 `data/local_capture/`。系統會：

- 移除 EXIF，將長邊限制為 1600 px
- 檢查模糊、曝光、完全重複及視覺過度相似照片
- 保存 session、類型、品質及標註狀態 metadata
- 將本地照片排除於 Git 追蹤之外

蒐集類型包括 `positive`（有垃圾）、`negative`（無垃圾）及 `hard`（困難案例）。`positive` 與 `hard` 必須具備人工確認的 bounding box 才能作為 YOLO 正樣本；`negative` 使用空標籤。

## API

主要端點：

- `POST /v1/predict-multiple`：YOLO 多物件定位、CNN 材質分類及回收說明
- `POST /v1/predict`：分類已裁切的單件物品，不執行 YOLO
- `POST /v1/decide`：用五類機率及可選感測資料測試決策引擎

範例：

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/predict-multiple `
  -Form @{ file = Get-Item ".\sample.jpg"; region = "Taiwan" }
```

## 資料集

| 資料來源 | 本專案用途 | 已匯入內容 |
|---|---|---:|
| RealWaste | CNN 五分類 | 4,752 張 |
| TACO | 實景垃圾定位 | 1,500 張、4,784 框 |
| CoxPWD | 戶外塑膠與小物件補強 | 4,195 張 |
| Bower | 手機多材質與安全背景裁切 | 1,578 張、2,546 框／裁切、129 張負樣本 |
| Mendeley Illegal Dumping | 乾淨與困難街景負樣本 | 去重後 428 張 |

偵測資料合計 7,830 張圖片、14,201 個框；train / validation / test 分別為 6,088 / 864 / 878 張。YOLO 的來源類別統一轉為單一 `waste`，材質由 CNN 判斷。

來源、轉換方式及授權記錄：

- [RealWaste](./data/REALWASTE_LICENSE.md)
- [TACO](./data/TACO_LICENSE.md)
- [CoxPWD](./data/COXPWD_LICENSE.md)
- [Bower](./data/BOWER_LICENSE.md)
- [Mendeley Illegal Dumping](./data/MENDELEY_ILLEGAL_DUMPING_LICENSE.md)

RealWaste 與 Mendeley 含非商業授權限制，因此目前專案定位為學術、非商業研究。

## 訓練與評估

### CNN 五分類

```powershell
python -m waste_system.training.train `
  --annotations data/annotations/combined_classification.csv `
  --output models/checkpoints_bower `
  --reports reports/cnn_bower `
  --epochs 20 `
  --freeze-epochs 5 `
  --batch-size 16 `
  --device auto

python -m waste_system.training.evaluate `
  --checkpoint models/checkpoints_bower/efficientnet_b0_best.pth `
  --annotations data/annotations/combined_classification.csv `
  --reports reports/cnn_bower_test `
  --device auto

python -m waste_system.training.export `
  --checkpoint models/checkpoints_bower/efficientnet_b0_best.pth `
  --output models/waste_classifier.pt
```

### YOLO26s 偵測

```powershell
python scripts/validate_detection_dataset.py data/detection

python -m waste_system.training.train_detector `
  --data configs/detection.yaml `
  --base-model yolo26s.pt `
  --epochs 60 `
  --image-size 640 `
  --batch-size 8 `
  --device 0 `
  --name phone_s640

python scripts/evaluate_detector.py `
  --weights runs/detection/phone_s640/weights/best.pt `
  --split test `
  --name phone_s640_test
```

中斷後可由完整 epoch 接續：

```powershell
python -m waste_system.training.train_detector `
  --resume-from runs/detection/phone_s640/weights/last.pt
```

手機只負責拍攝與傳送影格，模型實際在區域網路內的電腦推論；YOLO26s 是目前速度與偵測能力較平衡的正式模型。

## 測試

```powershell
python -m pytest
python scripts/validate_detection_dataset.py data/detection
```

## 專案結構

```text
trash-identify/
├── configs/                   # 系統與 YOLO 設定
├── data/                      # 標註、偵測資料及授權說明
├── models/                    # 正式 CNN 與 YOLO 權重
├── runs/detection/phone_s640/ # 正式 YOLO 訓練來源
├── scripts/                   # 資料匯入、驗證、評估及 HTTPS 工具
├── src/waste_system/
│   ├── api/                   # FastAPI、掃描頁及蒐集頁
│   ├── inference/             # OpenCV、YOLO、CNN 與決策流程
│   ├── llm/                   # 本機安全回收說明模板
│   └── training/              # CNN 與 YOLO 訓練、評估及匯出
├── tests/
├── update.md                  # 版本更新紀錄
└── README.md
```

## 使用限制

- 透明玻璃與透明塑膠只靠 RGB 圖片可能混淆。
- 鐵與鋁無法只靠外觀穩定區分，因此共同輸出為 `metal`。
- 紙杯、鋁箔包等複合材質會保守輸出 `unknown`。
- 自建 CA 僅供可信任的區域網路研究環境使用。
- 公開資料與 Ultralytics 套件各有授權條件；目前系統限學術、非商業用途。

每次變更記錄見 [update.md](./update.md)。

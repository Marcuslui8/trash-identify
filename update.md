# 更新紀錄

本檔案記錄專案每次更新。新紀錄請加在最上方，日期使用 `YYYY-MM-DD`。

## 2026-09-28 — v0.8.0 系統與文件整理

### 更新

- README 改為目前可直接使用的系統說明，整合架構、安裝、手機 HTTPS、API、資料集、模型指標、訓練、測試與限制。
- README 移除未完成事項、未來規劃、已失效操作與重複內容，不再作為待辦清單。
- 專案版本更新為 `0.8.0`。
- 清除 `.tmp`、`.pytest_cache`、圖表輸出與 Codex 建置暫存目錄；這些內容皆可由系統重新產生。

### 保留

- 正式部署的 `models/waste_detector.pt`、`models/waste_classifier.pt`、YOLO26s 基礎權重及正式 `phone_s640` 訓練紀錄均保留。
- 資料集、授權文件、匯入清單、測試、報告及手機蒐集功能均保留。

## 2026-09-28 — v0.7.1 公開乾淨街景負樣本

### 新增

- 核對 Bangladesh Illegal Dumping Dataset，確認授權為 CC BY-NC 4.0，符合目前學校研究的非商業用途。
- 新增 `scripts/import_mendeley_negatives.py`，只匯入 `Real/Negative` 與 `Real/Critical_negative`，不使用資料增強副本或沒有 bounding box 的正樣本。
- 匯入器支援官方 SHA-256 驗證、完全重複檔去除、失敗重試、EXIF 移除、長邊 1600 px 縮圖、空 YOLO label 與來源 manifest。
- 新增 `data/MENDELEY_ILLEGAL_DUMPING_LICENSE.md`，記錄作者、DOI、非商業限制、實際檔案數與專案處理方式。

### 結果與決策

- 公開 API 實際列出 478 個負樣本檔；去除 50 個完全重複檔後，成功匯入 428 張唯一圖片，其中一般乾淨街景 230 張、困難負樣本 198 張，保存後約 192.5 MB。
- 所有新圖片只加入 train；validation 與 test 保持不變，避免相同來源影響獨立評估。
- 合併偵測資料更新為 7,830 張圖片、14,201 個框；train / validation / test 為 6,088 / 864 / 878 張。
- 完整資料驗證通過。網路街景可補強誤報，但不能冒充本地手機資料，也不能取代實際部署場景驗收。
- 抽查並拒絕 `keremberke/garbage-object-detection`：其大量框覆蓋全圖或過小，且多為白底／圖庫素材，不加入正式資料。
- 訓練入口新增可重現的 optimizer、初始學習率、warm-up epochs 與 warm-up bias learning rate 參數。
- 以舊 `phone_s640/best.pt` 嘗試保守微調；第 1 epoch 為 P 0.795、R 0.640、mAP50 0.725，第 2 epoch 為 P 0.826、R 0.617、mAP50 0.714，低於原驗證 mAP50 約 0.749。候選模型因此拒絕，未覆蓋正式權重；後續應從 YOLO26s 基礎權重完整重訓並做來源平衡。
- 發現 `models/waste_detector.pt` 仍是 9/19 的舊 YOLO26n。先備份為 `models/waste_detector_yolo26n_legacy.pt`，再部署已通過獨立 test 的 `phone_s640/best.pt`。YOLO26s test 為 P 0.781、R 0.586、mAP50 0.662、mAP50-95 0.445，平均推論約 9.48 ms／張；網站服務需重啟才會載入新權重。
- `start_local_https.ps1` 現在會先檢查 8443；若健康的本專案服務已在背景執行，會顯示網址與健康狀態後正常結束，不再讓 Uvicorn 拋出 WinError 10048。若連接埠由其他程式佔用，則提供明確的 `netstat` 診斷指令。

## 2026-09-28 — v0.7.0 手機本地資料蒐集器

### 新增

- 新增 `/collector` 手機蒐集頁，支援後鏡頭、單張拍攝與每 3 秒自動連拍。
- 設定 500 張引導配額：有垃圾 350、無垃圾 100、困難案例 50，並即時顯示進度。
- 新增 `/v1/collection/images` 與 `/v1/collection/status`，照片與逐張 JSON metadata 只保存在本機 `data/local_capture/`。
- 每張照片記錄 session、類型、品質分數、亮度、模糊度、雜湊、備註及 bounding-box 標註狀態。

### 安全與品質

- 圖片重新編碼以移除 EXIF，長邊限制 1600 px，單檔限制 12 MB。
- Session 名稱嚴格限制安全字元，避免路徑越界。
- 拒絕完全相同與同 session 中視覺過度相似的照片；一般樣本若嚴重模糊／曝光則要求重拍，困難案例可保留。
- 本地蒐集照片加入 `.gitignore`，避免私人影像誤傳 GitHub。
- 新增 4 項蒐集器測試；全專案 24 項測試通過。

### 限制

- 程式可處理拍攝、品質、去重、配額與保存，但實際本地場景仍必須由使用者拿手機拍攝。
- 有垃圾與困難案例必須完成人工 bounding box 確認後才能加入 YOLO，避免把模型漏檢錯誤當成背景。

## 2026-09-27 — 手機資料擴充與 YOLO26s 重訓

### 新增

- 整合 Bower 公開手機垃圾資料：1,578 張唯一手機照片、2,546 個人工複核框／材質裁切。
- 從不與任何垃圾框重疊且保留安全邊距的區域產生 129 張背景負樣本，補強空桌面、地面等無垃圾畫面。
- 新增 `scripts/import_bower.py`、來源與 MIT 授權說明、五分類合併索引及匯入單元測試。
- 新增 `scripts/evaluate_detector.py`，固定以獨立 test split 輸出 Precision、Recall、mAP 與每張推論時間。

### 清理與修正

- 偵測訓練資料移除 Bower 物件內部零件的巢狀重複框；CNN 仍保留各材質裁切。
- 修正 YOLO 訓練入口的 `--project` 預設型別，避免相對輸出目錄錯誤。
- 移除 CoxPWD 一筆完全重複的 YOLO 標籤。
- YOLO 訓練入口新增 `--resume-from`，額度、關機或終端中斷後可從 `last.pt` 接續。
- 合併後偵測資料為 7,402 張圖片、14,201 個框；train / validation / test 為 5,660 / 864 / 878 張。

### 模型選擇

- 手機目前只負責拍攝與傳送，RTX 3050 電腦負責推論，因此選用 YOLO26s 而非 nano；1 epoch 壓力測試顯示約使用 1.4 GB VRAM，驗證推論約 5.9 ms／張。
- 正式訓練採 640 px、batch 8、最多 60 epochs 與 early stopping；最佳權重須通過獨立 test 評估後才替換正式模型。

## 2026-09-27 — 偵測能力改善規劃

### 分析

- 現有 YOLO 結果為 Precision 0.6584、Recall 0.4637、mAP50 0.5124、mAP50-95 0.3713；漏檢是目前主要瓶頸。
- CNN 公開測試 Accuracy 0.9054、Macro F1 0.8974，優於偵測器，但仍需以手機實拍裁切圖驗證 domain shift。
- 現有正式 YOLO 權重早於 CoxPWD 匯入，尚未使用目前共 5,695 張圖片及 12,087 個框重新訓練。
- CoxPWD 占訓練圖片約 75.5% 且全為戶外塑膠，必須補充本地室內、多材質與負樣本，並按資料來源分開評估。

### 新增

- 新增 `DETECTION_IMPROVEMENT_PLAN.md`，記錄本地測試集、重新訓練、手機資料蒐集、模型／解析度實驗、門檻校準、手機增強、追蹤穩定化及 CNN 微調方案。
- 設定第一階段量化驗收目標，避免只用主觀畫面或單一混合 mAP 判斷改善。

## 2026-09-26 — GitHub 提交整理

### 修改

- 補強 `.gitignore`，排除簡報建置暫存、`node_modules`、Office 鎖定檔、訓練 checkpoint、YOLO 基礎權重及資料快取。
- 保留 `deliverables` 中的正式評審簡報，可與程式及文件一起提交。
- 模型正式權重、資料集圖片、HTTPS 私鑰、虛擬環境及本機 `.env` 仍不會提交至 GitHub。

## 2026-09-24 — v0.6.0 README 現況整理

### 修改

- 重新編寫 README，使安裝、電腦啟動、手機 HTTPS、CA 安裝、API、資料匯入、訓練與測試流程符合目前程式。
- 刪除「未來才加入 YOLO」、模型尚未準備、僅支援單張上傳、舊固定 IP、Streamlit／React 等已過時或未實作的描述。
- 明確區分目前已完成、尚未完成與已知限制；回收建議目前使用安全模板，尚未連接外部 LLM。
- 手機網址改用 `<電腦IPv4>`，避免 IP 變動後照抄失效網址。
- 專案版本由 `0.5.1` 同步更新為 `0.6.0`。

### 驗證

- README 中的檔案、模型、腳本、API 路徑與目前專案結構逐項核對。

## 2026-09-22 — v0.6.0 CoxPWD 資料匯入流程

### 新增

- 確認 CoxPWD 2025 v3 官方資料：4,195 張 640×640 影像、14 種塑膠垃圾 COCO bounding box，授權為 CC BY 4.0。
- 新增 `scripts/import_coxpwd.py`，可下載約 382 MB 官方 ZIP、核對固定 SHA-256、保留官方 train/valid/test，並轉換為單類 `waste` YOLO 標籤。
- 官方 ZIP 支援 `.part` 中斷續傳，適合 Mendeley 下載速度較慢或連線不穩時分次執行。
- 匯入 manifest 保留來源 14 類統計、原始檔名、來源 split、CC BY 4.0、官方頁面與每張輸出圖片 SHA-256。
- 新增 `data/COXPWD_LICENSE.md` 與匯入器單元測試。

### 決策

- CoxPWD 與 TACO 混合用於垃圾位置偵測，不將 14 個塑膠外觀類別錯誤映射成五種材質。
- CNN 繼續負責 `glass / metal / paper / plastic / unknown`，避免塑膠資料量壓倒其他材質。
- CoxPWD 只能補強塑膠與戶外手機場景，仍需本地玻璃、金屬、紙類、室內場景及負樣本。

## 2026-09-22 — v0.5.1 即時鏡頭與區域網路 HTTPS

### 新增

- `/scanner` 加入瀏覽器即時鏡頭模式，預設要求後鏡頭並持續呼叫多物件辨識 API。
- 將即時影格長邊限制為 960 px、JPEG quality 0.76，降低手機網路與伺服器推論負擔。
- 同時間最多一個辨識請求，完成後間隔 450 ms 再取下一張，避免請求堆積。
- 以類別與 bounding-box IoU 進行兩個連續影格的穩定判定，降低辨識框閃爍。
- 在鏡頭畫面疊加中文類別、分類信心、偵測信心及回收說明。
- 頁面進入背景時暫停取樣；停止鏡頭時關閉 media track 並取消進行中的 HTTP 請求。
- 保留原本的單張照片辨識作為備援。
- 加入 `scripts/setup_local_https.ps1`，以本機 CA 簽發包含區域網路 IPv4、localhost 與 127.0.0.1 SAN 的伺服器憑證。
- 加入 `scripts/start_local_https.ps1`，以 Uvicorn TLS 在 `8443` 提供手機即時鏡頭頁面。
- 憑證與私鑰置於被 Git 忽略的 `certs/`；只將 CA 公開憑證安裝到自己的手機，影像不經第三方 Tunnel。
- 新增 `/local-ca.crt` 下載端點，只提供可公開的 CA 憑證，方便手機安裝；CA 與伺服器私鑰不會由 API 提供。

### 修正

- 修正 `/v1/predict` 建立分類決策後沒有回傳結果的程式流程。

### 驗證

- 掃描頁 JavaScript 語法檢查通過。
- API、決策引擎、CA 下載、資料匯入與多物件共 13 項測試全部通過。
- 使用正式 `waste_detector.pt` 與 `waste_classifier.pt` 完成端到端測試；API 成功在測試影像回傳 1 個物件、YOLO 偵測信心 0.826。

## 2026-09-19 — v0.4.0 TACO 多物件資料整合

### 新增

- 檢查 TACO 官方正式資料：1,500 張影像、4,784 個 COCO box/segmentation、60 個原始類別。
- 新增 `scripts/import_taco.py`，安全下載正式圖片並轉換為單類 `waste` YOLO 標籤。
- 以 `batch_N` 為群組進行 train/validation/test 切分，降低相同拍攝背景跨集合洩漏。
- 保留原始 COCO metadata、來源 URL、原始類別、圖片授權值及 SHA-256。
- 新增 `scripts/preview_detection_labels.py` 供人工抽查 bounding box。
- 新增 `data/TACO_LICENSE.md`，區分 MIT 程式碼、CC BY 4.0 標註與圖片個別授權。
- 更新 `USER_TASKS.md`，將 TACO 與手機本地資料的準備工作整合。
- 修正 YOLO 相對 `project` 路徑被 Ultralytics 再套入 `runs/detect` 的問題，後續訓練會直接輸出至 `runs/detection/<name>`。

### 決策

- TACO 用於訓練單類 `waste` 偵測器，不直接轉成五種材質。
- RealWaste 繼續訓練 CNN 材質分類器。
- TACO 的 segmentation metadata 保留供未來 instance segmentation 使用，第一版只轉 bounding box。

### 已知問題

- 正式 metadata 中 715 張圖片的 `license` 未指定；發布或商業化前需要人工確認。
- TACO 主要是戶外 litter，仍需要手機實際場景資料做 domain adaptation。
- 匯入時有 152 張手機照片的原始像素方向與 EXIF 顯示方向不同；匯入器現會先依 EXIF 自動轉正，再套用 COCO 標註。
- 若日後逐圖網址失效，支援從官方 Zenodo 封存解壓後以 `--local-image-root` 匯入。

### 驗證

- TACO 資料驗證：1,500 張圖片、4,784 個框；train/validation/test 分別為 1,090/198/212 張，所有圖片與 YOLO 標籤配對及座標皆有效。
- 人工抽查 train/validation/test 預覽，bounding box 正確；預覽工具會先縮圖，避免超大圖片顯示失真。
- 單元與 API 測試：11 項全部通過，包含 EXIF 轉正回歸測試。

## 2026-09-18 — v0.3.0 多物件與手機掃描架構

### 新增

- 新增可選的 Ultralytics YOLO 單類 `waste` 偵測介面。
- 新增 YOLO bounding box → OpenCV 裁切 → CNN 五分類的多物件管線。
- 新增 `/v1/predict-multiple` 多物件 API 與 `/scanner` 手機拍照頁面。
- 新增 EfficientNet-B0 CSV Dataset、加權 Loss、兩階段微調與 Early Stopping。
- 新增測試集分類報告、混淆矩陣、錯誤案例 CSV 與 TorchScript 匯出程式。
- 新增 YOLO 資料格式設定、資料驗證及訓練入口。
- 新增 `USER_TASKS.md`，記錄使用者需要拍攝、標註、訓練與驗收的工作。

### 保留

- 保留單物件 CNN API，作為分類器獨立測試及固定掃描台使用。
- 保留 RealWaste 原始分類資料與切分 CSV，用於 CNN 材質分類。

### 尚未完成

- 尚未取得多物件 bounding box 標註資料。
- 尚未訓練 `waste_classifier.pt` 與 `waste_detector.pt`。
- 手機頁面已完成程式骨架，但必須在兩個模型權重完成後才能正式辨識。

### 驗證

- 專案 `.venv`：Python 3.13.2、PyTorch 2.14.0+cu126、CUDA 可用。
- 單元與 API 測試：8 項全部通過。
- CNN 資料讀取：成功讀取 3,323 張訓練圖片及加權類別權重。
- GPU smoke test：1 epoch 成功，validation Macro F1 為 0.7572。
- 暫時 test 結果：Accuracy 0.7622、Macro F1 0.7462、平均推論 13.57 ms/張。
- 暫時模型已成功匯出並完成載入檢查；smoke checkpoint、報告及模型在驗證後清除，避免被誤認為正式模型。

## 2026-09-18 — v0.2.0 匯入 RealWaste 資料集

### 新增

- 新增可重複執行的 `scripts/import_realwaste.py` 匯入工具。
- 建立 RealWaste 九類到專案五類的明確映射。
- 匯入時檢查圖片解碼、尺寸、SHA-256 及重複內容。
- 新增 RealWaste 來源、論文引用與 CC BY-NC-SA 4.0 授權說明。
- 產生 `data/annotations/realwaste.csv` 資料索引。

### 類別映射

- Glass → glass
- Metal → metal
- Paper、Cardboard → paper
- Plastic → plastic
- Food Organics、Miscellaneous Trash、Textile Trash、Vegetation → unknown

### 驗證

- 成功匯入 4,752 張 JPG。
- 損壞或無法解碼圖片：0 張。
- 完全相同的 SHA-256 重複群組：0 組。
- 五分類數量：glass 420、metal 790、paper 961、plastic 921、unknown 1,660。

### 後續工作

- 根據 CSV 建立 train/validation/test 切分。
- 處理類別不平衡，避免 unknown 類別主導訓練。
- 建立 EfficientNet-B0 第一版訓練與評估程式。

## 2026-09-18 — v0.1.0 專案初始化

### 新增

- 建立 OpenCV、CNN、決策引擎與 LLM 分層架構。
- 建立 FastAPI 健康檢查、決策測試與影像上傳端點。
- 建立影像品質檢查、TorchScript 分類器及多角度融合介面。
- 建立低信心、前兩名差距、影像品質及金屬感測器衝突規則。
- 建立安全的模板式回收說明，作為未串接外部 LLM 前的替代方案。
- 建立集中式 JSON 設定、環境變數範例與 Python 專案設定。
- 建立決策引擎及 API 基礎測試。
- 完成系統介紹、架構、安裝、API 操作與後續優化文件。

### 目前限制

- 尚未建立實際垃圾影像資料集。
- 尚未訓練或放入 CNN 權重；影像推論端點在缺少模型時會回傳 `503`。
- 尚未串接實體攝影機、感測器、資料庫、前端與外部 LLM。
- 設定中的門檻是初始值，必須透過驗證集校準。

### 驗證

- `python -m pytest`：5 項決策及 API 測試全部通過。
- `python -m compileall src`：Python 語法檢查通過。

### 下一步

- 蒐集並標註五類垃圾影像。
- 建立自製 CNN 與遷移學習訓練管線。
- 產生第一版 TorchScript 模型並進行離線評估。

---

## 紀錄模板

```markdown
## YYYY-MM-DD — vX.Y.Z 更新名稱

### 新增
- ...

### 修改
- ...

### 修正
- ...

### 驗證
- ...

### 後續工作
- ...
```

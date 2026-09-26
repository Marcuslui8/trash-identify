# TACO 資料來源、引用與授權注意事項

本專案使用 TACO（Trash Annotations in Context）的正式標註資料，來源：

- 官方程式庫：<https://github.com/pedropro/TACO>
- 官方網站：<https://tacodataset.org/>
- 論文：<https://arxiv.org/abs/2003.06975>
- 本次檢查的官方 Git commit：`29de1a9ba05a647b83a90f18d7772e20bb23d846`

建議引用：

```bibtex
@article{taco2020,
  title={TACO: Trash Annotations in Context for Litter Detection},
  author={Pedro F Proença and Pedro Simões},
  journal={arXiv preprint arXiv:2003.06975},
  year={2020}
}
```

## 授權界線

- 官方 GitHub 程式碼採 MIT License。
- 官方網站說明標註採 CC BY 4.0。
- 圖片來自 Flickr、OpenLitterMap 等不同來源，可能具有不同公開授權。
- 正式 `annotations.json` 中每張圖片具有來源 URL 與 `license` 欄位，但並非每張都有完整授權值。

本次正式資料 metadata 統計：

- `ODBL (c) OpenLitterMap & Contributors`：466 張。
- `CC`：319 張。
- `license` 未指定：715 張。

因此，MIT 只代表官方程式碼，不能推論全部圖片都可用於任何商業目的。匯入工具會在 `data/annotations/taco_import.csv` 保留每張圖片的來源 URL 與原始 license 值。正式發表、散布圖片、發布衍生資料或商業化前，必須逐一確認來源授權；`UNSPECIFIED` 圖片應視為需要人工確認，而不是自動視為可自由使用。

本專案目前只將 TACO 用於學術、非商業的 `waste` 單類物件偵測研究。原始 60 類不會直接轉成玻璃、金屬、紙或塑膠，以避免對複合材質與模糊類別做不可靠映射。

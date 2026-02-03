# NBI + USGS ShakeMap 自動下載與處理 (移轉說明)

這份專案已從另一個專案移入以下檔案：

- `broker/utils/nbi_ingest.py`
  - 新增功能：
    - 自動下載 FHWA NBI (Delimited, all states single file)
    - 自動下載 USGS ShakeMap (Northridge 1994, event_id = `ci3144585`)
    - 下載後自動解壓 NBI zip 到 `data/nbi/raw/`
- `data/nbi/meta/nbi_pipeline.json`
  - 下載設定（年份、事件 ID、下載檔案、是否解壓）

## 使用方式

只下載：
```
python broker\utils\nbi_ingest.py --download --download-only
```

下載 + 立刻處理：
```
python broker\utils\nbi_ingest.py --download
```

如需覆寫年份或事件 ID：
```
python broker\utils\nbi_ingest.py --download --nbi-year 1994 --usgs-event ci3144585
```

## 下載後檔案位置

- NBI：`data/nbi/raw/`（同時保留 zip 與解壓後的 CSV）
- ShakeMap：`data/hazard/usgs/shakemap/raw/`
- 中介產出：`data/nbi/clean/`, `data/nbi/curated/`, `data/nbi/meta/`, `data/nbi/logs/`

## 注意

- 目前只支援 NBI 的 **Delimited** 版本（逗號分隔、單引號為文字限定符）。
- 若 FHWA 下載頁面連結格式變動，可能需微調下載解析邏輯。

# CAT411 CAT Modeling Task Board

## 目標
用最短路徑讓 CAT 主幹可穩定跑完：
`Hazard -> Exposure -> Vulnerability -> Loss/Finance -> Visualization`

## 里程碑（Milestones）

## M1: Backbone 可跑通（P0）
- 狀態: IN_PROGRESS
- 交付:
  - `run_data_analysis` 可在固定資料上穩定跑完
  - 產出 `bridge_damage_results.csv`
  - 產出核心圖檔（shakemap、bridge map、damage map、dashboard）
- 驗收:
  - [ ] 無例外中斷（待直接重跑 `run_data_analysis` 流程）
  - [x] 輸出檔案存在且非空
  - [x] 關鍵欄位完整（`hwb_class`, `P_*`；目前為 `sa_10/pga` 欄位，`im_selected` 尚未在舊檔中）

## M2: Deterministic 風險結果穩定（P0）
- 狀態: TODO
- 交付:
  - deterministic 模式重跑結果統計穩定
  - loss summary 指標一致
- 驗收:
  - [ ] 固定 seed 下 `mean_loss` 漂移在可接受閾值內
  - [ ] `count_by_ds` 合理（總和接近橋梁數）

## M3: Probabilistic（EP/AAL）可用（P1）
- 狀態: DONE
- 交付:
  - stochastic event set + EP curve + AAL
  - `plot_ep_curve` 產圖可讀
- 驗收:
  - [x] `AAL > 0`（本次執行 `$1,668`）
  - [x] `return_period`/`loss_thresholds` 序列可生成

## M4: Calibration 與模型治理（P1）
- 狀態: TODO
- 交付:
  - fragility override 流程固定
  - calibration factor 文件化與版本化
- 驗收:
  - [ ] 開/關 calibration 結果可解釋
  - [ ] 配置變更可追溯（config + metadata）

## M5: 報告與決策層（P2）
- 狀態: TODO
- 交付:
  - 決策報告模板（Top risk class/region、AAL 摘要）
  - 輸出目錄與命名規範固定
- 驗收:
  - [ ] 非技術人可讀摘要完成
  - [ ] 圖表與數值對齊

## 模組任務清單（按優先級）

## P0-1 Data/Config 契約鎖定
- [ ] 定義統一資料契約（IM、橋梁欄位、損失欄位）
- [x] 檢查 `%g -> g` 轉換一致性（程式邏輯已確認）
- [ ] 建立缺失值與欄位映射檢查
- 參考:
  - `src/data_loader.py`
  - `src/config.py`

## P0-2 Hazard 指派穩定化
- [ ] 固定 `im_source` 決策流程（先 `shakemap`）
- [ ] 插值方法 benchmark（nearest vs idw）
- [ ] 輸出每橋多 IM 與 `im_selected`
- 參考:
  - `main.py` (`_compute_bridge_damage`)
  - `src/hazard.py`
  - `src/interpolation.py`

## P0-3 Vulnerability/Loss 正確性守門
- [x] fragility 單調/邊界/總和檢查（`--fragility-only` PASS）
- [ ] `expected_loss` 與 `loss_ratio` 合理性檢查
- [ ] `count_by_ds` 與橋數一致性檢查
- 參考:
  - `src/fragility.py`
  - `src/loss.py`

## P1-1 Probabilistic 生產化
- [x] 事件集參數化（`n_events`, `seed`, mag range）
- [ ] EP/AAL 回歸測試
- [x] 風險指標輸出固定格式
- 參考:
  - `src/engine.py`
  - `src/loss.py`

## P1-2 Calibration/Validation
- [ ] 對 Northridge 結果做 baseline 比對
- [ ] class/global factor 敏感度測試
- [ ] 建立校準紀錄
- 參考:
  - `config.yaml`
  - `src/northridge_case.py`

## P2-1 視覺化與報告產品化
- [x] 關鍵圖表最小集合（現有輸出已存在）
- [ ] Dashboard 指標凍結
- [ ] 報告模板（技術版/管理版）
- 參考:
  - `src/plotting.py`
  - `output/analysis`

## 驗收清單（Definition of Done）

## 模組 DoD（每一模組都要過）
- [ ] 功能正確
- [ ] 數值合理
- [ ] 可追溯（config/seed/版本）
- [ ] 可整合（不破壞主流程）

## 系統 DoD（端到端）
- [ ] `python main.py --full-analysis` 可跑通（受網路下載步驟影響）
- [x] `python main.py --probabilistic` 可產生 EP/AAL
- [x] 輸出檔案與圖表齊全（離線可驗部分）
- [ ] 關鍵指標有簡報級摘要

## 角色建議（可一人多角）
- Modeling Lead: Hazard/Vulnerability 方法與假設
- Data Lead: NBI/ShakeMap 資料品質與契約
- Risk Lead: Loss/EP/AAL 指標定義
- Delivery Lead: 視覺化、報告、對外溝通

## 建議執行節奏（2 週 sprint）
- Week 1:
  - P0-1, P0-2, P0-3
- Week 2:
  - P1-1, P1-2, P2-1

## 每日站會輸出格式（建議）
- 昨日完成:
- 今日計畫:
- 阻塞:
- 指標變化:
  - mean_loss:
  - AAL:
  - highest-risk HWB:

## 執行紀錄（2026-02-21）
- 執行 `python main.py --fragility-only`：PASS，verification checks 全數通過。
- 執行 `python main.py --probabilistic --n-bridges 20 --n-events 5 --n-realizations 3`：PASS。
- 本次關鍵結果：
  - AAL = `$1,668`
  - AAL ratio = `0.0022%`
  - EP 圖輸出：`output/scenario/ep_curve.png`
- 現況差異：`bridge_damage_results.csv` 目前位於 `output/` 根目錄；看板原先寫在 `output/analysis/`。

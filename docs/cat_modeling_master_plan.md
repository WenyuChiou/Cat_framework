# CAT411 CAT Modeling Master Plan

## 1) Framework 主幹（Backbone）

CAT411 的主幹是「以地震事件為驅動、以橋梁資產為對象」的端到端 CAT pipeline：

1. Hazard：得到每座橋的 IM（PGA/SA03/SA10/SA30）
2. Exposure：把 NBI 轉成可計算的橋梁資產與重置成本
3. Vulnerability：把 IM 映射為損傷狀態機率（Hazus fragility）
4. Loss/Finance：把損傷機率轉成經濟損失、EP Curve、AAL
5. Visualization/Delivery：輸出風險地圖、儀表板、決策圖表

程式主流程入口與協調器：
- `main.py:97` (`run_data_analysis`)
- `main.py:602` (`run_full_analysis`)
- `src/engine.py:156` (`run_deterministic`)
- `src/engine.py:217` (`run_probabilistic`)

## 2) 水平計畫圖（模組與依賴）

```mermaid
flowchart LR
    A[Data Ingestion\nUSGS ShakeMap + NBI] --> B[Hazard Module\nIM source + interpolation]
    B --> C[Exposure Module\nNBI->HWB + replacement cost]
    C --> D[Vulnerability Module\nFragility P(DS|IM)]
    D --> E[Loss/Finance Module\nE[Loss], EP curve, AAL]
    E --> F[Visualization & Outputs\nmaps, dashboard, reports]
    F --> G[Model Governance\nverification, calibration, acceptance]

    B -. deterministic .-> E
    B -. probabilistic event set .-> E
```

## 3) 模組工作分解（要完成主幹所需工作）

### A. Data Ingestion（資料層）
- 建立穩定資料管線：ShakeMap `grid.xml`、`stationlist.json`、NBI `CA*.txt`
- 確保欄位標準化與單位一致（%g -> g）
- 建立資料品質檢查：缺值、座標、欄位映射版本

對應程式：
- `src/data_loader.py:35` `parse_shakemap_grid`
- `src/data_loader.py:229` `parse_nbi`
- `src/data_loader.py:444` `classify_nbi_to_hazus`

### B. Hazard（危害層）
- IM 來源雙軌：`shakemap` / `gmpe`
- 空間插值方法治理（nearest/idw/bilinear/natural_neighbor/kriging）
- 針對橋點輸出多 IM 並選定主 IM（建議 SA10）

對應程式：
- `src/hazard.py:91` `boore_atkinson_2008_sa10`
- `main.py:314` `_compute_bridge_damage`（包含 IM 指派與插值流程）

### C. Exposure（資產層）
- NBI -> Hazus HWB 分類穩定化（規則透明、可追溯）
- 重置成本模型校正（材料、長度、面積）
- 濾鏡策略：縣市、年代、材質、橋型

對應程式：
- `src/exposure.py`（BridgeExposure + 成本）
- `src/bridge_classes.py`（分類決策）

### D. Vulnerability（脆弱度層）
- Fragility 曲線正確性（單調、界限、機率和=1）
- IM/Fragility 一致性（非 SA10 時需 override 或轉換）
- 校準層（global/class factor）機制與版本化

對應程式：
- `src/fragility.py:73` `damage_state_probabilities`
- `main.py` 驗證流程 `_run_verification`

### E. Loss / Finance（財務層）
- 橋梁層：Expected Loss、Loss Ratio、Downtime
- 投組層：損失分佈、Loss by Class
- 風險層：EP Curve、AAL（供保險/再保險/資本評估）

對應程式：
- `src/loss.py:119` `compute_portfolio_loss`
- `src/loss.py:180` `compute_ep_curve`
- `src/engine.py:217` `run_probabilistic`

### F. Visualization / Decision（結果層）
- 風險地圖、橋梁疊圖、衰減曲線、儀表板
- 統一輸出結構與命名（analysis/scenario/fragility）
- 決策導向指標：Top risk classes、高損失區域、AAL

對應程式：
- `src/plotting.py`
- `main.py:97`（集中輸出流程）

## 4) 每模組開發的驗收標準（DoD）

每個模組都要通過四個維度：

1. 功能正確：輸入/輸出與理論一致
2. 數值合理：範圍、單位、統計行為正確
3. 可追溯：參數與來源可回放（config + metadata）
4. 可整合：可接回 end-to-end，不破壞主流程

最低測試矩陣：
- 單元測試：hazard / fragility / loss 各核心函式
- 整合測試：`run_data_analysis` 一次跑通
- 回歸測試：固定 seed 下損失統計不可異常漂移
- 輸出測試：關鍵 CSV 與圖檔存在且欄位完整

## 5) 推進順序（建議）

1. 先鎖主幹資料契約（IM 欄位、橋梁欄位、損失欄位）
2. 先做 deterministic E2E 穩定版
3. 再擴 probabilistic（event catalog -> EP/AAL）
4. 最後做 calibration + uncertainty + 報表產品化

## 6) 近期執行路線圖（4 週）

### Week 1: 主幹穩定
- 完成 Data/Hazard/Exposure 欄位契約
- 建立最小整合測試（一鍵跑通）

### Week 2: Vulnerability + Loss 完整化
- fragility 驗證與 override 流程固定
- 損失與 downtime 指標標準化

### Week 3: Probabilistic 化
- 隨機事件集與 EP/AAL 穩定
- 參數敏感度初步分析

### Week 4: 視覺化與交付
- 儀表板模板固定
- 決策報告格式與 QA checklist

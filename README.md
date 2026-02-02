# CAT411 — Bridge Earthquake Catastrophe Modeling Framework

A complete catastrophe (CAT) modeling pipeline for seismic risk assessment of highway bridges, built around the **FEMA Hazus 6.1** methodology and calibrated against the **1994 Northridge earthquake (Mw 6.7)**.

The framework implements the four core components of a catastrophe model — **Hazard, Exposure, Vulnerability, and Loss** — and supports both deterministic scenario analysis and probabilistic risk assessment.

---

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Quick Start](#quick-start)
- [Mathematical Foundations](#mathematical-foundations)
  - [Hazard: Ground Motion Prediction](#1-hazard-ground-motion-prediction-srchazardpy)
  - [Exposure: Bridge Inventory](#2-exposure-bridge-inventory--financial-data-srcexposurepy)
  - [Vulnerability: Fragility Curves](#3-vulnerability-fragility-curves-srcfragilitypy)
  - [Loss: Damage-to-Loss Translation](#4-loss-damage-to-loss-translation-srclosspy)
- [Pipeline Orchestrator](#5-pipeline-orchestrator-srcenginepy)
- [Supporting Modules](#supporting-modules)
- [Output Artifacts](#output-artifacts)
- [Data Sources](#data-sources)
- [API Usage Examples](#api-usage-examples)
- [References](#references)

---

## Pipeline Architecture

```
EarthquakeScenario (Mw, lat, lon, depth, fault_type)
        |
        v
   src/hazard.py
   |  GMPE: Boore-Atkinson 2008 -> median Sa(1.0s) + sigma
   |  Site amplification: Vs30-dependent F_S term
   |  Spatial correlation: Jayaram-Baker 2009 (rho = exp(-3h/b))
   |  Ground motion field: Cholesky(corr) x N(0,1) x sigma
   |
   +------------------------------+
   v                              v
 src/exposure.py            src/fragility.py
 Bridge inventory           P[DS >= ds | Sa] = Phi[(lnSa - ln_median) / beta]
 BridgeExposure dataclass   14 Hazus bridge classes (HWB1-HWB28)
 Replacement cost = f(material, deck_area)   Damage states: none/slight/moderate/
 Synthetic portfolio generator               extensive/complete
 NBI-to-exposure converter
   |                              |
   +--------------+---------------+
                  v
            src/loss.py
            E[Loss_i] = SUM_ds P(ds) x DR(ds) x RC_i
            DR: none=0%, slight=3%, moderate=8%, extensive=25%, complete=100%
            EP(L) = 1 - exp(-cumulative_rate)
            AAL = SUM_i lambda_i x L_i
                  |
                  v
           src/engine.py
           Deterministic: 1 scenario x N realizations -> mean/std loss
           Probabilistic: GR catalog x N realizations -> EP curve + AAL
                  |
                  v
             main.py (CLI)
             --pipeline / --probabilistic / --fragility-only
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
# Requires: numpy, scipy, matplotlib, pandas, requests
```

### Run modes

```bash
# 1. Full deterministic CAT model pipeline (Northridge Mw 6.7 scenario)
python main.py --pipeline

# 2. Probabilistic analysis: stochastic event catalog -> EP curve + AAL
python main.py --probabilistic

# 3. Fragility curves only (no hazard/loss, existing analysis)
python main.py --fragility-only

# 4. Download USGS ShakeMap + FHWA NBI data, then run full data analysis
python main.py --download
python main.py
```

### Configuration flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n-bridges` | 100 | Number of bridges in synthetic portfolio |
| `--n-realizations` | 50 | Monte Carlo ground motion field realizations |
| `--n-events` | 50 | Number of stochastic earthquake events (probabilistic mode) |

```bash
python main.py --pipeline --n-bridges 200 --n-realizations 100
python main.py --probabilistic --n-bridges 200 --n-events 100 --n-realizations 30
```

---

## Mathematical Foundations

### 1. Hazard: Ground Motion Prediction (`src/hazard.py`)

#### Boore & Atkinson (2008) GMPE

The ground motion prediction equation computes the median spectral acceleration Sa(1.0s) in g as:

```
ln(Sa) = F_M + F_D + F_S
```

where:

**Source (magnitude) term F_M:**
- For M <= M_h (hinge magnitude = 6.75):
  `F_M = e_i + e5*(M - M_h) + e6*(M - M_h)^2`
- For M > M_h:
  `F_M = e_i + e7*(M - M_h)`
- Coefficient e_i depends on fault mechanism:
  - e1 = -0.23898 (unspecified)
  - e2 = -0.28892 (strike-slip)
  - e4 = -0.20608 (reverse)

**Distance term F_D:**
```
R = sqrt(R_JB^2 + h^2)    where h = 2.54 km (fictitious depth)
F_D = (c1 + c2*(M - M_h)) * ln(R) + c3*(R - 1)
```
- c1 = -0.68898, c2 = 0.21521, c3 = -0.00707
- R_JB is the Joyner-Boore distance (km), approximated as epicentral distance for point sources

**Site amplification term F_S:**
```
F_lin = b_lin * ln(min(Vs30, V_ref) / V_ref)
```
- b_lin = -0.60, V_ref = 760 m/s
- Non-linear correction applied for Vs30 < 300 m/s using PGA on reference rock

**Aleatory uncertainty:**
- Inter-event: tau = 0.255
- Intra-event: sigma = 0.502
- Total: sigma_T = 0.564

#### Jayaram-Baker (2009) Spatial Correlation

The intra-event residuals at nearby sites are spatially correlated:

```
rho(h) = exp(-3h / b)
```

where h is the inter-site separation distance (km) and b = 40.7 km for T = 1.0s.

The correlation matrix C_ij = rho(h_ij) is constructed for all site pairs.

#### Ground Motion Field Generation

For each realization k:
1. Draw inter-event residual: eta_k ~ N(0, tau)  (shared by all sites)
2. Draw correlated intra-event residuals: eps = L * z * sigma
   - L = Cholesky decomposition of C (correlation matrix)
   - z ~ N(0, I) (independent standard normals)
3. Combine: `ln(Sa_i) = ln(median_i) + eta + eps_i`
4. Exponentiate: `Sa_i = exp(ln(Sa_i))`

### 2. Exposure: Bridge Inventory & Financial Data (`src/exposure.py`)

#### BridgeExposure Dataclass

Each bridge carries:
| Field | Type | Description |
|-------|------|-------------|
| `bridge_id` | str | Unique identifier |
| `lat`, `lon` | float | Geographic coordinates |
| `hwb_class` | str | Hazus bridge class (e.g. "HWB5") |
| `material` | str | "concrete", "steel", "prestressed_concrete", "wood", "other" |
| `length` | float | Total length (m) |
| `deck_area` | float | Deck area (m^2) |
| `replacement_cost` | float | Estimated replacement cost (USD) |
| `vs30` | float | Site Vs30 (m/s) |
| `skew_angle` | float | Skew angle (degrees) |

#### Replacement Cost Estimation

```
RC = unit_cost(material) x deck_area x length_factor
```

Unit costs by material (USD/m^2):
| Material | Unit Cost |
|----------|-----------|
| Concrete | $2,500 |
| Steel | $3,200 |
| Prestressed concrete | $2,800 |
| Wood | $1,800 |
| Other | $2,600 |

Length adjustment factor: `1.0 + 0.15 * max(0, (length - 100) / 200)` accounts for higher foundation costs on longer bridges.

#### Synthetic Portfolio Generation

`generate_synthetic_portfolio(n_bridges, center, radius_km, seed)` creates a portfolio with:
- Realistic Northridge-area HWB class distribution (e.g. HWB5: 14%, HWB3: 12%, HWB17: 10%)
- Uniform random placement within a disc around the center
- Random structural dimensions (length 15-120m, width 8-25m)
- Random Vs30 (200-800 m/s)

### 3. Vulnerability: Fragility Curves (`src/fragility.py`)

#### Lognormal Fragility Model

The probability of reaching or exceeding damage state `ds` given intensity measure `Sa`:

```
P[DS >= ds | Sa] = Phi( (ln(Sa) - ln(median_ds)) / beta_ds )
```

where Phi is the standard normal CDF, and (median_ds, beta_ds) are lognormal parameters from **Hazus Table 7.9**.

#### 14 Bridge Classes

Parameters are defined for 14 Hazus bridge classes in `src/hazus_params.py`. Example (HWB5 — Multi-Span Concrete Continuous, Conventional):

| Damage State | Median (g) | Beta |
|:------------|:----------:|:----:|
| Slight | 0.35 | 0.6 |
| Moderate | 0.45 | 0.6 |
| Extensive | 0.55 | 0.6 |
| Complete | 0.80 | 0.6 |

Seismic-designed bridges (even-numbered HWBs) have 1.5-2x higher median capacities.

#### Discrete Damage State Probabilities

From the exceedance curves, discrete probabilities are computed:
```
P[none]      = 1 - P[DS >= slight]
P[slight]    = P[DS >= slight]    - P[DS >= moderate]
P[moderate]  = P[DS >= moderate]  - P[DS >= extensive]
P[extensive] = P[DS >= extensive] - P[DS >= complete]
P[complete]  = P[DS >= complete]
```

These five probabilities sum to 1.0 for any given Sa value.

#### Skew Modification

For skewed bridges, the median capacity is reduced:
```
median_modified = median * sqrt(1 - (skew_angle / 90)^2)
```

### 4. Loss: Damage-to-Loss Translation (`src/loss.py`)

#### Hazus Damage Ratios (Table 7.11)

| Damage State | Damage Ratio (DR) | Downtime (days) |
|:------------|:-----------------:|:---------------:|
| None | 0.00 | 0 |
| Slight | 0.03 | 0.6 |
| Moderate | 0.08 | 2.5 |
| Extensive | 0.25 | 75 |
| Complete | 1.00 | 230 |

#### Expected Loss per Bridge

```
E[Loss_i] = SUM_ds  P(ds | Sa_i) x DR(ds) x RC_i
```

where P(ds | Sa_i) is the discrete damage state probability at the site-specific Sa, and RC_i is the replacement cost.

#### Portfolio Aggregation

Total expected loss:
```
E[L_portfolio] = SUM_i  E[Loss_i]
Loss_ratio = E[L_portfolio] / SUM_i RC_i
```

#### Loss Exceedance Probability (EP) Curve

For a set of scenarios with annual rates lambda_i and losses L_i:

1. Sort scenarios by loss (descending)
2. Cumulative rate: nu(L) = SUM_{L_i >= L} lambda_i
3. Annual exceedance probability (Poisson): EP(L) = 1 - exp(-nu(L))
4. Return period: RP(L) = 1 / nu(L)

#### Average Annual Loss (AAL)

```
AAL = SUM_i  lambda_i x L_i
```

where lambda_i is the annual rate of scenario i and L_i is the expected loss.

---

## 5. Pipeline Orchestrator (`src/engine.py`)

### Deterministic Mode

`run_deterministic(scenario, portfolio, n_realizations)`:

1. Compute median Sa(1.0s) at each bridge site via BA08
2. Generate N spatially-correlated ground motion fields (Cholesky + MVN)
3. For each realization: compute portfolio loss via fragility + damage ratios
4. Report mean, std, and loss distribution across realizations

### Probabilistic Mode

`run_probabilistic(portfolio, n_events, n_realizations)`:

1. Generate stochastic event catalog from Gutenberg-Richter: `log10(N) = a - b*M`
   - Default: a=4.0, b=1.0, M_range=[5.0, 7.5]
   - Random locations within 80 km radius of Northridge
   - Truncated exponential magnitude sampling
2. For each event: generate N ground motion fields, compute mean loss
3. Build EP curve from (loss, rate) pairs
4. Compute AAL

### Pre-defined Scenario

```python
NORTHRIDGE_SCENARIO = EarthquakeScenario(
    Mw=6.7, lat=34.213, lon=-118.537, depth_km=18.4, fault_type="reverse"
)
```

---

## Supporting Modules

| Module | Purpose |
|--------|---------|
| `src/hazus_params.py` | Hazus 6.1 Table 7.9 lognormal fragility parameters for 14 bridge classes |
| `src/bridge_classes.py` | `BridgeClassification` dataclass and `classify_bridge()` decision tree |
| `src/data_loader.py` | Download and parse USGS ShakeMap (grid.xml), station recordings (JSON), and FHWA NBI bridge inventory (delimited text) |
| `src/northridge_case.py` | 1994 Northridge observed damage statistics (1,600 bridges, 7 collapses) and prediction-vs-observation comparison |
| `src/plotting.py` | All visualization: fragility curves, ground motion map, loss bar chart, damage distribution, EP curve |

---

## Output Artifacts

All plots are saved to `output/` and can be regenerated at any time.

| File | CLI Mode | Description |
|------|----------|-------------|
| `ground_motion_field.png` | `--pipeline` | Scatter map of Sa(1.0s) at bridge sites, colored by intensity, epicenter marked |
| `loss_by_class.png` | `--pipeline` | Bar chart: expected loss per HWB class |
| `portfolio_damage.png` | `--pipeline` | Stacked horizontal bar: portfolio damage state distribution |
| `ep_curve.png` | `--probabilistic` | Loss EP curve (left) and loss vs return period (right) |
| `fragility_HWB*.png` | `--fragility-only` | 4-curve fragility plot per class (14 files) |
| `comparison_*.png` | `--fragility-only` | Cross-class comparison for slight and complete damage |
| `damage_distribution_*.png` | `--fragility-only` | Stacked bar charts at 10 intensity levels |
| `northridge_scenario.png` | `--fragility-only` | HWB5 fragility with observed PGA range overlay |
| `bridge_damage_results.csv` | default | Per-bridge damage probabilities (requires real data) |

---

## Data Sources & Integration

### Overview

本框架使用兩種外部資料來源，皆為**下載到本地後離線讀取**，不呼叫任何線上 API：

| 資料 | 來源機構 | 下載 URL | 本地路徑 | 格式 | 大小 |
|------|---------|---------|---------|------|------|
| ShakeMap 地震動格點 | **USGS** Earthquake Hazards Program | earthquake.usgs.gov/product/shakemap/ci3144585/.../grid.xml | `data/grid.xml` | XML (grid_field + grid_data) | ~3.4 MB |
| ShakeMap 測站記錄 | **USGS** Earthquake Hazards Program | earthquake.usgs.gov/product/shakemap/ci3144585/.../stationlist.json | `data/stationlist.json` | GeoJSON | ~3.4 MB |
| 橋梁清冊 | **FHWA** National Bridge Inventory (NBI) | www.fhwa.dot.gov/bridge/nbi/2024/delimited/CA24.txt | `data/CA24.txt` | Comma-delimited text | ~25 MB |

### Step 1: 下載資料

```bash
python main.py --download
```

這會透過 `src/data_loader.py` 中的 `download_shakemap()`、`download_shakemap_stations()`、`download_nbi()` 函數，從 USGS 和 FHWA 官網下載三個檔案到 `data/` 目錄。你也可以手動下載或從其他來源取得同格式的檔案，放到 `data/` 即可。

### Step 2: 框架如何讀取資料

下載完成後，`src/data_loader.py` 提供以下解析函數：

**ShakeMap (USGS 地震動資料):**

```python
from src.data_loader import load_shakemap, load_stations

# 讀取 grid.xml → DataFrame，欄位: LON, LAT, PGA, PGV, PSA03, PSA10, PSA30
# PGA 和 PSA 欄位自動從 %g 轉換為 g
shakemap = load_shakemap("data/grid.xml")

# 讀取 stationlist.json → DataFrame，欄位: station_code, lat, lon, pga, psa10, ...
stations = load_stations("data/stationlist.json")
```

- `grid.xml` 包含 ~10,000 個格點的 PGA、Sa(0.3s)、Sa(1.0s)、Sa(3.0s) 等地震動參數
- 原始數值單位為 %g，解析器自動除以 100 轉為 g
- 事件 metadata（震級、震央座標）從 XML root attributes 提取

**NBI (FHWA 橋梁清冊):**

```python
from src.data_loader import load_nbi, classify_nbi_to_hazus

# 讀取 NBI 分隔文字檔 → DataFrame
# 自動偵測欄位名稱（pattern matching）、轉換座標格式（DDMMSS.SS → 十進位）
# 以 bounding box 篩選 Northridge 地區 (33.7-34.8°N, 117.5-119.0°W)
nbi = load_nbi("data/CA24.txt")

# 根據材料代碼、跨數、建造年份分類為 Hazus HWB 橋梁類別
nbi = classify_nbi_to_hazus(nbi)
```

- NBI 欄位映射：material_code → 材料類型、num_spans → 跨型、year_built ≥ 1975 → 耐震設計
- 座標格式自動判斷：大於 200 視為 DDMMSS.SS 格式，否則視為十進位度數

### Step 3: 資料分析流程

```bash
python main.py          # 預設模式：讀取本地資料 → 完整分析
```

預設模式的處理流程：

1. **讀取 ShakeMap** → 取得格點上的 Sa(1.0s) 空間分布
2. **讀取 NBI** → 取得 Northridge 地區 ~1,600 座橋梁的位置與屬性
3. **HWB 分類** → 每座橋依結構屬性指派 Hazus 橋梁類別
4. **空間匹配** → 用 KD-tree 最近鄰搜尋，將 ShakeMap Sa(1.0s) 指派到每座橋的位置
5. **損壞機率** → 對每座橋計算 5 個損壞狀態的機率（none/slight/moderate/extensive/complete）
6. **輸出** → 結果存為 `output/bridge_damage_results.csv`

### 兩種分析模式的資料來源差異

| 模式 | 地震動來源 | 橋梁來源 | 說明 |
|------|-----------|---------|------|
| `python main.py` (預設) | **USGS ShakeMap** 實測/模型資料 | **FHWA NBI** 真實橋梁清冊 | 使用歷史地震的實際地震動 |
| `--pipeline` | **BA08 GMPE** 自行計算 | 合成組合 (或 NBI) | 可對任意地震情境預測 |
| `--probabilistic` | **BA08 GMPE** 自行計算 | 合成組合 (或 NBI) | 隨機地震目錄 + GMPE |
| `--fragility-only` | 不需要 | 不需要 | 純理論脆弱度曲線 |

### 使用自己的資料

你可以替換 `data/` 中的檔案來分析不同地震或不同地區的橋梁：

```python
# 使用不同州的 NBI 資料
nbi = load_nbi("data/TX24.txt", northridge_bbox={
    "lat_min": 29.0, "lat_max": 31.0,
    "lon_min": -98.0, "lon_max": -96.0,
})

# 使用不同地震的 ShakeMap
shakemap = load_shakemap("data/other_event_grid.xml")
```

只要檔案格式與 USGS ShakeMap XML / FHWA NBI delimited text 相同，框架即可讀取。

---

## API Usage Examples

### Custom earthquake scenario

```python
from src.hazard import EarthquakeScenario, boore_atkinson_2008_sa10
from src.exposure import generate_synthetic_portfolio, portfolio_to_sites
from src.engine import run_deterministic, print_deterministic_report

# Define a custom scenario
scenario = EarthquakeScenario(Mw=7.2, lat=34.0, lon=-118.3, depth_km=12.0,
                              fault_type="strike_slip")

# Generate or load a portfolio
portfolio = generate_synthetic_portfolio(n_bridges=200, seed=42)

# Run deterministic analysis
result = run_deterministic(scenario, portfolio, n_realizations=100)
print(print_deterministic_report(result))
```

### Using real NBI data

```python
from src.data_loader import load_nbi, classify_nbi_to_hazus
from src.exposure import create_portfolio_from_nbi
from src.engine import run_deterministic, NORTHRIDGE_SCENARIO

# Load and classify NBI bridges
nbi = load_nbi("data/CA24.txt")
nbi = classify_nbi_to_hazus(nbi)

# Convert to exposure objects
portfolio = create_portfolio_from_nbi(nbi, default_vs30=360.0)

# Run analysis
result = run_deterministic(NORTHRIDGE_SCENARIO, portfolio, n_realizations=50)
```

### Single bridge quick check

```python
from src.hazard import boore_atkinson_2008_sa10
from src.fragility import damage_state_probabilities
from src.loss import compute_bridge_loss

# Ground motion at a specific site
sa, sigma = boore_atkinson_2008_sa10(Mw=6.7, R_JB=15.0, Vs30=360.0)

# Damage probabilities
probs = damage_state_probabilities(sa, "HWB5")

# Expected loss
result = compute_bridge_loss(sa, "HWB5", replacement_cost=5_000_000)
print(f"Sa = {sa:.3f}g, E[Loss] = ${result.expected_loss:,.0f}")
```

---

## References

1. Boore, D.M. & Atkinson, G.M. (2008). Ground-Motion Prediction Equations for the Average Horizontal Component of PGA, PGV, and 5%-Damped PSA at Spectral Periods between 0.01s and 10.0s. *Earthquake Spectra*, 24(1), 99-138.
2. Jayaram, N. & Baker, J.W. (2009). Correlation model for spatially distributed ground-motion intensities. *Earthquake Engineering & Structural Dynamics*, 38(15), 1687-1708.
3. FEMA (2024). *Hazus 6.1 Earthquake Model Technical Manual*. Federal Emergency Management Agency.
4. Basoz, N. & Kiremidjian, A. (1998). Evaluation of Bridge Damage Data from the Loma Prieta and Northridge, CA Earthquakes. MCEER-98-0004.
5. Werner, S.D., et al. (2006). Seismic Risk Analysis of Highway Systems. MCEER-06-0011.
6. Caltrans (1994). The Northridge Earthquake: Post-Earthquake Investigation Report.

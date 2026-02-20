# CAT411 — Bridge Earthquake Catastrophe Modeling Framework

A complete catastrophe (CAT) modeling pipeline for seismic risk assessment of highway bridges, built around the **FEMA Hazus 6.1** methodology and calibrated against the **1994 Northridge earthquake (Mw 6.7)**.

The framework implements the four core components of a catastrophe model — **Hazard, Exposure, Vulnerability, and Loss** — and supports both deterministic scenario analysis and probabilistic risk assessment.

---

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Quick Start](#quick-start)
- [Configuration System](#configuration-system)
- [Data Sources](#data-sources)
- [Mathematical Foundations](#mathematical-foundations)
  - [Hazard: Ground Motion Prediction](#1-hazard-ground-motion-prediction-srchazardpy)
  - [Exposure: Bridge Inventory](#2-exposure-bridge-inventory--financial-data-srcexposurepy)
  - [Vulnerability: Fragility Curves](#3-vulnerability-fragility-curves-srcfragilitypy)
  - [Loss: Damage-to-Loss Translation](#4-loss-damage-to-loss-translation-srclosspy)
- [Spatial Interpolation Methods](#spatial-interpolation-methods)
- [Pipeline Orchestrator](#5-pipeline-orchestrator-srcenginepy)
- [Module Reference](#module-reference)
- [Output Structure](#output-structure)
- [API Usage Examples](#api-usage-examples)
- [References](#references)

---

## Pipeline Architecture

```
                    config.yaml
                        │
               ┌────────┴────────┐
               v                 v
    IM Source: ShakeMap      IM Source: GMPE
    (grid.xml via USGS)     (BA08 calculation)
               │                 │
               v                 v
       ┌── Spatial Interpolation ──┐
       │  nearest / idw / bilinear │
       │  natural_neighbor / kriging│
       └──────────┬───────────────┘
                  │
                  v
           Bridge IM values
           (PGA / SA03 / SA10 / SA30)
                  │
    ┌─────────────┼─────────────┐
    v             v             v
 Exposure     Fragility      Loss
 (NBI data)   P[DS≥ds|IM]    E[Loss] = ΣP(ds)·DR·RC
              Hazus 6.1      EP curve + AAL
                  │
                  v
          Visualizations
          (8 analysis plots)
```

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
# Requires: numpy, scipy, matplotlib, pandas, requests, pyyaml
```

### Run modes

```bash
# 1. Default: download data + full analysis with config.yaml
python main.py

# 2. Full deterministic CAT model pipeline (synthetic portfolio)
python main.py --pipeline

# 3. Probabilistic analysis: stochastic catalog → EP curve + AAL
python main.py --probabilistic

# 4. Fragility curves only
python main.py --fragility-only

# 5. Download USGS data only
python main.py --download-hazard

# 6. With configuration overrides
python main.py --config config.yaml --im-type PGA --nbi-filter "county=037" "year_built>1960"
```

### CLI flags

| Flag                | Default       | Description                                               |
| ------------------- | ------------- | --------------------------------------------------------- |
| `--config`          | `config.yaml` | Path to YAML configuration file                           |
| `--im-type`         | `SA10`        | Override IM type: `PGA`, `SA03`, `SA10`, `SA30`           |
| `--nbi-filter`      | —             | NBI column filters (e.g. `county=037`, `year_built>1960`) |
| `--pipeline`        | —             | Run deterministic CAT model (synthetic portfolio)         |
| `--probabilistic`   | —             | Run probabilistic analysis with event catalog             |
| `--fragility-only`  | —             | Generate fragility curves only                            |
| `--download-hazard` | —             | Download USGS ShakeMap data                               |
| `--n-bridges`       | 100           | Synthetic portfolio size                                  |
| `--n-realizations`  | 50            | Monte Carlo realizations per event                        |
| `--n-events`        | 50            | Stochastic events (probabilistic mode)                    |

---

## Configuration System

All analysis parameters are controlled via `config.yaml`:

```yaml
# ── IM Source ──────────────────────────────────
im_source: shakemap # "shakemap" or "gmpe"
im_type: SA10 # PGA, SA03, SA10, SA30

# ── Spatial Interpolation ─────────────────────
interpolation:
  method: nearest # nearest / idw / bilinear / natural_neighbor / kriging

# ── Region ────────────────────────────────────
region:
  lat_min: 33.8
  lat_max: 34.6
  lon_min: -118.9
  lon_max: -118.0

# ── Bridge Selection (any NBI column) ─────────
bridge_selection:
  county: "037" # exact match
  year_built: ">1960" # numeric comparison
  material: ["concrete"] # list match

# ── Fragility Overrides ───────────────────────
fragility_overrides:
  HWB5:
    slight: { median: 0.30, beta: 0.55 }
    moderate: { median: 0.50, beta: 0.55 }
    extensive: { median: 0.70, beta: 0.55 }
    complete: { median: 1.00, beta: 0.55 }

# ── Calibration ───────────────────────────────
calibration:
  global_median_factor: 1.0
  class_factors:
    HWB5: 0.90 # 10% more vulnerable
```

CLI arguments (`--im-type`, `--nbi-filter`) override `config.yaml` settings.

---

## Data Sources

The framework integrates **3 data sources**:

| Source            | File                    | Format        | Contents                                                                                |
| ----------------- | ----------------------- | ------------- | --------------------------------------------------------------------------------------- |
| **FHWA NBI**      | `data/CA24.txt`         | Delimited TXT | Bridge inventory — ~12,000 CA bridges with location, material, year, spans, condition   |
| **USGS ShakeMap** | `data/grid.xml`         | XML grid      | Ground motion intensities — PGA, PGV, PSA03, PSA10, PSA30, SVEL on regular lat/lon grid |
| **Station List**  | `data/stationlist.json` | JSON          | Real seismic station recordings for the event (used for ShakeMap validation)            |

### Data directory layout

```
data/
├── CA24.txt                    ← NBI raw file (top-level for quick access)
├── grid.xml                    ← ShakeMap grid (top-level for quick access)
├── stationlist.json            ← Station recordings
├── nbi/
│   ├── raw/                    ← Downloaded NBI zip + extracted files
│   ├── clean/                  ← Cleaned NBI CSV
│   ├── curated/                ← Curated NBI CSV (filtered, classified)
│   ├── meta/                   ← Processing metadata JSON
│   └── logs/                   ← Processing log files
└── hazard/usgs/
    ├── shakemap/               ← ShakeMap raw + processed files
    └── hazard_curves/          ← NSHMP probabilistic hazard curves
```

### Download commands

```bash
# Full download + processing pipeline
python main.py --download-pipeline

# Or manual NBI download
python broker/utils/nbi_ingest.py --download

# Download ShakeMap for a specific event
python main.py --download-hazard --hazard-event ci3144585
```

---

## Mathematical Foundations

### 1. Hazard: Ground Motion Prediction (`src/hazard.py`)

#### Boore & Atkinson (2008) GMPE

```
ln(Sa) = F_M + F_D + F_S
```

- **F_M** (source/magnitude): Uses hinge magnitude M_h = 6.75 with fault-type-dependent coefficients
- **F_D** (distance): `R = sqrt(R_JB² + h²)` with h = 2.54 km fictitious depth
- **F_S** (site): `F_lin = b_lin · ln(min(Vs30, V_ref) / V_ref)` with non-linear correction

**Aleatory uncertainty:** σ_total = 0.564 (inter-event τ = 0.255, intra-event σ = 0.502)

#### Jayaram-Baker (2009) Spatial Correlation

```
ρ(h) = exp(-3h / b),    b = 40.7 km for T = 1.0s
```

### 2. Exposure: Bridge Inventory & Financial Data (`src/exposure.py`)

#### Replacement Cost

```
RC = unit_cost(material) × deck_area × length_factor
```

| Material             | Unit Cost (USD/m²) |
| -------------------- | ------------------ |
| Concrete             | $2,500             |
| Steel                | $3,200             |
| Prestressed concrete | $2,800             |
| Wood                 | $1,800             |

### 3. Vulnerability: Fragility Curves (`src/fragility.py`)

#### Lognormal Fragility Model

```
P[DS ≥ ds | IM] = Φ( (ln(IM) - ln(median_ds)) / β_ds )
```

Parameters from **Hazus Table 7.9** for 14 bridge classes (HWB1–HWB28).

#### Discrete Damage State Probabilities

```
P[none]     = 1 - P[DS ≥ slight]
P[slight]   = P[DS ≥ slight]   - P[DS ≥ moderate]
P[moderate] = P[DS ≥ moderate] - P[DS ≥ extensive]
P[extensive]= P[DS ≥ extensive] - P[DS ≥ complete]
P[complete] = P[DS ≥ complete]
```

### 4. Loss: Damage-to-Loss Translation (`src/loss.py`)

#### Hazus Damage Ratios (Table 7.11)

| Damage State | Damage Ratio | Downtime (days) |
| :----------- | :----------: | :-------------: |
| None         |     0.00     |        0        |
| Slight       |     0.03     |       0.6       |
| Moderate     |     0.08     |       2.5       |
| Extensive    |     0.25     |       75        |
| Complete     |     1.00     |       230       |

```
E[Loss_i] = Σ_ds P(ds | IM_i) × DR(ds) × RC_i
```

---

## Spatial Interpolation Methods

The `src/interpolation.py` module provides **5 methods** for assigning IM values from ShakeMap grid to bridge locations:

| Method                         | Config Name        | Description                           | Best For                     |
| ------------------------------ | ------------------ | ------------------------------------- | ---------------------------- |
| **Nearest Neighbor**           | `nearest`          | KD-tree closest grid point            | Dense grids, fast processing |
| **Inverse Distance Weighting** | `idw`              | Weighted avg of k-nearest points      | Smooth fields, configurable  |
| **Bilinear**                   | `bilinear`         | RegularGridInterpolator               | Regular ShakeMap grids       |
| **Natural Neighbor**           | `natural_neighbor` | Voronoi/Delaunay triangulation        | Irregular station data       |
| **Ordinary Kriging**           | `kriging`          | Exponential variogram (Jayaram-Baker) | Geostatistical accuracy      |

Configure in `config.yaml`:

```yaml
interpolation:
  method: idw
  power: 2.0 # IDW: distance exponent
  n_neighbors: 8 # IDW/Kriging: neighbor count
  range_km: 50.0 # Kriging: variogram range
  nugget: 0.01 # Kriging: measurement noise
```

---

## 5. Pipeline Orchestrator (`src/engine.py`)

### Deterministic Mode

1. Compute median Sa(1.0s) at each site via BA08
2. Generate N spatially-correlated ground motion fields
3. Compute portfolio loss for each realization
4. Report mean, std, and loss distribution

### Probabilistic Mode

1. Generate stochastic event catalog from Gutenberg-Richter
2. For each event: compute mean loss across realizations
3. Build EP curve from (loss, rate) pairs
4. Compute Average Annual Loss (AAL)

---

## Module Reference

| Module                   | Purpose                                                        |
| ------------------------ | -------------------------------------------------------------- |
| `src/hazard.py`          | BA08 GMPE, spatial correlation, ground motion field generation |
| `src/exposure.py`        | Bridge portfolio, replacement cost, NBI-to-exposure conversion |
| `src/fragility.py`       | Lognormal fragility model, damage state probabilities          |
| `src/loss.py`            | Damage ratios, expected loss, EP curve, AAL                    |
| `src/engine.py`          | Pipeline orchestrator (deterministic + probabilistic)          |
| `src/config.py`          | Configuration loader (`config.yaml` → `AnalysisConfig`)        |
| `src/interpolation.py`   | 5 spatial interpolation methods for IM assignment              |
| `src/data_loader.py`     | Parse ShakeMap XML, station JSON, NBI text files               |
| `src/hazard_download.py` | USGS API download (ShakeMap, hazard curves, design maps)       |
| `src/hazus_params.py`    | Hazus 6.1 Table 7.9 fragility parameters (14 classes)          |
| `src/bridge_classes.py`  | NBI → Hazus bridge classification decision tree                |
| `src/northridge_case.py` | 1994 Northridge observed damage statistics                     |
| `src/plotting.py`        | All visualization functions (17 plot types)                    |

---

## Output Structure

All outputs are organized into subdirectories:

```
output/
├── analysis/                              ← Real data analysis results
│   ├── 00_analysis_dashboard.png          2×2 summary dashboard
│   ├── 01_shakemap_full_area.png          ShakeMap intensity grid
│   ├── 02_nbi_bridge_distribution_map.png Bridge locations
│   ├── 03_bridge_site_ground_motion.png   IM at bridge sites
│   ├── 04_bridge_damage_spatial.png       Damage probability map
│   ├── 05_bridges_on_shakemap.png         Bridges overlaid on ShakeMap contours
│   ├── 06_attenuation_curve.png           GMPE prediction vs observed IM
│   ├── 07_portfolio_damage_bars.png       Damage state distribution
│   └── bridge_damage_results.csv          Per-bridge damage probabilities
│
├── fragility/                             ← Fragility curve plots
│   ├── fragility_HWB*.png                 Individual class curves (14 files)
│   ├── comparison_*.png                   Cross-class comparisons
│   └── damage_distribution_HWB*.png       Damage probability distributions
│
└── scenario/                              ← Scenario-based analysis
    ├── northridge_scenario.png            Northridge PGA overlay
    ├── ground_motion_field.png            Synthetic GMF
    ├── loss_by_class.png                  Loss by bridge class
    ├── portfolio_damage.png               Portfolio damage summary
    └── ep_curve.png                       Exceedance probability curve
```

---

## API Usage Examples

### Custom analysis with configuration

```python
from src.config import load_config
from src.data_loader import load_shakemap, load_nbi, classify_nbi_to_hazus
from src.interpolation import interpolate_im

# Load config
config = load_config("config.yaml")

# Load data
sm = load_shakemap("data/grid.xml")
nbi = load_nbi("data/CA24.txt")
nbi = classify_nbi_to_hazus(nbi)

# Interpolate IM to bridge locations
from src.config import IM_COLUMN_MAP
im_col = IM_COLUMN_MAP[config.im_type]
bridge_ims = interpolate_im(
    sm["LAT"].values, sm["LON"].values, sm[im_col].values,
    nbi["latitude"].values, nbi["longitude"].values,
    method=config.interpolation_method,
)
```

### Single bridge quick check

```python
from src.hazard import boore_atkinson_2008_sa10
from src.fragility import damage_state_probabilities
from src.loss import compute_bridge_loss

sa, sigma = boore_atkinson_2008_sa10(Mw=6.7, R_JB=15.0, Vs30=360.0)
probs = damage_state_probabilities(sa, "HWB5")
result = compute_bridge_loss(sa, "HWB5", replacement_cost=5_000_000)
print(f"Sa = {sa:.3f}g, E[Loss] = ${result.expected_loss:,.0f}")
```

### Custom earthquake scenario

```python
from src.hazard import EarthquakeScenario
from src.exposure import generate_synthetic_portfolio
from src.engine import run_deterministic, print_deterministic_report

scenario = EarthquakeScenario(Mw=7.2, lat=34.0, lon=-118.3,
                              depth_km=12.0, fault_type="strike_slip")
portfolio = generate_synthetic_portfolio(n_bridges=200, seed=42)
result = run_deterministic(scenario, portfolio, n_realizations=100)
print(print_deterministic_report(result))
```

---

## References

1. Boore, D.M. & Atkinson, G.M. (2008). Ground-Motion Prediction Equations for the Average Horizontal Component of PGA, PGV, and 5%-Damped PSA. _Earthquake Spectra_, 24(1), 99-138.
2. Jayaram, N. & Baker, J.W. (2009). Correlation model for spatially distributed ground-motion intensities. _Earthquake Engineering & Structural Dynamics_, 38(15), 1687-1708.
3. FEMA (2024). _Hazus 6.1 Earthquake Model Technical Manual_. Federal Emergency Management Agency.
4. Basoz, N. & Kiremidjian, A. (1998). Evaluation of Bridge Damage Data from the Loma Prieta and Northridge Earthquakes. MCEER-98-0004.
5. Werner, S.D., et al. (2006). Seismic Risk Analysis of Highway Systems. MCEER-06-0011.
6. Caltrans (1994). The Northridge Earthquake: Post-Earthquake Investigation Report.
7. Worden, C.B., et al. (2018). Spatial and spectral interpolation of ground‐motion intensity measure observations. _BSSA_, 108(2), 866-875.

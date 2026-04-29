# CAT411 — Seismic Bridge Loss Estimation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Methodology: FEMA Hazus 6.1](https://img.shields.io/badge/Methodology-Hazus%206.1-orange.svg)](https://www.fema.gov/hazus)

A modular catastrophe modeling pipeline for **earthquake-induced bridge damage and loss estimation**, implementing the FEMA Hazus 6.1 methodology. Built for the CAT411 course at National Central University.

## What's in this repo

- **`src/`** — the framework itself: hazard, exposure, vulnerability, loss, and pipeline modules (~7,000 LOC, fully typed).
- **`tutorials/`** — six self-contained Jupyter notebooks that walk through the whole pipeline with the 1994 Northridge earthquake as the case study.
- **`Wenyu Chiou/`** (sibling directory) — an LLM-powered Streamlit chat agent that wraps the framework as 11 callable tools.

---

## Quick Start

```bash
git clone https://github.com/WenyuChiou/Cat_framework.git
cd Cat_framework
pip install -r requirements.txt
python main.py --full-analysis
```

For a guided walk-through, jump to [Live Demo](#live-demo-jupyter-notebooks). For the chat agent, see [`../Wenyu Chiou/`](../Wenyu%20Chiou/).

---

## Live Demo (Jupyter Notebooks)

Five-minute setup on a fresh machine. The six notebooks under `tutorials/` cover the whole pipeline (config → hazard → fragility → validation → loss).

### 1. Get the code

```bash
# A. Git clone (recommended if the demo machine has git + internet)
git clone https://github.com/WenyuChiou/Cat_framework.git
cd Cat_framework

# B. USB drive — copy CAT411_framework/ to the demo machine, then:
cd path/to/CAT411_framework
```

### 2. Check Python and install

```bash
python --version                           # must be 3.10 or newer
pip install -r requirements.txt
pip install jupyterlab geopandas contextily
```

If Python is missing, install **3.11** from <https://www.python.org/downloads/> with **"Add Python to PATH"** checked.

### 3. Launch JupyterLab

```bash
cd tutorials
jupyter lab
```

The browser opens at `http://localhost:8888` with the file panel on the left.

### 4. Notebooks to run on stage

Each notebook is self-contained — `Run → Run All Cells`.

| # | Notebook | What it shows | Time |
|---|---|---|---|
| 01 | `01_config_and_data.ipynb` | YAML config, NBI parsing, bridge map | 2 min |
| 02 | `02_hazard_shakemap.ipynb` | ShakeMap → Sa(1.0 s) at each bridge | 1.5 min |
| 04 | `04_fragility.ipynb` | Hazus fragility curves, HWB heatmap, MLE calibration | 2 min |
| 06 | `06_loss_and_cost.ipynb` | FHWA 2024 cost model, expected loss, EP curve | 2 min |

Demo flow: **01 → 02 → 04 → 06** (≈ 10 min). Notebooks 03 (GMPE) and 05 (validation) are kept for Q&A.

### 5. The night before

On your laptop, run the full sequence from a fresh clone, then `File → Save Notebook` for each. The saved output cells (figures, tables, prints) survive even if a cell errors on stage — you can read off the cached numbers.

```bash
git clone https://github.com/WenyuChiou/Cat_framework.git demo_dryrun
cd demo_dryrun && pip install -r requirements.txt
pip install jupyterlab geopandas contextily
cd tutorials && jupyter lab
```

### 6. If something breaks on stage

| Symptom | Fix |
|---|---|
| `pip install` blocked by firewall | Pre-build wheels: `pip download -r requirements.txt -d wheels && pip download jupyterlab geopandas contextily -d wheels`, copy `wheels/` to USB, install with `pip install --no-index --find-links wheels ...` |
| `jupyter: command not found` | Run `python -m jupyterlab` instead |
| Port 8888 in use | `jupyter lab --port 8889`, open the printed URL |
| A cell errors live | Scroll past it — saved output from the dry run is still visible |
| `geopandas` / `contextily` import error | Skip the basemap cell in notebook 01; the rest runs fine |
| Browser does not open | Copy the URL printed by `jupyter lab` into the browser manually |

### 7. Pre-flight checklist (5 min before you go on)

- [ ] `python --version` returns 3.10 or newer
- [ ] `cd CAT411_framework/tutorials && jupyter lab` opens the browser
- [ ] Notebook 01 runs to the end of the first cell
- [ ] Notebook 06 shows the loss number at the bottom
- [ ] PowerPoint is on slide 1

The 582 MB `data/vs30/global_vs30.grd` is **not** required by the tutorials — `data/vs30/california_vs30.npz` (4 MB) is the precomputed subset they actually load. Skip it when packing the USB.

---

## Architecture

```
  ┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────┐     ┌──────────────┐
  │  HAZARD  │────>│ EXPOSURE │────>│VULNERABILITY │────>│ LOSS │────>│   OUTPUT     │
  │ Sa(g) at │     │ 25,000+  │     │  P(DS|IM)    │     │ E[L] │     │  Maps, CSV,  │
  │ each site│     │ bridges  │     │  per bridge  │     │ EP   │     │  dashboards  │
  └──────────┘     └──────────┘     └──────────────┘     └──────┘     └──────────────┘
   hazard.py        exposure.py      fragility.py        loss.py       plotting.py
   gmpe_bssa21.py   data_loader.py   hazus_params.py     engine.py     validation.py
```

Two hazard paths converge at intensity-measure assignment:

- **Path A (ShakeMap)** — interpolates a USGS ShakeMap `grid.xml` to bridge sites (data-conditioned, historical).
- **Path B (GMPE)** — computes Sa via BSSA14/21 from scenario parameters (forward prediction, what-if).

See [`docs/`](docs/) for detailed dependency graphs.

---

## Tutorials

The [`tutorials/`](tutorials/README.md) folder contains **6 self-contained Jupyter notebooks**, one per pipeline stage.

| # | Notebook | Stage | What it covers |
|---|---|---|---|
| 01 | [Config & Data](tutorials/01_config_and_data.ipynb) | Setup | YAML config, NBI parsing, HWB classification, bridge map |
| 02 | [Hazard: ShakeMap](tutorials/02_hazard_shakemap.ipynb) | Hazard A | Sa interpolation to bridge sites, 5 spatial methods |
| 03 | [Hazard: GMPE](tutorials/03_hazard_gmpe.ipynb) | Hazard B | BSSA21 forward prediction with Vs30 enrichment |
| 04 | [Fragility](tutorials/04_fragility.ipynb) | Vulnerability | Hazus curves, 28-class heatmap, MLE calibration |
| 05 | [Validation](tutorials/05_validation.ipynb) | Validation | Attenuation curve, dual-pipeline damage distribution |
| 06 | [Loss & Cost](tutorials/06_loss_and_cost.ipynb) | Loss | FHWA 2024 multi-factor RCV, damage ratios, expected loss |

```bash
pip install jupyterlab geopandas contextily
jupyter lab tutorials/
```

---

## Run Modes

The framework supports three run paths.

### A. CLI pipeline (most common)

```bash
python main.py                              # default config.yaml
python main.py --full-analysis              # end-to-end automated
python main.py --full-analysis --validate   # + 3-level validation
python main.py --probabilistic              # stochastic catalog, EP curve, AAL
```

CLI arguments override `config.yaml`. The full flag list is in `python main.py --help`.

### B. Conversational agent (Streamlit chat UI)

The sibling `Wenyu Chiou/` directory contains an LLM agent that wraps the framework as 11 tools. Ask natural-language questions and get plots, maps, and Word reports inline.

```bash
cd "../Wenyu Chiou"
pip install -r requirements.txt
streamlit run app.py
```

Supports OpenAI, Anthropic, NVIDIA NIM (free), and local Ollama. Set the API key in the sidebar — no `.env` file required for the demo path.

### C. Python API (programmatic)

```python
from src.fragility import damage_state_probabilities
from src.loss import compute_bridge_loss

probs = damage_state_probabilities(sa=0.45, hwb_class="HWB5")
result = compute_bridge_loss(sa=0.45, hwb_class="HWB5",
                             replacement_cost=5_000_000)
print(f"E[Loss] = ${result.expected_loss:,.0f}")
```

---

## Configuration

All analysis parameters live in [`config.yaml`](config.yaml). Key sections:

```yaml
im_source: shakemap        # "shakemap" or "gmpe"
im_type:   SA10             # PGA | SA03 | SA10 | SA30
interpolation:
  method: nearest           # nearest | idw | bilinear | natural_neighbor | kriging
region:                     # study area bounding box
  lat_min: 33.8
  lat_max: 34.6
  lon_min: -118.9
  lon_max: -118.0
calibration:
  global_median_factor: 1.84   # MLE-fit fragility scaling
```

The loader enforces IM-fragility compatibility at load time — using a non-SA10 IM type without `fragility_overrides` raises an immediate error.

### Missing-value defaults

NBI bridge records often have missing fields. The classifier applies these conservative defaults:

| Field | Default | Rationale |
|---|---|---|
| `num_spans` | 1 | Single-span (most conservative) |
| `year_built` | 1960 | Pre-seismic era (conventional design) |
| `structure_length_m` | 30 | Typical short-span bridge |
| `deck_width_m` | 10 | Standard two-lane |
| `material_code` | `"other"` | Maps to HWB28 |
| `vs30` | 760 m/s | NEHRP B/C boundary (rock) |

ShakeMap NaN outside the convex hull is filled by nearest-neighbor.

---

## Validation & Calibration

The framework ships with a **3-level validation** against the 1994 Northridge earthquake:

| Level | What is validated | Headline finding |
|---|---|---|
| L1: GMPE | BSSA21 vs 185 station recordings | Near-field underestimation ≈46% (point-source); matches USGS GMPE r = 0.988 |
| L2: Event | Aggregate damage vs Basoz (1998) survey of 1,600 bridges | Baseline over-predicts ≈2.5×; calibration recovers 9.5% vs 10.6% observed |
| L3: Bridge | Per-bridge predicted vs observed (113 confirmed records) | 28.3% exact-match (data-quality limited) |

Run validation: `python main.py --full-analysis --validate`.

### MLE fragility calibration

Default Hazus parameters over-predict Northridge damage (27.2% vs 10.6% observed). The framework fits two global parameters against Basoz & Kiremidjian (1998):

| Parameter | Hazus default | Calibrated | Effect |
|---|---|---|---|
| `k` (median scale) | 1.00 | **1.84** | Curves shift right |
| `beta` (dispersion) | 0.60 | **0.26** | Steeper damage transitions |
| Damage fraction | 27.2% | **9.5%** | vs 10.6% observed |

```bash
python scripts/run_calibration.py        # fit k, beta
python scripts/anik_validation.py        # validate with calibrated params
```

To use the calibrated parameters in the main pipeline, set `calibration.global_median_factor: 1.8432` in `config.yaml`.

For methodology and diagnostic plots, see [Tutorial 04](tutorials/04_fragility.ipynb) §7 and [Tutorial 05](tutorials/05_validation.ipynb).

---

## Bring Your Own Data

The framework can analyze any earthquake plus bridge inventory.

### Bridge inventory (CSV, UTF-8, WGS84)

| Column | Type | Required |
|---|---|---|
| `structure_number` | string | yes |
| `latitude` | float | yes |
| `longitude` | float | yes |
| `year_built` | int | optional |
| `material_code` | string | optional |
| `num_spans` | int | optional |
| `hwb_class` | string | optional (skips auto-classification) |

### Earthquake scenario — pick one

- **ShakeMap:** `im_source: shakemap` + USGS event ID
- **GMPE:** `im_source: gmpe` + magnitude, depth, epicenter, fault style
- **Pre-computed Sa:** CSV with `structure_number` + `sa1s` columns (units of g)

### Observed damage (optional, for validation)

| Column | Type | Allowed values |
|---|---|---|
| `structure_number` | string | must match inventory |
| `observed_damage` | string | `none, slight, moderate, extensive, complete` |

See `CAT411_Technical_Documentation.docx` §13 for full specs.

---

## Project Structure

```
CAT411_framework/
├── main.py                     Entry point
├── config.yaml                 Analysis configuration
├── requirements.txt            Python dependencies
├── src/                        All source modules
├── data/
│   ├── CA24.txt                NBI inventory (California, 25,000+)
│   ├── grid.xml                USGS ShakeMap (1994 Northridge)
│   ├── stationlist.json        Seismic station recordings (1,378)
│   ├── nbi_classified_2024.csv Pre-classified inventory (cached)
│   ├── vs30/california_vs30.npz   4 MB precomputed Vs30 grid
│   └── validation/             Observed-damage datasets (113 records)
├── tutorials/                  6 Jupyter notebook walkthroughs
├── docs/                       Architecture diagrams & planning docs
├── scripts/                    Calibration / validation utilities
└── output/                     Generated results (plots, CSV, reports)
```

---

## Module Reference

| Layer | Module | Purpose |
|---|---|---|
| **Data** | `config.py` | YAML config loader with fail-fast validation |
| | `hazus_params.py` | Hazus 6.1 fragility parameters (28 HWB classes) |
| | `gmpe_base.py` | GMPE protocol + model registry |
| **Core** | `hazard.py` | BA08 GMPE, spatial correlation, GMF generation |
| | `gmpe_bssa21.py` | BSSA14/21 NGA-West2 (108 periods, auto-registered) |
| | `bridge_classes.py` | NBI → Hazus classification decision tree |
| | `interpolation.py` | 5 spatial interpolation methods |
| **Domain** | `fragility.py` | Lognormal fragility CDF, damage-state probabilities |
| | `exposure.py` | Bridge portfolio construction, replacement costs |
| | `data_loader.py` | ShakeMap XML, NBI text, station JSON parsers |
| **Pipeline** | `loss.py` | Damage-to-loss, EP curve, AAL |
| | `engine.py` | Pipeline orchestrator (deterministic + probabilistic) |
| | `calibration.py` | MLE fragility calibration |
| **Output** | `plotting.py` | 17 visualization functions |
| | `validation.py` | 3-level validation framework |
| | `hazard_download.py` | USGS API client for ShakeMap data |

All source modules live under `src/`. Each layer depends only on the layers above it.

---

## Limitations

| Area | Limitation | Mitigation path |
|---|---|---|
| GMPE | Point-source R_JB approximation | Future: finite-fault geometry |
| Fragility | Global 2-parameter calibration only | Per-class calibration with class-level damage counts |
| Site effects | USGS Vs30 grid; default fallback if unavailable | `vs30_provider.py` per-bridge lookup |
| Loss | Fixed Hazus damage ratios | Configurable via `fragility_overrides` |
| Temporal | Static inventory snapshot | Future: time-dependent fragility |
| Network | No connectivity / cascading failure | Out of scope for bridge-level CAT |
| Validation | L2 uses 2024 NBI vs 1994 observations | Future: use 1994 NBI for consistent comparison |

---

## Citation & References

If you use this framework, please cite:

```
Chiou, W., Albayrak, K., & Kedarsetty, S. (2026). CAT411 — A modular
catastrophe modeling framework for earthquake-induced bridge loss
estimation. CAT411 capstone project, National Central University.
https://github.com/WenyuChiou/Cat_framework
```

### Foundational references

1. Boore, D.M., Stewart, J.P., Seyhan, E. & Atkinson, G.M. (2014). NGA-West2 equations for predicting PGA, PGV, and 5%-damped PSA for shallow crustal earthquakes. *Earthquake Spectra*, 30(3), 1057–1085.
2. Boore, D.M. & Atkinson, G.M. (2008). Ground-motion prediction equations for the average horizontal component of PGA, PGV, and 5%-damped PSA. *Earthquake Spectra*, 24(1), 99–138.
3. Jayaram, N. & Baker, J.W. (2009). Correlation model for spatially distributed ground-motion intensities. *Earthquake Engineering & Structural Dynamics*, 38(15), 1687–1708.
4. FEMA (2022). *Hazus 6.1 Earthquake Model Technical Manual.* Federal Emergency Management Agency.
5. Basoz, N. & Kiremidjian, A. (1998). *Evaluation of Bridge Damage Data from the Loma Prieta and Northridge Earthquakes.* MCEER-98-0004.
6. Werner, S.D., et al. (2006). *Seismic Risk Analysis of Highway Systems.* MCEER-06-0011.
7. Worden, C.B., et al. (2018). Spatial and spectral interpolation of ground-motion intensity measure observations. *BSSA*, 108(2), 866–875.
8. FHWA (2024). National Bridge Inventory (NBI) data. https://www.fhwa.dot.gov/bridge/nbi.cfm

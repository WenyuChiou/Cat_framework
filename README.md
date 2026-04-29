# CAT411 -- Seismic Bridge Loss Estimation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Methodology: FEMA Hazus 6.1](https://img.shields.io/badge/Methodology-Hazus%206.1-orange.svg)](https://www.fema.gov/hazus)

A modular catastrophe modeling pipeline for **earthquake-induced bridge damage and loss estimation**, implementing the FEMA Hazus 6.1 methodology. Built for the CAT411 course at National Central University.

---

## Live Demo (Jupyter Notebooks)

Five-minute setup on a fresh machine. The six notebooks under `tutorials/` cover the whole pipeline (config → hazard → fragility → validation → loss).

### 1. Get the code

Pick one:

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

## Features

- **Dual hazard paths** -- ShakeMap (historical) and GMPE forward prediction (BSSA14/21, 108 spectral periods)
- **25,000+ bridge inventory** -- California NBI with automatic HWB classification (Hazus Table 7.3)
- **5 spatial interpolation methods** -- nearest, IDW, bilinear, natural neighbor, kriging
- **Lognormal fragility model** -- 28 HWB classes, skew-angle modification, per-class calibration
- **Probabilistic engine** -- Monte Carlo simulation with spatially correlated ground motion fields
- **3-level validation** -- GMPE accuracy, event-level damage distribution, per-bridge comparison
- **Batteries included** -- all data, parameters, and the 1994 Northridge case study ship with the repo

---

## Architecture

```
  ┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────┐     ┌──────────────┐
  │  HAZARD  │────>│ EXPOSURE │────>│VULNERABILITY │────>│ LOSS │────>│  OUTPUT       │
  │          │     │          │     │              │     │      │     │              │
  │ Sa(g) at │     │ 25,000+  │     │ P(DS|IM)     │     │E[L]  │     │ Maps, CSV,   │
  │ each site│     │ bridges  │     │ per bridge   │     │EP,AAL│     │ dashboards   │
  └──────────┘     └──────────┘     └──────────────┘     └──────┘     └──────────────┘
   hazard.py        exposure.py      fragility.py        loss.py       plotting.py
   gmpe_bssa21.py   data_loader.py   hazus_params.py     engine.py     validation.py
   interpolation.py bridge_classes.py
```

Two hazard paths converge at intensity measure assignment:

- **Path A (ShakeMap)** -- interpolates USGS ShakeMap grid.xml to bridge sites (data-conditioned, historical)
- **Path B (GMPE)** -- computes Sa via BSSA14/21 from scenario parameters (forward prediction, what-if)

See [`docs/`](docs/) for detailed dependency graphs and pipeline diagrams.

---

## Quick Start

### Prerequisites

- Python 3.10+
- NumPy, SciPy, pandas, matplotlib, requests, PyYAML

```bash
pip install -r requirements.txt
```

### 🤖 Conversational Agent (Streamlit Chat UI)

An LLM-powered chat interface that wraps the framework APIs as 11 tools.
Ask natural-language questions about earthquake bridge risk and get
results, plots, interactive maps, and exportable Word reports inline.

```bash
# 1. Copy template and add an API key (NVIDIA free, OpenAI, Anthropic, or local Ollama)
cp .env.example .env

# 2. Launch the web app
streamlit run app.py
```

**Supported LLM providers:**

| Provider | Cost | Setup |
|----------|------|-------|
| **NVIDIA NIM** (recommended for demo) | Free | Get key at [build.nvidia.com](https://build.nvidia.com/) |
| OpenAI | ~$0.01/query | `OPENAI_API_KEY` |
| Anthropic | ~$0.01/query | `ANTHROPIC_API_KEY` |
| Ollama (local) | Free | `ollama pull mistral-small:24b` |

**Example queries:**
- *"What are the fragility parameters for HWB5 at Sa=0.5g?"*
- *"Run a M6.7 earthquake scenario at Northridge with uncertainty analysis"*
- *"Show me a damage map for M6.7 Northridge with 100 bridges"*
- *"Export a Word report for a M7.0 scenario near LA"*

The agent auto-detects the framework layout — it works whether placed
inside the framework directory or as a sibling.

### Run the Northridge Case Study

```bash
# Full analysis with ShakeMap data
python main.py --full-analysis

# Full analysis + validation against observed damage
python main.py --full-analysis --validate

# GMPE-based forward scenario
python main.py --full-analysis --config config_gmpe.yaml
```

### Minimal Python Example

```python
from src.fragility import damage_state_probabilities
from src.loss import compute_bridge_loss

probs = damage_state_probabilities(sa=0.45, hwb_class="HWB5")
result = compute_bridge_loss(sa=0.45, hwb_class="HWB5", replacement_cost=5_000_000)
print(f"E[Loss] = ${result.expected_loss:,.0f}")
```

For detailed walkthroughs, see the [Tutorials](#tutorials) section.

---

## Run Modes & CLI

| Command | Description |
|---------|-------------|
| `python main.py` | Default analysis using `config.yaml` |
| `python main.py --full-analysis` | End-to-end automated analysis (recommended) |
| `python main.py --full-analysis --validate` | Full analysis + 3-level validation |
| `python main.py --pipeline` | Deterministic CAT model (synthetic portfolio) |
| `python main.py --probabilistic` | Stochastic catalog, EP curve, AAL |
| `python main.py --fragility-only` | Generate fragility curve plots only |
| `python main.py --download-hazard` | Download USGS ShakeMap data |
| `python scripts/run_validation_gmpe.py` | Standalone validation (no pipeline needed) |

### Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `config.yaml` | YAML configuration file |
| `--im-type TYPE` | `SA10` | Intensity measure: `PGA`, `SA03`, `SA10`, `SA30` |
| `--validate` | off | Enable 3-level validation |
| `--n-bridges N` | `100` | Synthetic portfolio size (pipeline mode) |
| `--n-realizations N` | `50` | Monte Carlo realizations per event |
| `--n-events N` | `50` | Stochastic events (probabilistic mode) |

CLI arguments override `config.yaml` settings.

---

## Module Reference

| Layer | Module | Purpose |
|-------|--------|---------|
| **Data** | `config.py` | YAML config loader with fail-fast validation |
| | `hazus_params.py` | Hazus 6.1 fragility parameters (28 HWB classes) |
| | `gmpe_base.py` | GMPE protocol + model registry |
| **Core** | `hazard.py` | BA08 GMPE, spatial correlation, GMF generation |
| | `gmpe_bssa21.py` | BSSA14/21 NGA-West2 (108 periods, auto-registered) |
| | `bridge_classes.py` | NBI-to-Hazus classification decision tree |
| | `interpolation.py` | 5 spatial interpolation methods |
| **Domain** | `fragility.py` | Lognormal fragility CDF, damage state probabilities |
| | `exposure.py` | Bridge portfolio construction, replacement costs |
| | `data_loader.py` | ShakeMap XML, NBI text, station JSON parsers |
| **Pipeline** | `loss.py` | Damage-to-loss, EP curve, AAL computation |
| | `engine.py` | Pipeline orchestrator (deterministic + probabilistic) |
| | `calibration.py` | MLE fragility calibration (k, beta) against observed data |
| **Output** | `plotting.py` | 17 visualization functions |
| | `validation.py` | 3-level validation framework (14 diagnostic plots) |
| | `hazard_download.py` | USGS API client for ShakeMap data |

All source modules live under `src/`. Each layer depends only on the layers above it.

---

## Configuration

All analysis parameters are centralized in [`config.yaml`](config.yaml). Key sections:

```yaml
im_source: shakemap        # "shakemap" or "gmpe"
im_type:   SA10             # PGA | SA03 | SA10 | SA30
interpolation:
  method: nearest           # nearest | idw | bilinear | natural_neighbor | kriging
region:                     # Study area bounding box
  lat_min: 33.8
  lat_max: 34.6
  lon_min: -118.9
  lon_max: -118.0
```

The config loader enforces IM-fragility compatibility at load time -- using a non-SA10 IM type without providing `fragility_overrides` raises an immediate error. See the full annotated config file and [Tutorial 01](tutorials/01_config_and_data.ipynb) for all options.

---

## Project Structure

```
CAT411_framework/
├── main.py                     Entry point
├── config.yaml                 Analysis configuration
├── requirements.txt            Python dependencies
├── src/                        All source modules (see Module Reference)
├── data/
│   ├── CA24.txt                NBI bridge inventory (California, 25,000+)
│   ├── grid.xml                USGS ShakeMap (1994 Northridge)
│   ├── stationlist.json        Seismic station recordings (1,378 stations)
│   ├── nbi_classified_2024.csv Pre-classified bridge inventory (NBI 2024, generated)
│   └── validation/             Validation datasets (113 confirmed observations)
├── tutorials/                  6 Jupyter notebook walkthroughs
├── docs/                       Architecture diagrams & planning docs
├── scripts/                    Standalone utility scripts
└── output/                     Generated results (plots, CSV, reports)
    ├── analysis/               ShakeMap-based analysis results
    ├── fragility/              Fragility curve library (14 HWB classes)
    ├── validation/             3-level validation plots & CSV
    └── scenario/               Probabilistic scenario outputs
```

---

## Tutorials

The [`tutorials/`](tutorials/README.md) folder contains **6 self-contained Jupyter notebooks**, one per pipeline stage, with inline outputs, DataFrames, and plots.

```bash
pip install jupyterlab geopandas contextily    # additional dependencies
jupyter lab tutorials/
```

| # | Notebook | Stage | What You Will Learn |
|---|----------|-------|---------------------|
| 01 | [Config & Data Loading](tutorials/01_config_and_data.ipynb) | Setup | Load config, parse ShakeMap/NBI, classify bridges to HWB classes |
| 02 | [Hazard: ShakeMap](tutorials/02_hazard_shakemap.ipynb) | Hazard (Path A) | Interpolate Sa to bridge sites, spatial IM maps |
| 03 | [Hazard: GMPE](tutorials/03_hazard_gmpe.ipynb) | Hazard (Path B) | Compute Sa via BSSA21, compare GMPE vs ShakeMap |
| 04 | [Fragility Curves](tutorials/04_fragility.ipynb) | Vulnerability | Fragility parameters, HWB lookup, curve plotting, MLE calibration |
| 05 | [Validation](tutorials/05_validation.ipynb) | Validation | Attenuation curve, dual-pipeline damage distribution vs Basoz 1998 |
| 06 | [Loss & Cost](tutorials/06_loss_and_cost.ipynb) | Loss | FHWA 2024 RCV model, damage ratios, expected loss per bridge |

Each notebook is self-contained -- no prior notebook execution required. See [`tutorials/README.md`](tutorials/README.md) for details.

---

## Validation Summary

The framework includes a **3-level validation** tested against the 1994 Northridge earthquake:

| Level | What is Validated | Key Finding |
|-------|-------------------|-------------|
| **L1: GMPE** | BSSA21 predictions vs 185 seismic station recordings | Near-field underestimation (~46%) due to point-source approximation; implementation matches ShakeMap GMPE (r = 0.988) |
| **L2: Event** | Aggregate damage distribution vs Basoz (1998) survey of 1,600 bridges | Baseline over-predicts ~2.5x; **MLE calibration (k=1.84) reduces to 9.5% vs 10.6% observed** |
| **L3: Bridge** | Per-bridge predicted vs observed for 113 confirmed damage records | 28.3% exact match; supplementary level due to data quality limitations |

Run validation: `python main.py --full-analysis --validate`

For methodology details, data sources, and diagnostic plots, see [Tutorial 05](tutorials/05_validation.ipynb).

---

## MLE Fragility Calibration

Default Hazus fragility parameters over-predict Northridge bridge damage (27.2% predicted vs 10.6% observed). The framework includes an MLE calibration module (`src/calibration.py`) that fits two global parameters against Basoz & Kiremidjian (1998) observed counts (N=1,600):

| Parameter | HAZUS Default | Calibrated | Meaning |
|-----------|--------------|------------|---------|
| **k** (median scale factor) | 1.0 | 1.84 | All fragility medians x1.84; curves shift right |
| **beta** (dispersion) | 0.6 | 0.26 | Steeper curves; sharper damage transitions |
| **Damage fraction** | 27.2% | 9.5% | vs observed 10.6% |

```bash
# Run calibration
python scripts/run_calibration.py

# Run validation with calibrated parameters (default)
python scripts/anik_validation.py

# Apply to main pipeline -- add to config.yaml:
#   calibration:
#     global_median_factor: 1.8432
```

Ground truth: Basoz & Kiremidjian (1998) aggregate damage counts from the 1994 Northridge earthquake, stored in `src/northridge_case.py`. See [Tutorial 04](tutorials/04_fragility.ipynb) Section 7 for details.

---

## Missing Value Handling

NBI bridge records may contain missing fields. The framework applies conservative defaults during classification:

| Field | Default | Rationale |
|-------|---------|-----------|
| `num_spans` | 1 | Single-span assumption (most conservative) |
| `year_built` | 1960 | Pre-seismic era (conventional design) |
| `structure_length_m` | 30 m | Typical short-span bridge |
| `deck_width_m` | 10 m | Standard two-lane width |
| `material_code` | "other" | Maps to HWB28 (general category) |
| `vs30` | 760 m/s | NEHRP B/C boundary (rock) |

ShakeMap interpolation NaN (outside convex hull) is filled with nearest-neighbor. See Technical Documentation Section 6.2 for full details.

---

## Using Your Own Data

The framework can analyze any earthquake + bridge inventory, not just Northridge.

### Required Input

1. **Bridge inventory CSV** with at minimum:

   | Column | Type | Example |
   |--------|------|---------|
   | `structure_number` | string | `53 0012` |
   | `latitude` | float | `34.0522` |
   | `longitude` | float | `-118.2437` |

   Optional: `year_built`, `material_code`, `design_code`, `num_spans`, `hwb_class` (if provided, skips auto-classification).

2. **Earthquake scenario** (choose one):
   - **ShakeMap:** `im_source: shakemap` + USGS event ID
   - **GMPE:** `im_source: gmpe` + magnitude, depth, epicenter, fault type
   - **Pre-computed Sa CSV:** with `structure_number` + `sa1s` columns

3. **Observed damage** (optional, for validation/calibration):

   | Column | Type | Allowed values |
   |--------|------|---------------|
   | `structure_number` | string | must match inventory |
   | `observed_damage` | string | none, slight, moderate, extensive, complete |

### Data Format Requirements

- CSV with UTF-8 encoding
- WGS84 coordinates (decimal degrees)
- Sa in units of g
- Damage states as lowercase strings

See Technical Documentation Section 13 for complete specifications and examples.

---

## Known Limitations

| Area | Limitation | Mitigation Path |
|------|-----------|-----------------|
| GMPE | Point-source R_JB approximation | Future: finite-fault geometry |
| Fragility | Global 2-parameter calibration; intermediate DS ordering reversal (<0.5 bridges) | Per-class calibration with class-level damage counts |
| Site effects | USGS Vs30 grid; default fallback if unavailable | `vs30_provider.py` per-bridge lookup |
| Loss | Fixed Hazus damage ratios | Configurable via `fragility_overrides` |
| Temporal | Static inventory snapshot | Future: time-dependent fragility |
| Network | No connectivity / cascading failure analysis | Out of scope for bridge-level CAT |
| Validation | L2 uses 2024 NBI (2,953 bridges) vs 1994 observations (1,600 bridges) | Future: use 1994 NBI for consistent comparison |
| Simplified GMPEs | 7 simplified models use representative coefficients, not official | Use `bssa21` for production; simplified models for teaching only |

---

## References

1. Boore, D.M., Stewart, J.P., Seyhan, E. & Atkinson, G.M. (2014). NGA-West2 Equations for Predicting PGA, PGV, and 5%-Damped PSA for Shallow Crustal Earthquakes. *Earthquake Spectra*, 30(3), 1057-1085. https://doi.org/10.1193/070113EQS184M
2. Boore, D.M. & Atkinson, G.M. (2008). Ground-Motion Prediction Equations for the Average Horizontal Component of PGA, PGV, and 5%-Damped PSA. *Earthquake Spectra*, 24(1), 99-138. https://doi.org/10.1193/1.2830434
3. Jayaram, N. & Baker, J.W. (2009). Correlation model for spatially distributed ground-motion intensities. *Earthquake Engineering & Structural Dynamics*, 38(15), 1687-1708. https://doi.org/10.1002/eqe.922
4. FEMA (2024). *Hazus 6.1 Earthquake Model Technical Manual*. Federal Emergency Management Agency. https://www.fema.gov/hazus
5. Basoz, N. & Kiremidjian, A. (1998). Evaluation of Bridge Damage Data from the Loma Prieta and Northridge Earthquakes. MCEER-98-0004.
6. Werner, S.D., et al. (2006). Seismic Risk Analysis of Highway Systems. MCEER-06-0011.
7. Caltrans (1994). The Northridge Earthquake: Post-Earthquake Investigation Report.
8. Worden, C.B., et al. (2018). Spatial and Spectral Interpolation of Ground-Motion Intensity Measure Observations. *BSSA*, 108(2), 866-875. https://doi.org/10.1785/0120170201
9. FHWA (2024). National Bridge Inventory (NBI) Data. https://www.fhwa.dot.gov/bridge/nbi.cfm

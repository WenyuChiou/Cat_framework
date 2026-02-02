# CAT411 — Bridge Earthquake Catastrophe Modeling Framework

A complete catastrophe (CAT) modeling pipeline for seismic risk assessment of highway bridges, built around the **FEMA Hazus 6.1** methodology and calibrated against the **1994 Northridge earthquake**.

## Pipeline Overview

```
EarthquakeScenario (Mw, lat, lon, depth)
        │
        ▼
   src/hazard.py ── GMPE (Boore-Atkinson 2008) → Sa(1.0s)
        │            Site amplification (Vs30)
        │            Spatial correlation (Jayaram-Baker 2009)
        │            Correlated ground motion fields (Cholesky)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  src/exposure.py               src/fragility.py
  Bridge inventory              P[DS ≥ ds | Sa]
  Replacement costs             Lognormal CDF model
  Synthetic portfolio           Hazus Table 7.9
        │                              │
        └──────────┬───────────────────┘
                   ▼
             src/loss.py
             Hazus damage ratios (Table 7.11)
             E[Loss] = Σ P(ds) × DR(ds) × RC
             EP curves, AAL
                   │
                   ▼
            src/engine.py
            Deterministic: single scenario → damage & loss
            Probabilistic: stochastic catalog → EP curve, AAL
```

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run modes

```bash
# Full deterministic CAT model pipeline (Northridge Mw 6.7 scenario)
python main.py --pipeline

# Probabilistic analysis — stochastic event catalog → EP curve + AAL
python main.py --probabilistic

# Fragility curves only (no hazard/loss computation)
python main.py --fragility-only

# Download USGS ShakeMap & FHWA NBI data, then run full data analysis
python main.py --download
python main.py
```

### Configuration flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n-bridges` | 100 | Synthetic portfolio size |
| `--n-realizations` | 50 | Monte Carlo ground motion realizations |
| `--n-events` | 50 | Stochastic catalog size (probabilistic mode) |

Example:

```bash
python main.py --pipeline --n-bridges 200 --n-realizations 100
python main.py --probabilistic --n-bridges 200 --n-events 100 --n-realizations 30
```

## Module Reference

### `src/hazard.py` — Ground Motion Prediction

- **Boore & Atkinson (2008)** GMPE for Sa(1.0s), with magnitude, distance, and site amplification terms
- **Jayaram & Baker (2009)** spatial correlation model (b = 40.7 km for T = 1.0 s)
- Correlated ground motion field generation via Cholesky decomposition of the correlation matrix
- Inter-event (τ) and intra-event (σ) variability

### `src/exposure.py` — Bridge Inventory & Financial Data

- `BridgeExposure` dataclass with location, HWB class, material, dimensions, and replacement cost
- Unit cost estimation by material ($2,500–$3,200/m² of deck area)
- Synthetic portfolio generator with realistic Southern California class distribution
- NBI-to-exposure converter for use with real FHWA bridge inventory data

### `src/fragility.py` — Vulnerability (Damage Probability)

- Lognormal CDF fragility model: P[DS ≥ ds | Sa] = Φ[(ln Sa − ln median) / β]
- 14 Hazus bridge classes (HWB1–HWB28) with parameters from Table 7.9
- Discrete damage state probabilities (none / slight / moderate / extensive / complete)
- Skew angle modification factor

### `src/loss.py` — Damage-to-Loss Translation

- Hazus damage ratios: slight 3%, moderate 8%, extensive 25%, complete 100%
- Expected loss: E[Loss] = Σ P(ds) × DR(ds) × Replacement Cost
- Loss Exceedance Probability (EP) curve with Poisson assumption
- Average Annual Loss (AAL) = Σ λᵢ × Lᵢ

### `src/engine.py` — Pipeline Orchestrator

- **Deterministic mode**: single earthquake → N ground motion realizations → mean/std loss
- **Probabilistic mode**: Gutenberg-Richter stochastic catalog → per-event losses → EP curve + AAL
- Pre-defined Northridge scenario (Mw 6.7, 34.213°N, 118.537°W, 18.4 km depth)

### Supporting modules

| Module | Purpose |
|--------|---------|
| `src/hazus_params.py` | Hazus 6.1 Table 7.9 fragility parameters |
| `src/bridge_classes.py` | HWB classification logic & dataclass |
| `src/data_loader.py` | USGS ShakeMap & FHWA NBI download/parse |
| `src/northridge_case.py` | 1994 Northridge observed damage data |
| `src/plotting.py` | All visualization functions |

## Output

Generated plots are saved to `output/`:

| Plot | Mode | Description |
|------|------|-------------|
| `ground_motion_field.png` | pipeline | Sa(1.0s) scatter map at bridge sites |
| `loss_by_class.png` | pipeline | Expected loss bar chart by HWB class |
| `portfolio_damage.png` | pipeline | Damage state distribution (stacked bar) |
| `ep_curve.png` | probabilistic | Loss exceedance probability & return period |
| `fragility_HWB*.png` | fragility-only | Individual class fragility curves |
| `comparison_*.png` | fragility-only | Cross-class comparison |
| `damage_distribution_*.png` | fragility-only | Stacked bar damage distributions |
| `northridge_scenario.png` | fragility-only | Fragility with observed PGA overlay |

## Data Sources

| Dataset | Source | File |
|---------|--------|------|
| ShakeMap grid | USGS (ci3144585) | `data/grid.xml` |
| Station recordings | USGS (ci3144585) | `data/stationlist.json` |
| Bridge inventory | FHWA NBI 2024 | `data/CA24.txt` |

Run `python main.py --download` to fetch these files automatically.

## References

- Boore, D.M. & Atkinson, G.M. (2008). Ground-Motion Prediction Equations for the Average Horizontal Component of PGA, PGV, and 5%-Damped PSA. *Earthquake Spectra*, 24(1), 99–138.
- Jayaram, N. & Baker, J.W. (2009). Correlation model for spatially distributed ground-motion intensities. *EESD*, 38(15), 1687–1708.
- FEMA (2024). *Hazus 6.1 Earthquake Model Technical Manual*.
- Basoz, N. & Kiremidjian, A. (1998). Evaluation of Bridge Damage Data from the Loma Prieta and Northridge Earthquakes. MCEER-98-0004.

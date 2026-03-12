# CAT411 Tutorials

Step-by-step Jupyter notebooks covering each stage of the seismic bridge loss estimation pipeline, using the **1994 Northridge earthquake** as a case study.

## Prerequisites

```bash
pip install -r ../requirements.txt
pip install jupyterlab geopandas contextily
```

## Notebooks

| # | Notebook | Pipeline Stage | Description |
|---|----------|---------------|-------------|
| 01 | [Config & Data Loading](01_config_and_data.ipynb) | Setup | Load `config.yaml`, parse ShakeMap grid and NBI inventory, classify bridges to HWB classes, bridge location map with basemap |
| 02 | [Hazard: ShakeMap](02_hazard_shakemap.ipynb) | Hazard (Path A) | Interpolate Sa(1.0s) to bridge sites via nearest-neighbor and kriging, spatial IM map with OpenStreetMap basemap |
| 03 | [Hazard: GMPE](03_hazard_gmpe.ipynb) | Hazard (Path B) | Compute Sa(1.0s) via BSSA21 GMPE with Vs30 enrichment, compare GMPE vs ShakeMap predictions |
| 04 | [Fragility Curves](04_fragility.ipynb) | Vulnerability | Load fragility parameter database (CSV), NBI → HWB → parameter lookup workflow, plot fragility curves, heatmap of 28 HWB classes, portfolio damage distribution |
| 05 | [Validation](05_validation.ipynb) | Validation | L1: BSSA21 attenuation curve vs 185 seismic stations; L2: dual-pipeline (ShakeMap + GMPE) damage distribution vs Basoz 1998 observations |

## Pipeline Flow

```
01 Config & Data ──> 02 ShakeMap ──┐
                     03 GMPE    ──┤──> 04 Fragility ──> 05 Validation
                                  │
                          (two hazard paths converge)
```

## How to Run

```bash
cd tutorials/
jupyter lab
```

Each notebook is **self-contained** — it loads its own data from `config.yaml` and the `data/` folder. No prior notebook execution is required.

## Important Notes

- **NBI Data Vintage:** All tutorials use the 2024 NBI (`CA24.txt`, ~2,953 bridges) as a proxy for the 1994 bridge stock (~1,600 bridges). See disclaimers within each notebook.
- **Replacement Cost:** The framework estimates replacement costs using a Hazus unit-cost model — these are proxy values, not market valuations. A dedicated loss tutorial is not included because NBI does not provide actual replacement cost values (RCV).
- **Basemap Dependencies:** Notebooks 01 and 02 require `geopandas` and `contextily` for OpenStreetMap basemaps.

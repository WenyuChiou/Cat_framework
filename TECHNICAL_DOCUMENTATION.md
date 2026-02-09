# CAT411 — Technical Documentation (Updated)

## 1. Introduction

CAT411 is a Bridge Earthquake Catastrophe Modeling Framework designed for seismic risk assessment of highway bridges. It implements Hazard, Exposure, Vulnerability, and Loss modules.

## 2. Hazard Module

The Hazard module computes ground motion intensity at bridge sites.

### 2.1 USGS Data Integration

The framework now supports direct download of USGS hazard data:

- **ShakeMap**: Downloads earthquake-specific intensity maps (grid.xml, shape.zip).
- **NSHMP Hazard Curves**: Downloads probabilistic seismic hazard curves for specific coordinates (integrated via NSHMP API).

### 2.2 CLI Usage

```bash
python main.py --download-hazard [--hazard-event <EVENT_ID>]
```

- Default event: `ci3144585` (1994 Northridge Earthquake).
- Auto-processes grid data and maps intensities to the bridge inventory.

### 2.3 Visualizations

The framework generates several maps in the `output/` directory:

- `shakemap_full_area.png`: Heatmap of the entire downloaded USGS ShakeMap grid.
- `bridge_site_ground_motion.png`: Scatter plot of intensities at specific bridge locations.
- `real_portfolio_damage.png`: Portfolio-wide damage state distribution based on real data.

## 3. Exposure Module

Parses FHWA NBI (National Bridge Inventory) data.

- **Data Source**: `CA24.txt` (California 2024 NBI data).
- **Classification**: Bridges are automatically classified into 14 Hazus HWB classes based on material, design, and structural parameters.

## 4. Vulnerability Module

Implements FEMA Hazus 6.1 fragility curves.

- Computes exceedance probabilities for five damage states: None, Slight, Moderate, Extensive, and Complete.

## 5. Loss Module

Translates damage probabilities into economic loss.

- Uses Hazus Table 7.11 damage ratios.
- Computes expected loss per bridge and aggregated portfolio loss.

## 6. Project Structure

- `src/hazard_download.py`: Core logic for USGS API communication.
- `src/plotting.py`: Enhanced with `plot_shakemap_grid` for spatial visualization.
- `main.py`: Entry point with integrated download-to-analysis pipeline.
- `data/hazard/usgs/`: Storage for downloaded hazard products.

---

_Note: This document replaces the legacy CAT411_Technical_Documentation.docx for consistent versioning with the codebase._

# NBI + ShakeMap Pipeline (1994 Northridge)

## Overview
This document describes the end-to-end workflow for pulling NBI (bridge inventory) and USGS ShakeMap (hazard) data into **CAT411_framework**, cleaning it, and producing outputs that downstream analysis can use without manual rework. The pipeline is designed for reproducibility: it uses a single config file, writes logs/metadata, and keeps raw data intact.

Data sources:
- **NBI**: FHWA annual ASCII release, **Delimited format** (comma-separated, single-quote text qualifier). The default year is **1994** for the Northridge case.
- **ShakeMap**: USGS GeoJSON detail API for the Northridge event (**`ci3144585`**). The pipeline downloads `grid.xml`, `shape.zip`, and `info.json`.

Directory layout:
- `data/nbi/raw/`: raw NBI zip + extracted TXT/CSV
- `data/nbi/clean/`: cleaned CSV (trimmed strings, normalized missing values)
- `data/nbi/curated/`: curated CSV for downstream use
- `data/nbi/meta/`: pipeline config + metadata
- `data/nbi/logs/`: run logs
- `data/hazard/usgs/shakemap/raw/`: downloaded ShakeMap files
- `data/hazard/usgs/shakemap/meta/`: ShakeMap metadata
- `data/hazard/usgs/shakemap/logs/`: ShakeMap run logs

## Download + Ingest
The main entry point is `broker/utils/nbi_ingest.py`. It reads settings from `data/nbi/meta/nbi_pipeline.json`, then:
1. Downloads NBI 1994 (delimited, all states single file)
2. Downloads USGS Northridge ShakeMap files
3. **Unzips the NBI zip into `data/nbi/raw/`**
4. Produces cleaned + curated outputs + metadata/logs

Commands:
- Download only:
```
python broker/utils/nbi_ingest.py --download --download-only
```
- Download + ingest:
```
python broker/utils/nbi_ingest.py --download
```
- Override year/event:
```
python broker/utils/nbi_ingest.py --download --nbi-year 1994 --usgs-event ci3144585
```

## What Gets Written (and Where)
After a successful `--download` run, you should see:
- `data/nbi/raw/`:
  - `nbi_1994_delimited_all_states_single.zip`
  - extracted TXT/CSV (FHWA often ships TXT inside the zip)
- `data/nbi/clean/nbi_latest_clean.csv`
- `data/nbi/curated/nbi_latest_curated.csv`
- `data/nbi/meta/nbi_latest_meta.json`
- `data/nbi/logs/nbi_latest_run.log`

For ShakeMap downloads:
- `data/hazard/usgs/shakemap/raw/grid.xml`
- `data/hazard/usgs/shakemap/raw/shape.zip`
- `data/hazard/usgs/shakemap/raw/info.json`
- `data/hazard/usgs/shakemap/meta/shakemap_ci3144585_meta.json`
- `data/hazard/usgs/shakemap/logs/shakemap_ci3144585_run.log`

## Data Handling Notes
- NBI delimited files can include malformed rows. The ingest parser logs warnings and skips bad lines to allow the pipeline to complete. Check `data/nbi/logs/nbi_latest_run.log` and `data/nbi/meta/nbi_latest_meta.json` for output summary.
- Large files are not committed. `.gitignore` excludes `data/nbi/raw/`, `data/nbi/clean/`, `data/nbi/curated/`, `data/nbi/logs/`, and `data/hazard/usgs/shakemap/{raw,logs,meta}/`. Only `data/nbi/meta/nbi_pipeline.json` is tracked.
- If you want a clean workspace after testing, delete contents under `data/` and re-run the pipeline when needed.

## Hazard Processing (Suggested Next Step)
The pipeline currently **downloads** ShakeMap data but does not convert it to a table. Recommended follow-up:
- Add a `hazard_processor.py` to parse `grid.xml` into CSV with fields like `lat`, `lon`, `pga`, `pgv`, `mmi`.
- Write outputs to `data/hazard/usgs/shakemap/clean/`.
- Join ShakeMap intensity to bridges via spatial nearest-neighbor (lat/lon). If bridge coordinates are missing, use county/FIPS as a coarse proxy.

## Notes / Common Issues
- **FHWA download pages can change**. If the FHWA page HTML changes, the link resolver in `nbi_ingest.py` may need a small adjustment.
- **Event timing**: The Northridge earthquake occurred on **1994-01-17**. For strict pre/post comparisons, consider adding **1993** (pre) and **1995** (post) NBI runs.
- **Format support**: Only NBI **Delimited** format is supported. Fixed-width ASCII is not supported.

# NBI + ShakeMap Pipeline (1994 Northridge)

## Overview
This document describes the end-to-end workflow for pulling NBI (bridge inventory) and USGS ShakeMap (hazard) data into **CAT411_framework**, cleaning it, and producing outputs that downstream analysis can use without manual rework. The pipeline is designed for reproducibility: it uses a single config file, writes logs/metadata, and keeps raw data intact.

Data sources:
- **NBI**: FHWA annual ASCII release, **Delimited format** (comma-separated, single-quote text qualifier). The default year is **1994** for the Northridge case.
- **ShakeMap**: USGS GeoJSON detail API for the Northridge event (**`ci3144585`**). The pipeline downloads `grid.xml.zip`, `shape.zip`, and `info.json`.

Directory layout:
- `data/nbi/raw/`: raw NBI zip + extracted CSV
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

## Hazard Processing (Suggested Next Step)
The pipeline currently **downloads** ShakeMap data but does not convert it to a table. Recommended follow-up:
- Add a `hazard_processor.py` to parse `grid.xml.zip` into CSV with fields like `lat`, `lon`, `pga`, `pgv`, `mmi`.
- Write outputs to `data/hazard/usgs/shakemap/clean/`.
- Join ShakeMap intensity to bridges via spatial nearest-neighbor (lat/lon). If bridge coordinates are missing, use county/FIPS as a coarse proxy.

## Notes / Common Issues
- **FHWA download pages can change**. If the FHWA page HTML changes, the link resolver in `nbi_ingest.py` may need a small adjustment.
- **Event timing**: The Northridge earthquake occurred on **1994-01-17**. For strict pre/post comparisons, consider adding **1993** (pre) and **1995** (post) NBI runs.
- **Format support**: Only NBI **Delimited** format is supported. Fixed-width ASCII is not supported.

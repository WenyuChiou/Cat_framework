# Example: Download NBI + Hazard Data

This example shows how to download NBI and USGS ShakeMap data using the pipeline.

## Prerequisites
1. Python 3.10+ installed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Steps
1. From the project root, run:

```bash
python broker/utils/nbi_ingest.py --download --download-only
```

2. To download and process (clean + curated outputs):

```bash
python broker/utils/nbi_ingest.py --download
```

3. To override the NBI year or USGS event:

```bash
python broker/utils/nbi_ingest.py --download --nbi-year 1994 --usgs-event ci3144585
```

4. Run the main analysis after data is ready:

```bash
python main.py
```

Optional modes:

```bash
python main.py --pipeline
python main.py --probabilistic
python main.py --fragility-only
```

## Expected Outputs
NBI:
- `data/nbi/raw/` contains the zip and extracted TXT/CSV
- `data/nbi/clean/` contains `nbi_latest_clean.csv`
- `data/nbi/curated/` contains `nbi_latest_curated.csv`
- `data/nbi/meta/` contains `nbi_latest_meta.json`
- `data/nbi/logs/` contains `nbi_latest_run.log`

ShakeMap:
- `data/hazard/usgs/shakemap/raw/grid.xml`
- `data/hazard/usgs/shakemap/raw/shape.zip`
- `data/hazard/usgs/shakemap/raw/info.json`
- `data/hazard/usgs/shakemap/meta/` and `data/hazard/usgs/shakemap/logs/`

## Notes
- Downloaded data is large and should not be committed.
- If downloads fail, try re-running the command.

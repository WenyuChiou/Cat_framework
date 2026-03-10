"""
Standalone validation script using BSSA21 GMPE.

Runs the full 3-level validation framework independently.
Core logic lives in src/validation.py.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.config import AnalysisConfig, validate_config
from src.validation import run_full_validation, run_validation, plot_validation_results

# ── Configuration ──
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
validation_csv = os.path.join(basedir, 'data/validation/northridge_1994_validation.csv')
stationlist_json = os.path.join(basedir, 'data/stationlist.json')

# Build a minimal config with Northridge scenario defaults
config = AnalysisConfig(
    validation_enabled=True,
    validation_data=validation_csv,
    validation_stationlist=stationlist_json,
    validation_im_source="gmpe",
    validation_levels=[1, 2, 3],
    gmpe_scenario={
        "Mw": 6.7,
        "lat": 34.213,
        "lon": -118.537,
        "depth_km": 18.4,
        "fault_type": "reverse",
        "vs30": 360.0,
    },
)
validate_config(config)

# ── Run full 3-level validation ──
output_dir = os.path.join(basedir, 'output/validation')
all_results = run_full_validation(
    config,
    bridges_df=pd.DataFrame(),
    output_dir=output_dir,
)

# ── Save per-level CSVs ──
for level, result in all_results.items():
    per_data = result.get("per_bridge") if "per_bridge" in result else result.get("per_station")
    if per_data is not None and len(per_data) > 0:
        csv_path = os.path.join(output_dir, f"validation_L{level}_results.csv")
        per_data.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\nSaved: {csv_path}")

# ── Backward-compatible Level 3 output ──
l3 = all_results.get(3, {})
per_bridge = l3.get("per_bridge")
if per_bridge is not None and len(per_bridge) > 0:
    rdf = per_bridge.copy()
    if "im_gmpe" in rdf.columns:
        rdf = rdf.rename(columns={"im_gmpe": "sa1s_gmpe", "im_shakemap": "sa1s_shakemap"})
    outpath = os.path.join(basedir, 'data/validation/validation_results_gmpe.csv')
    rdf.to_csv(outpath, index=False, encoding='utf-8')
    print(f"\nSaved (legacy): {outpath}")

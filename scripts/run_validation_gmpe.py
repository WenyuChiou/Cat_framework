"""
Standalone validation script using BSSA21 GMPE.

This script can be run independently for quick validation checks.
Core logic lives in src/validation.py.
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.config import AnalysisConfig, validate_config
from src.validation import run_validation, plot_validation_results

# ── Configuration ──
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
validation_csv = os.path.join(basedir, 'data/validation/northridge_validation_full.csv')

# Build a minimal config with Northridge scenario defaults
config = AnalysisConfig(
    validation_enabled=True,
    validation_data=validation_csv,
    validation_im_source="gmpe",
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

# ── Run validation ──
# bridges_df is not used for GMPE mode; pass empty DataFrame
metrics = run_validation(pd.DataFrame(), config, validation_csv)

# ── Plot results ──
output_dir = os.path.join(basedir, 'data/validation')
plot_validation_results(metrics, output_dir)

# ── Save per-bridge results ──
per_bridge = metrics.get("per_bridge")
if per_bridge is not None and len(per_bridge) > 0:
    # Rename columns for backward compatibility with original output
    rdf = per_bridge.copy()
    if "im_gmpe" in rdf.columns:
        rdf = rdf.rename(columns={"im_gmpe": "sa1s_gmpe", "im_shakemap": "sa1s_shakemap"})
    outpath = os.path.join(basedir, 'data/validation/validation_results_gmpe.csv')
    rdf.to_csv(outpath, index=False, encoding='utf-8')
    print(f"\nSaved: {outpath}")

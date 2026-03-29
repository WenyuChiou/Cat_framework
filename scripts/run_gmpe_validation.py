"""
Run GMPE Validation — NGA-West2 models vs Northridge observations.

Loads bridge data with ShakeMap-interpolated PGA, computes predictions
using 4 NGA-West2 GMPEs (ASK14, BSSA14, CB14, CY14) with site-specific
Vs30, and generates residual analysis by NEHRP site class.

Usage:
    python scripts/run_gmpe_validation.py

Input:
    data/northridge_validation_full.xlsx  (2,070 bridges with PGA + Vs30)

Output:
    output/gmpe_validation/
        bridges_gmpe_predictions.csv
        residual_statistics.csv
        01-05 diagnostic plots

Original analysis: Kubilay Albayrak
Integrated: March 2025
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.gmpe_validation import (
    FaultParams, NORTHRIDGE_FAULT, run_gmpe_validation,
)


def main():
    print("=" * 60)
    print("  GMPE Validation — NGA-West2 vs Northridge Observations")
    print("=" * 60)

    # Load data
    xlsx = PROJECT_ROOT / "data" / "northridge_validation_full.xlsx"
    if not xlsx.exists():
        sys.exit(f"ERROR: Data file not found: {xlsx}")

    df = pd.read_excel(xlsx)
    print(f"\n[load] {len(df)} bridges from {xlsx.name}")
    print(f"  Columns: {list(df.columns)}")

    # Filter to bridges with confirmed damage + ShakeMap PGA
    has_pga = df["pga_shakemap"].notna()
    print(f"  Bridges with PGA: {has_pga.sum()}")

    # Use ShakeMap-interpolated Sa values as "observed"
    # Need Vs30 for GMPE — estimate from Vs30 provider or use default
    # The xlsx doesn't have vs30, so we need to get it from the framework
    from src.vs30_provider import Vs30Provider
    print("\n[vs30] Interpolating Vs30 at bridge locations...")
    try:
        provider = Vs30Provider()
        vs30_arr = provider.get_vs30_array(
            df["latitude"].values, df["longitude"].values
        )
        df["vs30"] = vs30_arr
        print(f"  Vs30 range: {df['vs30'].min():.0f} - {df['vs30'].max():.0f} m/s")
    except Exception as e:
        print(f"  [warn] Vs30 interpolation failed: {e}")
        print(f"  [warn] Using default Vs30=760 m/s for all bridges")
        df["vs30"] = 760.0

    # Run validation
    results = run_gmpe_validation(
        bridges_df=df,
        fault=NORTHRIDGE_FAULT,
        obs_col="pga_shakemap",
        vs30_col="vs30",
        output_dir=str(PROJECT_ROOT / "output" / "gmpe_validation"),
    )

    # Summary
    stats = results["residual_stats"]
    if not stats.empty:
        print(f"\n{'='*60}")
        print("  SUMMARY — Overall Residual Statistics")
        print(f"{'='*60}")
        for name in ["ASK14", "BSSA14", "CB14", "CY14"]:
            s = stats[stats["GMPE"] == name]
            mean_all = s["Mean"].mean()
            std_all = s["Std"].mean()
            print(f"  {name:<8} mean={mean_all:+.3f}  avg_std={std_all:.3f}")

        # Best model
        overall = stats.groupby("GMPE")["Mean"].apply(lambda x: abs(x).mean())
        best = overall.idxmin()
        print(f"\n  Best model (lowest |bias|): {best}")

    print(f"\n[done] Outputs: {results['output_dir']}/")


if __name__ == "__main__":
    main()

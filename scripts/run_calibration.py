"""
Run MLE fragility calibration against Basoz (1998) Northridge observations.

Loads the 2,008 bridge dataset (data/northridge_observed.csv), computes
the optimal global median scale factor k and dispersion beta, and
generates diagnostic plots + config snippet.

Usage:
    python scripts/run_calibration.py
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.calibration import (
    ALL_DS,
    CalibrationResult,
    calibrate_global,
    generate_config_snippet,
    neg_log_likelihood,
    plot_before_after,
    plot_fragility_comparison,
    plot_likelihood_profile,
    predicted_distribution,
    save_results,
)
from src.northridge_case import NORTHRIDGE_DAMAGE_STATS


def main():
    print("=" * 60)
    print("FRAGILITY CALIBRATION — MLE Method")
    print("=" * 60)

    # ── Load bridge data ──────────────────────────────────────────
    csv_path = PROJECT_ROOT / "data" / "northridge_observed.csv"
    if not csv_path.exists():
        sys.exit(
            f"ERROR: Bridge data file not found: {csv_path}\n"
            f"Expected at: data/northridge_observed.csv relative to project root."
        )
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"\nLoaded {len(df)} bridges from {csv_path.name}")
    print(f"  HWB classes present: {sorted(df['hwb_class'].unique())}")
    print(f"  Sa(1.0s) range: {df['sa1s_shakemap'].min():.4f} - {df['sa1s_shakemap'].max():.4f} g")

    # ── Basoz observed counts ─────────────────────────────────────
    obs = NORTHRIDGE_DAMAGE_STATS["damage_summary"]
    n_total = sum(obs.values())
    print(f"\nBasoz (1998) observed counts (N={n_total}):")
    for ds in ALL_DS:
        print(f"  {ds:<12}: {obs[ds]:>5}  ({obs[ds]/n_total*100:.1f}%)")

    # ── Baseline prediction (k=1.0, beta=0.6) ────────────────────
    sa_arr = df["sa1s_shakemap"].values.astype(float)
    hwb_arr = df["hwb_class"].values
    obs_arr = np.array([obs[ds] for ds in ALL_DS], dtype=float)

    p_base = predicted_distribution(sa_arr, hwb_arr, k=1.0, beta=0.6)
    p_base = p_base / p_base.sum()
    print(f"\nBaseline HAZUS prediction (k=1.0, β=0.6):")
    for i, ds in enumerate(ALL_DS):
        cnt = p_base[i] * n_total
        print(f"  {ds:<12}: {cnt:>7.1f}  ({p_base[i]*100:.1f}%)")
    damage_frac_base = 1.0 - p_base[0]
    print(f"  Total damage fraction: {damage_frac_base*100:.1f}%  (observed: {(1-obs['none']/n_total)*100:.1f}%)")

    # ── Run MLE calibration ───────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Running MLE optimization...")
    result = calibrate_global(df, obs_counts=obs)

    print(f"\n{'─' * 60}")
    print("CALIBRATION RESULTS")
    print(f"{'─' * 60}")
    print(f"  Converged:     {result.converged}")
    print(f"  k (scale):     {result.k:.4f}  (95% CI: {result.k_ci[0]:.4f} - {result.k_ci[1]:.4f})")
    print(f"  beta (disp):   {result.beta:.4f}  (95% CI: {result.beta_ci[0]:.4f} - {result.beta_ci[1]:.4f})")
    print(f"  NLL:           {result.neg_log_likelihood:.2f}")
    print(f"  AIC:           {result.aic:.2f}")

    print(f"\nCalibrated prediction:")
    for ds in ALL_DS:
        pred = result.pred_counts[ds]
        obsv = result.obs_counts[ds]
        print(f"  {ds:<12}: pred={pred:>7.1f}  obs={obsv:>5}  "
              f"({result.pred_fractions[ds]*100:.1f}% vs {result.obs_fractions[ds]*100:.1f}%)")

    damage_frac_cal = 1.0 - result.pred_fractions["none"]
    damage_frac_obs = 1.0 - result.obs_fractions["none"]
    print(f"\n  Total damage fraction: {damage_frac_cal*100:.1f}%  (observed: {damage_frac_obs*100:.1f}%)")

    # ── Generate outputs ──────────────────────────────────────────
    out_dir = PROJECT_ROOT / "output" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving outputs to {out_dir}/")

    # JSON results
    save_results(result, out_dir)

    # Plots
    plot_before_after(result, save_path=out_dir / "before_after_distribution.png")
    plot_likelihood_profile(
        sa_arr, hwb_arr, obs_arr, result.k, result.beta,
        save_path=out_dir / "likelihood_profile.png",
    )
    plot_fragility_comparison(
        result.k, result.beta, hwb_class="HWB5",
        save_path=out_dir / "fragility_curves_comparison.png",
    )

    # Config snippet
    snippet = generate_config_snippet(result)
    snippet_path = out_dir / "config_snippet.yaml"
    with open(snippet_path, "w", encoding="utf-8") as f:
        f.write(snippet + "\n")
    print(f"  Saved: {snippet_path}")

    print(f"\n{'─' * 60}")
    print("CONFIG SNIPPET (paste into config.yaml):")
    print(f"{'─' * 60}")
    print(snippet)
    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()

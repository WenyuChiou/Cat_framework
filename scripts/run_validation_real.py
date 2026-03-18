"""
Real Data Validation — 1994 Northridge Earthquake Bridge Damage
===============================================================
Author : Anik Das (original analysis)
Revised: 2026-03-18 — fix HWB28 param bug, replace argmax with
         exceedance-threshold, integrate MLE calibration comparison.

Usage:
  python scripts/run_validation_real.py          # run both baseline & calibrated
  python scripts/run_validation_real.py --baseline   # baseline only
  python scripts/run_validation_real.py --calibrated # calibrated only

Input:
  data/northridge_observed.csv  (2008 bridges with observed damage + ShakeMap Sa)
  output/calibration/calibration_results.json  (MLE-calibrated k, beta)

Output:
  output/validation_real/baseline/    — 4 plots + acceptance_criteria.csv
  output/validation_real/calibrated/  — 4 plots + acceptance_criteria.csv
  output/validation_real/comparison_summary.png  — side-by-side comparison

Changes from Anik original:
  1. HWB28 bug fix: was using HWB5 medians; now uses correct HAZUS Table 7.9
     values via src.hazus_params.
  2. Prediction method: argmax → exceedance threshold (P(DS>=j) >= 0.5).
  3. Runs both baseline and calibrated, producing comparison plots.

HAZUS Source: Hazus 6.1 Table 7.9, cross-verified by Sirisha Kedarsetty
Hazard input: sa1s_shakemap column (ShakeMap-interpolated Sa at 1.0s)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from anik_validation import run_validation, DAMAGE_STATES, DS_INDEX
from src.hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "validation_real")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ══════════════════════════════════════════════════════════════════════
# 1.  FRAGILITY PREDICTION
# ══════════════════════════════════════════════════════════════════════

def predict_damage_state(
    sa: float,
    hwb_class: str,
    k: float = 1.0,
    beta: float = 0.6,
    threshold: float = 0.5,
) -> str:
    """
    Assign damage state via exceedance-threshold method.

    For each DS from complete -> slight, check if P(DS >= j) >= threshold.
    Return the highest DS that passes.
    """
    if sa <= 0:
        return "none"

    params = HAZUS_BRIDGE_FRAGILITY.get(hwb_class)
    if params is None:
        return "none"

    ds_params = params["damage_states"]
    for ds in reversed(DAMAGE_STATE_ORDER):
        median = ds_params[ds]["median"] * k
        p_exc = float(norm.cdf(np.log(sa / median) / beta))
        if p_exc >= threshold:
            return ds

    return "none"


# ══════════════════════════════════════════════════════════════════════
# 2.  RUN ONE VALIDATION PASS
# ══════════════════════════════════════════════════════════════════════

def run_single(obs_raw, k, beta, label, output_dir):
    """Run prediction + Anik validation for one parameter set."""
    os.makedirs(output_dir, exist_ok=True)

    param_label = f"k={k:.2f}, beta={beta:.2f}"
    print(f"\n{'='*60}")
    print(f"  {label}  ({param_label})")
    print(f"{'='*60}")

    # Build observed CSV
    observed_df = obs_raw[
        ["bridge_id", "latitude", "longitude", "observed_damage"]
    ].copy()

    # Predict
    print(f"[predict] Exceedance threshold, {param_label}")
    preds = obs_raw.apply(
        lambda row: predict_damage_state(
            row["sa1s_shakemap"], row["hwb_class"], k=k, beta=beta
        ),
        axis=1,
    )
    predicted_df = pd.DataFrame({
        "bridge_id": obs_raw["bridge_id"],
        "predicted_damage": preds,
        "sa_predicted": obs_raw["sa1s_shakemap"],
    })

    print(f"[check] Predicted: {predicted_df['predicted_damage'].value_counts().to_dict()}")

    # Save intermediate CSVs
    obs_path = os.path.join(output_dir, "observed.csv")
    pred_path = os.path.join(output_dir, "predicted.csv")
    observed_df.to_csv(obs_path, index=False, encoding="utf-8")
    predicted_df.to_csv(pred_path, index=False, encoding="utf-8")

    # Run Anik validation framework
    results = run_validation(
        observed_csv=obs_path,
        predicted_csv=pred_path,
        output_dir=output_dir,
    )
    return results


# ══════════════════════════════════════════════════════════════════════
# 3.  COMPARISON PLOT
# ══════════════════════════════════════════════════════════════════════

def plot_comparison(results_base, results_cal, k_cal, beta_cal, save_dir):
    """Generate side-by-side comparison: baseline vs calibrated."""

    merged_b = results_base["merged"]
    merged_c = results_cal["merged"]
    met_b = results_base["metrics"]
    met_c = results_cal["metrics"]
    res_b = results_base["residuals"]
    res_c = results_cal["residuals"]

    n = len(merged_b)
    DS = DAMAGE_STATES

    # Counts
    obs_c = [merged_b["observed_damage"].value_counts().get(d, 0) for d in DS]
    base_c = [merged_b["predicted_damage"].value_counts().get(d, 0) for d in DS]
    cal_c = [merged_c["predicted_damage"].value_counts().get(d, 0) for d in DS]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # ── Panel 1: Distribution ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(DS))
    w = 0.25
    ax.bar(x - w, obs_c, w, label="Observed", color="#2196F3", edgecolor="k")
    ax.bar(x, base_c, w, label="Baseline (k=1.0, \u03b2=0.6)", color="#FF9800", edgecolor="k")
    ax.bar(x + w, cal_c, w,
           label=f"Calibrated (k={k_cal:.2f}, \u03b2={beta_cal:.2f})",
           color="#4CAF50", edgecolor="k")
    for i in range(len(DS)):
        for vals, off in [(obs_c, -w), (base_c, 0), (cal_c, w)]:
            if vals[i] > 0:
                ax.text(x[i] + off, vals[i] + 10, str(vals[i]),
                        ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DS], fontsize=11)
    ax.set_ylabel("Number of Bridges", fontsize=11)
    ax.set_title("Damage Distribution: Observed vs Baseline vs Calibrated (N=2008)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: Accuracy ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    accs = [met_b["overall_accuracy"], met_c["overall_accuracy"]]
    colors = ["#FF9800", "#4CAF50"]
    labels = ["Baseline", "Calibrated"]
    bars = ax.bar(labels, accs, color=colors, edgecolor="k", width=0.5)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.1%}", ha="center", fontsize=12, fontweight="bold")
    ax.axhline(0.60, color="gray", ls=":", lw=1, label="Threshold (60%)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Accuracy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: Bias & RMSE ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    metrics_names = ["|Bias|", "RMSE"]
    base_vals = [abs(res_b["mean_residual"]), res_b["rmse"]]
    cal_vals = [abs(res_c["mean_residual"]), res_c["rmse"]]
    x2 = np.arange(2)
    w2 = 0.3
    b1 = ax.bar(x2 - w2 / 2, base_vals, w2, label="Baseline",
                color="#FF9800", edgecolor="k")
    b2 = ax.bar(x2 + w2 / 2, cal_vals, w2, label="Calibrated",
                color="#4CAF50", edgecolor="k")
    for i in range(2):
        ax.text(x2[i] - w2 / 2, base_vals[i] + 0.02,
                f"{base_vals[i]:.3f}", ha="center", fontsize=9)
        ax.text(x2[i] + w2 / 2, cal_vals[i] + 0.02,
                f"{cal_vals[i]:.3f}", ha="center", fontsize=9)
    ax.axhline(0.50, color="gray", ls=":", lw=1, label="Bias threshold")
    ax.axhline(1.50, color="gray", ls="--", lw=1, label="RMSE threshold")
    ax.set_xticks(x2)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("Value")
    ax.set_title("Bias & RMSE")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 4: Per-class recall ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    x3 = np.arange(len(DS))
    w3 = 0.3
    rec_b = [met_b["per_class"][d]["recall"] for d in DS]
    rec_c = [met_c["per_class"][d]["recall"] for d in DS]
    ax.bar(x3 - w3 / 2, rec_b, w3, label="Baseline", color="#FF9800", edgecolor="k")
    ax.bar(x3 + w3 / 2, rec_c, w3, label="Calibrated", color="#4CAF50", edgecolor="k")
    ax.axhline(0.30, color="gray", ls=":", lw=1, label="Threshold (30%)")
    ax.set_xticks(x3)
    ax.set_xticklabels([d.capitalize() for d in DS], fontsize=9)
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Validation Comparison: Baseline (HAZUS) vs MLE-Calibrated",
        fontsize=15, fontweight="bold", y=0.98,
    )
    plt.savefig(
        os.path.join(save_dir, "comparison_summary.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"\n[plot] Saved: {os.path.join(save_dir, 'comparison_summary.png')}")


# ══════════════════════════════════════════════════════════════════════
# 4.  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    only_baseline = "--baseline" in sys.argv and "--calibrated" not in sys.argv
    only_calibrated = "--calibrated" in sys.argv and "--baseline" not in sys.argv
    run_both = not only_baseline and not only_calibrated

    # Load calibration params
    cal_path = os.path.join(
        PROJECT_ROOT, "output", "calibration", "calibration_results.json"
    )
    k_cal, beta_cal = 1.0, 0.6
    has_cal = False
    if os.path.exists(cal_path):
        with open(cal_path, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
        k_cal = cal_data["k"]
        beta_cal = cal_data["beta"]
        has_cal = True
    else:
        print(f"[warn] No calibration file: {cal_path}")
        if only_calibrated or run_both:
            print("[warn] Cannot run calibrated — using baseline only.")
            only_baseline = True
            run_both = False

    # Load bridge data
    csv_path = os.path.join(DATA_DIR, "northridge_observed.csv")
    obs_raw = pd.read_csv(csv_path, encoding="utf-8")
    obs_raw = obs_raw.rename(columns={
        "structure_number": "bridge_id",
        "Observed_damage": "observed_damage",
    })
    obs_raw["observed_damage"] = obs_raw["observed_damage"].str.lower().str.strip()
    print(f"[load] {len(obs_raw)} bridges from northridge_observed.csv")
    print(f"[obs]  {obs_raw['observed_damage'].value_counts().to_dict()}")

    results_base = None
    results_cal = None

    # --- Baseline ---
    if only_baseline or run_both:
        results_base = run_single(
            obs_raw, k=1.0, beta=0.6,
            label="BASELINE (HAZUS defaults)",
            output_dir=os.path.join(OUTPUT_ROOT, "baseline"),
        )

    # --- Calibrated ---
    if only_calibrated or run_both:
        results_cal = run_single(
            obs_raw, k=k_cal, beta=beta_cal,
            label="CALIBRATED (MLE)",
            output_dir=os.path.join(OUTPUT_ROOT, "calibrated"),
        )

    # --- Comparison ---
    if run_both and results_base and results_cal:
        plot_comparison(results_base, results_cal, k_cal, beta_cal, OUTPUT_ROOT)

        # Print side-by-side summary
        mb = results_base["metrics"]
        mc = results_cal["metrics"]
        rb = results_base["residuals"]
        rc = results_cal["residuals"]
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Metric':<22} {'Baseline':>12} {'Calibrated':>12} {'Threshold':>12}")
        print(f"  {'-'*58}")
        print(f"  {'Accuracy':<22} {mb['overall_accuracy']:>11.1%} {mc['overall_accuracy']:>11.1%} {'>=60%':>12}")
        print(f"  {'|Mean Residual|':<22} {abs(rb['mean_residual']):>12.3f} {abs(rc['mean_residual']):>12.3f} {'<0.50':>12}")
        print(f"  {'RMSE':<22} {rb['rmse']:>12.3f} {rc['rmse']:>12.3f} {'<1.50':>12}")
        for ds in DAMAGE_STATES:
            rb_r = mb["per_class"][ds]["recall"]
            rc_r = mc["per_class"][ds]["recall"]
            print(f"  {'Recall-'+ds.capitalize():<22} {rb_r:>11.1%} {rc_r:>11.1%} {'>=30%':>12}")
        print(f"{'='*60}")

    # Also copy calibrated results to main output dir for NB05 compatibility
    if results_cal:
        cal_dir = os.path.join(OUTPUT_ROOT, "calibrated")
        for fname in ["01_confusion_matrix.png", "02_per_class_metrics.png",
                       "03_residual_distribution.png",
                       "04_damage_distribution_comparison.png",
                       "acceptance_criteria.csv"]:
            src = os.path.join(cal_dir, fname)
            dst = os.path.join(OUTPUT_ROOT, fname)
            if os.path.exists(src):
                import shutil
                shutil.copy2(src, dst)

    print("\n[done]")


if __name__ == "__main__":
    main()

"""
Real Data Validation — 1994 Northridge Earthquake Bridge Damage
===============================================================
Author : Anik Das (original analysis)
Revised: 2026-03-18 — fix HWB28 param bug, replace argmax with
         exceedance-threshold method, add calibrated-parameter support.
Source : Anik Das/Validation/run_real_validation.py

Usage:
  cd <project_root>
  python scripts/run_validation_real.py            # baseline (HAZUS defaults)
  python scripts/run_validation_real.py --calibrated  # with MLE-calibrated params

Input:
  data/northridge_observed.csv  (2008 bridges with observed damage + ShakeMap Sa)

Output:
  output/validation_real/  (4 PNG plots + acceptance_criteria.csv)
  data/real_observed.csv   (intermediate: observed in validation format)
  data/real_predicted.csv  (intermediate: HAZUS predictions)

Changes from original:
  1. HWB28 bug fix: was using HWB5 medians (most vulnerable); now uses correct
     HAZUS Table 7.9 values via src.hazus_params (slight=0.80 vs 0.25).
  2. Prediction method: argmax of discrete probabilities → exceedance threshold
     (assign highest DS where P(DS>=j) >= 0.5). The argmax method with beta=0.6
     *cannot* predict moderate for any HWB class in this dataset, because the
     high dispersion causes fragility curves to overlap so much that
     P(moderate) = P(>=mod) - P(>=ext) is never the largest probability.
  3. Calibrated parameters: --calibrated flag loads (k, beta) from
     output/calibration/calibration_results.json if available.

HAZUS Source: Hazus 6.1 Table 7.9, cross-verified by Sirisha Kedarsetty
Hazard input: sa1s_shakemap column (ShakeMap-interpolated Sa at 1.0s)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Cross-platform paths ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from anik_validation import run_validation
from src.hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "validation_real")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

DAMAGE_STATES = ["none", "slight", "moderate", "extensive", "complete"]


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

    For each DS from complete → slight, check if P(DS >= j) >= threshold.
    Return the highest DS that passes. This avoids the argmax problem where
    intermediate states (moderate, slight) are structurally unreachable
    with beta=0.6.

    Parameters
    ----------
    sa : float
        Sa(1.0s) in g.
    hwb_class : str
        HAZUS bridge class (e.g. "HWB5").
    k : float
        Median scale factor (1.0 = HAZUS defaults).
    beta : float
        Lognormal dispersion (0.6 = HAZUS default).
    threshold : float
        Exceedance probability threshold (default 0.5 = median).
    """
    if sa <= 0:
        return "none"

    params = HAZUS_BRIDGE_FRAGILITY.get(hwb_class)
    if params is None:
        return "none"

    ds_params = params["damage_states"]
    # Check from highest (complete) to lowest (slight)
    for ds in reversed(DAMAGE_STATE_ORDER):
        median = ds_params[ds]["median"] * k
        p_exc = float(norm.cdf(np.log(sa / median) / beta))
        if p_exc >= threshold:
            return ds

    return "none"


# ══════════════════════════════════════════════════════════════════════
# 2.  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    # Parse --calibrated flag
    use_calibrated = "--calibrated" in sys.argv

    # Load calibration parameters if requested
    k = 1.0
    beta = 0.6
    if use_calibrated:
        cal_path = os.path.join(
            PROJECT_ROOT, "output", "calibration", "calibration_results.json"
        )
        if os.path.exists(cal_path):
            with open(cal_path, "r", encoding="utf-8") as f:
                cal = json.load(f)
            k = cal["k"]
            beta = cal["beta"]
            print(f"[calibrated] Using k={k:.4f}, beta={beta:.4f}")
        else:
            print(f"[warn] Calibration file not found: {cal_path}")
            print("[warn] Falling back to HAZUS defaults (k=1.0, beta=0.6)")

    param_label = f"k={k:.2f}, β={beta:.2f}"

    print("=" * 60)
    print(f"  CAT411 - Real Data Validation")
    print(f"  Parameters: {param_label}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    obs_raw = pd.read_csv(
        os.path.join(DATA_DIR, "northridge_observed.csv"), encoding="utf-8"
    )
    print(f"\n[load] {len(obs_raw)} bridges loaded from northridge_observed.csv")

    obs_raw = obs_raw.rename(
        columns={"structure_number": "bridge_id", "Observed_damage": "observed_damage"}
    )

    # ── Build observed CSV ────────────────────────────────────────
    observed_df = obs_raw[
        ["bridge_id", "latitude", "longitude", "observed_damage"]
    ].copy()
    observed_df["observed_damage"] = (
        observed_df["observed_damage"].str.lower().str.strip()
    )

    # ── Generate predictions ──────────────────────────────────────
    print(f"[predict] Applying HAZUS fragility (exceedance threshold, {param_label})...")

    obs_raw["predicted_damage"] = obs_raw.apply(
        lambda row: predict_damage_state(
            row["sa1s_shakemap"], row["hwb_class"], k=k, beta=beta
        ),
        axis=1,
    )
    obs_raw["sa_predicted"] = obs_raw["sa1s_shakemap"]

    predicted_df = obs_raw[["bridge_id", "predicted_damage", "sa_predicted"]].copy()

    # ── Distribution check ────────────────────────────────────────
    print("\n[check] Observed damage distribution:")
    print(observed_df["observed_damage"].value_counts().to_string())
    print("\n[check] Predicted damage distribution:")
    print(predicted_df["predicted_damage"].value_counts().to_string())

    # ── Save CSVs ─────────────────────────────────────────────────
    obs_path = os.path.join(DATA_DIR, "real_observed.csv")
    pred_path = os.path.join(DATA_DIR, "real_predicted.csv")
    observed_df.to_csv(obs_path, index=False, encoding="utf-8")
    predicted_df.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"\n[save] {obs_path}")
    print(f"[save] {pred_path}")

    # ── Run validation framework ──────────────────────────────────
    print("\n[validate] Running full validation framework...")
    results = run_validation(
        observed_csv=obs_path,
        predicted_csv=pred_path,
        output_dir=OUTPUT_DIR,
    )

    # ── Acceptance summary ────────────────────────────────────────
    acceptance = results["acceptance"]
    passes = acceptance["Pass"].sum()
    total = len(acceptance)
    print(f"\n[result] {passes}/{total} acceptance criteria passed.")

    critical = acceptance[
        acceptance["Metric"].isin(
            ["Overall Accuracy", "Mean Residual (|bias|)", "RMSE (ordinal)"]
        )
    ]
    if not critical["Pass"].all():
        print("[warn] Some CRITICAL acceptance criteria FAILED.")
        sys.exit(1)
    else:
        print("[ok]  All critical acceptance criteria PASSED.")


if __name__ == "__main__":
    main()

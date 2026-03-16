"""
Real Data Validation — 1994 Northridge Earthquake Bridge Damage
===============================================================
Author : Anik Das (original analysis)
Integrated: 2026-03-15 into CAT411 framework
Source : Anik Das/Validation/run_real_validation.py

Usage:
  cd <project_root>
  python scripts/run_validation_real.py

Input:
  data/northridge_observed.csv  (2008 bridges with observed damage + ShakeMap Sa)

Output:
  output/validation_real/  (4 PNG plots + acceptance_criteria.csv)
  data/real_observed.csv   (intermediate: observed in validation format)
  data/real_predicted.csv  (intermediate: HAZUS predictions)

Description:
  Applies HAZUS fragility curves (12 HWB classes, Sa(1.0s), beta=0.6) to
  generate predicted damage states, then runs confusion matrix + residual
  analysis against observed damage from the 1994 Northridge earthquake.

Original docstring:
======================
CAT411 — T1b: Real Data Validation (Anik Das)

Pipeline:
  1. Load northridge_Observed.csv
  2. Apply HAZUS fragility curves (Sa1.0s, beta=0.6) per HWB class
     → generate predicted_damage for each bridge
  3. Save observed CSV and predicted CSV in validation.py-compatible format
  4. Run full validation framework (confusion matrix, metrics, plots)

HAZUS Source: Table provided by Sirisha (HazusTable.xlsx)
Hazard input: sa1s_shakemap column (ShakeMap-interpolated Sa at 1.0s)
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Cross-platform paths (adapted for CAT411 framework) ──────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from anik_validation import run_validation

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "validation_real")
DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 1.  HAZUS FRAGILITY PARAMETERS
#     Source: HazusTable.xlsx — Sa(1.0s) medians, beta = 0.6 (standard)
#     Rows = HWB class present in northridge_Observed.csv
# ══════════════════════════════════════════════════════════════════════

BETA = 0.6   # standard HAZUS lognormal dispersion for all bridge classes

# median Sa(1.0s) in g for each damage state threshold
HAZUS_MEDIANS = {
    "HWB1":  {"slight": 0.40, "moderate": 0.50, "extensive": 0.70, "complete": 0.90},
    "HWB2":  {"slight": 0.60, "moderate": 0.90, "extensive": 1.10, "complete": 1.70},
    "HWB3":  {"slight": 0.80, "moderate": 1.00, "extensive": 1.20, "complete": 1.70},
    "HWB4":  {"slight": 0.80, "moderate": 1.00, "extensive": 1.20, "complete": 1.70},
    "HWB5":  {"slight": 0.25, "moderate": 0.35, "extensive": 0.45, "complete": 0.70},
    "HWB6":  {"slight": 0.30, "moderate": 0.50, "extensive": 0.60, "complete": 0.90},
    "HWB7":  {"slight": 0.50, "moderate": 0.80, "extensive": 1.10, "complete": 1.70},
    "HWB8":  {"slight": 0.35, "moderate": 0.45, "extensive": 0.55, "complete": 0.80},
    "HWB10": {"slight": 0.60, "moderate": 0.90, "extensive": 1.10, "complete": 1.50},
    "HWB15": {"slight": 0.75, "moderate": 0.75, "extensive": 0.75, "complete": 1.10},
    "HWB16": {"slight": 0.90, "moderate": 0.90, "extensive": 1.10, "complete": 1.50},
    # HWB28 = "all other bridges not classified" → use HWB5 as conservative default
    "HWB28": {"slight": 0.25, "moderate": 0.35, "extensive": 0.45, "complete": 0.70},
}

DAMAGE_STATES = ["none", "slight", "moderate", "extensive", "complete"]


# ══════════════════════════════════════════════════════════════════════
# 2.  FRAGILITY CURVE FUNCTION
# ══════════════════════════════════════════════════════════════════════

def exceedance_prob(sa: float, median: float, beta: float = BETA) -> float:
    """
    P(DS >= ds | Sa) using lognormal CDF.
    P = Phi( ln(Sa / median) / beta )
    """
    if sa <= 0:
        return 0.0
    return float(norm.cdf(np.log(sa / median) / beta))


def predict_damage_state(sa: float, hwb_class: str) -> str:
    """
    Apply HAZUS fragility curves for a single bridge.

    Method: compute P(DS >= ds) for each damage state threshold,
    then derive individual state probabilities:
      P(none)      = 1 - P(DS >= slight)
      P(slight)    = P(DS >= slight)    - P(DS >= moderate)
      P(moderate)  = P(DS >= moderate)  - P(DS >= extensive)
      P(extensive) = P(DS >= extensive) - P(DS >= complete)
      P(complete)  = P(DS >= complete)

    Most-likely damage state = argmax of individual probabilities.
    """
    medians = HAZUS_MEDIANS.get(hwb_class, HAZUS_MEDIANS["HWB5"])

    p_slight    = exceedance_prob(sa, medians["slight"])
    p_moderate  = exceedance_prob(sa, medians["moderate"])
    p_extensive = exceedance_prob(sa, medians["extensive"])
    p_complete  = exceedance_prob(sa, medians["complete"])

    probs = {
        "none":      1.0 - p_slight,
        "slight":    p_slight    - p_moderate,
        "moderate":  p_moderate  - p_extensive,
        "extensive": p_extensive - p_complete,
        "complete":  p_complete,
    }

    return max(probs, key=probs.get)


# ══════════════════════════════════════════════════════════════════════
# 3.  LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  CAT411 - Real Data Validation (T1b)")
print("=" * 60)

obs_raw = pd.read_csv(os.path.join(DATA_DIR, "northridge_observed.csv"))
print(f"\n[load] {len(obs_raw)} bridges loaded from northridge_Observed.csv")

# Rename columns to match validation.py schema
obs_raw = obs_raw.rename(columns={
    "structure_number": "bridge_id",
    "Observed_damage":  "observed_damage",
})

# ── Build observed CSV (validation.py format) ──────────────────────
observed_df = obs_raw[["bridge_id", "latitude", "longitude", "observed_damage"]].copy()
observed_df["observed_damage"] = observed_df["observed_damage"].str.lower().str.strip()

# ── Generate predictions via HAZUS fragility curves ───────────────
print("[predict] Applying HAZUS fragility curves (Sa 1.0s, beta=0.6)...")

obs_raw["predicted_damage"] = obs_raw.apply(
    lambda row: predict_damage_state(row["sa1s_shakemap"], row["hwb_class"]),
    axis=1
)
obs_raw["sa_predicted"] = obs_raw["sa1s_shakemap"]   # keep hazard for residual plots

predicted_df = obs_raw[["bridge_id", "predicted_damage", "sa_predicted"]].copy()

# ── Distribution check ─────────────────────────────────────────────
print("\n[check] Observed damage distribution:")
print(observed_df["observed_damage"].value_counts().to_string())
print("\n[check] Predicted damage distribution:")
print(predicted_df["predicted_damage"].value_counts().to_string())

# ── HWB28 fallback report ──────────────────────────────────────────
n_hwb28 = (obs_raw["hwb_class"] == "HWB28").sum()
print(f"\n[note] HWB28 bridges (unclassified, used HWB5 fallback): {n_hwb28}")

# ── Save CSVs ──────────────────────────────────────────────────────
obs_path  = os.path.join(DATA_DIR, "real_observed.csv")
pred_path = os.path.join(DATA_DIR, "real_predicted.csv")
observed_df.to_csv(obs_path,  index=False)
predicted_df.to_csv(pred_path, index=False)
print(f"\n[save] {obs_path}")
print(f"[save] {pred_path}")

# ══════════════════════════════════════════════════════════════════════
# 4.  RUN VALIDATION FRAMEWORK
# ══════════════════════════════════════════════════════════════════════

print("\n[validate] Running full validation framework...")
results = run_validation(
    observed_csv  = obs_path,
    predicted_csv = pred_path,
    output_dir    = OUTPUT_DIR,
)

# ── Final acceptance summary ───────────────────────────────────────
acceptance = results["acceptance"]
passes = acceptance["Pass"].sum()
total  = len(acceptance)
print(f"\n[result] {passes}/{total} acceptance criteria passed.")

critical = acceptance[acceptance["Metric"].isin(
    ["Overall Accuracy", "Mean Residual (|bias|)", "RMSE (ordinal)"]
)]
if not critical["Pass"].all():
    print("[warn] Some CRITICAL acceptance criteria FAILED.")
    sys.exit(1)
else:
    print("[ok]  All critical acceptance criteria PASSED.")

"""
Validation Module for CAT411 Framework.

Compares pipeline-predicted bridge damage states against observed data.
Computes accuracy, MAE, bias, and confusion matrix metrics.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.fragility import damage_state_probabilities

DS_ORDER = ["none", "slight", "moderate", "extensive", "complete"]
DS_INDEX = {ds: i for i, ds in enumerate(DS_ORDER)}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def run_validation(
    bridges_df: pd.DataFrame,
    config,
    validation_csv_path: str,
    shakemap: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compare pipeline predictions vs observed damage data.

    Parameters
    ----------
    bridges_df : pd.DataFrame
        Pipeline output (used only if validation bridges match by structure_number).
    config : AnalysisConfig
        Analysis configuration.
    validation_csv_path : str
        Path to observed damage CSV file.
    shakemap : pd.DataFrame, optional
        ShakeMap grid data. Required for validation_im_source="shakemap" when
        validation bridges are not in the pipeline results.

    Returns
    -------
    dict with keys: accuracy, mae, bias, confusion_matrix, per_bridge (DataFrame)
    """
    # Load observed data
    val_df = pd.read_csv(validation_csv_path, encoding="utf-8")
    confirmed = val_df[val_df["damage_confirmed"] == True].copy()  # noqa: E712

    if len(confirmed) == 0:
        print("[Validation] No confirmed observations found in validation data.")
        return {"accuracy": 0.0, "mae": 0.0, "bias": 0.0,
                "confusion_matrix": {}, "per_bridge": pd.DataFrame()}

    print(f"[Validation] Confirmed observations: {len(confirmed)}")
    print(f"[Validation] Damage distribution:")
    print(confirmed["observed_damage_state"].value_counts().to_string())

    # Determine IM source for validation
    im_source = getattr(config, "validation_im_source", "gmpe")

    if im_source == "gmpe":
        results = _validate_with_gmpe(confirmed, config)
    else:
        # shakemap mode: try pipeline merge first, fall back to direct interpolation
        results = _validate_with_pipeline(confirmed, bridges_df)
        if len(results) == 0 and shakemap is not None:
            print("[Validation] No pipeline matches. "
                  "Interpolating ShakeMap IM directly for validation bridges...")
            results = _validate_with_shakemap(confirmed, shakemap, config)
        elif len(results) == 0 and shakemap is None:
            print("[Validation] WARNING: No pipeline matches and no ShakeMap grid "
                  "available. Falling back to GMPE mode.")
            results = _validate_with_gmpe(confirmed, config)

    if len(results) == 0:
        print("[Validation] No matching bridges for validation.")
        return {"accuracy": 0.0, "mae": 0.0, "bias": 0.0,
                "confusion_matrix": {}, "per_bridge": pd.DataFrame()}

    rdf = pd.DataFrame(results)
    metrics = compute_validation_metrics(rdf["predicted"], rdf["observed"])
    metrics["per_bridge"] = rdf

    # Print summary
    _print_validation_summary(rdf, metrics, config)

    return metrics


def _validate_with_gmpe(confirmed: pd.DataFrame, config) -> list[dict]:
    """Run validation using GMPE-computed IM values."""
    from src.gmpe_bssa21 import BSSA21

    sc = getattr(config, "gmpe_scenario", None)
    if sc is None:
        print("[Validation] WARNING: No gmpe_scenario in config. "
              "Using hard-coded Northridge defaults (Mw=6.7, lat=34.213, lon=-118.537).")
        sc = {
            "Mw": 6.7, "lat": 34.213, "lon": -118.537,
            "depth_km": 18.4, "fault_type": "reverse",
            "vs30": 360.0,
        }

    bssa = BSSA21()
    eq_mw = sc.get("Mw", 6.7)
    eq_lat = sc.get("lat", 34.213)
    eq_lon = sc.get("lon", -118.537)
    eq_depth = sc.get("depth_km", 18.4)
    eq_fault = sc.get("fault_type", "reverse")
    eq_period = 1.0  # Sa(1.0s) for Hazus bridge fragility
    default_vs30 = float(sc.get("vs30", 360.0))

    results = []
    for _, row in confirmed.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        hwb = row["hwb_class"]

        r_epi = haversine_km(lat, lon, eq_lat, eq_lon)
        r_jb = max(0.1, math.sqrt(max(0, r_epi**2 - eq_depth**2)))

        sa_gmpe, sigma = bssa.compute(eq_mw, r_jb, default_vs30, eq_fault, eq_period)

        probs = damage_state_probabilities(sa_gmpe, hwb)
        pred_ds = max(DS_ORDER, key=lambda ds: probs[ds])
        pred_idx = DS_INDEX[pred_ds]
        expected_idx = sum(DS_INDEX[ds] * probs[ds] for ds in DS_ORDER)

        obs_ds = row["observed_damage_state"]
        obs_idx = DS_INDEX.get(obs_ds, -1)
        if obs_idx < 0:
            continue

        results.append({
            "structure_number": row["structure_number"],
            "hwb_class": hwb,
            "r_jb_km": round(r_jb, 1),
            "im_gmpe": round(sa_gmpe, 4),
            "im_shakemap": round(row.get("sa1s_shakemap", 0.0), 4),
            "observed": obs_ds,
            "observed_idx": obs_idx,
            "predicted": pred_ds,
            "predicted_idx": pred_idx,
            "expected_idx": round(expected_idx, 2),
            "p_none": round(probs["none"], 3),
            "p_slight": round(probs["slight"], 3),
            "p_moderate": round(probs["moderate"], 3),
            "p_extensive": round(probs["extensive"], 3),
            "p_complete": round(probs["complete"], 3),
            "correct": pred_ds == obs_ds,
            "error": pred_idx - obs_idx,
        })

    return results


def _validate_with_pipeline(
    confirmed: pd.DataFrame, bridges_df: pd.DataFrame,
) -> list[dict]:
    """Validate using pipeline's pre-computed im_selected values."""
    # Match by structure_number
    required_cols = ["structure_number", "im_selected", "hwb_class",
                     "P_none", "P_slight", "P_moderate", "P_extensive", "P_complete"]
    missing = [c for c in required_cols if c not in bridges_df.columns]
    if missing:
        print(f"[Validation] WARNING: bridges_df missing columns {missing}. "
              f"Cannot use shakemap IM source for validation.")
        return []

    merged = confirmed.merge(
        bridges_df[required_cols],
        on="structure_number",
        how="inner",
        suffixes=("_obs", "_pipe"),
    )

    if len(merged) == 0:
        print("[Validation] No structure_number matches between pipeline and observations.")
        return []

    results = []
    for _, row in merged.iterrows():
        hwb = row.get("hwb_class_pipe", row.get("hwb_class_obs", ""))
        probs = {
            "none": row["P_none"],
            "slight": row["P_slight"],
            "moderate": row["P_moderate"],
            "extensive": row["P_extensive"],
            "complete": row["P_complete"],
        }
        pred_ds = max(DS_ORDER, key=lambda ds: probs[ds])
        pred_idx = DS_INDEX[pred_ds]
        expected_idx = sum(DS_INDEX[ds] * probs[ds] for ds in DS_ORDER)

        obs_ds = row["observed_damage_state"]
        obs_idx = DS_INDEX.get(obs_ds, -1)
        if obs_idx < 0:
            continue

        results.append({
            "structure_number": row["structure_number"],
            "hwb_class": hwb,
            "im_selected": round(float(row["im_selected"]), 4),
            "observed": obs_ds,
            "observed_idx": obs_idx,
            "predicted": pred_ds,
            "predicted_idx": pred_idx,
            "expected_idx": round(expected_idx, 2),
            "p_none": round(probs["none"], 3),
            "p_slight": round(probs["slight"], 3),
            "p_moderate": round(probs["moderate"], 3),
            "p_extensive": round(probs["extensive"], 3),
            "p_complete": round(probs["complete"], 3),
            "correct": pred_ds == obs_ds,
            "error": pred_idx - obs_idx,
        })

    return results


def _validate_with_shakemap(
    confirmed: pd.DataFrame, shakemap: pd.DataFrame, config,
) -> list[dict]:
    """Validate by interpolating ShakeMap IM directly for validation bridges."""
    from src.interpolation import interpolate_im
    from src.config import IM_COLUMN_MAP

    im_type = getattr(config, "im_type", "SA10")
    sm_col = IM_COLUMN_MAP.get(im_type, "PSA10")
    if sm_col not in shakemap.columns:
        print(f"[Validation] WARNING: ShakeMap missing column '{sm_col}' for {im_type}.")
        return []

    interp_method = getattr(config, "interpolation_method", "nearest")
    interp_params = getattr(config, "interpolation_params", {})

    bridge_lats = confirmed["latitude"].values
    bridge_lons = confirmed["longitude"].values
    grid_lats = shakemap["LAT"].values
    grid_lons = shakemap["LON"].values
    grid_vals = shakemap[sm_col].values

    im_values = interpolate_im(
        grid_lats, grid_lons, grid_vals,
        bridge_lats, bridge_lons,
        method=interp_method, **interp_params,
    )

    results = []
    for i, (_, row) in enumerate(confirmed.iterrows()):
        hwb = row["hwb_class"]
        sa_val = float(im_values[i])

        probs = damage_state_probabilities(sa_val, hwb)
        pred_ds = max(DS_ORDER, key=lambda ds: probs[ds])
        pred_idx = DS_INDEX[pred_ds]
        expected_idx = sum(DS_INDEX[ds] * probs[ds] for ds in DS_ORDER)

        obs_ds = row["observed_damage_state"]
        obs_idx = DS_INDEX.get(obs_ds, -1)
        if obs_idx < 0:
            continue

        results.append({
            "structure_number": row["structure_number"],
            "hwb_class": hwb,
            "im_shakemap": round(sa_val, 4),
            "observed": obs_ds,
            "observed_idx": obs_idx,
            "predicted": pred_ds,
            "predicted_idx": pred_idx,
            "expected_idx": round(expected_idx, 2),
            "p_none": round(probs["none"], 3),
            "p_slight": round(probs["slight"], 3),
            "p_moderate": round(probs["moderate"], 3),
            "p_extensive": round(probs["extensive"], 3),
            "p_complete": round(probs["complete"], 3),
            "correct": pred_ds == obs_ds,
            "error": pred_idx - obs_idx,
        })

    print(f"[Validation] ShakeMap interpolation: {len(results)} bridges processed, "
          f"IM range: {min(r['im_shakemap'] for r in results):.4f}g – "
          f"{max(r['im_shakemap'] for r in results):.4f}g")

    return results


def compute_validation_metrics(predicted: pd.Series, observed: pd.Series) -> dict:
    """
    Compute accuracy, MAE, bias, and confusion matrix.

    Parameters
    ----------
    predicted : pd.Series of damage state strings
    observed : pd.Series of damage state strings

    Returns
    -------
    dict with accuracy, mae, bias, confusion_matrix
    """
    pred_idx = predicted.map(DS_INDEX)
    obs_idx = observed.map(DS_INDEX)
    valid = (pred_idx >= 0) & (obs_idx >= 0)
    pred_idx = pred_idx[valid]
    obs_idx = obs_idx[valid]

    if len(pred_idx) == 0:
        return {"accuracy": 0.0, "mae": 0.0, "bias": 0.0, "confusion_matrix": {}}

    correct = (pred_idx == obs_idx).sum()
    accuracy = correct / len(pred_idx)
    error = pred_idx - obs_idx
    mae = error.abs().mean()
    bias = error.mean()

    # Confusion matrix as nested dict
    cm = {}
    for obs_ds in DS_ORDER:
        cm[obs_ds] = {}
        for pred_ds in DS_ORDER:
            cm[obs_ds][pred_ds] = int(
                ((observed == obs_ds) & (predicted == pred_ds)).sum()
            )

    return {
        "accuracy": float(accuracy),
        "mae": float(mae),
        "bias": float(bias),
        "confusion_matrix": cm,
    }


def plot_validation_results(metrics: dict, output_dir: str) -> list[str]:
    """
    Generate validation plots: confusion matrix heatmap and residual histogram.

    Returns list of saved file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # 1. Confusion matrix heatmap
    cm = metrics.get("confusion_matrix", {})
    if cm:
        fig, ax = plt.subplots(figsize=(7, 6))
        labels = DS_ORDER
        matrix = np.array([[cm.get(obs, {}).get(pred, 0)
                            for pred in labels] for obs in labels])

        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_title(
            f"Confusion Matrix  (Accuracy: {metrics['accuracy']:.1%}, "
            f"MAE: {metrics['mae']:.2f})"
        )

        # Annotate cells
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix[i, j]
                if val > 0:
                    color = "white" if val > matrix.max() / 2 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", color=color)

        fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
        fig.tight_layout()
        path = os.path.join(output_dir, "validation_confusion_matrix.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"  Saved: {path}")

    # 2. Residual histogram
    per_bridge = metrics.get("per_bridge")
    if per_bridge is not None and len(per_bridge) > 0 and "error" in per_bridge.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        errors = per_bridge["error"]
        bins = range(int(errors.min()) - 1, int(errors.max()) + 2)
        ax.hist(errors, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Perfect")
        ax.set_xlabel("Prediction Error (predicted - observed DS index)")
        ax.set_ylabel("Count")
        ax.set_title(f"Prediction Residuals  (Bias: {metrics['bias']:+.2f})")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(output_dir, "validation_residuals.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"  Saved: {path}")

    return saved


def _print_validation_summary(rdf: pd.DataFrame, metrics: dict, config) -> None:
    """Print validation results to console."""
    print(f"\n{'=' * 70}")
    print(f"VALIDATION RESULTS (N={len(rdf)})")
    print(f"{'=' * 70}")

    im_source = getattr(config, "validation_im_source", "gmpe")
    print(f"  IM source: {im_source}")
    print(f"  Exact match accuracy: {metrics['accuracy']:.1%}")
    print(f"  Mean absolute error:  {metrics['mae']:.2f} damage states")
    bias = metrics["bias"]
    print(f"  Bias: {bias:+.2f} ({'over-predicts' if bias > 0 else 'under-predicts'})")

    # Confusion matrix text
    cm = metrics.get("confusion_matrix", {})
    if cm:
        print(f"\n  Confusion matrix (rows=observed, cols=predicted):")
        print(f"  {'':>12s}  {'none':>6s}  {'slight':>6s}  {'mod':>6s}  {'ext':>6s}  {'comp':>6s}")
        print(f"  {'─' * 60}")
        for obs_ds in DS_ORDER:
            counts = [cm.get(obs_ds, {}).get(p, 0) for p in DS_ORDER]
            total = sum(counts)
            if total > 0:
                line = f"  {obs_ds:>12s}  " + "  ".join(f"{c:>6d}" for c in counts)
                print(line)

    print(f"{'=' * 70}")

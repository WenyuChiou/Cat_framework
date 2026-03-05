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

# Only observed_damage_state is truly required.
# Bridge identification: structure_number OR (latitude + longitude)
ENRICHABLE_COLS = ["latitude", "longitude", "hwb_class", "year_built", "material"]

# Spatial match threshold (km) for nearest-NBI lookup
SPATIAL_MATCH_THRESHOLD_KM = 0.5


def create_validation_template(output_path: str, n_examples: int = 5) -> str:
    """
    Generate a template CSV for users to fill in observed damage data.

    Two identification modes are supported:
    - By structure_number: auto-enriches lat/lon/hwb from NBI
    - By lat/lon: finds nearest NBI bridge within threshold

    Parameters
    ----------
    output_path : str
        Where to save the template CSV.
    n_examples : int
        Number of example rows to include.

    Returns
    -------
    str : path to created file
    """
    rows = [
        {"structure_number": "53-2795", "latitude": "", "longitude": "",
         "observed_damage_state": "complete", "hwb_class": "",
         "damage_description": "Column shear failure", "data_source": "field_inspection"},
        {"structure_number": "53-0566", "latitude": "", "longitude": "",
         "observed_damage_state": "slight", "hwb_class": "",
         "damage_description": "Minor spalling", "data_source": "field_inspection"},
        {"structure_number": "", "latitude": "34.328", "longitude": "-118.396",
         "observed_damage_state": "complete", "hwb_class": "",
         "damage_description": "Coordinate-only example", "data_source": "field_inspection"},
        {"structure_number": "", "latitude": "34.2", "longitude": "-118.5",
         "observed_damage_state": "moderate", "hwb_class": "HWB5",
         "damage_description": "User-provided HWB class", "data_source": "field_inspection"},
        {"structure_number": "YOUR_ID", "latitude": "", "longitude": "",
         "observed_damage_state": "none", "hwb_class": "",
         "damage_description": "", "data_source": ""},
    ][:n_examples]

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[Validation] Template saved: {output_path}")
    print(f"[Validation] Bridge identification (at least one):")
    print(f"  - structure_number: matches NBI by ID")
    print(f"  - latitude + longitude: matches nearest NBI bridge within {SPATIAL_MATCH_THRESHOLD_KM}km")
    print(f"[Validation] Required: observed_damage_state ({DS_ORDER})")
    print(f"[Validation] Optional: hwb_class (auto-enriched from NBI if missing)")
    return output_path


def load_validation_data(
    csv_path: str,
    nbi_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Load and normalize validation data from user-provided CSV.

    Supports three identification modes (per row):
    1. structure_number matches NBI → enrich from NBI
    2. latitude + longitude → spatial match to nearest NBI bridge
    3. latitude + longitude + hwb_class → no NBI needed

    Parameters
    ----------
    csv_path : str
        Path to validation CSV.
    nbi_df : pd.DataFrame, optional
        NBI bridge data for enriching missing columns.

    Returns
    -------
    pd.DataFrame with standardized columns.
    """
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Must have observed_damage_state
    if "observed_damage_state" not in df.columns:
        raise ValueError(
            "Validation CSV missing 'observed_damage_state' column. "
            "Use create_validation_template() to generate a template."
        )

    # Must have at least one identification method
    has_id = "structure_number" in df.columns
    has_coords = "latitude" in df.columns and "longitude" in df.columns
    if not has_id and not has_coords:
        raise ValueError(
            "Validation CSV must have 'structure_number' and/or "
            "'latitude'+'longitude' columns to identify bridges."
        )

    # Normalize damage state strings
    df["observed_damage_state"] = df["observed_damage_state"].str.strip().str.lower()
    invalid_ds = df[~df["observed_damage_state"].isin(DS_ORDER + ["unknown"])]
    if len(invalid_ds) > 0:
        bad_vals = invalid_ds["observed_damage_state"].unique().tolist()
        print(f"[Validation] WARNING: Unrecognized damage states {bad_vals} "
              f"will be excluded. Valid: {DS_ORDER}")

    # Add damage_confirmed if missing (assume all rows are confirmed)
    if "damage_confirmed" not in df.columns:
        df["damage_confirmed"] = True
        print(f"[Validation] No 'damage_confirmed' column — treating all "
              f"{len(df)} rows as confirmed observations.")

    # Ensure enrichable columns exist (even if NaN)
    # Use object dtype for string columns to avoid FutureWarning on mixed types
    for col in ENRICHABLE_COLS:
        if col not in df.columns:
            if col in ("hwb_class", "material"):
                df[col] = pd.Series([np.nan] * len(df), dtype="object")
            else:
                df[col] = np.nan

    # Ensure structure_number column exists
    if "structure_number" not in df.columns:
        df["structure_number"] = ""

    # Convert lat/lon to numeric (handles empty strings from template)
    if has_coords:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    if nbi_df is None or len(nbi_df) == 0:
        _check_completeness(df)
        return df

    # ── Enrichment Strategy ──
    # Phase 1: structure_number match
    n_enriched_by_id = _enrich_by_structure_number(df, nbi_df)

    # Phase 2: spatial match for rows still missing hwb_class or lat/lon
    n_enriched_by_spatial = _enrich_by_spatial_match(df, nbi_df)

    total = n_enriched_by_id + n_enriched_by_spatial
    if total > 0:
        print(f"[Validation] Enrichment summary: "
              f"{n_enriched_by_id} by ID, {n_enriched_by_spatial} by spatial match")

    _check_completeness(df)
    return df


def _enrich_by_structure_number(df: pd.DataFrame, nbi_df: pd.DataFrame) -> int:
    """Enrich validation rows by matching structure_number to NBI. Returns count enriched."""
    if "structure_number" not in nbi_df.columns:
        return 0

    # Rows that have a structure_number but are missing enrichable data
    has_sn = df["structure_number"].notna() & (df["structure_number"] != "")
    needs_enrich = has_sn & (df["hwb_class"].isna() | df["latitude"].isna())

    if needs_enrich.sum() == 0:
        return 0

    nbi_cols = ["structure_number"] + [c for c in ENRICHABLE_COLS if c in nbi_df.columns]
    nbi_lookup = nbi_df[nbi_cols].drop_duplicates("structure_number")

    n_matched = 0
    for idx in df.index[needs_enrich]:
        sn = df.at[idx, "structure_number"]
        nbi_row = nbi_lookup[nbi_lookup["structure_number"] == sn]
        if len(nbi_row) == 0:
            continue
        nbi_row = nbi_row.iloc[0]
        enriched = False
        for col in ENRICHABLE_COLS:
            if col in nbi_row.index and pd.isna(df.at[idx, col]):
                df.at[idx, col] = nbi_row[col]
                enriched = True
        if enriched:
            n_matched += 1

    if n_matched > 0:
        print(f"[Validation] Enriched {n_matched}/{needs_enrich.sum()} bridges by structure_number match.")

    return n_matched


def _enrich_by_spatial_match(df: pd.DataFrame, nbi_df: pd.DataFrame) -> int:
    """Enrich validation rows by finding nearest NBI bridge within threshold. Returns count enriched."""
    if "latitude" not in nbi_df.columns or "longitude" not in nbi_df.columns:
        return 0

    # Rows that have coordinates but still missing hwb_class
    has_coords = df["latitude"].notna() & df["longitude"].notna()
    needs_hwb = df["hwb_class"].isna() | (df["hwb_class"] == "")
    spatial_candidates = has_coords & needs_hwb

    if spatial_candidates.sum() == 0:
        return 0

    nbi_lats = nbi_df["latitude"].values
    nbi_lons = nbi_df["longitude"].values

    n_matched = 0
    for idx in df.index[spatial_candidates]:
        vlat = float(df.at[idx, "latitude"])
        vlon = float(df.at[idx, "longitude"])

        # Find nearest NBI bridge
        dists = np.array([
            haversine_km(vlat, vlon, nlat, nlon)
            for nlat, nlon in zip(nbi_lats, nbi_lons)
        ])
        min_idx = int(np.argmin(dists))
        min_dist = dists[min_idx]

        if min_dist <= SPATIAL_MATCH_THRESHOLD_KM:
            nbi_row = nbi_df.iloc[min_idx]
            for col in ENRICHABLE_COLS:
                if col in nbi_row.index and pd.isna(df.at[idx, col]):
                    df.at[idx, col] = nbi_row[col]
            # Also fill structure_number if missing
            if pd.isna(df.at[idx, "structure_number"]) or df.at[idx, "structure_number"] == "":
                df.at[idx, "structure_number"] = nbi_row["structure_number"]
            n_matched += 1
        else:
            print(f"[Validation] Row {idx}: nearest NBI bridge is {min_dist:.1f}km away "
                  f"(>{SPATIAL_MATCH_THRESHOLD_KM}km threshold). "
                  f"No spatial match for ({vlat:.4f}, {vlon:.4f}).")

    if n_matched > 0:
        print(f"[Validation] Enriched {n_matched}/{spatial_candidates.sum()} bridges "
              f"by spatial match (within {SPATIAL_MATCH_THRESHOLD_KM}km).")

    return n_matched


def _check_completeness(df: pd.DataFrame) -> None:
    """Print warnings for rows that still lack essential data after enrichment."""
    n_no_hwb = df["hwb_class"].isna().sum() + (df["hwb_class"] == "").sum()
    n_no_coords = df["latitude"].isna().sum()
    total = len(df)

    if n_no_hwb > 0:
        print(f"[Validation] WARNING: {n_no_hwb}/{total} bridges still missing hwb_class. "
              f"These cannot be validated (no fragility curve).")
    if n_no_coords > 0:
        print(f"[Validation] WARNING: {n_no_coords}/{total} bridges missing coordinates. "
              f"GMPE-based validation requires lat/lon.")


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
    # Load and normalize observed data (enrich from pipeline NBI if available)
    val_df = load_validation_data(validation_csv_path, nbi_df=bridges_df)
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
        lat = row.get("latitude")
        lon = row.get("longitude")
        hwb = row.get("hwb_class", "")

        if pd.isna(lat) or pd.isna(lon) or not hwb or pd.isna(hwb):
            continue

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
        hwb = row.get("hwb_class", "")
        if not hwb or pd.isna(hwb):
            continue
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

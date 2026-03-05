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
            "latitude": lat,
            "longitude": lon,
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
            "latitude": row.get("latitude", np.nan),
            "longitude": row.get("longitude", np.nan),
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
            "latitude": row.get("latitude", np.nan),
            "longitude": row.get("longitude", np.nan),
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
    Generate comprehensive validation plots.

    Produces:
    1. Confusion matrix heatmap
    2. Residual histogram
    3. Spatial residual map
    4. Observed vs predicted damage by IM bin (fragility validation)
    5. Per-class accuracy breakdown
    6. Residual vs distance from epicenter

    Returns list of saved file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    os.makedirs(output_dir, exist_ok=True)
    saved = []
    per_bridge = metrics.get("per_bridge")
    if per_bridge is None or len(per_bridge) == 0:
        return saved

    # ── 1. Confusion matrix heatmap ──
    cm = metrics.get("confusion_matrix", {})
    if cm:
        path = _plot_confusion_matrix(cm, metrics, output_dir, plt)
        saved.append(path)

    # ── 2. Residual histogram ──
    if "error" in per_bridge.columns:
        path = _plot_residual_histogram(per_bridge, metrics, output_dir, plt)
        saved.append(path)

    # ── 3. Spatial residual map ──
    if "latitude" in per_bridge.columns and "longitude" in per_bridge.columns:
        has_coords = per_bridge["latitude"].notna() & per_bridge["longitude"].notna()
        if has_coords.sum() >= 3:
            path = _plot_spatial_residual_map(per_bridge[has_coords], output_dir, plt, TwoSlopeNorm)
            saved.append(path)

    # ── 4. Observed vs predicted damage ratio by IM bin ──
    im_col = None
    for c in ["im_gmpe", "im_shakemap", "im_selected"]:
        if c in per_bridge.columns:
            im_col = c
            break
    if im_col:
        path = _plot_damage_ratio_by_im(per_bridge, im_col, output_dir, plt)
        saved.append(path)

    # ── 5. Per-class accuracy ──
    if "hwb_class" in per_bridge.columns:
        path = _plot_per_class_accuracy(per_bridge, output_dir, plt)
        saved.append(path)

    # ── 6. Residual vs distance ──
    if "r_jb_km" in per_bridge.columns:
        path = _plot_residual_vs_distance(per_bridge, output_dir, plt)
        saved.append(path)

    return saved


def _plot_confusion_matrix(cm, metrics, output_dir, plt):
    """Confusion matrix heatmap."""
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
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > matrix.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_01_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _plot_residual_histogram(per_bridge, metrics, output_dir, plt):
    """Residual histogram with normal overlay."""
    fig, ax = plt.subplots(figsize=(7, 5))
    errors = per_bridge["error"]
    bins = range(int(errors.min()) - 1, int(errors.max()) + 2)
    ax.hist(errors, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.axvline(errors.mean(), color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean bias: {errors.mean():+.2f}")
    ax.set_xlabel("Prediction Error (predicted - observed DS index)")
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction Residuals  (N={len(errors)}, Bias: {metrics['bias']:+.2f})")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_02_residuals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _plot_spatial_residual_map(df, output_dir, plt, TwoSlopeNorm):
    """Spatial map of prediction errors — shows where model over/under-predicts."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    errors = df["error"].values
    lats = df["latitude"].values
    lons = df["longitude"].values

    # Left: Spatial residual (error) map
    ax = axes[0]
    vmax = max(abs(errors.min()), abs(errors.max()), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sc = ax.scatter(lons, lats, c=errors, cmap="RdBu_r", norm=norm,
                    s=40, edgecolors="black", linewidths=0.3, alpha=0.8)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Error (pred - obs DS index)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Prediction Error\n(red=over-predict, blue=under-predict)")
    ax.set_aspect("equal")

    # Right: Observed vs predicted side-by-side
    ax = axes[1]
    obs_idx = df["observed_idx"].values
    pred_idx = df["predicted_idx"].values
    sc = ax.scatter(obs_idx + np.random.uniform(-0.15, 0.15, len(obs_idx)),
                    pred_idx + np.random.uniform(-0.15, 0.15, len(pred_idx)),
                    c=errors, cmap="RdBu_r", norm=norm,
                    s=40, edgecolors="black", linewidths=0.3, alpha=0.7)
    # 1:1 line
    ax.plot([-0.5, 4.5], [-0.5, 4.5], "k--", linewidth=1, label="Perfect")
    ax.set_xticks(range(5))
    ax.set_xticklabels(DS_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels(DS_ORDER)
    ax.set_xlabel("Observed Damage State")
    ax.set_ylabel("Predicted Damage State")
    ax.set_title("Observed vs Predicted\n(jittered for visibility)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    fig.suptitle(f"Validation Spatial Analysis (N={len(df)})", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_03_spatial_residual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _plot_damage_ratio_by_im(per_bridge, im_col, output_dir, plt):
    """Observed vs predicted damage distribution by IM bin — fragility validation."""
    df = per_bridge.copy()
    im_vals = df[im_col]

    # Create IM bins
    bin_edges = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, float("inf")]
    bin_labels = ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.8", "0.8-1.2", ">1.2"]
    df["im_bin"] = pd.cut(im_vals, bins=bin_edges, labels=bin_labels, right=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ds_colors = {
        "none": "#2ecc71", "slight": "#f1c40f", "moderate": "#e67e22",
        "extensive": "#e74c3c", "complete": "#8e44ad",
    }

    for ax_idx, (col_name, title) in enumerate([
        ("observed", "Observed Damage"),
        ("predicted", "Predicted Damage"),
    ]):
        ax = axes[ax_idx]
        # Count per bin per DS
        bin_groups = df.groupby("im_bin", observed=True)
        bin_counts = []
        for b in bin_labels:
            grp = df[df["im_bin"] == b]
            if len(grp) == 0:
                bin_counts.append({ds: 0 for ds in DS_ORDER})
                continue
            counts = grp[col_name].value_counts(normalize=True)
            bin_counts.append({ds: counts.get(ds, 0) for ds in DS_ORDER})

        # Stacked bar
        bottoms = np.zeros(len(bin_labels))
        for ds in DS_ORDER:
            vals = [bc[ds] for bc in bin_counts]
            ax.bar(bin_labels, vals, bottom=bottoms, label=ds.capitalize(),
                   color=ds_colors[ds], edgecolor="white", linewidth=0.5)
            bottoms += vals

        ax.set_xlabel(f"IM ({im_col}) bin (g)")
        ax.set_ylabel("Proportion")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_ylim(0, 1.05)

        # Add sample count labels
        for i, b in enumerate(bin_labels):
            n = len(df[df["im_bin"] == b])
            if n > 0:
                ax.text(i, 1.01, f"n={n}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Damage Distribution by IM Level — Observed vs Predicted", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_04_damage_by_im.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _plot_per_class_accuracy(per_bridge, output_dir, plt):
    """Per-HWB-class accuracy bar chart."""
    df = per_bridge.copy()
    class_stats = df.groupby("hwb_class").agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        mae=("error", lambda x: x.abs().mean()),
        bias=("error", "mean"),
    ).sort_values("n", ascending=False)

    # Only show classes with >= 2 bridges
    class_stats = class_stats[class_stats["n"] >= 2]
    if len(class_stats) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Accuracy + sample size
    ax = axes[0]
    colors = ["#2ecc71" if a >= 0.5 else "#e74c3c" for a in class_stats["accuracy"]]
    bars = ax.barh(range(len(class_stats)), class_stats["accuracy"],
                   color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(class_stats)))
    ax.set_yticklabels(class_stats.index)
    ax.set_xlabel("Accuracy")
    ax.set_title("Exact Match Accuracy by HWB Class")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, 1.0)
    for i, (_, row) in enumerate(class_stats.iterrows()):
        ax.text(row["accuracy"] + 0.02, i, f"n={int(row['n'])}", va="center", fontsize=8)

    # Right: Bias per class
    ax = axes[1]
    colors = ["#e74c3c" if b > 0 else "#3498db" for b in class_stats["bias"]]
    ax.barh(range(len(class_stats)), class_stats["bias"],
            color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(class_stats)))
    ax.set_yticklabels(class_stats.index)
    ax.set_xlabel("Mean Bias (positive = over-predicts)")
    ax.set_title("Prediction Bias by HWB Class")
    ax.axvline(0, color="black", linewidth=1)

    fig.suptitle("Per-Class Validation Performance", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_05_per_class_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _plot_residual_vs_distance(per_bridge, output_dir, plt):
    """Residual vs Rjb distance — tests GMPE attenuation correctness."""
    df = per_bridge[per_bridge["r_jb_km"].notna()].copy()
    if len(df) < 3:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Error vs distance
    ax = axes[0]
    ax.scatter(df["r_jb_km"], df["error"], alpha=0.5, s=30, c="steelblue", edgecolors="black", linewidths=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    # Running mean trend
    df_sorted = df.sort_values("r_jb_km")
    if len(df_sorted) >= 10:
        window = max(5, len(df_sorted) // 8)
        rolling_mean = df_sorted["error"].rolling(window, center=True).mean()
        ax.plot(df_sorted["r_jb_km"], rolling_mean, color="orange", linewidth=2,
                label=f"Rolling mean (w={window})")
        ax.legend()

    ax.set_xlabel("Distance R_JB (km)")
    ax.set_ylabel("Prediction Error (pred - obs DS index)")
    ax.set_title("Prediction Error vs Distance")

    # Right: IM vs distance (attenuation check)
    ax = axes[1]
    im_col = None
    for c in ["im_gmpe", "im_shakemap", "im_selected"]:
        if c in df.columns:
            im_col = c
            break
    if im_col:
        correct = df[df["correct"] == True]
        wrong = df[df["correct"] == False]
        if len(wrong) > 0:
            ax.scatter(wrong["r_jb_km"], wrong[im_col], alpha=0.5, s=30,
                       c="red", marker="x", label=f"Incorrect (n={len(wrong)})")
        if len(correct) > 0:
            ax.scatter(correct["r_jb_km"], correct[im_col], alpha=0.5, s=30,
                       c="green", marker="o", edgecolors="black", linewidths=0.3,
                       label=f"Correct (n={len(correct)})")
        ax.set_xlabel("Distance R_JB (km)")
        ax.set_ylabel(f"IM ({im_col}) (g)")
        ax.set_title("IM Attenuation with Prediction Accuracy")
        ax.legend()

    fig.suptitle("Distance-Dependent Validation", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_06_residual_vs_distance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


##############################################################################
# Level 1 — GMPE Component Validation (stations)
##############################################################################

def validate_gmpe_stations(
    stationlist_path: str,
    config,
    gmpe_model: str = "bssa21",
) -> dict:
    """
    Level 1: Compare BSSA21 predictions vs 185 seismic station observations.

    Parameters
    ----------
    stationlist_path : str
        Path to USGS stationlist.json.
    config : AnalysisConfig
        Analysis configuration (provides gmpe_scenario).
    gmpe_model : str
        GMPE model name (default: bssa21).

    Returns
    -------
    dict with level, metrics, per_station DataFrame, comparison_vs_shakemap.
    """
    from src.stationlist_parser import parse_stationlist
    from src.gmpe_bssa21 import BSSA21

    stations = parse_stationlist(stationlist_path, station_type="seismic")
    if len(stations) == 0:
        print("[L1] No seismic stations found.")
        return {"level": 1, "metrics": {}, "per_station": pd.DataFrame(),
                "comparison_vs_shakemap": {}}

    sc = getattr(config, "gmpe_scenario", None) or {
        "Mw": 6.7, "lat": 34.213, "lon": -118.537,
        "depth_km": 18.4, "fault_type": "reverse", "vs30": 360.0,
    }

    bssa = BSSA21()
    eq_mw = sc.get("Mw", 6.7)
    eq_fault = sc.get("fault_type", "reverse")
    target_period = 1.0  # Sa(1.0s)

    our_pred = np.empty(len(stations))
    our_sigma = np.empty(len(stations))
    for i, (_, row) in enumerate(stations.iterrows()):
        rjb = row["rjb"]
        if pd.isna(rjb) or rjb < 0.1:
            rjb = 0.1
        vs30 = row["vs30"] if pd.notna(row["vs30"]) else 360.0
        med_g, sig = bssa.compute(eq_mw, rjb, vs30, eq_fault, target_period)
        our_pred[i] = med_g
        our_sigma[i] = sig

    stations["our_pred_sa10"] = our_pred
    stations["our_sigma_sa10"] = our_sigma

    # Residuals in ln space
    obs = stations["obs_sa10"].values
    pred_ours = stations["our_pred_sa10"].values
    pred_sm = stations["pred_sa10"].values

    # Filter out invalid values
    valid = (obs > 0) & (pred_ours > 0) & (pred_sm > 0)
    obs_v = obs[valid]
    pred_ours_v = pred_ours[valid]
    pred_sm_v = pred_sm[valid]

    ln_obs = np.log(obs_v)
    ln_pred_ours = np.log(pred_ours_v)
    ln_pred_sm = np.log(pred_sm_v)

    residual_ours = ln_obs - ln_pred_ours
    residual_sm = ln_obs - ln_pred_sm

    stations["residual_ln"] = np.nan
    stations.loc[valid, "residual_ln"] = residual_ours
    stations["ratio_obs_pred"] = np.nan
    stations.loc[valid, "ratio_obs_pred"] = obs_v / pred_ours_v

    metrics = {
        "mean_residual": float(np.mean(residual_ours)),
        "std_residual": float(np.std(residual_ours)),
        "median_ratio": float(np.median(obs_v / pred_ours_v)),
        "rmse_ln": float(np.sqrt(np.mean(residual_ours**2))),
        "n_stations": int(valid.sum()),
        "inter_event_tau": float(np.mean(stations.loc[valid, "pred_ln_tau_sa10"].values)),
        "intra_event_phi": float(np.mean(stations.loc[valid, "pred_ln_phi_sa10"].values)),
        "total_sigma": float(np.mean(stations.loc[valid, "pred_ln_sigma_sa10"].values)),
    }

    # Correlation: our BSSA21 predicted vs station observed
    corr_pred_obs = float(np.corrcoef(pred_ours_v, obs_v)[0, 1])
    metrics["correlation_pred_obs"] = corr_pred_obs

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"LEVEL 1: GMPE COMPONENT VALIDATION (N={metrics['n_stations']} seismic stations)")
    print(f"{'=' * 70}")
    print(f"  Mean residual (ln):     {metrics['mean_residual']:+.3f}")
    print(f"  Std residual (ln):      {metrics['std_residual']:.3f}")
    print(f"  RMSE (ln):              {metrics['rmse_ln']:.3f}")
    print(f"  Median obs/pred ratio:  {metrics['median_ratio']:.3f}")
    print(f"  Correlation (pred vs obs): {corr_pred_obs:.3f}")
    print(f"  ShakeMap avg tau/phi/sigma: "
          f"{metrics['inter_event_tau']:.3f}/{metrics['intra_event_phi']:.3f}"
          f"/{metrics['total_sigma']:.3f}")
    print(f"{'=' * 70}")

    return {
        "level": 1,
        "metrics": metrics,
        "per_station": stations,
    }


def plot_level1_gmpe(results: dict, output_dir: str) -> list[str]:
    """
    Generate 5 Level 1 GMPE validation plots.

    1. Attenuation curve + station observations
    2. Residual vs Rjb distance
    3. Residual vs Vs30
    4. Residual histogram
    5. BSSA21 predicted vs station observed scatter
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    stations = results.get("per_station")
    if stations is None or len(stations) == 0:
        return saved

    valid = stations["residual_ln"].notna()
    df = stations[valid].copy()
    if len(df) == 0:
        return saved

    # ── 1. Attenuation curve + observations ──
    fig, ax = plt.subplots(figsize=(10, 7))
    from src.gmpe_bssa21 import BSSA21
    bssa = BSSA21()
    r_range = np.logspace(-0.5, 2.2, 200)
    pred_curve = np.array([bssa.compute(6.7, r, 360.0, "reverse", 1.0)[0] for r in r_range])
    ax.plot(r_range, pred_curve, "k-", linewidth=2, label="BSSA21 median (Vs30=360)")

    # +/- 1 sigma
    _, sig0 = bssa.compute(6.7, 10.0, 360.0, "reverse", 1.0)
    upper = pred_curve * np.exp(sig0)
    lower = pred_curve * np.exp(-sig0)
    ax.fill_between(r_range, lower, upper, alpha=0.15, color="gray", label=f"±1σ (σ={sig0:.2f})")

    # Color by Vs30
    vs30_vals = df["vs30"].values
    sc = ax.scatter(df["rjb"], df["obs_sa10"], c=vs30_vals, cmap="coolwarm_r",
                    s=30, edgecolors="black", linewidths=0.3, alpha=0.8,
                    vmin=200, vmax=800, zorder=5)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Vs30 (m/s)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("R_JB (km)")
    ax.set_ylabel("Sa(1.0s) (g)")
    ax.set_title("Level 1: Attenuation Curve + Seismic Station Observations\n"
                 "Northridge Mw 6.7, BSSA21")
    ax.legend(loc="upper right")
    ax.set_xlim(0.3, 200)
    ax.set_ylim(0.005, 3.0)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_L1_01_attenuation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    # ── 2. Residual vs distance ──
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["rjb"], df["residual_ln"], alpha=0.6, s=25, c="steelblue",
               edgecolors="black", linewidths=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

    # Moving average
    df_s = df.sort_values("rjb")
    if len(df_s) >= 10:
        window = max(5, len(df_s) // 8)
        rolling = df_s["residual_ln"].rolling(window, center=True).mean()
        ax.plot(df_s["rjb"], rolling, color="orange", linewidth=2,
                label=f"Moving average (w={window})")
        ax.legend()

    ax.set_xlabel("R_JB (km)")
    ax.set_ylabel("Residual: ln(obs) - ln(pred)")
    ax.set_title(f"Level 1: GMPE Residual vs Distance (N={len(df)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_L1_02_residual_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    # ── 3. Residual vs Vs30 ──
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["vs30"], df["residual_ln"], alpha=0.6, s=25, c="teal",
               edgecolors="black", linewidths=0.3)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

    # NEHRP boundaries
    nehrp = {"B/C": 760, "C/D": 360, "D/E": 180}
    for label, v in nehrp.items():
        ax.axvline(v, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.text(v + 5, ax.get_ylim()[1] * 0.9, label, fontsize=8, color="gray")

    ax.set_xlabel("Vs30 (m/s)")
    ax.set_ylabel("Residual: ln(obs) - ln(pred)")
    ax.set_title(f"Level 1: GMPE Residual vs Vs30 (N={len(df)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_L1_03_residual_vs30.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    # ── 4. Residual histogram + normal fit ──
    fig, ax = plt.subplots(figsize=(8, 6))
    residuals = df["residual_ln"].values
    n_bins = min(30, max(10, len(residuals) // 5))
    ax.hist(residuals, bins=n_bins, density=True, alpha=0.7, color="steelblue",
            edgecolor="black", label="Observed residuals")

    # Normal fit
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, pdf, "r-", linewidth=2,
            label=f"Normal fit (μ={mu:+.2f}, σ={sigma:.2f})")

    ax.axvline(0, color="green", linestyle="--", linewidth=1.5, label="Zero (unbiased)")
    ax.set_xlabel("Residual: ln(obs) - ln(pred)")
    ax.set_ylabel("Density")
    ax.set_title(f"Level 1: Residual Distribution (N={len(residuals)})")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_L1_04_residual_hist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    # ── 5. BSSA21 predicted vs station observed scatter ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df["our_pred_sa10"], df["obs_sa10"], alpha=0.6, s=30,
               c="steelblue", edgecolors="black", linewidths=0.3)
    lims = [0.005, max(df["obs_sa10"].max(), df["our_pred_sa10"].max()) * 1.2]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="1:1 line (perfect prediction)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BSSA21 Predicted Sa(1.0s) (g)")
    ax.set_ylabel("Station Observed Sa(1.0s) (g)")
    metrics = results.get("metrics", {})
    ax.set_title(f"Level 1: GMPE Predicted vs Station Observed\n"
                 f"mean residual(ln)={metrics.get('mean_residual', 0):+.3f}, "
                 f"r={metrics.get('correlation_pred_obs', 0):.3f}, "
                 f"N={len(df)}")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_L1_05_pred_vs_obs.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    # ── 6. Spatial residual map ──
    fig, ax = plt.subplots(figsize=(10, 8))

    # Earthquake epicenter
    ax.plot(-118.537, 34.213, "r*", markersize=18, zorder=10, label="Epicenter")

    # Stations colored by residual
    res_vals = df["residual_ln"].values
    vmax = max(abs(res_vals.min()), abs(res_vals.max()))
    sc = ax.scatter(df["lon"], df["lat"], c=res_vals, cmap="RdBu_r",
                    s=50, edgecolors="black", linewidths=0.4, alpha=0.85,
                    vmin=-vmax, vmax=vmax, zorder=5)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, label="Residual: ln(obs) − ln(pred)")

    # Add basemap if contextily available
    try:
        import contextily as cx
        ax.set_xlim(df["lon"].min() - 0.15, df["lon"].max() + 0.15)
        ax.set_ylim(df["lat"].min() - 0.1, df["lat"].max() + 0.1)
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron,
                       zoom=10, alpha=0.6)
    except ImportError:
        pass

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    m_res = results.get("metrics", {})
    ax.set_title(f"Level 1: Spatial Distribution of GMPE Residuals (N={len(df)})\n"
                 f"Red = GMPE under-predicts, Blue = GMPE over-predicts")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_L1_06_spatial_residual.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    return saved


##############################################################################
# Level 2 — Event-Level Damage Distribution Validation
##############################################################################

def validate_event_damage(
    bridges_csv_path: str,
    config,
    reference_stats: Optional[dict] = None,
) -> dict:
    """
    Level 2: Compare aggregate predicted damage distribution vs Basoz (1998).

    Parameters
    ----------
    bridges_csv_path : str
        Path to full bridge inventory CSV (2,123 bridges with sa1s_shakemap + hwb_class).
    config : AnalysisConfig
        Analysis configuration.
    reference_stats : dict, optional
        Override reference damage fractions. Defaults to NORTHRIDGE_DAMAGE_STATS.

    Returns
    -------
    dict with level, metrics, predicted/observed distributions, per_bridge DataFrame.
    """
    from src.northridge_case import NORTHRIDGE_DAMAGE_STATS

    if reference_stats is None:
        reference_stats = NORTHRIDGE_DAMAGE_STATS

    # Load full bridge inventory
    df = pd.read_csv(bridges_csv_path, encoding="utf-8")
    print(f"[L2] Loaded {len(df)} bridges from {bridges_csv_path}")

    # Need sa1s_shakemap and hwb_class
    if "sa1s_shakemap" not in df.columns:
        print("[L2] WARNING: 'sa1s_shakemap' column not found. Cannot compute Level 2.")
        return {"level": 2, "metrics": {}, "predicted_distribution": {},
                "observed_distribution": {}, "per_bridge": pd.DataFrame()}

    if "hwb_class" not in df.columns:
        print("[L2] WARNING: 'hwb_class' column not found. Cannot compute Level 2.")
        return {"level": 2, "metrics": {}, "predicted_distribution": {},
                "observed_distribution": {}, "per_bridge": pd.DataFrame()}

    # Filter to bridges with valid SA and HWB
    valid = df["sa1s_shakemap"].notna() & (df["sa1s_shakemap"] > 0) & df["hwb_class"].notna()
    df_valid = df[valid].copy()
    print(f"[L2] Valid bridges (SA > 0 + HWB): {len(df_valid)}")

    # Compute damage probabilities for each bridge
    prob_cols = {ds: [] for ds in DS_ORDER}
    pred_ds_list = []
    expected_idx_list = []

    for _, row in df_valid.iterrows():
        sa = float(row["sa1s_shakemap"])
        hwb = str(row["hwb_class"])
        probs = damage_state_probabilities(sa, hwb)

        for ds in DS_ORDER:
            prob_cols[ds].append(probs[ds])

        pred_ds = max(DS_ORDER, key=lambda d: probs[d])
        pred_ds_list.append(pred_ds)
        expected_idx_list.append(sum(DS_INDEX[d] * probs[d] for d in DS_ORDER))

    for ds in DS_ORDER:
        df_valid[f"p_{ds}"] = prob_cols[ds]
    df_valid["predicted_ds"] = pred_ds_list
    df_valid["expected_idx"] = expected_idx_list

    # Aggregate: probability-expectation method
    pred_dist = {ds: float(np.mean(prob_cols[ds])) for ds in DS_ORDER}
    obs_dist = reference_stats.get("observed_damage_fractions", {})

    # Compute distance from epicenter if lat/lon available
    if "latitude" in df_valid.columns and "longitude" in df_valid.columns:
        sc = getattr(config, "gmpe_scenario", None) or {"lat": 34.213, "lon": -118.537}
        eq_lat, eq_lon = sc.get("lat", 34.213), sc.get("lon", -118.537)
        df_valid["r_epi_km"] = df_valid.apply(
            lambda r: haversine_km(r["latitude"], r["longitude"], eq_lat, eq_lon)
            if pd.notna(r["latitude"]) else np.nan, axis=1)

    # Chi-squared test
    # Merge extensive+complete for chi-squared (low expected counts)
    pred_counts = np.array([pred_dist[ds] * len(df_valid) for ds in DS_ORDER])
    obs_fracs = np.array([obs_dist.get(ds, 0) for ds in DS_ORDER])
    obs_counts = obs_fracs * reference_stats.get("total_bridges_in_area", 1600)

    # Merge last two categories if expected < 5
    if pred_counts[-1] < 5 or obs_counts[-1] < 5:
        pred_counts_merged = np.concatenate([pred_counts[:-2], [pred_counts[-2] + pred_counts[-1]]])
        obs_counts_merged = np.concatenate([obs_counts[:-2], [obs_counts[-2] + obs_counts[-1]]])
    else:
        pred_counts_merged = pred_counts
        obs_counts_merged = obs_counts

    # Normalize to same total for fair comparison
    total_n = reference_stats.get("total_bridges_in_area", 1600)
    pred_norm = (pred_counts_merged / pred_counts_merged.sum()) * total_n
    obs_norm = (obs_counts_merged / obs_counts_merged.sum()) * total_n

    from scipy import stats as sp_stats
    # Chi-squared: compare predicted vs observed distributions
    # Use observed as expected frequencies
    chi2_stat = float(np.sum((pred_norm - obs_norm) ** 2 / np.maximum(obs_norm, 1)))
    chi2_dof = len(pred_norm) - 1
    chi2_pval = float(1.0 - sp_stats.chi2.cdf(chi2_stat, chi2_dof))

    # KL divergence: D_KL(pred || obs)
    pred_f = np.array([pred_dist[ds] for ds in DS_ORDER])
    obs_f = np.array([obs_dist.get(ds, 0) for ds in DS_ORDER])
    # Smooth zeros
    eps = 1e-8
    pred_f = np.clip(pred_f, eps, None)
    obs_f = np.clip(obs_f, eps, None)
    pred_f = pred_f / pred_f.sum()
    obs_f = obs_f / obs_f.sum()
    kl_div = float(np.sum(pred_f * np.log(pred_f / obs_f)))

    # Total damage ratio
    damage_weights = {"none": 0, "slight": 0.03, "moderate": 0.08, "extensive": 0.25, "complete": 1.0}
    tdr_pred = sum(pred_dist[ds] * damage_weights[ds] for ds in DS_ORDER)
    tdr_obs = sum(obs_dist.get(ds, 0) * damage_weights[ds] for ds in DS_ORDER)

    max_err = max(abs(pred_dist[ds] - obs_dist.get(ds, 0)) for ds in DS_ORDER)

    metrics = {
        "chi_squared": chi2_stat,
        "chi_squared_pvalue": chi2_pval,
        "kl_divergence": kl_div,
        "total_damage_ratio_pred": float(tdr_pred),
        "total_damage_ratio_obs": float(tdr_obs),
        "max_category_error": float(max_err),
    }

    print(f"\n{'=' * 70}")
    print(f"LEVEL 2: EVENT-LEVEL DAMAGE DISTRIBUTION (N={len(df_valid)} bridges)")
    print(f"{'=' * 70}")
    print(f"  {'State':<12} {'Predicted':>10} {'Observed':>10} {'Diff':>10}")
    print(f"  {'─' * 45}")
    for ds in DS_ORDER:
        p = pred_dist[ds]
        o = obs_dist.get(ds, 0)
        print(f"  {ds:<12} {p:>10.4f} {o:>10.4f} {p-o:>+10.4f}")
    print(f"  {'─' * 45}")
    print(f"  Chi-squared: {chi2_stat:.2f} (p={chi2_pval:.4f})")
    print(f"  KL divergence: {kl_div:.4f}")
    print(f"  Total damage ratio: pred={tdr_pred:.4f}, obs={tdr_obs:.4f}")
    print(f"{'=' * 70}")

    return {
        "level": 2,
        "metrics": metrics,
        "predicted_distribution": pred_dist,
        "observed_distribution": dict(obs_dist),
        "per_bridge": df_valid,
    }


def plot_level2_event(results: dict, output_dir: str) -> list[str]:
    """
    Generate 3 Level 2 event-level damage validation plots.

    1. Predicted vs observed damage distribution bar chart
    2. Damage rate vs distance
    3. Damage index by HWB class
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    pred_dist = results.get("predicted_distribution", {})
    obs_dist = results.get("observed_distribution", {})
    per_bridge = results.get("per_bridge")
    metrics = results.get("metrics", {})

    if not pred_dist or not obs_dist:
        return saved

    ds_colors = {
        "none": "#2ecc71", "slight": "#f1c40f", "moderate": "#e67e22",
        "extensive": "#e74c3c", "complete": "#8e44ad",
    }

    # ── 1. Side-by-side bar chart ──
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DS_ORDER))
    width = 0.35
    pred_vals = [pred_dist.get(ds, 0) for ds in DS_ORDER]
    obs_vals = [obs_dist.get(ds, 0) for ds in DS_ORDER]

    bars1 = ax.bar(x - width / 2, pred_vals, width, label="Predicted (this model)",
                   color="steelblue", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, obs_vals, width, label="Observed (Basoz 1998)",
                   color="coral", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([ds.capitalize() for ds in DS_ORDER])
    ax.set_ylabel("Fraction of Bridges")
    ax.set_title(f"Level 2: Predicted vs Observed Damage Distribution\n"
                 f"χ²={metrics.get('chi_squared', 0):.2f}, "
                 f"p={metrics.get('chi_squared_pvalue', 0):.4f}, "
                 f"KL={metrics.get('kl_divergence', 0):.4f}")
    ax.legend()

    # Value labels
    for bar_set in [bars1, bars2]:
        for bar in bar_set:
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, max(max(pred_vals), max(obs_vals)) * 1.2)
    fig.tight_layout()
    path = os.path.join(output_dir, "validation_L2_01_damage_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    if per_bridge is None or len(per_bridge) == 0:
        return saved

    # ── 2. Damage rate vs distance ──
    if "r_epi_km" in per_bridge.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        df = per_bridge[per_bridge["r_epi_km"].notna()].copy()
        if len(df) > 0:
            dist_bins = [0, 10, 20, 30, 50, 80, 120, 200]
            dist_labels = ["0-10", "10-20", "20-30", "30-50", "50-80", "80-120", "120+"]
            df["dist_bin"] = pd.cut(df["r_epi_km"], bins=dist_bins, labels=dist_labels, right=False)

            # Stacked area: mean probability by distance bin
            bin_probs = []
            for bl in dist_labels:
                grp = df[df["dist_bin"] == bl]
                if len(grp) > 0:
                    bin_probs.append({ds: grp[f"p_{ds}"].mean() for ds in DS_ORDER})
                else:
                    bin_probs.append({ds: 0 for ds in DS_ORDER})

            bottoms = np.zeros(len(dist_labels))
            for ds in DS_ORDER:
                vals = [bp[ds] for bp in bin_probs]
                ax.bar(dist_labels, vals, bottom=bottoms, label=ds.capitalize(),
                       color=ds_colors[ds], edgecolor="white", linewidth=0.5)
                bottoms += np.array(vals)

            # Sample count annotations
            for i, bl in enumerate(dist_labels):
                n = len(df[df["dist_bin"] == bl])
                if n > 0:
                    ax.text(i, bottoms[i] + 0.01, f"n={n}", ha="center", fontsize=7)

            ax.set_xlabel("Epicentral Distance (km)")
            ax.set_ylabel("Mean Damage Probability")
            ax.set_title("Level 2: Damage Distribution by Distance")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_ylim(0, 1.1)
            fig.tight_layout()
            path = os.path.join(output_dir, "validation_L2_02_damage_by_dist.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)
            print(f"  Saved: {path}")

    # ── 3. Damage index by HWB class ──
    if "hwb_class" in per_bridge.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        df = per_bridge.copy()
        class_stats = df.groupby("hwb_class").agg(
            n=("expected_idx", "size"),
            mean_di=("expected_idx", "mean"),
        ).sort_values("n", ascending=False)
        # Only classes with >= 5 bridges
        class_stats = class_stats[class_stats["n"] >= 5]

        if len(class_stats) > 0:
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(class_stats)))
            bars = ax.bar(range(len(class_stats)), class_stats["mean_di"],
                          color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(class_stats)))
            ax.set_xticklabels(class_stats.index, rotation=45, ha="right")
            ax.set_ylabel("Mean Expected Damage Index")
            ax.set_title("Level 2: Predicted Damage Index by HWB Class")

            for i, (_, row) in enumerate(class_stats.iterrows()):
                ax.text(i, row["mean_di"] + 0.02, f"n={int(row['n'])}",
                        ha="center", fontsize=7)

            fig.tight_layout()
            path = os.path.join(output_dir, "validation_L2_03_damage_by_hwb.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved.append(path)
            print(f"  Saved: {path}")

    return saved


##############################################################################
# Level 3 — Per-Bridge Validation (wrapper around existing logic)
##############################################################################

def validate_per_bridge(
    bridges_df: pd.DataFrame,
    config,
    validation_csv_path: str,
    shakemap: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Level 3 (supplementary): Per-bridge damage state comparison.

    Wraps existing run_validation logic with a data-quality caveat.

    Returns
    -------
    dict with level, metrics, per_bridge DataFrame, caveat string.
    """
    # Use existing core logic
    val_df = load_validation_data(validation_csv_path, nbi_df=bridges_df)
    confirmed = val_df[val_df["damage_confirmed"] == True].copy()  # noqa: E712

    if len(confirmed) == 0:
        print("[L3] No confirmed observations found.")
        return {"level": 3, "metrics": {"accuracy": 0, "mae": 0, "bias": 0},
                "per_bridge": pd.DataFrame(),
                "caveat": "No confirmed observations available."}

    print(f"[L3] Confirmed observations: {len(confirmed)}")

    im_source = getattr(config, "validation_im_source", "gmpe")
    if im_source == "gmpe":
        results = _validate_with_gmpe(confirmed, config)
    else:
        results = _validate_with_pipeline(confirmed, bridges_df)
        if len(results) == 0 and shakemap is not None:
            results = _validate_with_shakemap(confirmed, shakemap, config)
        elif len(results) == 0 and shakemap is None:
            results = _validate_with_gmpe(confirmed, config)

    if len(results) == 0:
        return {"level": 3, "metrics": {"accuracy": 0, "mae": 0, "bias": 0},
                "per_bridge": pd.DataFrame(),
                "caveat": "No matching bridges for validation."}

    rdf = pd.DataFrame(results)
    metrics = compute_validation_metrics(rdf["predicted"], rdf["observed"])
    metrics["per_bridge"] = rdf

    _print_validation_summary(rdf, metrics, config)

    caveat = (
        "Level 3 per-bridge validation is supplementary. NBI condition-rating "
        "differences are unreliable proxies for earthquake damage states. "
        "Accuracy is expected to be low (28-37%) due to data quality issues."
    )

    return {
        "level": 3,
        "metrics": {k: v for k, v in metrics.items() if k != "per_bridge"},
        "per_bridge": rdf,
        "caveat": caveat,
    }


def plot_level3_per_bridge(metrics_and_bridge: dict, output_dir: str) -> list[str]:
    """
    Generate Level 3 validation plots (renamed from validation_0N to validation_L3_0N).

    Reuses existing plot logic with updated filenames.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    per_bridge = metrics_and_bridge.get("per_bridge")
    level3_metrics = metrics_and_bridge.get("metrics", {})

    if per_bridge is None or len(per_bridge) == 0:
        return saved

    # Reconstruct metrics dict in the format plot functions expect
    cm = {}
    if "observed" in per_bridge.columns and "predicted" in per_bridge.columns:
        for obs_ds in DS_ORDER:
            cm[obs_ds] = {}
            for pred_ds in DS_ORDER:
                cm[obs_ds][pred_ds] = int(
                    ((per_bridge["observed"] == obs_ds) & (per_bridge["predicted"] == pred_ds)).sum()
                )

    plot_metrics = {
        "accuracy": level3_metrics.get("accuracy", 0),
        "mae": level3_metrics.get("mae", 0),
        "bias": level3_metrics.get("bias", 0),
        "confusion_matrix": cm,
        "per_bridge": per_bridge,
    }

    # 1. Confusion matrix
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
        ax.set_title(f"L3 Confusion Matrix  (Accuracy: {plot_metrics['accuracy']:.1%}, "
                     f"MAE: {plot_metrics['mae']:.2f})")
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix[i, j]
                if val > 0:
                    color = "white" if val > matrix.max() / 2 else "black"
                    ax.text(j, i, str(val), ha="center", va="center", color=color)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
        fig.tight_layout()
        path = os.path.join(output_dir, "validation_L3_01_confusion_matrix.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"  Saved: {path}")

    # 2. Residual histogram
    if "error" in per_bridge.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        errors = per_bridge["error"]
        bins = range(int(errors.min()) - 1, int(errors.max()) + 2)
        ax.hist(errors, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.axvline(errors.mean(), color="orange", linestyle="-", linewidth=1.5,
                   label=f"Mean bias: {errors.mean():+.2f}")
        ax.set_xlabel("Prediction Error (predicted - observed DS index)")
        ax.set_ylabel("Count")
        ax.set_title(f"L3 Residuals (N={len(errors)}, Bias: {plot_metrics['bias']:+.2f})")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(output_dir, "validation_L3_02_residuals.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"  Saved: {path}")

    # 3. Spatial residual
    if "latitude" in per_bridge.columns and "longitude" in per_bridge.columns:
        has_coords = per_bridge["latitude"].notna() & per_bridge["longitude"].notna()
        if has_coords.sum() >= 3 and "error" in per_bridge.columns:
            df_sp = per_bridge[has_coords]
            fig, ax = plt.subplots(figsize=(9, 7))
            errors = df_sp["error"].values
            vmax = max(abs(errors.min()), abs(errors.max()), 1)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            sc = ax.scatter(df_sp["longitude"], df_sp["latitude"], c=errors,
                            cmap="RdBu_r", norm=norm, s=40,
                            edgecolors="black", linewidths=0.3, alpha=0.8)
            fig.colorbar(sc, ax=ax, shrink=0.8, label="Error (pred - obs)")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("L3 Spatial Prediction Error")
            ax.set_aspect("equal")
            fig.tight_layout()
            path = os.path.join(output_dir, "validation_L3_03_spatial_residual.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
            print(f"  Saved: {path}")

    # 4. Damage by IM
    im_col = None
    for c in ["im_gmpe", "im_shakemap", "im_selected"]:
        if c in per_bridge.columns:
            im_col = c
            break
    if im_col:
        path = _plot_damage_ratio_by_im(per_bridge, im_col, output_dir, plt)
        if path:
            # Rename to L3 scheme
            import shutil
            new_path = os.path.join(output_dir, "validation_L3_04_damage_by_im.png")
            shutil.move(path, new_path)
            saved.append(new_path)

    # 5. Per-class accuracy
    if "hwb_class" in per_bridge.columns and "correct" in per_bridge.columns:
        path = _plot_per_class_accuracy(per_bridge, output_dir, plt)
        if path:
            import shutil
            new_path = os.path.join(output_dir, "validation_L3_05_per_class_accuracy.png")
            shutil.move(path, new_path)
            saved.append(new_path)

    # 6. Residual vs distance
    if "r_jb_km" in per_bridge.columns:
        path = _plot_residual_vs_distance(per_bridge, output_dir, plt)
        if path:
            import shutil
            new_path = os.path.join(output_dir, "validation_L3_06_residual_vs_distance.png")
            shutil.move(path, new_path)
            saved.append(new_path)

    return saved


##############################################################################
# Orchestrator — run_full_validation()
##############################################################################

def run_full_validation(
    config,
    bridges_df: Optional[pd.DataFrame] = None,
    shakemap: Optional[pd.DataFrame] = None,
    output_dir: str = "output/validation",
    levels: Optional[list[int]] = None,
) -> dict[int, dict]:
    """
    Execute full 3-level validation.

    Parameters
    ----------
    config : AnalysisConfig
        Analysis configuration.
    bridges_df : pd.DataFrame, optional
        NBI bridge data (for Level 3).
    shakemap : pd.DataFrame, optional
        ShakeMap grid data (for Level 3 shakemap mode).
    output_dir : str
        Directory for output plots and CSVs.
    levels : list[int], optional
        Which levels to run. Defaults to config.validation_levels or [1, 2, 3].

    Returns
    -------
    dict mapping level number to result dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    if levels is None:
        levels = getattr(config, "validation_levels", [1, 2, 3])

    all_results = {}

    # ── Level 1: GMPE Station Validation ──
    if 1 in levels:
        stationlist_path = getattr(config, "validation_stationlist", None)
        if stationlist_path is None:
            # Try default path
            import pathlib
            default = pathlib.Path(__file__).parent.parent / "data" / "stationlist.json"
            if default.exists():
                stationlist_path = str(default)

        if stationlist_path:
            try:
                print("\n[Validation] Running Level 1: GMPE Component Validation...")
                l1 = validate_gmpe_stations(stationlist_path, config)
                all_results[1] = l1
                plot_level1_gmpe(l1, output_dir)
            except Exception as e:
                print(f"[Validation] Level 1 FAILED: {e}")
                all_results[1] = {"level": 1, "error": str(e)}
        else:
            print("[Validation] Level 1 SKIPPED: no stationlist.json found.")

    # ── Level 2: Event-Level Damage Distribution ──
    if 2 in levels:
        val_csv = getattr(config, "validation_data", None)
        if val_csv:
            try:
                print("\n[Validation] Running Level 2: Event-Level Damage Distribution...")
                l2 = validate_event_damage(val_csv, config)
                all_results[2] = l2
                plot_level2_event(l2, output_dir)
            except Exception as e:
                print(f"[Validation] Level 2 FAILED: {e}")
                all_results[2] = {"level": 2, "error": str(e)}
        else:
            print("[Validation] Level 2 SKIPPED: no validation data path.")

    # ── Level 3: Per-Bridge Validation ──
    if 3 in levels:
        val_csv = getattr(config, "validation_data", None)
        if val_csv:
            try:
                print("\n[Validation] Running Level 3: Per-Bridge Validation...")
                if bridges_df is None:
                    bridges_df = pd.DataFrame()
                l3 = validate_per_bridge(bridges_df, config, val_csv, shakemap)
                all_results[3] = l3
                plot_level3_per_bridge(l3, output_dir)
            except Exception as e:
                print(f"[Validation] Level 3 FAILED: {e}")
                all_results[3] = {"level": 3, "error": str(e)}
        else:
            print("[Validation] Level 3 SKIPPED: no validation data path.")

    # Summary
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    for lv in sorted(all_results.keys()):
        r = all_results[lv]
        if "error" in r:
            print(f"  Level {lv}: FAILED — {r['error']}")
        else:
            m = r.get("metrics", {})
            if lv == 1:
                print(f"  Level 1 (GMPE):  mean_residual={m.get('mean_residual', '?'):+.3f}, "
                      f"std={m.get('std_residual', '?'):.3f}, "
                      f"N={m.get('n_stations', 0)}")
            elif lv == 2:
                print(f"  Level 2 (Event): chi2={m.get('chi_squared', '?'):.2f}, "
                      f"p={m.get('chi_squared_pvalue', '?'):.4f}, "
                      f"KL={m.get('kl_divergence', '?'):.4f}")
            elif lv == 3:
                print(f"  Level 3 (Bridge): accuracy={m.get('accuracy', '?'):.1%}, "
                      f"MAE={m.get('mae', '?'):.2f}, "
                      f"bias={m.get('bias', '?'):+.2f}")
    print(f"{'=' * 70}")

    return all_results


##############################################################################
# Existing helper functions (unchanged)
##############################################################################

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

"""
USGS ShakeMap stationlist.json parser.

Parses the GeoJSON station list into a flat DataFrame with observed
amplitudes, GMPE predictions, and distance metrics.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd


def parse_stationlist(
    filepath: str,
    station_type: Optional[str] = "seismic",
) -> pd.DataFrame:
    """
    Parse USGS ShakeMap stationlist.json into a flat DataFrame.

    Raw amplitude values are in %g and are converted to g (divided by 100).
    PGV remains in cm/s.

    Parameters
    ----------
    filepath : str
        Path to stationlist.json.
    station_type : str or None
        Filter by station_type (e.g. "seismic" for 185 instrument stations).
        None returns all stations.

    Returns
    -------
    pd.DataFrame with columns:
        station_id, name, lat, lon, vs30, station_type,
        obs_pga, obs_sa03, obs_sa10, obs_sa30, obs_pgv,
        pred_pga, pred_sa03, pred_sa10, pred_sa30, pred_pgv,
        pred_ln_tau_sa10, pred_ln_phi_sa10, pred_ln_sigma_sa10, pred_ln_bias_sa10,
        repi, rhypo, rjb, rrup, rx, ry0
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    rows = []

    # Mapping from stationlist amplitude names to our column suffixes
    obs_amp_map = {
        "pga": "obs_pga",
        "sa(0.3)": "obs_sa03",
        "sa(1.0)": "obs_sa10",
        "sa(3.0)": "obs_sa30",
        "pgv": "obs_pgv",
    }
    pred_amp_map = {
        "pga": "pred_pga",
        "sa(0.3)": "pred_sa03",
        "sa(1.0)": "pred_sa10",
        "sa(3.0)": "pred_sa30",
        "pgv": "pred_pgv",
    }
    # Units that need %g -> g conversion (everything except pgv)
    pct_g_names = {"pga", "sa(0.3)", "sa(1.0)", "sa(3.0)"}

    for feat in features:
        props = feat.get("properties", {})
        stype = props.get("station_type", "")
        if station_type is not None and stype != station_type:
            continue

        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])

        row = {
            "station_id": feat.get("id", ""),
            "name": props.get("name", ""),
            "lat": coords[1] if len(coords) > 1 else None,
            "lon": coords[0] if len(coords) > 0 else None,
            "vs30": props.get("vs30"),
            "station_type": stype,
        }

        # --- Observed amplitudes from channels ---
        channels = props.get("channels", [])
        obs_values = {}
        for ch in channels:
            for amp in ch.get("amplitudes", []):
                name = amp.get("name", "")
                if name in obs_amp_map:
                    val = amp.get("value")
                    if val is not None:
                        # Convert %g to g for acceleration IMs
                        if name in pct_g_names:
                            val = val / 100.0
                        obs_values[obs_amp_map[name]] = val

        for col in obs_amp_map.values():
            row[col] = obs_values.get(col, np.nan)

        # --- GMPE predictions ---
        predictions = props.get("predictions", [])
        pred_values = {}
        sa10_pred = {}
        for pred in predictions:
            name = pred.get("name", "")
            if name in pred_amp_map:
                val = pred.get("value")
                if val is not None:
                    if name in pct_g_names:
                        val = val / 100.0
                    pred_values[pred_amp_map[name]] = val
            # Capture uncertainty params for SA(1.0)
            if name == "sa(1.0)":
                sa10_pred = pred

        for col in pred_amp_map.values():
            row[col] = pred_values.get(col, np.nan)

        # SA(1.0) uncertainty parameters
        row["pred_ln_tau_sa10"] = sa10_pred.get("ln_tau", np.nan)
        row["pred_ln_phi_sa10"] = sa10_pred.get("ln_phi", np.nan)
        row["pred_ln_sigma_sa10"] = sa10_pred.get("ln_sigma", np.nan)
        row["pred_ln_bias_sa10"] = sa10_pred.get("ln_bias", np.nan)

        # --- Distance metrics ---
        distances = props.get("distances", {})
        for dkey in ("repi", "rhypo", "rjb", "rrup", "rx", "ry0"):
            row[dkey] = distances.get(dkey, np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # Ensure numeric types
    numeric_cols = [c for c in df.columns if c not in ("station_id", "name", "station_type")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

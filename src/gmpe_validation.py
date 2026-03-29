"""
GMPE Validation — Residual Analysis for NGA-West2 Models.

Compares GMPE-predicted ground motion against observed intensity measures
(ShakeMap or instrumental) and computes residuals stratified by NEHRP site
class.  Produces diagnostic plots and summary statistics.

Models supported: ASK14, BSSA14, CB14, CY14 (NGA-West2 suite).

Original analysis: Kubilay Albayrak (enhanced_gmpe_bridge_analysis.py)
Integrated into CAT411 framework, March 2025.

References:
  - Abrahamson, Silva & Kamai (2014). Earthquake Spectra 30(3), 1025-1055.
  - Boore, Stewart, Seyhan & Atkinson (2014). Earthquake Spectra 30(3), 1057-1085.
  - Campbell & Bozorgnia (2014). Earthquake Spectra 30(3), 1087-1115.
  - Chiou & Youngs (2014). Earthquake Spectra 30(3), 1117-1153.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Fault Parameters ──────────────────────────────────────────────────

@dataclass
class FaultParams:
    """Finite fault rupture geometry."""
    Mw: float = 6.7
    epi_lat: float = 34.213
    epi_lon: float = -118.537
    depth_km: float = 17.5
    strike: float = 122.0   # degrees
    dip: float = 40.0       # degrees
    rake: float = 101.0     # degrees
    length_km: float = 18.0
    width_km: float = 22.0
    ztor_km: float = 5.0
    frv: int = 1
    fnm: int = 0


# Northridge default
NORTHRIDGE_FAULT = FaultParams()


# ── NEHRP Site Classification ─────────────────────────────────────────

NEHRP_BREAKS = [180, 360, 760, 1500]
NEHRP_LABELS = ["E", "D", "C", "B/C", "B"]


def classify_nehrp(vs30: float) -> str:
    """Classify Vs30 into NEHRP site class."""
    if vs30 < 180:
        return "E"
    elif vs30 < 360:
        return "D"
    elif vs30 < 760:
        return "C"
    elif vs30 < 1500:
        return "B/C"
    else:
        return "B"


def classify_nehrp_series(vs30_series: pd.Series) -> pd.Series:
    """Vectorized NEHRP classification."""
    return pd.cut(
        vs30_series,
        bins=[0, 180, 360, 760, 1500, 99999],
        labels=NEHRP_LABELS,
        right=False,
    )


# ── Finite Fault Distance ────────────────────────────────────────────

def _local_cart(site_lat, site_lon, fault: FaultParams):
    """Site position in km relative to epicenter (E=x, N=y)."""
    kml = 111.32
    kmn = 111.32 * np.cos(np.radians(fault.epi_lat))
    return (site_lon - fault.epi_lon) * kmn, (site_lat - fault.epi_lat) * kml


def _fault_anchor(fault: FaultParams):
    """Compute corrected fault anchor (top-center of rupture)."""
    dip_r = np.radians(fault.dip)
    strike_r = np.radians(fault.strike)
    along_dip_to_top = (fault.depth_km - fault.ztor_km) / np.sin(dip_r)
    e2h_x = np.cos(strike_r) * np.cos(dip_r)
    e2h_y = -np.sin(strike_r) * np.cos(dip_r)
    ax = -e2h_x * along_dip_to_top
    ay = -e2h_y * along_dip_to_top
    az = -fault.ztor_km
    return ax, ay, az


def rrup_finite(site_lat, site_lon, fault: FaultParams) -> float:
    """Closest 3D distance to finite fault plane (km)."""
    sx, sy = _local_cart(site_lat, site_lon, fault)
    dip_r = np.radians(fault.dip)
    strike_r = np.radians(fault.strike)
    e1 = np.array([np.sin(strike_r), np.cos(strike_r), 0.0])
    e2 = np.array([np.cos(strike_r) * np.cos(dip_r),
                    -np.sin(strike_r) * np.cos(dip_r),
                    -np.sin(dip_r)])
    hx, hy, hz = _fault_anchor(fault)
    dx, dy, dz = sx - hx, sy - hy, 0.0 - hz
    pa = dx * e1[0] + dy * e1[1] + dz * e1[2]
    pd_ = dx * e2[0] + dy * e2[1] + dz * e2[2]
    pa_c = np.clip(pa, -fault.length_km / 2, fault.length_km / 2)
    pd_c = np.clip(pd_, 0.0, fault.width_km)
    cpx = hx + pa_c * e1[0] + pd_c * e2[0]
    cpy = hy + pa_c * e1[1] + pd_c * e2[1]
    cpz = hz + pa_c * e1[2] + pd_c * e2[2]
    return float(np.sqrt((sx - cpx)**2 + (sy - cpy)**2 + (0 - cpz)**2))


def rjb_finite(site_lat, site_lon, fault: FaultParams) -> float:
    """Closest horizontal distance to rupture projection (km)."""
    sx, sy = _local_cart(site_lat, site_lon, fault)
    strike_r = np.radians(fault.strike)
    dip_r = np.radians(fault.dip)
    e1 = np.array([np.sin(strike_r), np.cos(strike_r)])
    e2h = np.array([np.cos(strike_r) * np.cos(dip_r),
                     -np.sin(strike_r) * np.cos(dip_r)])
    hx, hy, _ = _fault_anchor(fault)
    dx, dy = sx - hx, sy - hy
    pa = dx * e1[0] + dy * e1[1]
    pd_ = dx * e2h[0] + dy * e2h[1]
    pa_c = np.clip(pa, -fault.length_km / 2, fault.length_km / 2)
    pd_c = np.clip(pd_, 0.0, fault.width_km)
    cpx = hx + pa_c * e1[0] + pd_c * e2h[0]
    cpy = hy + pa_c * e1[1] + pd_c * e2h[1]
    return float(np.sqrt((sx - cpx)**2 + (sy - cpy)**2))


def rx_finite(site_lat, site_lon, fault: FaultParams) -> float:
    """Horizontal distance perpendicular to strike, down-dip direction (km)."""
    sx, sy = _local_cart(site_lat, site_lon, fault)
    strike_r = np.radians(fault.strike)
    dip_r = np.radians(fault.dip)
    e2h = np.array([np.cos(strike_r) * np.cos(dip_r),
                     -np.sin(strike_r) * np.cos(dip_r)])
    hx, hy, _ = _fault_anchor(fault)
    dx, dy = sx - hx, sy - hy
    pd_ = dx * e2h[0] + dy * e2h[1]
    return float(pd_) if pd_ >= 0 else float(-pd_)


# ── NGA-West2 GMPE implementations ───────────────────────────────────

def _get_gmpe_models():
    """Load NGA-West2 simplified GMPE models from framework.

    Uses the calibrated simplified implementations in gmpe_nga_simplified.py
    which have proper coefficient structures (unlike the raw NGA coefficient
    approach which has numerical overflow issues with near-source terms).
    """
    from .gmpe_nga_simplified import ASK14, BSSA14_Simplified, CB14, CY14
    return {
        "ASK14": ASK14(),
        "BSSA14": BSSA14_Simplified(),
        "CB14": CB14(),
        "CY14": CY14(),
    }


GMPE_MODEL_NAMES = ["ASK14", "BSSA14", "CB14", "CY14"]

GMPE_COLORS = {
    "ASK14": "#e41a1c",
    "BSSA14": "#377eb8",
    "CB14": "#4daf4a",
    "CY14": "#ff7f00",
}


# ── Core Validation Functions ─────────────────────────────────────────

def compute_distances(df: pd.DataFrame, fault: FaultParams,
                      lat_col: str = "latitude",
                      lon_col: str = "longitude") -> pd.DataFrame:
    """Compute finite-fault distances for all bridges."""
    df = df.copy()
    df["Rrup"] = df.apply(lambda r: rrup_finite(r[lat_col], r[lon_col], fault), axis=1)
    df["Rjb"] = df.apply(lambda r: rjb_finite(r[lat_col], r[lon_col], fault), axis=1)
    df["Rx"] = df.apply(lambda r: rx_finite(r[lat_col], r[lon_col], fault), axis=1)
    return df


def compute_gmpe_predictions(df: pd.DataFrame, fault: FaultParams,
                              vs30_col: str = "vs30") -> pd.DataFrame:
    """Compute PGA predictions for all 4 NGA-West2 models."""
    df = df.copy()
    models = _get_gmpe_models()
    fault_type = "reverse" if fault.frv else ("normal" if fault.fnm else "strike_slip")

    for name, model in models.items():
        preds = []
        for _, row in df.iterrows():
            vs30 = float(row.get(vs30_col, 760.0))
            if pd.isna(vs30):
                vs30 = 760.0
            rjb = row["Rjb"]
            pga, _ = model.compute(fault.Mw, rjb, vs30, fault_type, period=0.0)
            preds.append(pga)
        df[f"{name}_PGA"] = preds
    return df


def compute_residuals(df: pd.DataFrame,
                       obs_col: str = "pga_shakemap") -> pd.DataFrame:
    """Compute ln-scale residuals for each GMPE.

    Residual = ln(observed) - ln(predicted).
    Positive = GMPE underpredicts.
    """
    df = df.copy()
    obs = pd.to_numeric(df[obs_col], errors="coerce")
    for name in GMPE_MODEL_NAMES:
        pred = df[f"{name}_PGA"]
        df[f"{name}_Residual"] = np.log(obs + 1e-10) - np.log(pred + 1e-10)
    return df


def residual_statistics(df: pd.DataFrame,
                         site_col: str = "SiteClass") -> pd.DataFrame:
    """Compute residual statistics by GMPE x site class."""
    rows = []
    site_classes = [sc for sc in NEHRP_LABELS if sc in df[site_col].values]
    for name in GMPE_MODEL_NAMES:
        resid_col = f"{name}_Residual"
        for sc in site_classes:
            mask = df[site_col] == sc
            if not mask.any():
                continue
            r = df.loc[mask, resid_col].dropna()
            rows.append({
                "GMPE": name,
                "SiteClass": sc,
                "N": len(r),
                "Mean": r.mean(),
                "Median": r.median(),
                "Std": r.std(),
                "Min": r.min(),
                "Max": r.max(),
            })
    return pd.DataFrame(rows)


# ── Plotting Functions ────────────────────────────────────────────────

def plot_prediction_vs_observation(df, obs_col="pga_shakemap",
                                    save_path=None):
    """4-panel scatter: predicted vs observed, colored by site class."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Predicted vs Observed PGA by GMPE (1994 Northridge)",
                 fontsize=14, fontweight="bold")

    obs = pd.to_numeric(df[obs_col], errors="coerce")
    for idx, name in enumerate(GMPE_MODEL_NAMES):
        ax = axes[idx // 2, idx % 2]
        pred = df[f"{name}_PGA"]
        for sc in NEHRP_LABELS:
            mask = df["SiteClass"] == sc
            if mask.any():
                ax.scatter(obs[mask], pred[mask], label=f"{sc} (n={mask.sum()})",
                           s=60, alpha=0.6, edgecolors="k", linewidth=0.3)
        lims = [min(obs.min(), pred.min()) * 0.5, max(obs.max(), pred.max()) * 2]
        ax.plot(lims, lims, "k--", lw=1.5, alpha=0.5, label="1:1")
        corr = np.corrcoef(np.log(obs + 1e-10), np.log(pred + 1e-10))[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=11, va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax.set_xlabel("Observed PGA (g)")
        ax.set_ylabel(f"{name} PGA (g)")
        ax.set_title(name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_vs_vs30(df, vs30_col="vs30", save_path=None):
    """4-panel: residuals vs Vs30, colored by site class."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("GMPE Residuals vs VS30 (ln(obs) - ln(pred))",
                 fontsize=14, fontweight="bold")
    for idx, name in enumerate(GMPE_MODEL_NAMES):
        ax = axes[idx // 2, idx % 2]
        resid_col = f"{name}_Residual"
        for sc in NEHRP_LABELS:
            mask = df["SiteClass"] == sc
            if mask.any():
                ax.scatter(df.loc[mask, vs30_col], df.loc[mask, resid_col],
                           label=sc, s=60, alpha=0.6, edgecolors="k", linewidth=0.3)
        ax.axhline(0, color="k", ls="--", lw=1.5, alpha=0.7)
        ax.set_xlabel("VS30 (m/s)")
        ax.set_ylabel("Residual (ln)")
        ax.set_title(f"{name}")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residual_distributions(df, save_path=None):
    """Violin plots of residuals by site class and GMPE."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    site_classes = [sc for sc in NEHRP_LABELS if sc in df["SiteClass"].values]
    fig, ax = plt.subplots(figsize=(12, 7))

    for gi, name in enumerate(GMPE_MODEL_NAMES):
        resid_col = f"{name}_Residual"
        data, positions = [], []
        for ci, sc in enumerate(site_classes):
            mask = df["SiteClass"] == sc
            if mask.any():
                data.append(df.loc[mask, resid_col].dropna().values)
                positions.append(ci + gi * 0.2)
        if data:
            parts = ax.violinplot(data, positions=positions, widths=0.15,
                                  showmeans=True, showmedians=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(GMPE_COLORS[name])
                pc.set_alpha(0.7)

    ax.axhline(0, color="k", ls="--", lw=1.5, alpha=0.7)
    ax.set_xticks(np.arange(len(site_classes)))
    ax.set_xticklabels(site_classes)
    ax.set_xlabel("Site Class (NEHRP)")
    ax.set_ylabel("Residual (ln)")
    ax.set_title("Residual Distribution by Site Class (1994 Northridge)")
    ax.grid(True, alpha=0.3, axis="y")
    legend_elements = [mpatches.Patch(facecolor=GMPE_COLORS[n], alpha=0.7,
                                       edgecolor="k", label=n)
                       for n in GMPE_MODEL_NAMES]
    ax.legend(handles=legend_elements, fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_site_class_comparison(df, save_path=None):
    """Box plot + bar chart of predictions by site class."""
    import matplotlib.pyplot as plt

    site_classes = [sc for sc in NEHRP_LABELS if sc in df["SiteClass"].values]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GMPE Predictions by NEHRP Site Class", fontsize=14, fontweight="bold")

    # Box plot
    ax = axes[0]
    box_data, box_labels = [], []
    for sc in site_classes:
        mask = df["SiteClass"] == sc
        if mask.any():
            vals = []
            for name in GMPE_MODEL_NAMES:
                vals.extend(df.loc[mask, f"{name}_PGA"].values)
            box_data.append(vals)
            box_labels.append(f"{sc}\n(n={mask.sum()})")
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#7fc97f")
        patch.set_alpha(0.7)
    ax.set_ylabel("PGA (g)")
    ax.set_title("PGA Distribution")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    # Mean bar chart
    ax = axes[1]
    x = np.arange(len(site_classes))
    w = 0.2
    for gi, name in enumerate(GMPE_MODEL_NAMES):
        means = []
        for sc in site_classes:
            mask = df["SiteClass"] == sc
            means.append(df.loc[mask, f"{name}_PGA"].mean() if mask.any() else 0)
        ax.bar(x + gi * w, means, w, label=name,
               color=GMPE_COLORS[name], alpha=0.8)
    ax.set_ylabel("Mean PGA (g)")
    ax.set_title("Mean PGA by GMPE")
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(site_classes)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gmpe_vs_vs30(df, vs30_col="vs30", save_path=None):
    """4-panel: GMPE predictions vs VS30, colored by distance."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("GMPE Predictions vs VS30 (colored by Rrup)",
                 fontsize=14, fontweight="bold")
    for idx, name in enumerate(GMPE_MODEL_NAMES):
        ax = axes[idx // 2, idx % 2]
        sc = ax.scatter(df[vs30_col], df[f"{name}_PGA"],
                        c=df["Rrup"], cmap="viridis",
                        s=60, alpha=0.6, edgecolors="k", linewidth=0.3)
        ax.set_xlabel("VS30 (m/s)")
        ax.set_ylabel(f"{name} PGA (g)")
        ax.set_title(name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        plt.colorbar(sc, ax=ax, label="Rrup (km)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Full Validation Pipeline ─────────────────────────────────────────

def run_gmpe_validation(
    bridges_df: pd.DataFrame,
    fault: FaultParams = NORTHRIDGE_FAULT,
    obs_col: str = "pga_shakemap",
    vs30_col: str = "vs30",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    output_dir: str = "output/gmpe_validation",
) -> dict:
    """
    Full GMPE validation pipeline.

    Parameters
    ----------
    bridges_df : DataFrame
        Must have lat, lon, vs30, and observed IM column.
    fault : FaultParams
    obs_col : str
        Column with observed intensity measures.
    vs30_col : str
        Column with Vs30 values.
    output_dir : str

    Returns
    -------
    dict with keys: df, residual_stats, output_dir
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = bridges_df.copy()

    # Site classification
    vs30 = pd.to_numeric(df[vs30_col], errors="coerce").fillna(760.0)
    df["SiteClass"] = classify_nehrp_series(vs30)

    # Distances
    print("[gmpe] Computing finite-fault distances...")
    df = compute_distances(df, fault, lat_col, lon_col)
    print(f"  Rrup: {df['Rrup'].min():.1f} - {df['Rrup'].max():.1f} km")

    # GMPE predictions
    print("[gmpe] Computing NGA-West2 predictions (4 models)...")
    df = compute_gmpe_predictions(df, fault, vs30_col)

    # Residuals
    if obs_col in df.columns:
        print(f"[gmpe] Computing residuals vs '{obs_col}'...")
        df = compute_residuals(df, obs_col)
        stats = residual_statistics(df)

        print("\n  Residual Statistics (ln scale):")
        for name in GMPE_MODEL_NAMES:
            r = df[f"{name}_Residual"].dropna()
            print(f"    {name}: mean={r.mean():+.3f}, std={r.std():.3f}, n={len(r)}")
    else:
        print(f"[gmpe] No observed column '{obs_col}' — skipping residuals.")
        stats = pd.DataFrame()

    # Save data
    csv_path = out / "bridges_gmpe_predictions.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n[save] {csv_path}")

    if not stats.empty:
        stats_path = out / "residual_statistics.csv"
        stats.to_csv(stats_path, index=False, encoding="utf-8")
        print(f"[save] {stats_path}")

    # Plots
    print("[plot] Generating validation plots...")
    plot_gmpe_vs_vs30(df, vs30_col, out / "01_gmpe_vs_vs30.png")
    plot_site_class_comparison(df, out / "02_site_class_comparison.png")

    if obs_col in df.columns and f"ASK14_Residual" in df.columns:
        plot_residuals_vs_vs30(df, vs30_col, out / "03_residuals_vs_vs30.png")
        plot_residual_distributions(df, out / "04_residual_distributions.png")
        plot_prediction_vs_observation(df, obs_col, out / "05_prediction_vs_observation.png")

    print("[done] GMPE validation complete.")

    return {
        "df": df,
        "residual_stats": stats,
        "output_dir": str(out),
    }

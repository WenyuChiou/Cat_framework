"""
CAT411 Demo: ShakeMap vs BSSA21 GMPE Comparison
================================================
Generates comparison plots and CSV for the 1994 Northridge earthquake.

Output (saved to output/demo_W2_0303/):
  01_shakemap_vs_gmpe_spatial.png   — Side-by-side IM maps
  02_attenuation_comparison.png     — IM vs distance scatter
  03_damage_comparison_nearest.png  — Damage probability bar chart
  04_im_histogram.png               — IM distribution histograms
  05_residual_map.png               — Spatial residual (SM/GMPE ratio)
  bridge_comparison_results.csv     — Full per-bridge results
"""

import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
import contextily as ctx

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_shakemap, load_nbi, classify_nbi_to_hazus
from src.interpolation import interpolate_im
from src.config import IM_COLUMN_MAP
from src.hazard import haversine_distance_km
from src.fragility import damage_state_probabilities

import src.gmpe_bssa21  # noqa: F401
from src.gmpe_base import get_gmpe, IM_TYPE_TO_PERIOD

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "output", "demo_W2_0303")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Earthquake parameters ──
EQ_LAT, EQ_LON = 34.213, -118.537
MW = 6.7
FAULT_TYPE = "reverse"
VS30 = 360.0  # average valley soil

print("=" * 70)
print("CAT411 Demo: ShakeMap vs BSSA21 GMPE Comparison")
print("1994 Northridge Earthquake (Mw 6.7)")
print("=" * 70)

# ── Load and filter data ──
sm = load_shakemap()
nbi = load_nbi()
nbi = classify_nbi_to_hazus(nbi)

mask = (
    (nbi["latitude"] >= 33.8) & (nbi["latitude"] <= 34.6)
    & (nbi["longitude"] >= -118.9) & (nbi["longitude"] <= -118.0)
)
nbi = nbi[mask].copy()
print(f"Bridges in Northridge region: {len(nbi)}")

# ── Path A: ShakeMap ──
im_col = IM_COLUMN_MAP["SA10"]
nbi["im_shakemap"] = interpolate_im(
    sm["LAT"].values, sm["LON"].values, sm[im_col].values,
    nbi["latitude"].values, nbi["longitude"].values,
    method="kriging",
)

# ── Path B: BSSA21 GMPE ──
gmpe = get_gmpe("bssa21")
period = IM_TYPE_TO_PERIOD["SA10"]

im_gmpe = np.empty(len(nbi))
for i, (_, row) in enumerate(nbi.iterrows()):
    R_JB = max(haversine_distance_km(EQ_LAT, EQ_LON, row["latitude"], row["longitude"]), 0.1)
    med, _ = gmpe.compute(Mw=MW, R_JB=R_JB, Vs30=VS30, fault_type=FAULT_TYPE, period=period)
    im_gmpe[i] = med

nbi["im_gmpe"] = im_gmpe
nbi["R_JB_km"] = [
    haversine_distance_km(EQ_LAT, EQ_LON, r["latitude"], r["longitude"])
    for _, r in nbi.iterrows()
]
nbi["ratio_sm_gmpe"] = nbi["im_shakemap"] / nbi["im_gmpe"].replace(0, np.nan)

print("IM computation complete.\n")

# ══════════════════════════════════════════════════════════════════════
# Plot 1: Side-by-side spatial IM maps (with basemap)
# ══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

vmin, vmax = 0, min(nbi["im_shakemap"].quantile(0.98), 1.2)

INTERP_METHOD = "kriging"  # interpolation method used

for ax, col, title_label in [
    (ax1, "im_shakemap", f"Path A: ShakeMap Interpolation\nSa(1.0s) — method: {INTERP_METHOD}"),
    (ax2, "im_gmpe",     "Path B: BSSA21 GMPE Prediction\nSa(1.0s) — point-source R_JB"),
]:
    sc = ax.scatter(nbi["longitude"], nbi["latitude"], c=nbi[col],
                    cmap="YlOrRd", s=10, vmin=vmin, vmax=vmax, alpha=0.85, zorder=3)
    ax.plot(EQ_LON, EQ_LAT, "k*", markersize=18, label="Epicenter", zorder=5)
    ax.set_title(title_label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    plt.colorbar(sc, ax=ax, label="Sa(1.0s) [g]", shrink=0.75)
    # Set extent for basemap
    ax.set_xlim(nbi["longitude"].min() - 0.02, nbi["longitude"].max() + 0.02)
    ax.set_ylim(nbi["latitude"].min() - 0.02, nbi["latitude"].max() + 0.02)
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron,
                        zoom=11, alpha=0.5)
    except Exception as e:
        print(f"  Basemap warning: {e} (plot still saved without basemap)")

fig.suptitle("CAT411: ShakeMap vs BSSA21 GMPE — Spatial IM Distribution\n"
             "1994 Northridge Earthquake (Mw 6.7)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_shakemap_vs_gmpe_spatial.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved: 01_shakemap_vs_gmpe_spatial.png")

# ══════════════════════════════════════════════════════════════════════
# Plot 2: Attenuation comparison (IM vs distance)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))

# +/- 1 sigma band (draw first so it's behind everything)
r_range = np.logspace(np.log10(0.5), np.log10(100), 100)
gmpe_curve = np.array([
    gmpe.compute(Mw=MW, R_JB=r, Vs30=VS30, fault_type=FAULT_TYPE, period=period)[0]
    for r in r_range
])
_, sig_ref = gmpe.compute(Mw=MW, R_JB=20, Vs30=VS30, fault_type=FAULT_TYPE, period=period)
ax.fill_between(r_range, gmpe_curve * np.exp(-sig_ref), gmpe_curve * np.exp(sig_ref),
                color="#FFA500", alpha=0.15, label=f"BSSA21 ±1σ ({sig_ref:.2f})")

# Scatter: ShakeMap observed
ax.scatter(nbi["R_JB_km"], nbi["im_shakemap"], s=12, alpha=0.5, c="#4682B4",
           edgecolors="none", marker="o", label="ShakeMap (nearest interp.)", zorder=3)

# Scatter: GMPE per-bridge predictions
ax.scatter(nbi["R_JB_km"], nbi["im_gmpe"], s=12, alpha=0.5, c="#2CA02C",
           edgecolors="none", marker="s", label="BSSA21 per-bridge", zorder=3)

# GMPE median curve
ax.plot(r_range, gmpe_curve, color="#D62728", linewidth=2.5,
        label="BSSA21 median curve", zorder=4)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Distance to Epicenter R_JB (km)", fontsize=12)
ax.set_ylabel("Sa(1.0s) [g]", fontsize=12)
ax.set_title("Attenuation Comparison: ShakeMap vs BSSA21\n"
             f"Northridge Mw {MW}, Vs30={VS30} m/s, {FAULT_TYPE}", fontsize=13, fontweight="bold")
legend = ax.legend(fontsize=10, loc="upper right", framealpha=0.9, edgecolor="gray")
for lh in legend.legend_handles:
    lh.set_alpha(1.0)  # full opacity in legend
ax.grid(True, alpha=0.3, which="both")
ax.set_xlim(0.5, 100)
ax.set_ylim(0.01, 2.0)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_attenuation_comparison.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved: 02_attenuation_comparison.png")

# ══════════════════════════════════════════════════════════════════════
# Plot 3: Damage comparison for nearest bridge
# ══════════════════════════════════════════════════════════════════════
nearest = nbi.loc[nbi["R_JB_km"].idxmin()]
probs_sm = damage_state_probabilities(nearest["im_shakemap"], nearest["hwb_class"])
probs_gm = damage_state_probabilities(nearest["im_gmpe"], nearest["hwb_class"])

ds_names = ["none", "slight", "moderate", "extensive", "complete"]
x = np.arange(len(ds_names))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - w / 2, [probs_sm[ds] for ds in ds_names], w,
               label=f"ShakeMap Sa={nearest['im_shakemap']:.3f}g", color="steelblue", edgecolor="white")
bars2 = ax.bar(x + w / 2, [probs_gm[ds] for ds in ds_names], w,
               label=f"BSSA21 Sa={nearest['im_gmpe']:.3f}g", color="orangered", edgecolor="white")

for bar in bars1:
    if bar.get_height() > 0.02:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    if bar.get_height() > 0.02:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([ds.capitalize() for ds in ds_names], fontsize=11)
ax.set_ylabel("Probability", fontsize=12)
ax.set_title(f"Damage State Probabilities — Nearest Bridge\n"
             f"{nearest['hwb_class']}, R_JB={nearest['R_JB_km']:.1f} km, "
             f"lat={nearest['latitude']:.3f}, lon={nearest['longitude']:.3f}",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0, max(max(probs_sm.values()), max(probs_gm.values())) * 1.25)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_damage_comparison_nearest.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved: 03_damage_comparison_nearest.png")

# ══════════════════════════════════════════════════════════════════════
# Plot 4: IM histogram comparison
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
bins_hist = np.linspace(0, 1.0, 40)
ax.hist(nbi["im_shakemap"], bins=bins_hist, alpha=0.6, color="steelblue",
        label=f"ShakeMap (mean={nbi['im_shakemap'].mean():.3f}g)", edgecolor="white")
ax.hist(nbi["im_gmpe"], bins=bins_hist, alpha=0.6, color="orangered",
        label=f"BSSA21 (mean={nbi['im_gmpe'].mean():.3f}g)", edgecolor="white")
ax.set_xlabel("Sa(1.0s) [g]", fontsize=12)
ax.set_ylabel("Number of Bridges", fontsize=12)
ax.set_title("IM Distribution: ShakeMap vs BSSA21 GMPE\n"
             f"{len(nbi)} bridges in Northridge region", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_im_histogram.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved: 04_im_histogram.png")

# ══════════════════════════════════════════════════════════════════════
# Plot 5: Residual map (ShakeMap / GMPE ratio) with basemap
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
valid = nbi["ratio_sm_gmpe"].notna() & (nbi["ratio_sm_gmpe"] < 10)
sc = ax.scatter(nbi.loc[valid, "longitude"], nbi.loc[valid, "latitude"],
                c=np.log2(nbi.loc[valid, "ratio_sm_gmpe"]),
                cmap="RdBu_r", s=10, alpha=0.85, vmin=-2, vmax=2, zorder=3)
ax.plot(EQ_LON, EQ_LAT, "k*", markersize=18, label="Epicenter", zorder=5)
cb = plt.colorbar(sc, ax=ax, shrink=0.8)
cb.set_label("log2(ShakeMap / GMPE)", fontsize=11)
cb.set_ticks([-2, -1, 0, 1, 2])
cb.set_ticklabels(["0.25x", "0.5x", "1x", "2x", "4x"])
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Spatial Residual: ShakeMap / BSSA21 GMPE\n"
             "Red = ShakeMap higher, Blue = GMPE higher", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
# Basemap
ax.set_xlim(nbi["longitude"].min() - 0.02, nbi["longitude"].max() + 0.02)
ax.set_ylim(nbi["latitude"].min() - 0.02, nbi["latitude"].max() + 0.02)
try:
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron,
                    zoom=11, alpha=0.5)
except Exception as e:
    print(f"  Basemap warning: {e} (plot still saved without basemap)")
# Annotation: interpolation method
ax.text(0.02, 0.02, f"ShakeMap interp: {INTERP_METHOD}",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_residual_map.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved: 05_residual_map.png")

# ══════════════════════════════════════════════════════════════════════
# CSV output
# ══════════════════════════════════════════════════════════════════════
out_cols = ["latitude", "longitude", "hwb_class", "R_JB_km",
            "im_shakemap", "im_gmpe", "ratio_sm_gmpe"]
nbi[out_cols].to_csv(os.path.join(OUT_DIR, "bridge_comparison_results.csv"),
                     index=False, float_format="%.4f", encoding="utf-8")
print("Saved: bridge_comparison_results.csv")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")

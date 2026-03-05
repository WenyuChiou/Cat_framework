"""Generate GMPE comparison plots for next week's presentation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.gmpe_nga_simplified import ALL_SIMPLIFIED_MODELS, attenuation_curves, vs30_sensitivity
from src.gmpe_bssa21 import BSSA21

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "demo_W3_0310")
os.makedirs(OUT_DIR, exist_ok=True)

# Color scheme for models
COLORS = {
    "bssa21": "#d62728",          # red (full model, highlighted)
    "ask14": "#1f77b4",
    "bssa14_simplified": "#ff7f0e",
    "cb14": "#2ca02c",
    "cy14": "#9467bd",
    "idriss14": "#8c564b",
    "gk15": "#e377c2",
    "nga_east": "#7f7f7f",
}

bssa21 = BSSA21()

# ── Plot 1: Attenuation curves (all models) ─────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))

Mw, Vs30 = 6.7, 360.0
distances = np.logspace(0, np.log10(200), 150)

# Simplified models
curves = attenuation_curves(Mw, Vs30, distances)
for name in ALL_SIMPLIFIED_MODELS:
    ax.loglog(distances, curves[name], color=COLORS[name], linewidth=1.5,
              alpha=0.7, label=f"{name.upper()}")

# Full BSSA21
pga_full = np.array([bssa21.compute(Mw, float(r), Vs30, "reverse", 0.0)[0] for r in distances])
ax.loglog(distances, pga_full, color=COLORS["bssa21"], linewidth=3,
          linestyle="-", label="BSSA21 (full coefficients)", zorder=10)

ax.set_xlabel("Joyner-Boore Distance R_JB (km)", fontsize=12)
ax.set_ylabel("PGA (g)", fontsize=12)
ax.set_title(f"NGA GMPE Comparison — Attenuation Curves\nMw={Mw}, Vs30={Vs30} m/s, Reverse Fault",
             fontsize=13)
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(1, 200)
ax.set_ylim(1e-4, 2)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gmpe_attenuation_comparison.png"), dpi=150)
print(f"[OK] Saved attenuation comparison plot")

# ── Plot 2: Vs30 sensitivity (all models) ───────────────────────────────

fig, ax = plt.subplots(figsize=(10, 7))

Mw, R_JB = 6.7, 20.0
vs30_vals = np.linspace(150, 1200, 100)

sens = vs30_sensitivity(Mw, R_JB, vs30_vals)
for name in ALL_SIMPLIFIED_MODELS:
    ax.semilogy(vs30_vals, sens[name], color=COLORS[name], linewidth=1.5,
                alpha=0.7, label=f"{name.upper()}")

# Full BSSA21
pga_vs30 = np.array([bssa21.compute(Mw, R_JB, float(v), "reverse", 0.0)[0] for v in vs30_vals])
ax.semilogy(vs30_vals, pga_vs30, color=COLORS["bssa21"], linewidth=3,
            label="BSSA21 (full coefficients)", zorder=10)

# NEHRP class boundaries
nehrp = {"E": 180, "D": 360, "C": 760, "B": 1500}
for cls, boundary in nehrp.items():
    if 150 <= boundary <= 1200:
        ax.axvline(boundary, color="gray", linestyle=":", alpha=0.5)
        ax.text(boundary + 5, ax.get_ylim()[1] * 0.7, cls, fontsize=9, color="gray")

ax.set_xlabel("Vs30 (m/s)", fontsize=12)
ax.set_ylabel("PGA (g)", fontsize=12)
ax.set_title(f"NGA GMPE Comparison — Vs30 Site Response\nMw={Mw}, R_JB={R_JB} km, Reverse Fault",
             fontsize=13)
ax.legend(fontsize=9, loc="upper right", ncol=2)
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gmpe_vs30_sensitivity.png"), dpi=150)
print(f"[OK] Saved Vs30 sensitivity plot")

# ── Plot 3: Model comparison bar chart (Northridge scenario) ─────────────

fig, ax = plt.subplots(figsize=(10, 6))

scenarios = {
    "Near-field\n(R=10km)":  (6.7, 10.0, 360.0),
    "Mid-field\n(R=30km)":   (6.7, 30.0, 360.0),
    "Far-field\n(R=80km)":   (6.7, 80.0, 360.0),
    "Soft soil\n(Vs30=200)": (6.7, 20.0, 200.0),
    "Rock\n(Vs30=760)":      (6.7, 20.0, 760.0),
}

model_names = list(ALL_SIMPLIFIED_MODELS.keys()) + ["bssa21"]
n_models = len(model_names)
n_scenarios = len(scenarios)
x = np.arange(n_scenarios)
width = 0.10

for i, name in enumerate(model_names):
    pga_vals = []
    for label, (mw, r, v) in scenarios.items():
        if name == "bssa21":
            pga, _ = bssa21.compute(mw, r, v, "reverse", 0.0)
        else:
            pga, _ = ALL_SIMPLIFIED_MODELS[name].compute(mw, r, v)
        pga_vals.append(pga)

    offset = (i - n_models / 2 + 0.5) * width
    color = COLORS[name]
    lw = 1.5 if name == "bssa21" else 0.5
    edge = "black" if name == "bssa21" else color
    label = f"{name.upper()} (full)" if name == "bssa21" else name.upper()
    ax.bar(x + offset, pga_vals, width, label=label, color=color,
           edgecolor=edge, linewidth=lw, alpha=0.85)

ax.set_ylabel("PGA (g)", fontsize=12)
ax.set_title("NGA GMPE Comparison — Multiple Scenarios (Mw=6.7, Reverse Fault)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(list(scenarios.keys()), fontsize=10)
ax.legend(fontsize=8, ncol=4, loc="upper right")
ax.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "gmpe_scenario_comparison.png"), dpi=150)
print(f"[OK] Saved scenario comparison bar chart")

print(f"\nAll plots saved to: {OUT_DIR}")

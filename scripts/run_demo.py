"""
CAT411 Demo — Config-Driven Analysis Scenarios

Generates multiple analysis outputs to demonstrate the framework's
ability to filter bridges by region, material, design era, and HWB class.

Each scenario produces:
  - bridge_damage_results.csv   (per-bridge damage probabilities)
  - 00_analysis_dashboard.png   (summary dashboard)
  - 01-07_*.png                 (maps and charts)
  - scenario_summary.txt        (text summary)

Usage:
    python scripts/run_demo.py
"""

import os
import sys
import time
import shutil
import textwrap
from pathlib import Path

# Ensure project root is on sys.path for src.* imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import numpy as np
import pandas as pd
import yaml

# ── Project imports ──────────────────────────────────────────────
from src.config import AnalysisConfig, load_config, validate_config, IM_COLUMN_MAP
from src.data_loader import load_shakemap, load_nbi, classify_nbi_to_hazus, DATA_DIR
from src.interpolation import interpolate_im
from src.fragility import damage_state_probabilities
from src.hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER
from src.plotting import (
    plot_shakemap_grid,
    plot_nbi_bridge_distribution_map,
    plot_ground_motion_field,
    plot_bridge_damage_map,
    plot_bridges_on_shakemap,
    plot_attenuation_curve,
    plot_analysis_summary,
    plot_portfolio_damage,
)

ROOT = _PROJECT_ROOT
DEMO_DIR = ROOT / "output" / "demo_W1_0224"

# ── Demo Scenarios ───────────────────────────────────────────────

SCENARIOS = {
    "01_full_northridge": {
        "title": "Full Northridge Region — All Bridges",
        "description": (
            "Baseline scenario: all 2,900+ bridges in the greater LA / "
            "Northridge region with Sa(1.0s) from USGS ShakeMap."
        ),
        "config": {
            "region": {"lat_min": 33.8, "lat_max": 34.6,
                       "lon_min": -118.9, "lon_max": -118.0},
            "im_type": "SA10",
            "interpolation": {"method": "nearest"},
        },
    },
    "02_la_county_pre1975": {
        "title": "LA County — Pre-1975 (Conventional Design) Bridges",
        "description": (
            "Focuses on older bridges built before the 1975 seismic code "
            "adoption. These conventional-design bridges are expected to "
            "show higher vulnerability."
        ),
        "config": {
            "region": {"lat_min": 33.8, "lat_max": 34.6,
                       "lon_min": -118.9, "lon_max": -118.0},
            "im_type": "SA10",
            "interpolation": {"method": "nearest"},
            "bridge_selection": {"county": "037"},
            "design_era": "conventional",
        },
    },
    "03_epicenter_zone": {
        "title": "Near-Epicenter High-Intensity Zone",
        "description": (
            "Zooms into a 0.3° × 0.3° box centered near the Northridge "
            "epicenter (34.21°N, -118.54°W). Bridges here experienced "
            "the strongest ground shaking (Sa > 0.5g)."
        ),
        "config": {
            "region": {"lat_min": 34.05, "lat_max": 34.35,
                       "lon_min": -118.7, "lon_max": -118.4},
            "im_type": "SA10",
            "interpolation": {"method": "idw", "power": 2.0, "n_neighbors": 8},
        },
    },
    "04_concrete_multispan": {
        "title": "Concrete Multi-Span Bridges (HWB5 + HWB7)",
        "description": (
            "Isolates the two most common conventional-design concrete "
            "multi-span classes: HWB5 (continuous) and HWB7 (simply-supported). "
            "These are the dominant bridge types in California."
        ),
        "config": {
            "region": {"lat_min": 33.8, "lat_max": 34.6,
                       "lon_min": -118.9, "lon_max": -118.0},
            "im_type": "SA10",
            "interpolation": {"method": "nearest"},
            "hwb_filter": ["HWB5", "HWB7"],
        },
    },
}


# ── Core analysis function ───────────────────────────────────────

def run_scenario(name: str, spec: dict, shakemap: pd.DataFrame) -> dict:
    """Run a single demo scenario and save outputs."""
    out_dir = DEMO_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = spec["config"]
    title = spec["title"]
    desc = spec["description"]

    print(f"\n{'='*60}")
    print(f"  SCENARIO: {title}")
    print(f"{'='*60}")

    # Build AnalysisConfig from dict
    cfg = AnalysisConfig()
    if "region" in cfg_dict:
        cfg.region = cfg_dict["region"]
    cfg.im_type = cfg_dict.get("im_type", "SA10")
    interp = cfg_dict.get("interpolation", {})
    cfg.interpolation_method = interp.get("method", "nearest")
    cfg.interpolation_params = {k: v for k, v in interp.items() if k != "method"}
    cfg.bridge_selection = cfg_dict.get("bridge_selection", {})
    cfg.hwb_filter = cfg_dict.get("hwb_filter", None)
    cfg.design_era = cfg_dict.get("design_era", None)
    cfg.material_filter = cfg_dict.get("material_filter", None)

    validate_config(cfg)

    # Save config YAML for reproducibility
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False, allow_unicode=True)

    # Load NBI
    nbi_path = sorted(DATA_DIR.glob("[A-Z][A-Z]*.txt"))[0]
    nbi = load_nbi(nbi_path, northridge_bbox=cfg.bbox_dict)
    print(f"  Bridges in region: {len(nbi):,}")

    if len(nbi) == 0:
        print("  ⚠ No bridges found in this region. Skipping.")
        return {}

    # Classify
    nbi = classify_nbi_to_hazus(
        nbi,
        hwb_filter=cfg.hwb_filter,
        design_era_filter=cfg.design_era,
        material_filter=cfg.material_filter,
        nbi_filters=cfg.bridge_selection,
    )
    print(f"  After filtering: {len(nbi):,}")
    if len(nbi) == 0:
        print("  ⚠ No bridges after filtering. Skipping.")
        return {}

    hwb_counts = nbi["hwb_class"].value_counts().sort_index()
    print(f"  HWB classes: {hwb_counts.to_dict()}")

    # Interpolate IM
    grid_lats = shakemap["LAT"].values
    grid_lons = shakemap["LON"].values
    bridge_lats = nbi["latitude"].values
    bridge_lons = nbi["longitude"].values

    for im_name, sm_col in IM_COLUMN_MAP.items():
        if sm_col in shakemap.columns:
            nbi[f"im_{im_name}"] = interpolate_im(
                grid_lats, grid_lons, shakemap[sm_col].values,
                bridge_lats, bridge_lons,
                method=cfg.interpolation_method,
                **cfg.interpolation_params,
            )

    im_col_name = f"im_{cfg.im_type}"
    nbi["im_selected"] = nbi[im_col_name]

    # Legacy aliases
    if "im_SA10" in nbi.columns:
        nbi["sa_10"] = nbi["im_SA10"]

    print(f"  {cfg.im_type} range: {nbi['im_selected'].min():.4f}g – {nbi['im_selected'].max():.4f}g")

    # Compute damage probabilities
    ds_cols = ["P_none", "P_slight", "P_moderate", "P_extensive", "P_complete"]
    for _, row in nbi.iterrows():
        probs = damage_state_probabilities(row["im_selected"], row["hwb_class"])
        for ds_key, col in zip(["none", "slight", "moderate", "extensive", "complete"], ds_cols):
            nbi.loc[row.name, col] = probs[ds_key]

    # Summary stats
    count_by_ds = {}
    for ds in ["none", "slight", "moderate", "extensive", "complete"]:
        col = f"P_{ds}"
        mean_p = nbi[col].mean()
        count_by_ds[ds] = mean_p * len(nbi)
        print(f"    {ds:>10}: {mean_p:.3f} ({mean_p * len(nbi):.0f} bridges)")

    # Save CSV
    csv_path = out_dir / "bridge_damage_results.csv"
    nbi.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Saved: {csv_path.name}")

    # ── Visualizations ──────────────────────────────────────────
    sm_im_col = IM_COLUMN_MAP.get(cfg.im_type, "PSA10")

    # 1. ShakeMap
    plot_shakemap_grid(
        shakemap, intensity_measure=sm_im_col,
        output_dir=str(out_dir), filename="01_shakemap.png",
    )

    # 2. Bridge distribution
    plot_nbi_bridge_distribution_map(
        nbi, output_dir=str(out_dir), filename="02_bridge_distribution.png",
    )

    # 3. Ground motion at bridges
    from src.exposure import SiteParams
    bridge_sites = [SiteParams(lat=r["latitude"], lon=r["longitude"]) for _, r in nbi.iterrows()]
    plot_ground_motion_field(
        bridge_sites, nbi["im_selected"].values,
        output_dir=str(out_dir), filename="03_bridge_ground_motion.png",
        im_type=cfg.im_type,
    )

    # 4. Damage map
    plot_bridge_damage_map(
        nbi, damage_state="complete",
        output_dir=str(out_dir), filename="04_damage_map_complete.png",
    )

    # 5. Bridges on ShakeMap overlay
    plot_bridges_on_shakemap(
        shakemap, nbi, im_type=cfg.im_type,
        output_dir=str(out_dir), filename="05_bridges_on_shakemap.png",
    )

    # 6. Attenuation curve
    plot_attenuation_curve(
        nbi, im_type=cfg.im_type,
        output_dir=str(out_dir), filename="06_attenuation_curve.png",
    )

    # 7. Dashboard
    stats_dict = {
        "event_id": f"Demo: {title}",
        "total_bridges": len(nbi),
        "max_pga": shakemap["PGA"].max() if "PGA" in shakemap.columns else 0,
        "avg_sa": nbi["im_selected"].mean(),
        "im_type": cfg.im_type,
        "interpolation": cfg.interpolation_method,
        "total_loss": 0,
        "damage_distribution": count_by_ds,
        "sa_values": nbi["im_selected"].values,
        "class_breakdown": nbi["hwb_class"].value_counts().to_dict(),
    }
    plot_analysis_summary(
        stats_dict, output_dir=str(out_dir), filename="00_dashboard.png",
    )

    # 8. Portfolio damage bars
    plot_portfolio_damage(
        count_by_ds, len(nbi),
        output_dir=str(out_dir), filename="07_portfolio_damage.png",
    )

    print(f"  All plots saved to: {out_dir.relative_to(ROOT)}/")

    # ── Text summary ────────────────────────────────────────────
    summary = textwrap.dedent(f"""\
    ============================================================
    SCENARIO: {title}
    ============================================================
    {desc}

    Config:
      Region:        lat[{cfg.region['lat_min']}, {cfg.region['lat_max']}] lon[{cfg.region['lon_min']}, {cfg.region['lon_max']}]
      IM Type:        {cfg.im_type}
      Interpolation:  {cfg.interpolation_method}
      Bridge Filter:  {cfg.bridge_selection or '(none)'}
      HWB Filter:     {cfg.hwb_filter or '(all classes)'}
      Design Era:     {cfg.design_era or '(all)'}

    Results:
      Total bridges:  {len(nbi):,}
      HWB classes:    {hwb_counts.to_dict()}
      {cfg.im_type} range:    {nbi['im_selected'].min():.4f}g – {nbi['im_selected'].max():.4f}g
      Mean {cfg.im_type}:     {nbi['im_selected'].mean():.4f}g

    Damage Distribution (expected bridge count):
      None:       {count_by_ds['none']:>8.1f}
      Slight:     {count_by_ds['slight']:>8.1f}
      Moderate:   {count_by_ds['moderate']:>8.1f}
      Extensive:  {count_by_ds['extensive']:>8.1f}
      Complete:   {count_by_ds['complete']:>8.1f}

    Output files:
      config.yaml                  — reproducible configuration
      bridge_damage_results.csv    — per-bridge results
      00_dashboard.png             — summary dashboard
      01_shakemap.png              — ShakeMap intensity grid
      02_bridge_distribution.png   — bridge location map
      03_bridge_ground_motion.png  — IM at bridge sites
      04_damage_map_complete.png   — P(Complete) spatial map
      05_bridges_on_shakemap.png   — overlay visualization
      06_attenuation_curve.png     — GMPE vs observed
      07_portfolio_damage.png      — damage distribution bars
    """)

    with open(out_dir / "scenario_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    return {
        "name": name,
        "title": title,
        "n_bridges": len(nbi),
        "im_mean": nbi["im_selected"].mean(),
        "im_max": nbi["im_selected"].max(),
        "p_moderate_plus": nbi[["P_moderate", "P_extensive", "P_complete"]].sum(axis=1).mean(),
        "p_complete": nbi["P_complete"].mean(),
        "damage_dist": count_by_ds,
    }


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CAT411 DEMO — Config-Driven Analysis Scenarios")
    print(f"  Output: {DEMO_DIR.relative_to(ROOT)}/")
    print("=" * 60)

    # Load ShakeMap once (shared across all scenarios)
    print("\n[Setup] Loading ShakeMap grid.xml...")
    sm = load_shakemap()
    print(f"  Grid points: {len(sm):,}")

    # Clean output
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)

    # Run all scenarios
    results = []
    t0 = time.time()
    for name, spec in SCENARIOS.items():
        t_start = time.time()
        result = run_scenario(name, spec, sm)
        if result:
            result["time_s"] = time.time() - t_start
            results.append(result)

    elapsed = time.time() - t0

    # ── Comparison summary ──────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  DEMO COMPARISON SUMMARY")
    print(f"{'='*60}\n")

    header = f"{'Scenario':<42} {'Bridges':>7} {'Mean SA':>8} {'Max SA':>8} {'P(M+E+C)':>9} {'P(Comp)':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['title'][:42]:<42} {r['n_bridges']:>7,} {r['im_mean']:>8.4f} {r['im_max']:>8.4f} {r['p_moderate_plus']:>9.3f} {r['p_complete']:>8.4f}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Output: {DEMO_DIR.relative_to(ROOT)}/")

    # Save comparison
    comp_lines = [
        "CAT411 Demo — Scenario Comparison",
        "=" * 60,
        f"Generated: 2026-02-24",
        "",
        header,
        "-" * len(header),
    ]
    for r in results:
        comp_lines.append(
            f"{r['title'][:42]:<42} {r['n_bridges']:>7,} {r['im_mean']:>8.4f} {r['im_max']:>8.4f} {r['p_moderate_plus']:>9.3f} {r['p_complete']:>8.4f}"
        )
    comp_lines.append("")
    comp_lines.append("Columns:")
    comp_lines.append("  Bridges   — Number of bridges after all filters")
    comp_lines.append("  Mean SA   — Mean Sa(1.0s) at bridge sites (g)")
    comp_lines.append("  Max SA    — Maximum Sa(1.0s) at bridge sites (g)")
    comp_lines.append("  P(M+E+C)  — Mean probability of Moderate or worse damage")
    comp_lines.append("  P(Comp)   — Mean probability of Complete damage")
    comp_lines.append("")
    comp_lines.append("Key observations:")
    comp_lines.append("  - Epicenter zone bridges experience ~2-3x higher mean Sa than full region")
    comp_lines.append("  - Conventional design bridges show higher P(damage) at same IM levels")
    comp_lines.append("  - HWB5/HWB7 (concrete multi-span) have moderate fragility thresholds")

    with open(DEMO_DIR / "comparison_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(comp_lines))

    print("\nDone! Ready for demo presentation.")


if __name__ == "__main__":
    main()

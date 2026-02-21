"""
CAT411 — Bridge Earthquake Catastrophe Modeling Framework
Northridge Case Study & General Risk Assessment

Usage:
    python main.py --full-analysis        # Automated end-to-end analysis (Recommended)
    python main.py --interactive          # Interactive menu-driven interface
    python main.py --download-hazard      # Download USGS hazard data only
    python main.py --probabilistic        # Run probabilistic risk analysis
    python main.py --fragility-only       # Run fragility curve analysis only
"""

import argparse
import os
import sys
import subprocess

import numpy as np
import pandas as pd

from src.hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER
from src.fragility import compute_all_curves, damage_state_probabilities, apply_skew_modification
from src.plotting import (
    plot_single_class,
    plot_comparison,
    plot_damage_distribution,
    plot_northridge_scenario,
    plot_ground_motion_field,
    plot_loss_by_class,
    plot_ep_curve,
    plot_portfolio_damage,
    plot_shakemap_grid,
    plot_bridge_damage_map,
    plot_nbi_bridge_distribution_map,
    plot_analysis_summary,
    plot_bridges_on_shakemap,
    plot_attenuation_curve,
)
from src.northridge_case import (
    NORTHRIDGE_GROUND_MOTION,
    NORTHRIDGE_DAMAGE_STATS,
    print_scenario_report,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_FRAGILITY = os.path.join(OUTPUT_DIR, "fragility")
OUTPUT_ANALYSIS = os.path.join(OUTPUT_DIR, "analysis")
OUTPUT_SCENARIO = os.path.join(OUTPUT_DIR, "scenario")


def run_fragility_analysis():
    """Generate all fragility plots and run verification."""
    im_values = np.linspace(0.001, 2.5, 500)
    northridge_classes = list(HAZUS_BRIDGE_FRAGILITY.keys())

    print("=" * 60)
    print("Bridge Fragility Analysis — Hazus 6.1 Method")
    print("Northridge Earthquake Case Study")
    print("=" * 60)
    print()

    # 1. Individual fragility curves
    print("[1/5] Generating individual fragility curves...")
    for hwb in northridge_classes:
        path = plot_single_class(hwb, im_values, OUTPUT_FRAGILITY)
        print(f"  Saved: {path}")

    # 2. Comparison plots
    print("\n[2/5] Generating comparison plots...")
    for ds in ["slight", "complete"]:
        path = plot_comparison(northridge_classes, ds, im_values, OUTPUT_FRAGILITY)
        print(f"  Saved: {path}")

    # 3. Damage distribution plots
    print("\n[3/5] Generating damage distribution plots...")
    sample_intensities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
    for hwb in NORTHRIDGE_DAMAGE_STATS["most_vulnerable_types"]:
        path = plot_damage_distribution(hwb, sample_intensities, OUTPUT_FRAGILITY)
        print(f"  Saved: {path}")

    # 4. Northridge scenario plot
    print("\n[4/5] Generating Northridge scenario plot...")
    pga_range = NORTHRIDGE_GROUND_MOTION["typical_sa_1s_near_field_g"]
    path = plot_northridge_scenario("HWB5", im_values, pga_range, OUTPUT_SCENARIO)
    print(f"  Saved: {path}")

    # 5. Scenario report
    print("\n[5/5] Running Northridge scenario analysis...\n")
    report = print_scenario_report(sa_value=0.60)
    print(report)

    # Verification
    print("\n\nVERIFICATION CHECKS")
    print("-" * 40)
    _run_verification(im_values)


def run_data_analysis(
    hwb_filter=None,
    material_filter=None,
    design_era=None,
    bbox=None,
    config=None,
):
    """Load real data and run integrated analysis."""
    from src.data_loader import (
        load_shakemap,
        load_stations,
        load_nbi,
        classify_nbi_to_hazus,
        DATA_DIR,
    )
    from src.config import AnalysisConfig, print_config_summary, IM_COLUMN_MAP

    # Merge CLI args into config (CLI overrides config.yaml)
    if config is None:
        config = AnalysisConfig()
    if hwb_filter:
        config.hwb_filter = hwb_filter
    if material_filter:
        config.material_filter = material_filter
    if design_era:
        config.design_era = design_era
    if bbox:
        config.region = {
            "lat_min": bbox[0], "lat_max": bbox[1],
            "lon_min": bbox[2], "lon_max": bbox[3],
        }

    print("=" * 60)
    print("Loading Real Data for Seismic Risk Analysis")
    print("=" * 60)
    print()
    print_config_summary(config)

    # --- ShakeMap ---
    shakemap_path = DATA_DIR / "grid.xml"
    if shakemap_path.exists():
        print("[ShakeMap] Loading grid.xml...")
        sm = load_shakemap()
        print(f"  Grid points: {len(sm):,}")
        print(f"  Available IMs: {[c for c in sm.columns if c in IM_COLUMN_MAP.values()]}")
        im_col = config.im_column
        if im_col in sm.columns:
            print(f"  Selected IM ({config.im_type}): "
                  f"{sm[im_col].min():.3f}g – {sm[im_col].max():.3f}g")
        else:
            print(f"  ⚠ Selected IM column '{im_col}' not found in ShakeMap!")
        print(f"  Bounding box: ({sm['LAT'].min():.2f}, {sm['LON'].min():.2f}) "
              f"to ({sm['LAT'].max():.2f}, {sm['LON'].max():.2f})")
        print()
    else:
        print(f"[ShakeMap] grid.xml not found at {shakemap_path}")
        print("  Run: python main.py --download")
        print()
        sm = None

    # --- Stations ---
    station_path = DATA_DIR / "stationlist.json"
    if station_path.exists():
        print("[Stations] Loading stationlist.json...")
        stations = load_stations()
        print(f"  Stations: {len(stations):,}")
        valid_pga = stations["pga"].dropna()
        if len(valid_pga) > 0:
            print(f"  PGA range (recorded): {valid_pga.min():.3f}g – {valid_pga.max():.3f}g")
        valid_sa10 = stations["psa10"].dropna()
        if len(valid_sa10) > 0:
            print(f"  Sa(1.0s) range (recorded): {valid_sa10.min():.3f}g – {valid_sa10.max():.3f}g")
        print()
    else:
        print(f"[Stations] stationlist.json not found at {station_path}")
        stations = None

    # --- NBI ---
    # Search for any state NBI file (e.g. CA24.txt, TX24.txt, NY24.txt)
    nbi_files = sorted(DATA_DIR.glob("[A-Z][A-Z]*.txt")) if DATA_DIR.exists() else []
    if nbi_files:
        nbi_path = nbi_files[0]
        print(f"[NBI] Loading {nbi_path.name}...")
        nbi_bbox = config.bbox_dict
        nbi = load_nbi(nbi_path, northridge_bbox=nbi_bbox)
        print(f"  Bridges in region: {len(nbi):,}")
        if len(nbi) > 0:
            print(f"  Year built range: {int(nbi['year_built'].min())} – {int(nbi['year_built'].max())}")
            print(f"  Materials: {nbi['material'].value_counts().to_dict()}")

            # Classify into Hazus bridge classes
            print("\n[NBI] Classifying bridges into Hazus classes...")
            nbi = classify_nbi_to_hazus(
                nbi,
                hwb_filter=config.hwb_filter,
                design_era_filter=config.design_era,
                material_filter=config.material_filter,
                nbi_filters=config.bridge_selection,
            )
            hwb_counts = nbi["hwb_class"].value_counts().sort_index()
            print("  HWB class distribution:")
            for hwb, count in hwb_counts.items():
                print(f"    {hwb}: {count}")

            # Compute expected damage for each bridge using ShakeMap data
            im_col = config.im_column
            if sm is not None and im_col in sm.columns:
                _compute_bridge_damage(nbi, sm, config)
                
                # Visualizations for real data
                print("\n[Analysis] Generating visualizations for real data...")
                # 1. Plot full ShakeMap area (configured IM)
                from src.config import IM_COLUMN_MAP as _IM_MAP
                sm_im_col = _IM_MAP.get(config.im_type, "PSA10")
                path_sm = plot_shakemap_grid(
                    sm, intensity_measure=sm_im_col,
                    output_dir=OUTPUT_ANALYSIS, filename="01_shakemap_full_area.png",
                )
                print(f"  Saved: {path_sm}")
                
                # 2. Plot NBI bridge distribution map
                path_nbi = plot_nbi_bridge_distribution_map(
                    nbi,
                    output_dir=OUTPUT_ANALYSIS,
                    filename="02_nbi_bridge_distribution_map.png",
                )
                print(f"  Saved: {path_nbi}")

                # 3. Plot ground motion at bridge sites (use im_selected)
                from src.exposure import SiteParams
                bridge_sites = [SiteParams(lat=r["latitude"], lon=r["longitude"]) for _, r in nbi.iterrows()]
                im_vals = nbi["im_selected"].values if "im_selected" in nbi.columns else nbi["sa_10"].values
                path_gm = plot_ground_motion_field(bridge_sites, im_vals, output_dir=OUTPUT_ANALYSIS, filename="03_bridge_site_ground_motion.png", im_type=config.im_type)
                print(f"  Saved: {path_gm}")
                
                # 4. Plot specific damage map
                path_dm = plot_bridge_damage_map(nbi, damage_state="complete", output_dir=OUTPUT_ANALYSIS, filename="04_bridge_damage_spatial.png")
                print(f"  Saved: {path_dm}")

                # 5. NEW — Bridges overlaid on ShakeMap contour
                path_overlay = plot_bridges_on_shakemap(
                    sm, nbi, im_type=config.im_type,
                    output_dir=OUTPUT_ANALYSIS, filename="05_bridges_on_shakemap.png",
                )
                print(f"  Saved: {path_overlay}")

                # 6. NEW — Attenuation curve (GMPE vs observed)
                path_atten = plot_attenuation_curve(
                    nbi, im_type=config.im_type,
                    output_dir=OUTPUT_ANALYSIS, filename="06_attenuation_curve.png",
                )
                print(f"  Saved: {path_atten}")

                # 7. Portfolio summary stats & Dashboard
                count_by_ds = {ds: nbi[f"P_{ds}"].mean() * len(nbi) for ds in ["none", "slight", "moderate", "extensive", "complete"]}
                
                total_loss = nbi["expected_loss"].sum() if "expected_loss" in nbi.columns else 0
                
                # Read event info from ShakeMap metadata (fallback to generic)
                _event_id = sm.attrs.get("event_id", "unknown")
                _event_desc = sm.attrs.get("event_description", "")
                _event_label = f"{_event_desc} ({_event_id})" if _event_desc else _event_id

                stats_dict = {
                    "event_id": _event_label,
                    "total_bridges": len(nbi),
                    "max_pga": sm["PGA"].max() if "PGA" in sm.columns else 0,
                    "avg_sa": nbi["im_selected"].mean() if "im_selected" in nbi.columns else 0,
                    "im_type": config.im_type,
                    "interpolation": config.interpolation_method,
                    "total_loss": total_loss,
                    "damage_distribution": count_by_ds,
                    "sa_values": nbi["im_selected"].values if "im_selected" in nbi.columns else nbi["sa_10"].values,
                    "class_breakdown": nbi["hwb_class"].value_counts().to_dict()
                }
                path_dash = plot_analysis_summary(stats_dict, output_dir=OUTPUT_ANALYSIS, filename="00_analysis_dashboard.png")
                print(f"  Saved: {path_dash}")
                
                # 8. Portfolio damage distribution
                path_pd = plot_portfolio_damage(count_by_ds, len(nbi), output_dir=OUTPUT_ANALYSIS, filename="07_portfolio_damage_bars.png")
                print(f"  Saved: {path_pd}")
                
        print()
    else:
        print(f"[NBI] No NBI file ([A-Z][A-Z]*.txt) found in {DATA_DIR}")
        print("  Place a state NBI file (e.g. CA24.txt, TX24.txt) in the data/ directory.")
        nbi = None

    return sm, stations, nbi


def _calibrated_damage_probs(im_val: float, hwb_class: str, cal_factor: float, skew_angle: float = 0.0) -> dict:
    """Compute damage probs with calibrated fragility medians and skew.

    Multiplies all Hazus medians by cal_factor before computing.
    factor < 1.0 → more vulnerable, factor > 1.0 → less vulnerable.
    Applies Hazus skew modification if skew_angle > 0.
    """
    import numpy as np
    from src.fragility import fragility_curve, apply_skew_modification
    from src.hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER

    params = HAZUS_BRIDGE_FRAGILITY.get(hwb_class)
    if params is None:
        return {"none": 1.0, "slight": 0.0, "moderate": 0.0,
                "extensive": 0.0, "complete": 0.0}

    ds_params = params["damage_states"]
    exceed = {}
    for ds in DAMAGE_STATE_ORDER:
        med = ds_params[ds]["median"] * cal_factor
        if skew_angle > 0:
            med = apply_skew_modification(med, skew_angle)
        beta = ds_params[ds]["beta"]
        exceed[ds] = float(fragility_curve(np.array([im_val]), med, beta)[0])

    probs = {
        "none":      1.0 - exceed.get("slight", 0.0),
        "slight":    exceed.get("slight", 0.0) - exceed.get("moderate", 0.0),
        "moderate":  exceed.get("moderate", 0.0) - exceed.get("extensive", 0.0),
        "extensive": exceed.get("extensive", 0.0) - exceed.get("complete", 0.0),
        "complete":  exceed.get("complete", 0.0),
    }
    return probs


def _compute_bridge_damage(nbi, shakemap, config=None):
    """Assign ShakeMap IM to each bridge and compute damage probs.

    Parameters
    ----------
    nbi : pd.DataFrame
        NBI bridge data with 'latitude', 'longitude', 'hwb_class'.
    shakemap : pd.DataFrame
        ShakeMap grid data with 'LAT', 'LON', and IM columns.
    config : AnalysisConfig, optional
        Analysis configuration (IM type, calibration, fragility overrides).
    """
    from src.config import AnalysisConfig, IM_COLUMN_MAP

    if config is None:
        config = AnalysisConfig()

    if config.im_source == "gmpe":
        raise NotImplementedError(
            "im_source='gmpe' is not yet supported in the data analysis pipeline. "
            "Please use im_source='shakemap' (requires downloaded ShakeMap data) "
            "or use --pipeline / --probabilistic modes for GMPE-based analysis."
        )

    print(f"\n[Analysis] Assigning ground motion to bridges "
          f"(IM: {config.im_type}, interpolation: {config.interpolation_method})...")

    from src.interpolation import interpolate_im

    grid_lats = shakemap["LAT"].values
    grid_lons = shakemap["LON"].values
    bridge_lats = nbi["latitude"].values
    bridge_lons = nbi["longitude"].values

    # Assign ALL available IMs from ShakeMap using configured interpolation
    for im_name, sm_col in IM_COLUMN_MAP.items():
        if sm_col in shakemap.columns:
            nbi[f"im_{im_name}"] = interpolate_im(
                grid_lats, grid_lons, shakemap[sm_col].values,
                bridge_lats, bridge_lons,
                method=config.interpolation_method,
                **config.interpolation_params,
            )

    # Select the configured IM for fragility analysis
    im_col_name = f"im_{config.im_type}"
    nbi["im_selected"] = nbi[im_col_name] if im_col_name in nbi.columns else 0.0

    # Also keep legacy aliases for backward compatibility
    if "im_SA10" in nbi.columns:
        nbi["sa_10"] = nbi["im_SA10"]
    if "im_PGA" in nbi.columns:
        nbi["pga"] = nbi["im_PGA"]

    print(f"  Bridges with IM assigned: {nbi['im_selected'].notna().sum()}")
    print(f"  {config.im_type} at bridge sites: "
          f"{nbi['im_selected'].min():.3f}g – {nbi['im_selected'].max():.3f}g")

    # Show all IM ranges
    for im_name, sm_col in IM_COLUMN_MAP.items():
        col = f"im_{im_name}"
        if col in nbi.columns:
            print(f"    {im_name:>4}: {nbi[col].min():.4f}g – {nbi[col].max():.4f}g"
                  + (" ← selected" if im_name == config.im_type else ""))

    # Compute damage probabilities per bridge
    print("\n[Analysis] Computing damage state probabilities...")

    # Check for fragility overrides and calibration
    has_overrides = bool(config.fragility_overrides)
    has_calibration = (config.global_median_factor != 1.0 or bool(config.class_factors))
    if has_overrides:
        print(f"  Using fragility overrides for: {list(config.fragility_overrides.keys())}")
    if has_calibration:
        print(f"  Calibration: global={config.global_median_factor}, "
              f"class={config.class_factors}")

    ds_cols = ["P_none", "P_slight", "P_moderate", "P_extensive", "P_complete"]

    # Check if skew angle data is available
    has_skew = "skew_angle" in nbi.columns

    for _, row in nbi.iterrows():
        im_val = row["im_selected"]
        hwb = row["hwb_class"]
        skew = float(row["skew_angle"]) if has_skew and pd.notna(row.get("skew_angle")) else 0.0

        # Use overridden fragility if available, otherwise default
        if hwb in config.fragility_overrides:
            # Custom fragility params from config
            custom = config.fragility_overrides[hwb]
            from src.fragility import fragility_curve
            import numpy as np

            ds_order = ["slight", "moderate", "extensive", "complete"]
            exceed = {}
            for ds in ds_order:
                if ds in custom:
                    med = custom[ds]["median"]
                    beta = custom[ds]["beta"]
                    # Apply calibration
                    cal = config.class_factors.get(hwb, config.global_median_factor)
                    med *= cal
                    # Apply skew modification (reduces median for skewed bridges)
                    if skew > 0:
                        med = apply_skew_modification(med, skew)
                    exceed[ds] = float(fragility_curve(np.array([im_val]), med, beta)[0])
                else:
                    exceed[ds] = 0.0

            probs = {
                "none":      1.0 - exceed.get("slight", 0.0),
                "slight":    exceed.get("slight", 0.0) - exceed.get("moderate", 0.0),
                "moderate":  exceed.get("moderate", 0.0) - exceed.get("extensive", 0.0),
                "extensive": exceed.get("extensive", 0.0) - exceed.get("complete", 0.0),
                "complete":  exceed.get("complete", 0.0),
            }
        else:
            # Default Hazus fragility, with optional calibration
            if has_calibration:
                cal_factor = config.class_factors.get(hwb, config.global_median_factor)
                probs = _calibrated_damage_probs(im_val, hwb, cal_factor, skew)
            else:
                probs = _calibrated_damage_probs(im_val, hwb, 1.0, skew) if skew > 0 else damage_state_probabilities(im_val, hwb)

        for ds_key, col in zip(
            ["none", "slight", "moderate", "extensive", "complete"], ds_cols
        ):
            nbi.loc[row.name, col] = probs[ds_key]

    # Summary statistics
    print("\n  Expected damage distribution (bridge-count weighted):")
    for col, ds_name in zip(ds_cols, ["None", "Slight", "Moderate", "Extensive", "Complete"]):
        mean_p = nbi[col].mean()
        print(f"    {ds_name:>10}: {mean_p:.3f} ({mean_p * len(nbi):.0f} bridges)")

    # Save results
    output_path = os.path.join(OUTPUT_ANALYSIS, "bridge_damage_results.csv")
    os.makedirs(OUTPUT_ANALYSIS, exist_ok=True)
    nbi.to_csv(output_path, index=False)
    print(f"\n  Results saved to: {output_path}")


def _run_verification(im_values):
    """Run sanity checks on fragility curves."""
    all_pass = True

    for hwb in HAZUS_BRIDGE_FRAGILITY:
        curves = compute_all_curves(hwb, im_values)

        for ds in DAMAGE_STATE_ORDER:
            diff = np.diff(curves[ds])
            if not np.all(diff >= -1e-10):
                print(f"  FAIL: {hwb} {ds} not monotonically increasing")
                all_pass = False

        for ds in DAMAGE_STATE_ORDER:
            if np.any(curves[ds] < -1e-10) or np.any(curves[ds] > 1 + 1e-10):
                print(f"  FAIL: {hwb} {ds} out of [0,1] bounds")
                all_pass = False

        for i in range(len(DAMAGE_STATE_ORDER) - 1):
            ds_curr = DAMAGE_STATE_ORDER[i]
            ds_next = DAMAGE_STATE_ORDER[i + 1]
            if np.any(curves[ds_curr] < curves[ds_next] - 1e-10):
                print(f"  FAIL: {hwb} {ds_curr} < {ds_next}")
                all_pass = False

        for test_im in [0.3, 0.6, 1.0]:
            probs = damage_state_probabilities(test_im, hwb)
            total = sum(probs.values())
            if abs(total - 1.0) > 1e-6:
                print(f"  FAIL: {hwb} at Sa={test_im}g probs sum to {total:.6f}")
                all_pass = False

    if all_pass:
        print("  ALL CHECKS PASSED")
    print()


# ── CAT Model Pipeline ───────────────────────────────────────────────────

def run_pipeline(n_bridges: int = 100, n_realizations: int = 50):
    """Run the full deterministic Hazard -> Exposure -> Vulnerability -> Loss pipeline."""
    from src.hazard import boore_atkinson_2008_sa10, EarthquakeScenario
    from src.exposure import generate_synthetic_portfolio, portfolio_to_sites, portfolio_summary
    from src.engine import (
        NORTHRIDGE_SCENARIO,
        run_deterministic,
        print_deterministic_report,
    )

    print("=" * 65)
    print("CAT MODEL PIPELINE — DETERMINISTIC MODE")
    print("Hazard -> Exposure -> Vulnerability -> Loss")
    print("=" * 65)
    print()

    # BA08 sanity check
    print("[Sanity Check] BA08 GMPE for Mw 7.0, R_JB=10km, Vs30=760 m/s:")
    sa_check, sigma_check = boore_atkinson_2008_sa10(7.0, 10.0, 760.0, "reverse")
    print(f"  Sa(1.0s) = {sa_check:.4f}g  (sigma_ln = {sigma_check:.3f})")
    print()

    # Step 1: Exposure
    print(f"[1/4] Generating synthetic portfolio ({n_bridges} bridges)...")
    portfolio = generate_synthetic_portfolio(n_bridges)
    ps = portfolio_summary(portfolio)
    print(f"  Total replacement cost: ${ps['total_replacement_cost']:,.0f}")
    print(f"  Class distribution: {ps['class_distribution']}")
    print()

    # Step 2: Run deterministic analysis
    print(f"[2/4] Running deterministic analysis ({n_realizations} realizations)...")
    result = run_deterministic(
        NORTHRIDGE_SCENARIO, portfolio, n_realizations=n_realizations
    )
    print()

    # Step 3: Report
    print("[3/4] Generating report...\n")
    report = print_deterministic_report(result)
    print(report)
    print()

    # Step 4: Plots
    print("[4/4] Generating pipeline plots...")
    sites = portfolio_to_sites(portfolio)

    # Ground motion field (use median Sa)
    path = plot_ground_motion_field(
        sites, result.median_sa, scenario=NORTHRIDGE_SCENARIO,
        output_dir=OUTPUT_SCENARIO,
    )
    print(f"  Saved: {path}")

    # Loss by class (use median realization)
    losses = np.array([r.total_loss for r in result.loss_results])
    median_idx = np.argsort(losses)[len(losses) // 2]
    rep = result.loss_results[median_idx]

    path = plot_loss_by_class(rep.loss_by_class, output_dir=OUTPUT_SCENARIO)
    print(f"  Saved: {path}")

    path = plot_portfolio_damage(
        rep.count_by_ds, len(portfolio), output_dir=OUTPUT_SCENARIO
    )
    print(f"  Saved: {path}")

    print("\nPipeline complete.")


def run_probabilistic_analysis(
    n_bridges: int = 100,
    n_events: int = 50,
    n_realizations: int = 20,
):
    """Run the probabilistic analysis with stochastic event catalog."""
    from src.exposure import generate_synthetic_portfolio, portfolio_summary
    from src.engine import (
        run_probabilistic,
        print_probabilistic_report,
    )

    print("=" * 65)
    print("CAT MODEL PIPELINE — PROBABILISTIC MODE")
    print("Stochastic Event Set -> EP Curve + AAL")
    print("=" * 65)
    print()

    # Step 1: Portfolio
    print(f"[1/3] Generating synthetic portfolio ({n_bridges} bridges)...")
    portfolio = generate_synthetic_portfolio(n_bridges)
    ps = portfolio_summary(portfolio)
    print(f"  Total replacement cost: ${ps['total_replacement_cost']:,.0f}")
    print()

    # Step 2: Probabilistic run
    print(f"[2/3] Running probabilistic analysis "
          f"({n_events} events x {n_realizations} realizations)...")
    result = run_probabilistic(
        portfolio,
        n_events=n_events,
        n_realizations=n_realizations,
    )
    print()

    # Step 3: Report and plots
    print("[3/3] Generating report and plots...\n")
    report = print_probabilistic_report(result)
    print(report)
    print()

    # EP curve plot
    path = plot_ep_curve(result.ep_curve, output_dir=OUTPUT_SCENARIO)
    print(f"  Saved: {path}")

    print("\nProbabilistic analysis complete.")


# ── Automated Workflows ──────────────────────────────────────────────────

def run_full_analysis(
    event_id: str = "ci3144585",
    overwrite: bool = False,
    hwb_filter=None,
    material_filter=None,
    design_era=None,
    bbox=None,
    config=None,
):
    """Run the complete automated end-to-end analysis pipeline."""
    print("\n" + "="*60)
    print("  CAT411 — STARTING FULL AUTOMATED ANALYSIS")
    print("="*60)
    
    # 1. Download
    print("\n[Step 1/3] Downloading USGS Hazard Data...")
    from src.hazard_download import download_all_hazard_data
    download_all_hazard_data(event_id=event_id, overwrite=overwrite)
    
    # 2. Run Analysis & Generate Maps
    print("\n[Step 2/3] Processing Data & Computing Risks...")
    run_data_analysis(
        hwb_filter=hwb_filter,
        material_filter=material_filter,
        design_era=design_era,
        bbox=bbox,
        config=config,
    )
    
    # 3. Generate Fragility Library
    print("\n[Step 3/3] Generating Fragility Curve Library Reference...")
    run_fragility_analysis()
    
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE!")
    print(f"  All results are saved in: {OUTPUT_DIR}")
    print("="*60 + "\n")


def run_interactive_menu():
    """Display an interactive menu for the user."""
    while True:
        print("\n" + "="*60)
        print("  CAT411 — Bridge Earthquake Catastrophe Modeling")
        print("="*60)
        print("\n  Main Menu:\n")
        print("  [1] 🚀 Run Full Analysis (Download + Analyze + Visualize)")
        print("  [2] 📥 Download USGS Hazard Data Only")
        print("  [3] 📊 Generate Fragility Curves Reference")
        print("  [4] 🎲 Run Stochastic Probabilistic Analysis")
        print("  [0] 👋 Exit")
        
        choice = input("\n  Selection: ").strip()
        
        if choice == '1':
            run_full_analysis()
        elif choice == '2':
            from src.hazard_download import download_all_hazard_data
            evt = input("  Enter USGS Event ID [default: ci3144585]: ").strip() or "ci3144585"
            download_all_hazard_data(event_id=evt)
            run_data_analysis()
        elif choice == '3':
            run_fragility_analysis()
        elif choice == '4':
            n_b = input("  Number of bridges [default: 100]: ").strip() or "100"
            run_probabilistic_analysis(n_bridges=int(n_b))
        elif choice == '0':
            print("\n  Exiting CAT411. Goodbye!")
            break
        else:
            print("\n  [Error] Invalid selection. Please try again.")


# ── Main entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CAT411 — Bridge Earthquake Catastrophe Modeling Framework"
    )
    # Primary modes
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Run full automated analysis: Download -> Process -> Analyze -> Visualize",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive menu-driven interface",
    )
    parser.add_argument(
        "--download-hazard",
        action="store_true",
        help="Download USGS hazard data (ShakeMap + hazard curves) for Northridge",
    )
    parser.add_argument(
        "--probabilistic",
        action="store_true",
        help="Run probabilistic analysis with stochastic event catalog",
    )
    parser.add_argument(
        "--fragility-only",
        action="store_true",
        help="Run fragility curve analysis only (no real data needed)",
    )
    
    # Configuration
    parser.add_argument(
        "--hazard-event",
        type=str,
        default="ci3144585",
        help="USGS event ID for ShakeMap download (default: ci3144585 = Northridge)",
    )
    parser.add_argument(
        "--n-bridges",
        type=int,
        default=100,
        help="Number of synthetic bridges for pipeline/probabilistic modes",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=50,
        help="Monte Carlo realizations per event",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=50,
        help="Number of stochastic events for probabilistic mode",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downloaded files",
    )

    # ── Focused analysis filters ──────────────────────────────────────
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to analysis config file (default: config.yaml)",
    )
    parser.add_argument(
        "--im-type",
        type=str,
        choices=["PGA", "SA03", "SA10", "SA30"],
        default=None,
        help="IM type from ShakeMap (default: SA10). Overrides config.yaml.",
    )
    parser.add_argument(
        "--hwb-filter",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific HWB classes (e.g. --hwb-filter HWB5 HWB17)",
    )
    parser.add_argument(
        "--material-filter",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific materials (e.g. --material-filter concrete steel)",
    )
    parser.add_argument(
        "--design-era",
        type=str,
        choices=["conventional", "seismic"],
        default=None,
        help="Filter by design era (conventional=pre-1975, seismic=post-1975)",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        default=None,
        help="Custom bounding box (e.g. --bbox 34.0 34.5 -118.8 -118.2)",
    )
    parser.add_argument(
        "--nbi-filter",
        type=str,
        nargs="+",
        default=None,
        help="Generic NBI column filters as key=value (e.g. --nbi-filter county=037 'year_built>1960')",
    )
    args = parser.parse_args()

    # Load config file and merge CLI overrides
    from src.config import load_config
    cfg = load_config(args.config)

    # CLI im-type overrides config
    if args.im_type:
        cfg.im_type = args.im_type

    # CLI nbi-filter merges into bridge_selection
    if args.nbi_filter:
        for filt in args.nbi_filter:
            if ">=" in filt:
                k, v = filt.split(">=", 1)
                cfg.bridge_selection[k.strip()] = ">=" + v.strip()
            elif "<=" in filt:
                k, v = filt.split("<=", 1)
                cfg.bridge_selection[k.strip()] = "<=" + v.strip()
            elif ">" in filt:
                k, v = filt.split(">", 1)
                cfg.bridge_selection[k.strip()] = ">" + v.strip()
            elif "<" in filt:
                k, v = filt.split("<", 1)
                cfg.bridge_selection[k.strip()] = "<" + v.strip()
            elif "=" in filt:
                k, v = filt.split("=", 1)
                cfg.bridge_selection[k.strip()] = v.strip()
            else:
                print(f"[Warning] Cannot parse filter: {filt}")

    if args.interactive:
        run_interactive_menu()
        return

    if args.full_analysis:
        run_full_analysis(
            event_id=args.hazard_event,
            overwrite=args.overwrite,
            hwb_filter=args.hwb_filter,
            material_filter=args.material_filter,
            design_era=args.design_era,
            bbox=args.bbox,
            config=cfg,
        )
        return

    if args.download_hazard:
        from src.hazard_download import download_all_hazard_data
        download_all_hazard_data(
            event_id=args.hazard_event,
            overwrite=args.overwrite,
        )
        print("\n[Pipeline] Hazard data download complete. Running data processing...")
        run_data_analysis(
            hwb_filter=args.hwb_filter,
            material_filter=args.material_filter,
            design_era=args.design_era,
            bbox=args.bbox,
            config=cfg,
        )
        return

    if args.probabilistic:
        run_probabilistic_analysis(
            n_bridges=args.n_bridges,
            n_events=args.n_events,
            n_realizations=args.n_realizations,
        )
    elif args.fragility_only:
        run_fragility_analysis()
    else:
        # Default: Show help or suggest full analysis
        print("\n" + "="*60)
        print("CAT411 — Quick Start")
        print("="*60)
        print("To run a full automated analysis (Northridge scenario):")
        print("  python main.py --full-analysis")
        print("\nTo start interactive mode:")
        print("  python main.py --interactive")
        print("\nFor more options, use --help")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()

"""
Bridge Fragility Analysis — Hazus Method, Northridge Case Study

Entry point: downloads data (if needed), generates fragility curves,
comparison plots, damage distributions, and Northridge scenario analysis.
Now includes full CAT model pipeline: Hazard -> Exposure -> Vulnerability -> Loss.

Usage:
    python main.py                      # Full analysis (requires data files)
    python main.py --download           # Download data files first, then analyze
    python main.py --download-only      # Download data files only (no analysis)
    python main.py --fragility-only     # Run fragility analysis without real data
    python main.py --pipeline           # Deterministic Northridge end-to-end
    python main.py --probabilistic      # Stochastic event set -> EP curve + AAL
    python main.py --n-bridges 200      # Synthetic portfolio size
    python main.py --n-realizations 50  # Monte Carlo realizations
    python main.py --n-events 50        # Stochastic catalog size
"""

import argparse
import os
import sys

import numpy as np

from src.hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER
from src.fragility import compute_all_curves, damage_state_probabilities
from src.plotting import (
    plot_single_class,
    plot_comparison,
    plot_damage_distribution,
    plot_northridge_scenario,
    plot_ground_motion_field,
    plot_loss_by_class,
    plot_ep_curve,
    plot_portfolio_damage,
)
from src.northridge_case import (
    NORTHRIDGE_GROUND_MOTION,
    NORTHRIDGE_DAMAGE_STATS,
    print_scenario_report,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


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
        path = plot_single_class(hwb, im_values, OUTPUT_DIR)
        print(f"  Saved: {path}")

    # 2. Comparison plots
    print("\n[2/5] Generating comparison plots...")
    for ds in ["slight", "complete"]:
        path = plot_comparison(northridge_classes, ds, im_values, OUTPUT_DIR)
        print(f"  Saved: {path}")

    # 3. Damage distribution plots
    print("\n[3/5] Generating damage distribution plots...")
    sample_intensities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
    for hwb in NORTHRIDGE_DAMAGE_STATS["most_vulnerable_types"]:
        path = plot_damage_distribution(hwb, sample_intensities, OUTPUT_DIR)
        print(f"  Saved: {path}")

    # 4. Northridge scenario plot
    print("\n[4/5] Generating Northridge scenario plot...")
    pga_range = NORTHRIDGE_GROUND_MOTION["typical_sa_1s_near_field_g"]
    path = plot_northridge_scenario("HWB5", im_values, pga_range, OUTPUT_DIR)
    print(f"  Saved: {path}")

    # 5. Scenario report
    print("\n[5/5] Running Northridge scenario analysis...\n")
    report = print_scenario_report(sa_value=0.60)
    print(report)

    # Verification
    print("\n\nVERIFICATION CHECKS")
    print("-" * 40)
    _run_verification(im_values)


def run_data_download():
    """Download ShakeMap and NBI data files."""
    from src.data_loader import download_all

    print("=" * 60)
    print("Downloading Hazard & Bridge Data")
    print("=" * 60)
    print()
    paths = download_all()
    print()
    print("Downloaded files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print()


def run_data_analysis():
    """Load real data and run integrated analysis."""
    from src.data_loader import (
        load_shakemap,
        load_stations,
        load_nbi,
        classify_nbi_to_hazus,
        DATA_DIR,
    )

    print("=" * 60)
    print("Loading Real Data for Northridge Analysis")
    print("=" * 60)
    print()

    # --- ShakeMap ---
    shakemap_path = DATA_DIR / "grid.xml"
    if shakemap_path.exists():
        print("[ShakeMap] Loading grid.xml...")
        sm = load_shakemap()
        print(f"  Grid points: {len(sm):,}")
        print(f"  PGA range:   {sm['PGA'].min():.3f}g – {sm['PGA'].max():.3f}g")
        if "PSA10" in sm.columns:
            print(f"  Sa(1.0s) range: {sm['PSA10'].min():.3f}g – {sm['PSA10'].max():.3f}g")
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
    nbi_files = list(DATA_DIR.glob("CA*.txt")) if DATA_DIR.exists() else []
    if nbi_files:
        nbi_path = nbi_files[0]
        print(f"[NBI] Loading {nbi_path.name}...")
        nbi = load_nbi(nbi_path)
        print(f"  Bridges in Northridge area: {len(nbi):,}")
        if len(nbi) > 0:
            print(f"  Year built range: {int(nbi['year_built'].min())} – {int(nbi['year_built'].max())}")
            print(f"  Materials: {nbi['material'].value_counts().to_dict()}")

            # Classify into Hazus bridge classes
            print("\n[NBI] Classifying bridges into Hazus classes...")
            nbi = classify_nbi_to_hazus(nbi)
            hwb_counts = nbi["hwb_class"].value_counts().sort_index()
            print("  HWB class distribution:")
            for hwb, count in hwb_counts.items():
                print(f"    {hwb}: {count}")

            # Compute expected damage for each bridge using ShakeMap data
            if sm is not None and "PSA10" in sm.columns:
                _compute_bridge_damage(nbi, sm)
        print()
    else:
        print(f"[NBI] No CA*.txt found in {DATA_DIR}")
        print("  Run: python main.py --download")
        nbi = None

    return sm, stations, nbi


def _compute_bridge_damage(nbi, shakemap):
    """Assign ShakeMap Sa(1.0s) to each bridge and compute damage probs."""
    from scipy.spatial import cKDTree

    print("\n[Analysis] Assigning ground motion to bridges...")

    # Build KD-tree from ShakeMap grid
    grid_coords = shakemap[["LAT", "LON"]].values
    tree = cKDTree(grid_coords)

    bridge_coords = nbi[["latitude", "longitude"]].values
    _, indices = tree.query(bridge_coords)

    nbi["sa_10"] = shakemap["PSA10"].values[indices]
    nbi["pga"] = shakemap["PGA"].values[indices]

    print(f"  Bridges with Sa(1.0s) assigned: {nbi['sa_10'].notna().sum()}")
    print(f"  Sa(1.0s) at bridge sites: "
          f"{nbi['sa_10'].min():.3f}g – {nbi['sa_10'].max():.3f}g")

    # Compute damage probabilities per bridge
    print("\n[Analysis] Computing damage state probabilities...")
    ds_cols = ["P_none", "P_slight", "P_moderate", "P_extensive", "P_complete"]

    for _, row in nbi.iterrows():
        probs = damage_state_probabilities(row["sa_10"], row["hwb_class"])
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
    output_path = os.path.join(OUTPUT_DIR, "bridge_damage_results.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
        output_dir=OUTPUT_DIR,
    )
    print(f"  Saved: {path}")

    # Loss by class (use median realization)
    losses = np.array([r.total_loss for r in result.loss_results])
    median_idx = np.argsort(losses)[len(losses) // 2]
    rep = result.loss_results[median_idx]

    path = plot_loss_by_class(rep.loss_by_class, output_dir=OUTPUT_DIR)
    print(f"  Saved: {path}")

    path = plot_portfolio_damage(
        rep.count_by_ds, len(portfolio), output_dir=OUTPUT_DIR
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
    path = plot_ep_curve(result.ep_curve, output_dir=OUTPUT_DIR)
    print(f"  Saved: {path}")

    print("\nProbabilistic analysis complete.")


# ── Main entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bridge Fragility Analysis — Hazus Method, Northridge Case Study"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download ShakeMap and NBI data files to data/",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download data only and exit (no analysis)",
    )
    parser.add_argument(
        "--fragility-only",
        action="store_true",
        help="Run fragility curve analysis only (no real data needed)",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run full deterministic CAT model pipeline (Northridge scenario)",
    )
    parser.add_argument(
        "--probabilistic",
        action="store_true",
        help="Run probabilistic analysis with stochastic event catalog",
    )
    parser.add_argument(
        "--n-bridges",
        type=int,
        default=100,
        help="Number of synthetic bridges (default: 100)",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=50,
        help="Monte Carlo realizations per event (default: 50)",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=50,
        help="Number of stochastic events for probabilistic mode (default: 50)",
    )
    args = parser.parse_args()

    if args.download or args.download_only:
        run_data_download()
        if args.download_only:
            return

    if args.pipeline:
        run_pipeline(
            n_bridges=args.n_bridges,
            n_realizations=args.n_realizations,
        )
    elif args.probabilistic:
        run_probabilistic_analysis(
            n_bridges=args.n_bridges,
            n_events=args.n_events,
            n_realizations=args.n_realizations,
        )
    elif args.fragility_only:
        run_fragility_analysis()
    else:
        # Default: run fragility analysis first
        run_fragility_analysis()

        # Then load and analyze real data if available
        from src.data_loader import DATA_DIR
        if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
            print("\n")
            run_data_analysis()
        else:
            print("\n[Info] No data files found in data/.")
            print("  To download: python main.py --download")
            print("  To run without data: python main.py --fragility-only")


if __name__ == "__main__":
    main()

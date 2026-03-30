"""Tool: run_scenario — run a deterministic earthquake scenario analysis."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.tools.registry import register_tool


def _run_scenario(
    magnitude: float = 6.7,
    lat: float = 34.213,
    lon: float = -118.537,
    depth_km: float = 10.0,
    fault_type: str = "reverse",
    n_bridges: int = 100,
    radius_km: float = 30.0,
    n_realizations: int = 20,
    seed: int = 42,
    use_nbi: bool = False,
) -> str:
    """Run a deterministic earthquake scenario with synthetic or NBI portfolio."""
    import numpy as np
    from src.hazard import EarthquakeScenario
    from src.exposure import generate_synthetic_portfolio, portfolio_summary
    from src.engine import run_deterministic
    from src.loss import loss_summary_table

    scenario = EarthquakeScenario(
        Mw=magnitude,
        lat=lat,
        lon=lon,
        depth_km=depth_km,
        fault_type=fault_type,
    )

    if use_nbi:
        # Load real NBI bridges near the epicenter
        from src.data_loader import parse_nbi, classify_nbi_to_hazus, DATA_DIR
        from src.exposure import create_portfolio_from_nbi

        nbi_path = DATA_DIR / "CA24.txt"
        if not nbi_path.exists():
            import pandas as pd
            csv_path = DATA_DIR / "nbi_classified_2024.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, encoding="utf-8")
            else:
                return "Error: NBI data not found"
        else:
            df = parse_nbi(nbi_path)
            df = classify_nbi_to_hazus(df)

        # Filter to region around epicenter
        import math
        lat_margin = radius_km / 111.0
        lon_margin = radius_km / (111.0 * math.cos(math.radians(lat)))
        mask = (
            (df["latitude"] >= lat - lat_margin) &
            (df["latitude"] <= lat + lat_margin) &
            (df["longitude"] >= lon - lon_margin) &
            (df["longitude"] <= lon + lon_margin)
        )
        df = df[mask]
        if len(df) == 0:
            return f"No NBI bridges found within {radius_km}km of ({lat}, {lon})"

        # Limit to avoid very long computation
        if len(df) > 500:
            df = df.sample(500, random_state=seed)

        portfolio = create_portfolio_from_nbi(df)
    else:
        portfolio = generate_synthetic_portfolio(
            n_bridges=n_bridges,
            center=(lat, lon),
            radius_km=radius_km,
            seed=seed,
        )

    result = run_deterministic(
        scenario=scenario,
        portfolio=portfolio,
        n_realizations=n_realizations,
        seed=seed,
    )

    ps = result.portfolio_summary

    lines = [
        "DETERMINISTIC SCENARIO ANALYSIS",
        "=" * 60,
        f"  Earthquake:  M{scenario.Mw} at ({scenario.lat:.3f}, {scenario.lon:.3f})",
        f"  Depth:       {scenario.depth_km} km",
        f"  Fault type:  {scenario.fault_type}",
        f"  Portfolio:   {result.portfolio_size} bridges ({'NBI' if use_nbi else 'synthetic'})",
        f"  Realizations:{result.n_realizations}",
        "",
        f"  Mean Total Loss:    ${result.mean_loss:,.0f}",
        f"  Std Total Loss:     ${result.std_loss:,.0f}",
        f"  Mean Loss Ratio:    {result.mean_loss_ratio:.4f} ({result.mean_loss_ratio*100:.2f}%)",
        f"  Total Replacement:  ${ps.get('total_replacement_cost', 0):,.0f}",
        "",
    ]

    # Use the last realization's detailed breakdown
    if not result.loss_results:
        lines.append("(No loss results — all realizations may have failed)")
    else:
        last_result = result.loss_results[-1]
        lines.append(loss_summary_table(last_result))

    # Sa statistics
    lines.append("")
    lines.append("Ground Motion Statistics:")
    lines.append(f"  Median Sa range: {float(np.min(result.median_sa)):.4f} – {float(np.max(result.median_sa)):.4f} g")
    lines.append(f"  Mean median Sa:  {float(np.mean(result.median_sa)):.4f} g")

    return "\n".join(lines)


register_tool(
    name="run_scenario",
    description=(
        "Run a deterministic earthquake scenario analysis. "
        "Specify earthquake parameters (magnitude, location, depth, fault type) "
        "and portfolio options (number of bridges, radius, or use real NBI data). "
        "Returns total loss, loss ratio, damage distribution, and ground motion stats."
    ),
    parameters={
        "type": "object",
        "properties": {
            "magnitude": {
                "type": "number",
                "description": "Moment magnitude (e.g. 6.7)",
                "default": 6.7,
            },
            "lat": {
                "type": "number",
                "description": "Epicenter latitude (e.g. 34.213 for Northridge)",
                "default": 34.213,
            },
            "lon": {
                "type": "number",
                "description": "Epicenter longitude (e.g. -118.537 for Northridge)",
                "default": -118.537,
            },
            "depth_km": {
                "type": "number",
                "description": "Focal depth in km (default 10)",
                "default": 10.0,
            },
            "fault_type": {
                "type": "string",
                "enum": ["reverse", "strike_slip", "normal", "unspecified"],
                "description": "Fault mechanism type",
                "default": "reverse",
            },
            "n_bridges": {
                "type": "integer",
                "description": "Number of synthetic bridges (ignored if use_nbi=true)",
                "default": 100,
            },
            "radius_km": {
                "type": "number",
                "description": "Radius in km for bridge portfolio around epicenter",
                "default": 30.0,
            },
            "n_realizations": {
                "type": "integer",
                "description": "Number of ground motion realizations (default 20)",
                "default": 20,
            },
            "seed": {
                "type": "integer",
                "description": "Random seed for reproducibility",
                "default": 42,
            },
            "use_nbi": {
                "type": "boolean",
                "description": "If true, use real NBI bridge inventory instead of synthetic",
                "default": False,
            },
        },
        "required": [],
    },
    function=_run_scenario,
)

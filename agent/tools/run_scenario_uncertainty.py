"""Tool: run_scenario_with_uncertainty — scenario analysis with confidence intervals."""

import agent.config  # noqa: F401
from agent.tools.registry import register_tool


def _run_scenario_with_uncertainty(
    magnitude: float = 6.7,
    lat: float = 34.213,
    lon: float = -118.537,
    depth_km: float = 10.0,
    fault_type: str = "reverse",
    n_bridges: int = 100,
    radius_km: float = 30.0,
    n_realizations: int = 50,
    seed: int = 42,
) -> str:
    """Run scenario with full uncertainty quantification."""
    import math
    import numpy as np
    from src.hazard import EarthquakeScenario
    from src.exposure import generate_synthetic_portfolio, portfolio_to_sites
    from src.engine import run_deterministic
    from src.loss import loss_summary_table

    scenario = EarthquakeScenario(
        Mw=magnitude, lat=lat, lon=lon,
        depth_km=depth_km, fault_type=fault_type,
    )

    portfolio = generate_synthetic_portfolio(
        n_bridges=n_bridges, center=(lat, lon),
        radius_km=radius_km, seed=seed,
    )

    result = run_deterministic(
        scenario=scenario, portfolio=portfolio,
        n_realizations=n_realizations, seed=seed,
    )

    if not result.loss_results:
        return "Error: no loss results generated"

    # Extract per-realization losses
    losses = np.array([r.total_loss for r in result.loss_results])

    # Percentiles
    p5 = float(np.percentile(losses, 5))
    p25 = float(np.percentile(losses, 25))
    p50 = float(np.percentile(losses, 50))
    p75 = float(np.percentile(losses, 75))
    p95 = float(np.percentile(losses, 95))
    mean = float(np.mean(losses))
    std = float(np.std(losses))
    cv = std / mean if mean > 0 else 0

    # Per-realization loss ratios
    total_rc = result.portfolio_summary.get("total_replacement_cost", 1)
    ratios = losses / total_rc

    # Damage state uncertainty across realizations
    ds_names = ["none", "slight", "moderate", "extensive", "complete"]
    ds_stats = {}
    for ds in ds_names:
        counts = [r.count_by_ds.get(ds, 0) for r in result.loss_results]
        ds_stats[ds] = {
            "mean": float(np.mean(counts)),
            "std": float(np.std(counts)),
            "p5": float(np.percentile(counts, 5)),
            "p95": float(np.percentile(counts, 95)),
        }

    lines = [
        "SCENARIO ANALYSIS WITH UNCERTAINTY QUANTIFICATION",
        "=" * 62,
        f"  Earthquake:    M{scenario.Mw} at ({scenario.lat:.3f}, {scenario.lon:.3f})",
        f"  Depth:         {scenario.depth_km} km, {scenario.fault_type}",
        f"  Portfolio:     {result.portfolio_size} bridges (synthetic)",
        f"  Realizations:  {n_realizations}",
        f"  Replacement:   ${total_rc:,.0f}",
        "",
        "LOSS DISTRIBUTION (across realizations)",
        "-" * 62,
        f"  Mean:          ${mean:,.0f}",
        f"  Std Dev:       ${std:,.0f}  (CV = {cv:.2f})",
        f"  Median (P50):  ${p50:,.0f}",
        f"  5th pctile:    ${p5:,.0f}",
        f"  25th pctile:   ${p25:,.0f}",
        f"  75th pctile:   ${p75:,.0f}",
        f"  95th pctile:   ${p95:,.0f}",
        "",
        f"  90% Confidence Interval: ${p5:,.0f} - ${p95:,.0f}",
        f"  Mean Loss Ratio:         {mean/total_rc:.4f} ({mean/total_rc*100:.2f}%)",
        f"  P95 Loss Ratio:          {p95/total_rc:.4f} ({p95/total_rc*100:.2f}%)",
        "",
        "DAMAGE STATE UNCERTAINTY (expected bridge counts)",
        "-" * 62,
        f"  {'State':<12s} {'Mean':>8s} {'Std':>8s} {'P5':>8s} {'P95':>8s}",
    ]
    for ds in ds_names:
        s = ds_stats[ds]
        lines.append(
            f"  {ds:<12s} {s['mean']:>8.1f} {s['std']:>8.1f} "
            f"{s['p5']:>8.1f} {s['p95']:>8.1f}"
        )

    lines.extend([
        "",
        "INTERPRETATION",
        "-" * 62,
        f"  There is a 90% probability that total loss falls between",
        f"  ${p5:,.0f} and ${p95:,.0f}.",
        f"  The coefficient of variation (CV={cv:.2f}) indicates",
        f"  {'high' if cv > 0.5 else 'moderate' if cv > 0.3 else 'low'} "
        f"uncertainty in the loss estimate.",
        "",
        "  Note: Uncertainty arises from spatial correlation of ground",
        "  motion (Jayaram-Baker 2009 model). Epistemic uncertainty",
        "  (GMPE model choice, fragility parameter uncertainty) is",
        "  not included in this analysis.",
    ])

    return "\n".join(lines)


register_tool(
    name="run_scenario_with_uncertainty",
    description=(
        "Run a deterministic earthquake scenario with full uncertainty "
        "quantification. Reports mean, median, percentiles (P5/P25/P50/P75/P95), "
        "90% confidence interval, coefficient of variation, and per-damage-state "
        "uncertainty across multiple ground motion realizations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "magnitude": {"type": "number", "default": 6.7},
            "lat": {"type": "number", "default": 34.213},
            "lon": {"type": "number", "default": -118.537},
            "depth_km": {"type": "number", "default": 10.0},
            "fault_type": {
                "type": "string",
                "enum": ["reverse", "strike_slip", "normal"],
                "default": "reverse",
            },
            "n_bridges": {"type": "integer", "default": 100},
            "radius_km": {"type": "number", "default": 30.0},
            "n_realizations": {
                "type": "integer",
                "description": "More realizations = more stable uncertainty estimates (default 50)",
                "default": 50,
            },
            "seed": {"type": "integer", "default": 42},
        },
        "required": [],
    },
    function=_run_scenario_with_uncertainty,
)

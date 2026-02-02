"""
CAT Model Pipeline Orchestrator.

Provides deterministic and probabilistic run modes that combine the
hazard, exposure, vulnerability (fragility), and loss modules into
end-to-end catastrophe model analyses.

Deterministic mode: single earthquake scenario → ground motion field
    → damage probabilities → expected loss.

Probabilistic mode: stochastic event catalog (Gutenberg-Richter) →
    multiple scenarios → EP curve + AAL.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .hazard import (
    EarthquakeScenario,
    compute_sa_at_sites,
    generate_ground_motion_fields,
)
from .exposure import (
    BridgeExposure,
    generate_synthetic_portfolio,
    portfolio_to_sites,
    portfolio_summary,
)
from .loss import (
    compute_portfolio_loss,
    compute_ep_curve,
    compute_aal,
    loss_summary_table,
    PortfolioLossResult,
)


# ── Pre-defined scenarios ─────────────────────────────────────────────────

NORTHRIDGE_SCENARIO = EarthquakeScenario(
    Mw=6.7,
    lat=34.213,
    lon=-118.537,
    depth_km=18.4,
    fault_type="reverse",
)


# ── Result dataclasses ────────────────────────────────────────────────────

@dataclass
class DeterministicResult:
    """Result from a deterministic (single-scenario) analysis."""
    scenario: EarthquakeScenario
    portfolio_size: int
    n_realizations: int
    median_sa: np.ndarray         # median Sa at each site
    mean_loss: float              # mean over realizations
    std_loss: float
    mean_loss_ratio: float
    loss_results: list[PortfolioLossResult]  # one per realization
    portfolio_summary: dict


@dataclass
class ProbabilisticResult:
    """Result from a probabilistic (stochastic catalog) analysis."""
    n_events: int
    n_realizations: int
    portfolio_size: int
    scenario_losses: np.ndarray   # mean loss per event
    annual_rates: np.ndarray
    ep_curve: dict
    aal: float
    portfolio_summary: dict


# ── Stochastic event set generation ───────────────────────────────────────

def generate_stochastic_event_set(
    n_events: int = 50,
    mag_range: tuple[float, float] = (5.0, 7.5),
    center: tuple[float, float] = (34.213, -118.537),
    radius_km: float = 80.0,
    a_value: float = 4.0,
    b_value: float = 1.0,
    seed: int = 123,
) -> tuple[list[EarthquakeScenario], np.ndarray]:
    """
    Generate a stochastic earthquake catalog using Gutenberg-Richter.

    log10(N) = a - b*M  →  rate(M) ∝ 10^(a - b*M)

    Parameters
    ----------
    n_events : int
        Number of events to sample.
    mag_range : (Mmin, Mmax)
    center : (lat, lon)
        Geographic center for event locations.
    radius_km : float
    a_value, b_value : float
        Gutenberg-Richter parameters.
    seed : int

    Returns
    -------
    (scenarios, annual_rates) : tuple
        List of EarthquakeScenario and array of annual rates.
    """
    rng = np.random.default_rng(seed)
    Mmin, Mmax = mag_range

    # Sample magnitudes from truncated exponential (GR) distribution
    # CDF: F(M) = (1 - 10^(-b*(M-Mmin))) / (1 - 10^(-b*(Mmax-Mmin)))
    beta = b_value * np.log(10)
    u = rng.uniform(size=n_events)
    denom = 1.0 - np.exp(-beta * (Mmax - Mmin))
    magnitudes = Mmin - np.log(1.0 - u * denom) / beta
    magnitudes = np.clip(magnitudes, Mmin, Mmax)
    magnitudes = np.sort(magnitudes)[::-1]  # largest first

    # Annual rate for each event bin
    # Total rate above Mmin: N(Mmin) = 10^(a - b*Mmin)
    total_rate = 10 ** (a_value - b_value * Mmin)
    annual_rates = np.full(n_events, total_rate / n_events)

    # Random locations within radius
    fault_types = ["reverse", "strike_slip", "normal"]
    scenarios = []
    for i, mw in enumerate(magnitudes):
        angle = rng.uniform(0, 2 * np.pi)
        r = radius_km * np.sqrt(rng.uniform())
        dlat = r / 111.0
        dlon = r / (111.0 * np.cos(np.radians(center[0])))
        lat = center[0] + dlat * np.sin(angle)
        lon = center[1] + dlon * np.cos(angle)
        depth = rng.uniform(5, 25)
        ft = rng.choice(fault_types, p=[0.5, 0.3, 0.2])

        scenarios.append(EarthquakeScenario(
            Mw=float(mw),
            lat=float(lat),
            lon=float(lon),
            depth_km=float(depth),
            fault_type=ft,
        ))

    return scenarios, annual_rates


# ── Deterministic analysis ────────────────────────────────────────────────

def run_deterministic(
    scenario: EarthquakeScenario,
    portfolio: list[BridgeExposure],
    n_realizations: int = 50,
    seed: int = 42,
) -> DeterministicResult:
    """
    Run a deterministic scenario analysis.

    Generates n_realizations of spatially-correlated ground motion fields
    and computes loss for each, then reports mean and standard deviation.

    Parameters
    ----------
    scenario : EarthquakeScenario
    portfolio : list[BridgeExposure]
    n_realizations : int
    seed : int

    Returns
    -------
    DeterministicResult
    """
    sites = portfolio_to_sites(portfolio)
    median_sa, _ = compute_sa_at_sites(scenario, sites)

    # Generate correlated ground motion fields
    sa_fields = generate_ground_motion_fields(
        scenario, sites, n_realizations, seed=seed
    )

    # Compute loss for each realization
    loss_results = []
    total_losses = []
    for k in range(n_realizations):
        result = compute_portfolio_loss(portfolio, sa_fields[k, :])
        loss_results.append(result)
        total_losses.append(result.total_loss)

    total_losses = np.array(total_losses)
    mean_loss = float(np.mean(total_losses))
    std_loss = float(np.std(total_losses))

    ps = portfolio_summary(portfolio)
    mean_ratio = mean_loss / ps["total_replacement_cost"] if ps["total_replacement_cost"] > 0 else 0.0

    return DeterministicResult(
        scenario=scenario,
        portfolio_size=len(portfolio),
        n_realizations=n_realizations,
        median_sa=median_sa,
        mean_loss=mean_loss,
        std_loss=std_loss,
        mean_loss_ratio=mean_ratio,
        loss_results=loss_results,
        portfolio_summary=ps,
    )


# ── Probabilistic analysis ────────────────────────────────────────────────

def run_probabilistic(
    portfolio: list[BridgeExposure],
    n_events: int = 50,
    n_realizations: int = 20,
    seed: int = 99,
) -> ProbabilisticResult:
    """
    Run a probabilistic analysis with stochastic event catalog.

    For each event, generates ground motion fields and computes the
    mean loss.  Then builds EP curve and AAL.

    Parameters
    ----------
    portfolio : list[BridgeExposure]
    n_events : int
    n_realizations : int
        Realizations per event for averaging.
    seed : int

    Returns
    -------
    ProbabilisticResult
    """
    sites = portfolio_to_sites(portfolio)
    scenarios, annual_rates = generate_stochastic_event_set(
        n_events=n_events, seed=seed
    )

    scenario_losses = np.empty(n_events)
    rng = np.random.default_rng(seed + 1000)

    for i, sc in enumerate(scenarios):
        # Generate fields for this event
        event_seed = int(rng.integers(0, 2**31))
        sa_fields = generate_ground_motion_fields(
            sc, sites, n_realizations, seed=event_seed
        )
        # Average loss over realizations
        losses = []
        for k in range(n_realizations):
            result = compute_portfolio_loss(portfolio, sa_fields[k, :])
            losses.append(result.total_loss)
        scenario_losses[i] = np.mean(losses)

    # EP curve and AAL
    ep_curve = compute_ep_curve(scenario_losses, annual_rates)
    aal = compute_aal(scenario_losses, annual_rates)

    ps = portfolio_summary(portfolio)

    return ProbabilisticResult(
        n_events=n_events,
        n_realizations=n_realizations,
        portfolio_size=len(portfolio),
        scenario_losses=scenario_losses,
        annual_rates=annual_rates,
        ep_curve=ep_curve,
        aal=aal,
        portfolio_summary=ps,
    )


# ── Convenience functions ─────────────────────────────────────────────────

def run_northridge_deterministic(
    portfolio: list[BridgeExposure] | None = None,
    n_bridges: int = 100,
    n_realizations: int = 50,
    seed: int = 42,
) -> DeterministicResult:
    """
    Run deterministic analysis for the 1994 Northridge earthquake.

    Parameters
    ----------
    portfolio : list[BridgeExposure], optional
        If None, generates a synthetic portfolio.
    n_bridges : int
        Number of bridges if generating synthetic portfolio.
    n_realizations : int
    seed : int

    Returns
    -------
    DeterministicResult
    """
    if portfolio is None:
        portfolio = generate_synthetic_portfolio(n_bridges, seed=seed)

    return run_deterministic(
        NORTHRIDGE_SCENARIO, portfolio, n_realizations, seed=seed
    )


# ── Report printing ──────────────────────────────────────────────────────

def print_deterministic_report(result: DeterministicResult) -> str:
    """Format a deterministic analysis report."""
    sc = result.scenario
    ps = result.portfolio_summary

    lines = []
    lines.append("=" * 65)
    lines.append("DETERMINISTIC SCENARIO ANALYSIS")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"  Earthquake:  Mw {sc.Mw:.1f}")
    lines.append(f"  Epicenter:   ({sc.lat:.3f}, {sc.lon:.3f})")
    lines.append(f"  Depth:       {sc.depth_km:.1f} km")
    lines.append(f"  Fault type:  {sc.fault_type}")
    lines.append("")
    lines.append(f"  Portfolio:   {result.portfolio_size} bridges")
    lines.append(f"  Total TIV:   ${ps['total_replacement_cost']:,.0f}")
    lines.append(f"  Realizations: {result.n_realizations}")
    lines.append("")
    lines.append("-" * 65)
    lines.append("GROUND MOTION AT BRIDGE SITES")
    lines.append("-" * 65)
    lines.append(f"  Median Sa(1.0s) range: {result.median_sa.min():.4f}g "
                 f"- {result.median_sa.max():.4f}g")
    lines.append(f"  Median Sa(1.0s) mean:  {result.median_sa.mean():.4f}g")
    lines.append("")

    # Use the mean realization for the loss summary
    losses = np.array([r.total_loss for r in result.loss_results])
    lines.append("-" * 65)
    lines.append("LOSS RESULTS (across realizations)")
    lines.append("-" * 65)
    lines.append(f"  Mean expected loss:   ${result.mean_loss:,.0f}")
    lines.append(f"  Std deviation:        ${result.std_loss:,.0f}")
    lines.append(f"  CV:                   {result.std_loss / result.mean_loss:.2f}"
                 if result.mean_loss > 0 else "  CV:                   N/A")
    lines.append(f"  Mean loss ratio:      {result.mean_loss_ratio:.4f} "
                 f"({result.mean_loss_ratio * 100:.2f}%)")
    lines.append(f"  Min scenario loss:    ${losses.min():,.0f}")
    lines.append(f"  Max scenario loss:    ${losses.max():,.0f}")
    lines.append("")

    # Show representative loss detail from median realization
    median_idx = np.argsort(losses)[len(losses) // 2]
    rep = result.loss_results[median_idx]
    lines.append(loss_summary_table(rep))

    return "\n".join(lines)


def print_probabilistic_report(result: ProbabilisticResult) -> str:
    """Format a probabilistic analysis report."""
    ps = result.portfolio_summary
    ep = result.ep_curve

    lines = []
    lines.append("=" * 65)
    lines.append("PROBABILISTIC ANALYSIS RESULTS")
    lines.append("=" * 65)
    lines.append("")
    lines.append(f"  Portfolio:      {result.portfolio_size} bridges")
    lines.append(f"  Total TIV:      ${ps['total_replacement_cost']:,.0f}")
    lines.append(f"  Events:         {result.n_events}")
    lines.append(f"  Realizations:   {result.n_realizations} per event")
    lines.append("")
    lines.append("-" * 65)
    lines.append("AVERAGE ANNUAL LOSS")
    lines.append("-" * 65)
    aal_ratio = result.aal / ps["total_replacement_cost"] if ps["total_replacement_cost"] > 0 else 0
    lines.append(f"  AAL:            ${result.aal:,.0f}")
    lines.append(f"  AAL ratio:      {aal_ratio:.6f} ({aal_ratio * 100:.4f}%)")
    lines.append("")

    lines.append("-" * 65)
    lines.append("LOSS EXCEEDANCE PROBABILITY (selected return periods)")
    lines.append("-" * 65)
    lines.append(f"  {'Return Period':>15}  {'Loss ($)':>15}  {'EP':>8}")
    lines.append(f"  {'-'*15}  {'-'*15}  {'-'*8}")

    # Show key return periods
    rp = ep["return_period"]
    losses = ep["loss_thresholds"]
    ep_vals = ep["exceedance_prob"]

    target_rps = [10, 25, 50, 100, 250, 500]
    for target in target_rps:
        # Find closest return period
        idx = np.argmin(np.abs(rp - target))
        if rp[idx] < np.inf:
            lines.append(f"  {rp[idx]:>12.0f} yr  ${losses[idx]:>14,.0f}  {ep_vals[idx]:>7.4f}")

    lines.append("")
    lines.append("-" * 65)
    lines.append("SCENARIO LOSS STATISTICS")
    lines.append("-" * 65)
    sl = result.scenario_losses
    lines.append(f"  Mean scenario loss:    ${np.mean(sl):,.0f}")
    lines.append(f"  Median scenario loss:  ${np.median(sl):,.0f}")
    lines.append(f"  Max scenario loss:     ${np.max(sl):,.0f}")
    lines.append(f"  Min scenario loss:     ${np.min(sl):,.0f}")
    lines.append("")
    lines.append("=" * 65)

    return "\n".join(lines)

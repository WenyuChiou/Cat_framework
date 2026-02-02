"""
Damage-to-Loss Translation & Portfolio Aggregation.

Converts damage state probabilities into direct economic losses using
Hazus damage ratios (Table 7.11) and computes portfolio-level metrics
including loss exceedance probability (EP) curves and average annual loss
(AAL).

Reference: FEMA Hazus 6.1 Earthquake Model Technical Manual, Table 7.11
"""

from dataclasses import dataclass, field

import numpy as np

from .fragility import damage_state_probabilities
from .hazus_params import DAMAGE_STATE_ORDER


# ── Hazus damage ratios (Table 7.11) ─────────────────────────────────────

# Mean damage ratio (fraction of replacement cost) by damage state
HAZUS_DAMAGE_RATIOS = {
    "none":      0.00,
    "slight":    0.03,
    "moderate":  0.08,
    "extensive": 0.25,
    "complete":  1.00,
}

# Estimated restoration time (days) by damage state
HAZUS_DOWNTIME_DAYS = {
    "none":        0,
    "slight":      0.6,
    "moderate":    2.5,
    "extensive":  75,
    "complete":  230,
}


# ── Result dataclasses ────────────────────────────────────────────────────

@dataclass
class BridgeLossResult:
    """Loss result for a single bridge."""
    bridge_id: str
    sa: float
    damage_probs: dict[str, float]
    expected_loss: float        # USD
    expected_downtime: float    # days
    replacement_cost: float     # USD
    loss_ratio: float           # expected_loss / replacement_cost


@dataclass
class PortfolioLossResult:
    """Aggregated loss result for a portfolio."""
    total_loss: float
    total_replacement_cost: float
    loss_ratio: float
    bridge_results: list[BridgeLossResult]
    loss_by_class: dict[str, float]
    count_by_ds: dict[str, float]      # expected count per damage state
    total_downtime_bridge_days: float   # sum of expected downtime across bridges


# ── Single bridge loss ────────────────────────────────────────────────────

def compute_bridge_loss(
    sa: float,
    hwb_class: str,
    replacement_cost: float,
    bridge_id: str = "",
) -> BridgeLossResult:
    """
    Compute expected loss for a single bridge.

    E[Loss] = sum over ds: P(ds) * DR(ds) * RC

    Parameters
    ----------
    sa : float
        Sa(1.0s) in g at the bridge site.
    hwb_class : str
        Hazus bridge class.
    replacement_cost : float
        Bridge replacement cost (USD).
    bridge_id : str
        Identifier.

    Returns
    -------
    BridgeLossResult
    """
    probs = damage_state_probabilities(sa, hwb_class)
    all_ds = ["none"] + DAMAGE_STATE_ORDER

    expected_loss = 0.0
    expected_downtime = 0.0
    for ds in all_ds:
        expected_loss += probs[ds] * HAZUS_DAMAGE_RATIOS[ds] * replacement_cost
        expected_downtime += probs[ds] * HAZUS_DOWNTIME_DAYS[ds]

    loss_ratio = expected_loss / replacement_cost if replacement_cost > 0 else 0.0

    return BridgeLossResult(
        bridge_id=bridge_id,
        sa=sa,
        damage_probs=probs,
        expected_loss=expected_loss,
        expected_downtime=expected_downtime,
        replacement_cost=replacement_cost,
        loss_ratio=loss_ratio,
    )


# ── Portfolio loss ────────────────────────────────────────────────────────

def compute_portfolio_loss(
    portfolio,
    sa_values: np.ndarray,
) -> PortfolioLossResult:
    """
    Compute aggregate loss for a portfolio of bridges.

    Parameters
    ----------
    portfolio : list[BridgeExposure]
        Bridge inventory with hwb_class and replacement_cost.
    sa_values : np.ndarray
        Sa(1.0s) at each bridge site (length must match portfolio).

    Returns
    -------
    PortfolioLossResult
    """
    bridge_results = []
    loss_by_class = {}
    count_by_ds = {ds: 0.0 for ds in ["none"] + DAMAGE_STATE_ORDER}
    total_loss = 0.0
    total_rc = 0.0
    total_downtime = 0.0

    for i, bridge in enumerate(portfolio):
        sa = float(sa_values[i])
        result = compute_bridge_loss(
            sa=sa,
            hwb_class=bridge.hwb_class,
            replacement_cost=bridge.replacement_cost,
            bridge_id=bridge.bridge_id,
        )
        bridge_results.append(result)
        total_loss += result.expected_loss
        total_rc += bridge.replacement_cost
        total_downtime += result.expected_downtime

        # Accumulate by class
        cls = bridge.hwb_class
        loss_by_class[cls] = loss_by_class.get(cls, 0.0) + result.expected_loss

        # Expected damage state counts
        for ds, p in result.damage_probs.items():
            count_by_ds[ds] += p

    loss_ratio = total_loss / total_rc if total_rc > 0 else 0.0

    return PortfolioLossResult(
        total_loss=total_loss,
        total_replacement_cost=total_rc,
        loss_ratio=loss_ratio,
        bridge_results=bridge_results,
        loss_by_class=dict(sorted(loss_by_class.items())),
        count_by_ds=count_by_ds,
        total_downtime_bridge_days=total_downtime,
    )


# ── Exceedance Probability (EP) curve ─────────────────────────────────────

def compute_ep_curve(
    scenario_losses: np.ndarray,
    annual_rates: np.ndarray | None = None,
) -> dict:
    """
    Compute loss exceedance probability curve.

    For each loss threshold L, EP(L) = 1 - prod(1 - rate_i) for all
    scenarios where loss_i >= L.  If annual_rates is None, assumes
    equal probability (1/N) for each scenario.

    Parameters
    ----------
    scenario_losses : np.ndarray
        Total portfolio loss for each scenario, shape (n_events,).
    annual_rates : np.ndarray, optional
        Annual occurrence rate for each scenario.  If None, uses 1/N.

    Returns
    -------
    dict with keys:
        "loss_thresholds" : np.ndarray
        "exceedance_prob" : np.ndarray
        "return_period"   : np.ndarray (years, where EP > 0)
    """
    n = len(scenario_losses)
    if annual_rates is None:
        annual_rates = np.full(n, 1.0 / n)

    # Sort losses for EP computation
    sorted_idx = np.argsort(scenario_losses)[::-1]
    sorted_losses = scenario_losses[sorted_idx]
    sorted_rates = annual_rates[sorted_idx]

    # Cumulative rate of exceedance
    cum_rate = np.cumsum(sorted_rates)

    # EP = 1 - exp(-cum_rate) (Poisson assumption)
    ep = 1.0 - np.exp(-cum_rate)

    # Return period = 1 / cum_rate
    return_period = np.where(cum_rate > 0, 1.0 / cum_rate, np.inf)

    return {
        "loss_thresholds": sorted_losses,
        "exceedance_prob": ep,
        "return_period": return_period,
    }


# ── Average Annual Loss (AAL) ────────────────────────────────────────────

def compute_aal(
    scenario_losses: np.ndarray,
    annual_rates: np.ndarray,
) -> float:
    """
    Compute Average Annual Loss.

    AAL = sum(rate_i * loss_i)

    Parameters
    ----------
    scenario_losses : np.ndarray
        Loss for each scenario.
    annual_rates : np.ndarray
        Annual occurrence rate for each scenario.

    Returns
    -------
    float
        Average Annual Loss in USD.
    """
    return float(np.sum(scenario_losses * annual_rates))


# ── Reporting ─────────────────────────────────────────────────────────────

def loss_summary_table(result: PortfolioLossResult) -> str:
    """Format a text summary of portfolio loss results."""
    lines = []
    lines.append("=" * 65)
    lines.append("PORTFOLIO LOSS SUMMARY")
    lines.append("=" * 65)
    lines.append(f"  Bridges analyzed:        {len(result.bridge_results):,}")
    lines.append(f"  Total replacement cost:  ${result.total_replacement_cost:,.0f}")
    lines.append(f"  Expected total loss:     ${result.total_loss:,.0f}")
    lines.append(f"  Portfolio loss ratio:    {result.loss_ratio:.4f} "
                 f"({result.loss_ratio * 100:.2f}%)")
    lines.append(f"  Total downtime:          {result.total_downtime_bridge_days:,.0f} bridge-days")
    lines.append("")

    lines.append("-" * 65)
    lines.append("EXPECTED DAMAGE STATE DISTRIBUTION")
    lines.append("-" * 65)
    n_total = len(result.bridge_results)
    for ds in ["none"] + DAMAGE_STATE_ORDER:
        count = result.count_by_ds[ds]
        pct = count / n_total * 100 if n_total > 0 else 0
        lines.append(f"  {ds.capitalize():>12}: {count:6.1f} bridges ({pct:5.1f}%)")
    lines.append("")

    lines.append("-" * 65)
    lines.append("LOSS BY BRIDGE CLASS")
    lines.append("-" * 65)
    for cls, loss in sorted(result.loss_by_class.items(),
                            key=lambda x: x[1], reverse=True):
        pct = loss / result.total_loss * 100 if result.total_loss > 0 else 0
        lines.append(f"  {cls:<8}  ${loss:>14,.0f}  ({pct:5.1f}%)")
    lines.append("")
    lines.append("=" * 65)

    return "\n".join(lines)

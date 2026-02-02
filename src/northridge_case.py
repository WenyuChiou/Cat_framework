"""
1994 Northridge Earthquake case study data and analysis.

Ground motion and observed damage data from the Mw 6.7 Northridge
earthquake (January 17, 1994). Bridge damage statistics are based on
Basoz & Kiremidjian (1998), Caltrans post-earthquake reports, and
Werner et al. (2006).
"""

import numpy as np

from .fragility import damage_state_probabilities, compute_all_curves
from .hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER


# --- Ground motion observations ---

NORTHRIDGE_GROUND_MOTION = {
    "event": "1994 Northridge Earthquake",
    "magnitude": 6.7,
    "date": "1994-01-17",
    "epicenter": {"lat": 34.213, "lon": -118.537},
    "pga_range_g": (0.10, 1.82),
    "sa_1s_range_g": (0.05, 1.20),
    "typical_sa_1s_near_field_g": (0.40, 0.90),
    "description": (
        "The Northridge earthquake produced PGA values from ~0.1g in the "
        "San Fernando Valley periphery to 1.82g at the Tarzana strong-motion "
        "station. Sa(1.0s) values in the near-field zone where most bridge "
        "damage occurred ranged from approximately 0.4g to 0.9g."
    ),
}


# --- Observed bridge damage statistics ---

NORTHRIDGE_DAMAGE_STATS = {
    "total_bridges_in_area": 1600,
    "total_damaged": 170,
    "total_collapsed": 7,
    "damage_summary": {
        "none": 1430,
        "slight": 98,
        "moderate": 44,
        "extensive": 21,
        "complete": 7,
    },
    "observed_damage_fractions": {
        "none": 1430 / 1600,
        "slight": 98 / 1600,
        "moderate": 44 / 1600,
        "extensive": 21 / 1600,
        "complete": 7 / 1600,
    },
    "key_collapses": [
        "I-5 / SR-14 Interchange (multi-span concrete)",
        "I-5 / SR-118 connector",
        "SR-118 at Mission-Gothic Undercrossing",
        "I-10 La Cienega-Venice Undercrossing",
        "I-10 Fairfax-Washington Undercrossing",
        "SR-118 at Bull Creek Bridge",
        "I-5 Gavin Canyon Undercrossing",
    ],
    "most_vulnerable_types": [
        "HWB5",   # Multi-span concrete continuous, conventional
        "HWB7",   # Multi-span concrete simply-supported, conventional
        "HWB17",  # Multi-span concrete box girder, conventional
        "HWB22",  # Multi-span concrete frame, conventional
        "HWB1",   # Major bridges, conventional
    ],
    "references": [
        "Basoz, N. & Kiremidjian, A. (1998). Evaluation of Bridge Damage "
        "Data from the Loma Prieta and Northridge, CA Earthquakes. "
        "MCEER-98-0004.",
        "Werner, S.D., et al. (2006). Seismic Risk Analysis of Highway "
        "Systems. MCEER-06-0011.",
        "Caltrans (1994). The Northridge Earthquake: Post-Earthquake "
        "Investigation Report.",
    ],
}


def compute_northridge_scenario(
    sa_value: float = 0.60,
) -> dict[str, dict[str, float]]:
    """
    Compute Hazus-predicted damage probabilities for all Northridge-relevant
    bridge classes at a representative Sa(1.0s) level.

    Parameters
    ----------
    sa_value : float
        Representative Sa(1.0s) in g for the scenario (default 0.60g,
        a typical near-field value during Northridge).

    Returns
    -------
    dict
        {hwb_class: {damage_state: probability}} for each vulnerable class.
    """
    results = {}
    for hwb in NORTHRIDGE_DAMAGE_STATS["most_vulnerable_types"]:
        results[hwb] = damage_state_probabilities(sa_value, hwb)
    return results


def compare_predicted_vs_observed(sa_value: float = 0.60) -> dict:
    """
    Compare Hazus predictions with observed Northridge damage fractions.

    This is an aggregate comparison: the Hazus prediction is averaged over
    the most vulnerable bridge types (weighted equally), while the observed
    fractions are for the entire bridge inventory.

    Parameters
    ----------
    sa_value : float
        Representative Sa(1.0s) in g.

    Returns
    -------
    dict
        Contains "predicted_avg", "observed", and "sa_value" keys.
    """
    scenario = compute_northridge_scenario(sa_value)

    # Average predicted probabilities across vulnerable classes
    all_ds = ["none"] + DAMAGE_STATE_ORDER
    avg = {ds: 0.0 for ds in all_ds}
    n = len(scenario)
    for hwb_probs in scenario.values():
        for ds in all_ds:
            avg[ds] += hwb_probs[ds] / n

    return {
        "sa_value": sa_value,
        "predicted_avg": avg,
        "observed": NORTHRIDGE_DAMAGE_STATS["observed_damage_fractions"],
        "note": (
            "Predicted values are averaged over the most vulnerable bridge "
            "types at a single Sa level. Observed fractions represent the "
            "entire inventory exposed to spatially varying ground motion. "
            "Direct comparison is illustrative, not exact."
        ),
    }


def print_scenario_report(sa_value: float = 0.60) -> str:
    """
    Generate a formatted text report of the Northridge scenario analysis.

    Parameters
    ----------
    sa_value : float
        Representative Sa(1.0s) in g.

    Returns
    -------
    str
        Formatted report string.
    """
    comparison = compare_predicted_vs_observed(sa_value)
    scenario = compute_northridge_scenario(sa_value)

    lines = []
    lines.append("=" * 70)
    lines.append("NORTHRIDGE EARTHQUAKE BRIDGE FRAGILITY ANALYSIS")
    lines.append("Hazus 6.1 Method — Scenario Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Event:       {NORTHRIDGE_GROUND_MOTION['event']}")
    lines.append(f"Magnitude:   Mw {NORTHRIDGE_GROUND_MOTION['magnitude']}")
    lines.append(f"PGA range:   {NORTHRIDGE_GROUND_MOTION['pga_range_g'][0]:.2f}g "
                 f"- {NORTHRIDGE_GROUND_MOTION['pga_range_g'][1]:.2f}g")
    lines.append(f"Sa(1.0s):    {sa_value:.2f}g (scenario value)")
    lines.append("")

    lines.append("-" * 70)
    lines.append("PREDICTED DAMAGE PROBABILITIES BY BRIDGE CLASS")
    lines.append("-" * 70)
    header = f"{'Class':<8} {'None':>8} {'Slight':>8} {'Moderate':>8} {'Extensive':>10} {'Complete':>10}"
    lines.append(header)
    lines.append("-" * 70)

    for hwb, probs in scenario.items():
        row = (
            f"{hwb:<8} "
            f"{probs['none']:>8.3f} "
            f"{probs['slight']:>8.3f} "
            f"{probs['moderate']:>8.3f} "
            f"{probs['extensive']:>10.3f} "
            f"{probs['complete']:>10.3f}"
        )
        lines.append(row)

    lines.append("")
    lines.append("-" * 70)
    lines.append("COMPARISON: PREDICTED (AVG) vs OBSERVED")
    lines.append("-" * 70)
    all_ds = ["none"] + DAMAGE_STATE_ORDER
    header2 = f"{'State':<12} {'Predicted':>10} {'Observed':>10}"
    lines.append(header2)
    lines.append("-" * 70)

    for ds in all_ds:
        pred = comparison["predicted_avg"][ds]
        obs = comparison["observed"][ds]
        lines.append(f"{ds.capitalize():<12} {pred:>10.3f} {obs:>10.3f}")

    lines.append("")
    lines.append(f"Note: {comparison['note']}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("KEY BRIDGE COLLAPSES")
    lines.append("-" * 70)
    for collapse in NORTHRIDGE_DAMAGE_STATS["key_collapses"]:
        lines.append(f"  - {collapse}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("REFERENCES")
    lines.append("-" * 70)
    for ref in NORTHRIDGE_DAMAGE_STATS["references"]:
        lines.append(f"  {ref}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)

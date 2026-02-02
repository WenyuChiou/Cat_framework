"""
Core fragility curve computation using the lognormal CDF model.

The probability of reaching or exceeding a damage state is:
    P[DS >= ds | IM] = Phi[ (ln(IM) - ln(median)) / beta ]

where Phi is the standard normal CDF, median and beta are lognormal
parameters from Hazus Table 7.9, and IM is the spectral acceleration
Sa(1.0s) in g.
"""

import numpy as np
from scipy.stats import norm

from .hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER


def fragility_curve(
    im_values: np.ndarray, median: float, beta: float
) -> np.ndarray:
    """
    Compute exceedance probability using lognormal CDF.

    Parameters
    ----------
    im_values : np.ndarray
        Intensity measure values (Sa in g). Values <= 0 yield probability 0.
    median : float
        Median intensity for the damage state (g).
    beta : float
        Lognormal standard deviation (dispersion).

    Returns
    -------
    np.ndarray
        Exceedance probabilities in [0, 1].
    """
    im_values = np.asarray(im_values, dtype=float)
    prob = np.zeros_like(im_values)
    mask = im_values > 0
    prob[mask] = norm.cdf(
        (np.log(im_values[mask]) - np.log(median)) / beta
    )
    return prob


def compute_all_curves(
    hwb_class: str, im_values: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Compute fragility curves for all 4 damage states of a bridge class.

    Parameters
    ----------
    hwb_class : str
        Hazus bridge class (e.g. "HWB5").
    im_values : np.ndarray
        Array of spectral acceleration values (g).

    Returns
    -------
    dict
        Mapping from damage state name to exceedance probability array.
    """
    params = HAZUS_BRIDGE_FRAGILITY[hwb_class]["damage_states"]
    curves = {}
    for ds in DAMAGE_STATE_ORDER:
        p = params[ds]
        curves[ds] = fragility_curve(im_values, p["median"], p["beta"])
    return curves


def damage_state_probabilities(
    im: float, hwb_class: str
) -> dict[str, float]:
    """
    Compute discrete damage state probabilities at a single IM level.

    Returns P[DS = none], P[DS = slight], ..., P[DS = complete] such that
    they sum to 1.0.

    Parameters
    ----------
    im : float
        Spectral acceleration Sa(1.0s) in g.
    hwb_class : str
        Hazus bridge class.

    Returns
    -------
    dict
        Mapping from damage state name (including "none") to probability.
    """
    im_arr = np.array([im])
    curves = compute_all_curves(hwb_class, im_arr)
    exceedance = {ds: float(curves[ds][0]) for ds in DAMAGE_STATE_ORDER}

    probs = {}
    probs["none"] = 1.0 - exceedance["slight"]
    for i, ds in enumerate(DAMAGE_STATE_ORDER):
        if i < len(DAMAGE_STATE_ORDER) - 1:
            next_ds = DAMAGE_STATE_ORDER[i + 1]
            probs[ds] = exceedance[ds] - exceedance[next_ds]
        else:
            probs[ds] = exceedance[ds]

    return probs


def apply_skew_modification(
    median: float, skew_angle_deg: float
) -> float:
    """
    Apply Hazus skew angle modification factor to fragility median.

    The modification factor reduces the median capacity for skewed bridges:
        median_modified = median * sqrt(1 - (skew / 90)^2)

    Parameters
    ----------
    median : float
        Original median Sa value (g).
    skew_angle_deg : float
        Skew angle in degrees (0 = no skew, max 90).

    Returns
    -------
    float
        Modified median value.
    """
    skew_angle_deg = min(max(skew_angle_deg, 0.0), 90.0)
    factor = np.sqrt(1.0 - (skew_angle_deg / 90.0) ** 2)
    return median * factor

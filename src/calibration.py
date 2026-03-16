"""
Fragility curve calibration using Maximum Likelihood Estimation (MLE).

Calibrates a global median scale factor `k` and dispersion `beta` to
minimize the discrepancy between Hazus-predicted and observed damage
distributions.  The observed reference is Basoz & Kiremidjian (1998)
aggregate damage counts from the 1994 Northridge earthquake (N=1600).

Method
------
All Hazus medians are multiplied by k (preserving relative ordering),
and beta is shared across damage states.  The multinomial log-likelihood
is maximised over the bridge inventory:

    LL(k, beta) = sum_j  n_j * ln(p_j_pred(k, beta))

where p_j_pred is the predicted fraction of bridges in damage state j,
averaged over individual bridges weighted by their site-specific Sa and
HWB class.

Reference: Basoz, N. & Kiremidjian, A. (1998). MCEER-98-0004.
Data preparation: Sirisha Kedarsetty (damage state mapping, HAZUS parameter tables).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER
from .northridge_case import NORTHRIDGE_DAMAGE_STATS

# Damage states including "none"
ALL_DS = ["none"] + DAMAGE_STATE_ORDER


# ── Dataclass for results ─────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """Container for MLE calibration output."""
    k: float                          # optimal median scale factor
    beta: float                       # optimal common dispersion
    neg_log_likelihood: float         # NLL at optimum
    aic: float                        # Akaike Information Criterion
    k_ci: tuple[float, float] = (0.0, 0.0)   # 95% CI for k
    beta_ci: tuple[float, float] = (0.0, 0.0) # 95% CI for beta
    pred_counts: dict[str, float] = field(default_factory=dict)
    obs_counts: dict[str, int] = field(default_factory=dict)
    pred_fractions: dict[str, float] = field(default_factory=dict)
    obs_fractions: dict[str, float] = field(default_factory=dict)
    n_bridges: int = 0
    converged: bool = False


# ── Core likelihood function ──────────────────────────────────────────

def _exceedance_prob(sa: float, median: float, beta: float) -> float:
    """Lognormal CDF: P[DS >= ds | Sa]."""
    if sa <= 0 or median <= 0 or beta <= 0:
        return 0.0
    return float(norm.cdf((np.log(sa) - np.log(median)) / beta))


def _bridge_ds_probs(
    sa: float, hwb_class: str, k: float, beta: float
) -> np.ndarray:
    """Compute discrete DS probabilities for one bridge.

    Returns array of length 5: [none, slight, moderate, extensive, complete].
    """
    params = HAZUS_BRIDGE_FRAGILITY.get(hwb_class)
    if params is None:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    ds_params = params["damage_states"]
    exceed = []
    for ds in DAMAGE_STATE_ORDER:
        med = ds_params[ds]["median"] * k
        exceed.append(_exceedance_prob(sa, med, beta))

    # Discrete probabilities from exceedance differences
    probs = np.zeros(5)
    probs[0] = 1.0 - exceed[0]                    # none
    probs[1] = exceed[0] - exceed[1]               # slight
    probs[2] = exceed[1] - exceed[2]               # moderate
    probs[3] = exceed[2] - exceed[3]               # extensive
    probs[4] = exceed[3]                            # complete
    # Clip to avoid log(0) in NLL. Safe because k scales all medians equally,
    # preserving ordering, so differences can be zero but not negative.
    return np.clip(probs, 1e-15, None)


def predicted_distribution(
    sa_arr: np.ndarray, hwb_arr: np.ndarray, k: float, beta: float
) -> np.ndarray:
    """Average predicted DS distribution over all bridges.

    Returns array of length 5 (fractions summing to ~1).
    """
    n = len(sa_arr)
    if n == 0:
        return np.zeros(5)
    total = np.zeros(5)
    for i in range(n):
        total += _bridge_ds_probs(sa_arr[i], hwb_arr[i], k, beta)
    return total / n


def neg_log_likelihood(
    params: np.ndarray,
    sa_arr: np.ndarray,
    hwb_arr: np.ndarray,
    obs_counts: np.ndarray,
) -> float:
    """Multinomial negative log-likelihood.

    Parameters
    ----------
    params : array [k, beta]
    sa_arr : Sa(1.0s) for each bridge
    hwb_arr : HWB class string for each bridge
    obs_counts : observed counts [none, slight, moderate, extensive, complete]
    """
    k, beta = params
    if k <= 0 or beta <= 0:
        return 1e12

    p_pred = predicted_distribution(sa_arr, hwb_arr, k, beta)
    p_pred = np.clip(p_pred, 1e-15, None)
    p_pred = p_pred / p_pred.sum()  # renormalize

    # Multinomial log-likelihood: sum(n_j * ln(p_j))
    ll = np.sum(obs_counts * np.log(p_pred))
    return -ll


# ── Main calibration function ────────────────────────────────────────

def calibrate_global(
    bridges_df: pd.DataFrame,
    obs_counts: Optional[dict[str, int]] = None,
    sa_column: str = "sa1s_shakemap",
    hwb_column: str = "hwb_class",
    x0: tuple[float, float] = (1.5, 0.6),
    bounds: tuple[tuple[float, float], ...] = ((0.5, 5.0), (0.2, 1.5)),
) -> CalibrationResult:
    """Run global MLE calibration for (k, beta).

    Parameters
    ----------
    bridges_df : DataFrame
        Must contain columns for Sa and HWB class.
    obs_counts : dict, optional
        Observed damage counts {none: N, slight: N, ...}.
        Defaults to Basoz (1998) Northridge counts.
    sa_column : str
        Column name for Sa(1.0s) values.
    hwb_column : str
        Column name for HWB class.
    x0 : tuple
        Initial guess [k, beta].
    bounds : tuple
        Bounds for [k, beta].

    Returns
    -------
    CalibrationResult
    """
    if len(bridges_df) == 0:
        raise ValueError("bridges_df must contain at least one bridge record.")
    for col in (sa_column, hwb_column):
        if col not in bridges_df.columns:
            raise ValueError(
                f"Column '{col}' not found in bridges_df. "
                f"Available: {list(bridges_df.columns)}"
            )

    if obs_counts is None:
        obs_counts = NORTHRIDGE_DAMAGE_STATS["damage_summary"]

    obs_arr = np.array([obs_counts[ds] for ds in ALL_DS], dtype=float)
    n_total = obs_arr.sum()
    if n_total == 0:
        raise ValueError("obs_counts must have a positive total count.")

    sa_arr = bridges_df[sa_column].values.astype(float)
    hwb_arr = bridges_df[hwb_column].values

    # Optimize
    result = minimize(
        neg_log_likelihood,
        x0=np.array(x0),
        args=(sa_arr, hwb_arr, obs_arr),
        method="L-BFGS-B",
        bounds=bounds,
    )

    k_opt, beta_opt = result.x
    nll = result.fun
    aic = 2 * 2 + 2 * nll  # 2 params; uses kernel NLL (multinomial coeff omitted)

    # Predicted distribution at optimum
    p_pred = predicted_distribution(sa_arr, hwb_arr, k_opt, beta_opt)
    p_pred = p_pred / p_pred.sum()
    pred_counts_val = {ds: float(p_pred[i] * n_total) for i, ds in enumerate(ALL_DS)}
    pred_fracs = {ds: float(p_pred[i]) for i, ds in enumerate(ALL_DS)}
    obs_fracs = {ds: obs_counts[ds] / n_total for ds in ALL_DS}

    # Profile likelihood CIs
    k_ci = profile_likelihood_ci(
        sa_arr, hwb_arr, obs_arr, k_opt, beta_opt,
        param_index=0, bounds=bounds,
    )
    beta_ci = profile_likelihood_ci(
        sa_arr, hwb_arr, obs_arr, k_opt, beta_opt,
        param_index=1, bounds=bounds,
    )

    return CalibrationResult(
        k=k_opt,
        beta=beta_opt,
        neg_log_likelihood=nll,
        aic=aic,
        k_ci=k_ci,
        beta_ci=beta_ci,
        pred_counts=pred_counts_val,
        obs_counts=dict(obs_counts),
        pred_fractions=pred_fracs,
        obs_fractions=obs_fracs,
        n_bridges=len(sa_arr),
        converged=result.success,
    )


# ── Profile likelihood confidence interval ────────────────────────────

def profile_likelihood_ci(
    sa_arr: np.ndarray,
    hwb_arr: np.ndarray,
    obs_arr: np.ndarray,
    k_opt: float,
    beta_opt: float,
    param_index: int = 0,
    chi2_crit: float = 3.841,  # chi2(1, 0.95)
    bounds: tuple[tuple[float, float], ...] = ((0.5, 5.0), (0.2, 1.5)),
) -> tuple[float, float]:
    """Compute 95% profile likelihood CI for one parameter.

    Uses coarse scan then bisection refinement for precision.
    """
    nll_opt = neg_log_likelihood(
        np.array([k_opt, beta_opt]), sa_arr, hwb_arr, obs_arr
    )

    opt_vals = [k_opt, beta_opt]
    bounds_list = list(bounds)
    other_idx = 1 - param_index

    def _profile_nll(fixed_val: float) -> float:
        """NLL with one param fixed, other optimized."""
        def _obj(x, fv=fixed_val):
            p = [0.0, 0.0]
            p[param_index] = fv
            p[other_idx] = x[0]
            return neg_log_likelihood(np.array(p), sa_arr, hwb_arr, obs_arr)

        res = minimize(
            _obj,
            x0=[opt_vals[other_idx]],
            bounds=[bounds_list[other_idx]],
            method="L-BFGS-B",
        )
        return res.fun

    threshold = nll_opt + chi2_crit / 2.0

    def _bisect(inside: float, outside: float, tol: float = 0.005) -> float:
        """Bisect between a point inside CI and one outside to find boundary."""
        for _ in range(20):
            if abs(outside - inside) < tol:
                break
            mid = (inside + outside) / 2.0
            if _profile_nll(mid) > threshold:
                outside = mid
            else:
                inside = mid
        return inside  # last point still inside CI

    param_lo, param_hi = bounds_list[param_index]

    # Search lower bound: coarse scan then bisect
    lo = opt_vals[param_index]
    lo_prev = lo
    step = 0.05
    while lo > param_lo + step:
        lo -= step
        if _profile_nll(lo) > threshold:
            lo = _bisect(lo_prev, lo)
            break
        lo_prev = lo
    else:
        lo = param_lo

    # Search upper bound: coarse scan then bisect
    hi = opt_vals[param_index]
    hi_prev = hi
    while hi < param_hi - step:
        hi += step
        if _profile_nll(hi) > threshold:
            hi = _bisect(hi_prev, hi)
            break
        hi_prev = hi
    else:
        hi = param_hi

    return (round(lo, 4), round(hi, 4))


# ── Plotting functions ────────────────────────────────────────────────

def plot_before_after(
    result: CalibrationResult,
    save_path: Optional[str | Path] = None,
) -> None:
    """Bar chart comparing observed vs HAZUS-original vs calibrated counts."""
    import matplotlib.pyplot as plt

    obs = result.obs_counts
    pred = result.pred_counts
    ds_labels = ALL_DS
    x = np.arange(len(ds_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_obs = ax.bar(x - width / 2, [obs[ds] for ds in ds_labels], width,
                      label="Observed (Basoz 1998)", color="#2196F3", edgecolor="k")
    bars_pred = ax.bar(x + width / 2, [pred[ds] for ds in ds_labels], width,
                       label=f"Calibrated (k={result.k:.2f}, β={result.beta:.2f})",
                       color="#FF9800", edgecolor="k")

    # Add count labels
    for bar in bars_obs:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h:.0f}",
                ha="center", va="bottom", fontsize=9)
    for bar in bars_pred:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h:.0f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Damage State")
    ax.set_ylabel("Number of Bridges")
    ax.set_title("Observed vs Calibrated Damage Distribution (N=1600)")
    ax.set_xticks(x)
    ax.set_xticklabels([ds.capitalize() for ds in ds_labels])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_likelihood_profile(
    sa_arr: np.ndarray,
    hwb_arr: np.ndarray,
    obs_arr: np.ndarray,
    k_opt: float,
    beta_opt: float,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot NLL as a function of k (with beta re-optimized)."""
    import matplotlib.pyplot as plt

    nll_opt = neg_log_likelihood(
        np.array([k_opt, beta_opt]), sa_arr, hwb_arr, obs_arr
    )

    k_range = np.linspace(max(0.5, k_opt - 1.0), min(5.0, k_opt + 1.0), 40)
    nll_values = []

    for k_val in k_range:
        # Re-optimize beta for each k
        def _obj(x, k=k_val):
            return neg_log_likelihood(
                np.array([k, x[0]]), sa_arr, hwb_arr, obs_arr
            )
        res = minimize(_obj, x0=[beta_opt], bounds=[(0.2, 1.5)], method="L-BFGS-B")
        nll_values.append(res.fun)

    nll_values = np.array(nll_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, nll_values, "b-", lw=2)
    ax.axvline(k_opt, color="red", ls="--", lw=1.5, label=f"k* = {k_opt:.3f}")
    ax.axhline(nll_opt + 3.841 / 2, color="gray", ls=":", lw=1,
               label="95% CI threshold")
    ax.set_xlabel("Median Scale Factor k")
    ax.set_ylabel("Profile Negative Log-Likelihood")
    ax.set_title("Likelihood Profile for k (beta profiled out)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_fragility_comparison(
    k: float,
    beta: float,
    hwb_class: str = "HWB5",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot original vs calibrated fragility curves for one HWB class."""
    import matplotlib.pyplot as plt

    params = HAZUS_BRIDGE_FRAGILITY[hwb_class]["damage_states"]
    im_range = np.linspace(0.01, 2.5, 200)
    colors = {"slight": "#2196F3", "moderate": "#FF9800",
              "extensive": "#F44336", "complete": "#9C27B0"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ds in DAMAGE_STATE_ORDER:
        med_orig = params[ds]["median"]
        beta_orig = params[ds]["beta"]
        med_cal = med_orig * k

        # Original
        p_orig = norm.cdf((np.log(im_range) - np.log(med_orig)) / beta_orig)
        ax1.plot(im_range, p_orig, color=colors[ds], lw=2, label=ds.capitalize())

        # Calibrated
        p_cal = norm.cdf((np.log(im_range) - np.log(med_cal)) / beta)
        ax2.plot(im_range, p_cal, color=colors[ds], lw=2, label=ds.capitalize())

    beta_orig = params[DAMAGE_STATE_ORDER[0]]["beta"]
    ax1.set_title(f"{hwb_class} — Original (β={beta_orig})")
    ax2.set_title(f"{hwb_class} — Calibrated (k={k:.2f}, β={beta:.2f})")
    for ax in (ax1, ax2):
        ax.set_xlabel("Sa(1.0s) [g]")
        ax.set_ylabel("P[DS ≥ ds | IM]")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 1.05)

    plt.suptitle(f"Fragility Curve Comparison: {hwb_class}", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ── Config snippet generator ─────────────────────────────────────────

def generate_config_snippet(result: CalibrationResult) -> str:
    """Generate a YAML snippet for config.yaml."""
    lines = [
        "# --- Calibrated fragility parameters ---",
        "# MLE calibration against Basoz (1998) Northridge observations",
        f"# k = {result.k:.4f} (95% CI: {result.k_ci[0]:.4f} - {result.k_ci[1]:.4f})",
        f"# beta = {result.beta:.4f} (95% CI: {result.beta_ci[0]:.4f} - {result.beta_ci[1]:.4f})",
        f"# NLL = {result.neg_log_likelihood:.2f}, AIC = {result.aic:.2f}",
        "calibration:",
        f"  global_median_factor: {result.k:.4f}",
        "  # class_factors: {}  # override per-class if needed",
    ]
    return "\n".join(lines)


def save_results(
    result: CalibrationResult,
    output_dir: str | Path = "output/calibration",
) -> None:
    """Save calibration results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "k": result.k,
        "beta": result.beta,
        "k_ci_95": list(result.k_ci),
        "beta_ci_95": list(result.beta_ci),
        "neg_log_likelihood": result.neg_log_likelihood,
        "aic": result.aic,
        "n_bridges": result.n_bridges,
        "converged": result.converged,
        "predicted_counts": result.pred_counts,
        "observed_counts": result.obs_counts,
        "predicted_fractions": result.pred_fractions,
        "observed_fractions": result.obs_fractions,
        "config_snippet": generate_config_snippet(result),
    }

    path = output_dir / "calibration_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")

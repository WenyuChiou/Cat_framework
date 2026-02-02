"""
Fragility curve and CAT model visualization functions.

Generates publication-quality plots for bridge fragility analysis,
ground motion fields, loss distributions, and EP curves.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .fragility import compute_all_curves, damage_state_probabilities
from .hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER

# Consistent styling
DS_COLORS = {
    "slight": "#2196F3",
    "moderate": "#FF9800",
    "extensive": "#F44336",
    "complete": "#9C27B0",
}
DS_LINESTYLES = {
    "slight": "-",
    "moderate": "--",
    "extensive": "-.",
    "complete": ":",
}


def _setup_axes(ax: plt.Axes, title: str) -> None:
    """Apply consistent formatting to axes."""
    ax.set_xlabel("Sa(1.0s) [g]", fontsize=12)
    ax.set_ylabel("P(Exceedance)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")


def plot_single_class(
    hwb_class: str,
    im_values: np.ndarray,
    output_dir: str = "output",
) -> str:
    """
    Plot fragility curves for all 4 damage states of one bridge class.

    Parameters
    ----------
    hwb_class : str
        Hazus bridge class identifier.
    im_values : np.ndarray
        Intensity measure values.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to the saved figure.
    """
    curves = compute_all_curves(hwb_class, im_values)
    bridge_name = HAZUS_BRIDGE_FRAGILITY[hwb_class]["name"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for ds in DAMAGE_STATE_ORDER:
        ax.plot(
            im_values,
            curves[ds],
            color=DS_COLORS[ds],
            linestyle=DS_LINESTYLES[ds],
            linewidth=2,
            label=ds.capitalize(),
        )

    _setup_axes(ax, f"Fragility Curves — {hwb_class}: {bridge_name}")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"fragility_{hwb_class}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_comparison(
    hwb_classes: list[str],
    damage_state: str,
    im_values: np.ndarray,
    output_dir: str = "output",
) -> str:
    """
    Compare one damage state across multiple bridge classes.

    Parameters
    ----------
    hwb_classes : list[str]
        Bridge classes to compare.
    damage_state : str
        Damage state to plot ("slight", "moderate", "extensive", "complete").
    im_values : np.ndarray
        Intensity measure values.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(hwb_classes)))

    for i, hwb in enumerate(hwb_classes):
        curves = compute_all_curves(hwb, im_values)
        ax.plot(
            im_values,
            curves[damage_state],
            color=cmap[i],
            linewidth=1.8,
            label=hwb,
        )

    _setup_axes(
        ax,
        f"Bridge Class Comparison — {damage_state.capitalize()} Damage",
    )
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"comparison_{damage_state}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_damage_distribution(
    hwb_class: str,
    im_values_sample: list[float],
    output_dir: str = "output",
) -> str:
    """
    Plot discrete damage state probability distribution as stacked bars.

    Parameters
    ----------
    hwb_class : str
        Hazus bridge class.
    im_values_sample : list[float]
        Sa values at which to evaluate probabilities.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to the saved figure.
    """
    all_ds = ["none"] + DAMAGE_STATE_ORDER
    ds_colors = {
        "none": "#4CAF50",
        "slight": "#2196F3",
        "moderate": "#FF9800",
        "extensive": "#F44336",
        "complete": "#9C27B0",
    }

    probs_by_ds = {ds: [] for ds in all_ds}
    for im in im_values_sample:
        p = damage_state_probabilities(im, hwb_class)
        for ds in all_ds:
            probs_by_ds[ds].append(p[ds])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(im_values_sample))
    width = 0.7
    bottom = np.zeros(len(im_values_sample))

    for ds in all_ds:
        vals = np.array(probs_by_ds[ds])
        ax.bar(
            x,
            vals,
            width,
            bottom=bottom,
            label=ds.capitalize(),
            color=ds_colors[ds],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += vals

    bridge_name = HAZUS_BRIDGE_FRAGILITY[hwb_class]["name"]
    ax.set_xlabel("Sa(1.0s) [g]", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(
        f"Damage State Distribution — {hwb_class}: {bridge_name}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in im_values_sample], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"damage_distribution_{hwb_class}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_northridge_scenario(
    hwb_class: str,
    im_values: np.ndarray,
    pga_range: tuple[float, float],
    output_dir: str = "output",
    observed_label: str = "Northridge PGA Range",
) -> str:
    """
    Fragility curves with Northridge observed PGA range overlay.

    Parameters
    ----------
    hwb_class : str
        Hazus bridge class.
    im_values : np.ndarray
        Intensity measure values.
    pga_range : tuple[float, float]
        (low, high) PGA range observed in Northridge (g).
    output_dir : str
        Directory to save the plot.
    observed_label : str
        Label for the shaded PGA region.

    Returns
    -------
    str
        Path to the saved figure.
    """
    curves = compute_all_curves(hwb_class, im_values)
    bridge_name = HAZUS_BRIDGE_FRAGILITY[hwb_class]["name"]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Shaded PGA region
    ax.axvspan(
        pga_range[0],
        pga_range[1],
        alpha=0.15,
        color="red",
        label=observed_label,
    )
    ax.axvline(pga_range[0], color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(pga_range[1], color="red", linewidth=0.8, linestyle="--", alpha=0.5)

    # Fragility curves
    for ds in DAMAGE_STATE_ORDER:
        ax.plot(
            im_values,
            curves[ds],
            color=DS_COLORS[ds],
            linestyle=DS_LINESTYLES[ds],
            linewidth=2,
            label=ds.capitalize(),
        )

    _setup_axes(
        ax,
        f"Northridge Scenario — {hwb_class}: {bridge_name}",
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "northridge_scenario.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── CAT Model Pipeline Plots ─────────────────────────────────────────────


def plot_ground_motion_field(
    sites,
    sa_values: np.ndarray,
    scenario=None,
    output_dir: str = "output",
    filename: str = "ground_motion_field.png",
) -> str:
    """
    Scatter map of Sa(1.0s) at bridge sites, colored by intensity.

    Parameters
    ----------
    sites : list of objects with .lat, .lon attributes
    sa_values : np.ndarray
        Sa(1.0s) in g at each site.
    scenario : EarthquakeScenario, optional
        If provided, marks the epicenter.
    output_dir : str
    filename : str

    Returns
    -------
    str
        Path to saved figure.
    """
    lats = [s.lat for s in sites]
    lons = [s.lon for s in sites]

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        lons, lats, c=sa_values, cmap="YlOrRd",
        s=40, edgecolors="k", linewidths=0.3,
        vmin=0, vmax=max(0.5, np.percentile(sa_values, 95)),
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Sa(1.0s) [g]", fontsize=11)

    if scenario is not None:
        ax.plot(scenario.lon, scenario.lat, "r*", markersize=18,
                markeredgecolor="k", markeredgewidth=0.8, label="Epicenter")
        ax.legend(fontsize=10)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    title = "Ground Motion Field — Sa(1.0s)"
    if scenario is not None:
        title += f" | Mw {scenario.Mw:.1f}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loss_by_class(
    loss_by_class: dict[str, float],
    output_dir: str = "output",
    filename: str = "loss_by_class.png",
) -> str:
    """
    Bar chart of expected loss by HWB bridge class.

    Parameters
    ----------
    loss_by_class : dict
        {hwb_class: loss_usd}
    output_dir : str
    filename : str

    Returns
    -------
    str
        Path to saved figure.
    """
    classes = sorted(loss_by_class.keys())
    losses = [loss_by_class[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    bars = ax.bar(classes, losses, color=colors, edgecolor="k", linewidth=0.5)

    ax.set_xlabel("Bridge Class", fontsize=12)
    ax.set_ylabel("Expected Loss (USD)", fontsize=12)
    ax.set_title("Expected Loss by Bridge Class", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"
    ))
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_ep_curve(
    ep_data: dict,
    output_dir: str = "output",
    filename: str = "ep_curve.png",
) -> str:
    """
    Plot loss Exceedance Probability curve with return period axis.

    Parameters
    ----------
    ep_data : dict
        Output from loss.compute_ep_curve() with keys
        "loss_thresholds", "exceedance_prob", "return_period".
    output_dir : str
    filename : str

    Returns
    -------
    str
        Path to saved figure.
    """
    losses = ep_data["loss_thresholds"]
    ep = ep_data["exceedance_prob"]
    rp = ep_data["return_period"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: EP vs Loss
    ax1.semilogy(losses, ep, "b-", linewidth=2)
    ax1.set_xlabel("Loss (USD)", fontsize=12)
    ax1.set_ylabel("Annual Exceedance Probability", fontsize=12)
    ax1.set_title("Loss Exceedance Probability Curve", fontsize=13,
                  fontweight="bold")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x:,.0f}"
    ))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=1e-4)

    # Right: Loss vs Return Period
    valid = rp < np.inf
    ax2.semilogx(rp[valid], losses[valid], "r-", linewidth=2)
    ax2.set_xlabel("Return Period (years)", fontsize=12)
    ax2.set_ylabel("Loss (USD)", fontsize=12)
    ax2.set_title("Loss vs Return Period", fontsize=13, fontweight="bold")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x:,.0f}"
    ))
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_portfolio_damage(
    count_by_ds: dict[str, float],
    n_bridges: int,
    output_dir: str = "output",
    filename: str = "portfolio_damage.png",
) -> str:
    """
    Stacked horizontal bar showing portfolio-level damage state distribution.

    Parameters
    ----------
    count_by_ds : dict
        {damage_state: expected_count}
    n_bridges : int
        Total number of bridges.
    output_dir : str
    filename : str

    Returns
    -------
    str
        Path to saved figure.
    """
    ds_colors_all = {
        "none": "#4CAF50",
        "slight": "#2196F3",
        "moderate": "#FF9800",
        "extensive": "#F44336",
        "complete": "#9C27B0",
    }
    all_ds = ["none", "slight", "moderate", "extensive", "complete"]

    fig, ax = plt.subplots(figsize=(10, 3))
    left = 0.0
    for ds in all_ds:
        count = count_by_ds.get(ds, 0)
        pct = count / n_bridges * 100 if n_bridges > 0 else 0
        bar = ax.barh(0, pct, left=left, color=ds_colors_all[ds],
                      edgecolor="white", linewidth=0.5, height=0.5,
                      label=f"{ds.capitalize()} ({pct:.1f}%)")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of Bridges (%)", fontsize=12)
    ax.set_title("Portfolio Damage State Distribution", fontsize=13,
                 fontweight="bold")
    ax.set_yticks([])
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=5)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

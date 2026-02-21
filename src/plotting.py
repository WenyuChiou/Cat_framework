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
import pandas as pd

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


IM_LABEL_MAP = {
    "PGA": "PGA [g]",
    "SA03": "Sa(0.3s) [g]",
    "SA10": "Sa(1.0s) [g]",
    "SA30": "Sa(3.0s) [g]",
}


def _setup_axes(ax: plt.Axes, title: str, im_type: str = "SA10") -> None:
    """Apply consistent formatting to axes."""
    ax.set_xlabel(IM_LABEL_MAP.get(im_type, "Sa(1.0s) [g]"), fontsize=12)
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
    ax.set_xlabel(IM_LABEL_MAP.get("SA10", "Sa(1.0s) [g]"), fontsize=12)
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
    im_type: str = "SA10",
) -> str:
    """
    Scatter map of IM at bridge sites, colored by intensity.

    Parameters
    ----------
    sites : list of objects with .lat, .lon attributes
    sa_values : np.ndarray
        IM values in g at each site.
    scenario : EarthquakeScenario, optional
        If provided, marks the epicenter.
    output_dir : str
    filename : str
    im_type : str
        IM type label for axis/title (e.g. "SA10", "PGA").

    Returns
    -------
    str
        Path to saved figure.
    """
    im_label = IM_LABEL_MAP.get(im_type, "Sa(1.0s) [g]")
    lats = [s.lat for s in sites]
    lons = [s.lon for s in sites]

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        lons, lats, c=sa_values, cmap="YlOrRd",
        s=40, edgecolors="k", linewidths=0.3,
        vmin=0, vmax=max(0.5, np.percentile(sa_values, 95)),
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(im_label, fontsize=11)

    if scenario is not None:
        ax.plot(scenario.lon, scenario.lat, "r*", markersize=18,
                markeredgecolor="k", markeredgewidth=0.8, label="Epicenter")
        ax.legend(fontsize=10)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    im_short = im_label.split(" [")[0]
    title = f"Ground Motion Field — {im_short}"
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


def plot_shakemap_grid(
    shakemap_df: pd.DataFrame,
    intensity_measure: str = "PSA10",
    output_dir: str = "output",
    filename: str = "shakemap_map.png",
) -> str:
    """
    Plot the full ShakeMap intensity grid to show the downloaded area.

    Parameters
    ----------
    shakemap_df : pd.DataFrame
        DataFrame from data_loader.load_shakemap().
    intensity_measure : str
        Column name to plot (e.g. "PSA10", "PGA").
    output_dir : str
    filename : str

    Returns
    -------
    str
        Path to saved figure.
    """
    if intensity_measure not in shakemap_df.columns:
        # Fallback logic
        candidates = ["PSA10", "PGA", "PSA03"]
        for c in candidates:
            if c in shakemap_df.columns:
                intensity_measure = c
                break
        else:
            # Last resort: first numeric column after LAT/LON
            intensity_measure = shakemap_df.select_dtypes(include=[np.number]).columns[2]

    lats = shakemap_df["LAT"].values
    lons = shakemap_df["LON"].values
    vals = shakemap_df[intensity_measure].values

    # Sample for performance if grid is huge (matplotlib scatter is slow)
    if len(shakemap_df) > 50000:
        idx = np.random.choice(len(shakemap_df), 50000, replace=False)
        lats, lons, vals = lats[idx], lons[idx], vals[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        lons, lats, c=vals, cmap="YlOrRd",
        s=2, alpha=0.6, edgecolors="none",
        vmin=0, vmax=max(0.5, np.percentile(vals, 98)),
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(f"{intensity_measure} [g]", fontsize=11)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(f"USGS ShakeMap Grid — {intensity_measure}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_bridge_damage_map(
    nbi_df: pd.DataFrame,
    damage_state: str = "complete",
    output_dir: str = "output",
    filename: str = "bridge_damage_map.png",
) -> str:
    """
    Plot a map of bridges colored by their probability of being in a certain damage state.
    """
    col = f"P_{damage_state}"
    if col not in nbi_df.columns:
        col = "sa_10"  # fallback to ground motion

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        nbi_df["longitude"], nbi_df["latitude"], 
        c=nbi_df[col], cmap="Reds", s=15, edgecolors="k", linewidths=0.2,
        vmin=0, vmax=max(0.1, nbi_df[col].max())
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(f"Probability of {damage_state.capitalize()} Damage", fontsize=11)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Bridge Damage Spatial Distribution — {damage_state.capitalize()}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_nbi_bridge_distribution_map(
    nbi_df: pd.DataFrame,
    output_dir: str = "output",
    filename: str = "nbi_bridge_distribution_map.png",
) -> str:
    """
    Plot NBI bridge locations as a spatial distribution map.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        nbi_df["longitude"],
        nbi_df["latitude"],
        s=8,
        c="#1f77b4",
        alpha=0.55,
        edgecolors="none",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("NBI Bridge Spatial Distribution", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_analysis_summary(
    stats_dict: dict,
    output_dir: str = "output",
    filename: str = "analysis_dashboard.png",
) -> str:
    """
    Create a 2x2 dashboard summary of the analysis results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CAT411 — Bridge Risk Analysis Dashboard", fontsize=18, fontweight="bold")

    # 1. Damage Distribution (Ax 0,0)
    ds_counts = stats_dict.get("damage_distribution", {})
    all_ds = ["none", "slight", "moderate", "extensive", "complete"]
    counts = [ds_counts.get(ds, 0) for ds in all_ds]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    axes[0,0].bar(all_ds, counts, color=colors, edgecolor="k")
    axes[0,0].set_title("Portfolio Damage State Distribution")
    axes[0,0].set_ylabel("Bridge Count")

    # 2. Key Metrics Text (Ax 0,1)
    axes[0,1].axis('off')
    metrics_text = (
        f"Total Bridges: {stats_dict.get('total_bridges', 'N/A')}\n\n"
        f"Scenario: {stats_dict.get('event_id', 'Northridge')}\n"
        f"Max PGA: {stats_dict.get('max_pga', 0):.3f}g\n"
        f"Avg Sa(1.0s): {stats_dict.get('avg_sa', 0):.3f}g\n\n"
        f"Estimated Economic Impact:\n"
        f"Expected Loss: ${stats_dict.get('total_loss', 0):,.0f}"
    )
    axes[0,1].text(0.1, 0.5, metrics_text, fontsize=14, family='monospace', va='center')
    axes[0,1].set_title("Core Risk Metrics")

    # 3. Ground Motion at Site (Ax 1,0)
    sa_vals = stats_dict.get("sa_values", [])
    if len(sa_vals) > 0:
        axes[1,0].hist(sa_vals, bins=20, color="gray", alpha=0.7, edgecolor="black")
        _im_label = IM_LABEL_MAP.get(stats_dict.get("im_type", "SA10"), "Sa(1.0s) [g]")
        axes[1,0].set_title("Ground Motion Intensity Histogram")
        axes[1,0].set_xlabel(_im_label)
        axes[1,0].set_ylabel("Frequency")

    # 4. Bridge Class Breakdown (Ax 1,1)
    class_counts = stats_dict.get("class_breakdown", {})
    if class_counts:
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        labels = [c[0] for c in top_classes]
        sizes = [c[1] for c in top_classes]
        axes[1,1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
        axes[1,1].set_title("Top 5 Bridge Classes")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ── New: Combined ShakeMap + Bridges Overlay ─────────────────────────────

def plot_bridges_on_shakemap(
    shakemap_df: pd.DataFrame,
    nbi_df: pd.DataFrame,
    im_type: str = "SA10",
    output_dir: str = "output",
    filename: str = "bridges_on_shakemap.png",
) -> str:
    """
    Overlay filtered bridge locations on a ShakeMap IM contour map.

    Parameters
    ----------
    shakemap_df : pd.DataFrame
        ShakeMap grid with LAT, LON, and IM columns.
    nbi_df : pd.DataFrame
        Bridge data with latitude, longitude, im_selected columns.
    im_type : str
        IM type label for axis/title (e.g. "SA10", "PGA").
    output_dir : str
    filename : str

    Returns
    -------
    str
        Path to saved figure.
    """
    from src.config import IM_COLUMN_MAP
    sm_col = IM_COLUMN_MAP.get(im_type, "PSA10")
    if sm_col not in shakemap_df.columns:
        sm_col = "PSA10"

    im_label_map = {
        "PGA": "PGA [g]",
        "SA03": "Sa(0.3s) [g]",
        "SA10": "Sa(1.0s) [g]",
        "SA30": "Sa(3.0s) [g]",
    }
    im_label = im_label_map.get(im_type, f"{im_type} [g]")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Background: ShakeMap contour
    sm_lats = shakemap_df["LAT"].values
    sm_lons = shakemap_df["LON"].values
    sm_vals = shakemap_df[sm_col].values

    try:
        from scipy.interpolate import griddata
        grid_lon = np.linspace(sm_lons.min(), sm_lons.max(), 200)
        grid_lat = np.linspace(sm_lats.min(), sm_lats.max(), 200)
        glon, glat = np.meshgrid(grid_lon, grid_lat)
        grid_vals = griddata(
            np.column_stack([sm_lons, sm_lats]), sm_vals,
            (glon, glat), method="linear",
        )
        contour = ax.contourf(
            glon, glat, grid_vals, levels=20,
            cmap="YlOrRd", alpha=0.6,
        )
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
    except Exception:
        sc_bg = ax.scatter(
            sm_lons, sm_lats, c=sm_vals, cmap="YlOrRd",
            s=1, alpha=0.3, edgecolors="none",
        )
        cbar = fig.colorbar(sc_bg, ax=ax, shrink=0.8)

    cbar.set_label(im_label, fontsize=12)

    # Overlay: Bridges
    im_col = "im_selected" if "im_selected" in nbi_df.columns else "sa_10"
    bridge_vals = nbi_df[im_col].values if im_col in nbi_df.columns else np.zeros(len(nbi_df))

    sc = ax.scatter(
        nbi_df["longitude"], nbi_df["latitude"],
        c=bridge_vals, cmap="cool", s=30,
        edgecolors="black", linewidths=0.5,
        vmin=0, vmax=max(0.3, np.percentile(bridge_vals, 95)),
        zorder=5,
    )
    cbar2 = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.06, location="left")
    cbar2.set_label(f"Bridge {im_label}", fontsize=10)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(
        f"Bridges on ShakeMap — {im_type} ({len(nbi_df)} bridges)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── New: GMPE Attenuation Curve ──────────────────────────────────────────

def plot_attenuation_curve(
    nbi_df: pd.DataFrame,
    epicenter_lat: float = 34.213,
    epicenter_lon: float = -118.537,
    Mw: float = 6.7,
    im_type: str = "SA10",
    output_dir: str = "output",
    filename: str = "attenuation_curve.png",
) -> str:
    """
    Plot GMPE attenuation prediction vs observed bridge IM (like USGS page).

    X-axis: distance from epicenter (km).
    Y-axis: IM (g).
    Line: BA08 GMPE median prediction ± 1σ.
    Scatter: actual bridge IM values from ShakeMap.

    Parameters
    ----------
    nbi_df : pd.DataFrame
        Bridge data with latitude, longitude, im_selected, hwb_class.
    epicenter_lat, epicenter_lon : float
        Earthquake epicenter coordinates.
    Mw : float
        Moment magnitude.
    im_type : str
        IM type label.
    output_dir : str
    filename : str
    """
    from src.hazard import haversine_distance_km, boore_atkinson_2008_sa10

    im_label_map = {
        "PGA": "PGA [g]",
        "SA03": "Sa(0.3s) [g]",
        "SA10": "Sa(1.0s) [g]",
        "SA30": "Sa(3.0s) [g]",
    }
    im_label = im_label_map.get(im_type, f"{im_type} [g]")

    # Compute distance from epicenter for each bridge
    distances = np.array([
        haversine_distance_km(epicenter_lat, epicenter_lon, lat, lon)
        for lat, lon in zip(nbi_df["latitude"], nbi_df["longitude"])
    ])

    # Bridge IMs
    im_col = "im_selected" if "im_selected" in nbi_df.columns else "sa_10"
    bridge_ims = nbi_df[im_col].values if im_col in nbi_df.columns else np.zeros(len(nbi_df))

    # GMPE prediction curve
    d_range = np.logspace(np.log10(1), np.log10(max(200, distances.max() * 1.2)), 100)
    gmpe_median = np.zeros_like(d_range)
    gmpe_sigma = np.zeros_like(d_range)

    for i, d in enumerate(d_range):
        med, sig = boore_atkinson_2008_sa10(Mw, d, Vs30=760.0, fault_type="reverse")
        gmpe_median[i] = med
        gmpe_sigma[i] = sig

    fig, ax = plt.subplots(figsize=(10, 7))

    # ±1σ band
    upper = gmpe_median * np.exp(gmpe_sigma)
    lower = gmpe_median * np.exp(-gmpe_sigma)
    ax.fill_between(d_range, lower, upper, alpha=0.15, color="blue",
                    label="BA08 ±1σ")

    # Median line
    ax.plot(d_range, gmpe_median, "b-", linewidth=2.5,
            label=f"BA08 Median (Mw {Mw:.1f})")

    # Bridge scatter (colored by HWB class)
    if "hwb_class" in nbi_df.columns:
        unique_classes = sorted(nbi_df["hwb_class"].unique())
        cmap = plt.cm.Set1(np.linspace(0, 1, max(len(unique_classes), 1)))
        for j, hwb in enumerate(unique_classes[:8]):  # max 8 classes for clarity
            mask = nbi_df["hwb_class"] == hwb
            ax.scatter(
                distances[mask], bridge_ims[mask],
                color=cmap[j], s=20, alpha=0.6,
                edgecolors="k", linewidths=0.3,
                label=hwb, zorder=5,
            )
    else:
        ax.scatter(distances, bridge_ims, c="red", s=20, alpha=0.5,
                   edgecolors="k", linewidths=0.3, label="Bridges", zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance from Epicenter (km)", fontsize=12)
    ax.set_ylabel(im_label, fontsize=12)
    ax.set_title(
        f"Ground Motion Attenuation — Mw {Mw:.1f} | {im_type}\n"
        f"(BA08 GMPE vs ShakeMap bridge values)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1, max(200, distances.max() * 1.2))
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

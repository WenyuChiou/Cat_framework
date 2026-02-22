"""Draw CAT411 architecture diagrams using matplotlib."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Common style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
})

COLORS = {
    "hazard":    ("#E3F2FD", "#1565C0"),  # (fill, border)
    "exposure":  ("#E8F5E9", "#2E7D32"),
    "vuln":      ("#FFF3E0", "#E65100"),
    "loss":      ("#FCE4EC", "#C62828"),
    "viz":       ("#F3E5F5", "#6A1B9A"),
    "L0":        ("#EDE7F6", "#5E35B1"),
    "L1":        ("#E3F2FD", "#1565C0"),
    "L2":        ("#E8F5E9", "#2E7D32"),
    "L3":        ("#FFF3E0", "#E65100"),
    "L4":        ("#FCE4EC", "#C62828"),
    "entry":     ("#F5F5F5", "#424242"),
}


def _box(ax, x, y, w, h, title, subtitle, data, fill, border):
    """Draw a rounded box with title, subtitle, and data label."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          facecolor=fill, edgecolor=border, linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h*0.68, title,
            ha="center", va="center", fontsize=13, fontweight="bold", color=border)
    ax.text(x + w/2, y + h*0.38, subtitle,
            ha="center", va="center", fontsize=9, color="#666666")
    # Data label below box
    ax.text(x + w/2, y - 0.06, data,
            ha="center", va="top", fontsize=7.5, color="#888888",
            style="italic")


def _arrow(ax, x1, y1, x2, y2, label="", color="#666666"):
    """Draw an arrow with optional label."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5,
                                connectionstyle="arc3,rad=0"))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2 + 0.04
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=7,
                color=color, fontweight="bold")


# =====================================================================
# Diagram 1: Backbone Pipeline
# =====================================================================
def draw_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(16, 5.5))
    ax.set_xlim(-0.1, 5.5)
    ax.set_ylim(-0.35, 1.15)
    ax.axis("off")

    # Title
    ax.text(2.7, 1.08, "CAT411 Earthquake Bridge Loss Framework",
            ha="center", va="center", fontsize=18, fontweight="bold", color="#1a1a1a")
    ax.text(2.7, 0.97, "Backbone Pipeline",
            ha="center", va="center", fontsize=13, color="#666666")

    # 5 boxes
    bw, bh = 0.85, 0.45
    gap = 1.1
    boxes = [
        (0,    "HAZARD",        "Ground Motion",     "Sa(g) at each\nbridge site",     "hazard"),
        (gap,  "EXPOSURE",      "Bridge Assets",     "4,853 bridges\nHWB class + cost", "exposure"),
        (2*gap,"VULNERABILITY", "Fragility Curves",  "P(DS|IM)\nper bridge",           "vuln"),
        (3*gap,"LOSS",          "Financial Impact",  "E[Loss], EP curve\nAAL",          "loss"),
        (4*gap,"VISUALIZATION", "Maps & Reports",    "Dashboard\nrisk maps, CSV",      "viz"),
    ]

    y = 0.3
    for x, title, sub, data, ckey in boxes:
        fill, border = COLORS[ckey]
        _box(ax, x, y, bw, bh, title, sub, data, fill, border)

    # Arrows between boxes
    labels = ["IM values", "Portfolio", "P(DS)", "Risk metrics"]
    for i in range(4):
        x1 = boxes[i][0] + bw
        x2 = boxes[i+1][0]
        _arrow(ax, x1 + 0.02, y + bh/2, x2 - 0.02, y + bh/2, labels[i])

    # Bottom data flow strip
    ax.text(2.7, -0.02,
            "grid.xml / GMPE  >>>  Sa(g) per bridge  >>>  P(DS)  >>>  E[L] = \u03a3(P \u00d7 DR \u00d7 Cost)  >>>  PNG + CSV",
            ha="center", va="top", fontsize=8.5, color="#999999",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#E0E0E0"))

    # Module labels under each box
    mod_labels = [
        "hazard.py\ninterpolation.py",
        "exposure.py\nbridge_classes.py\ndata_loader.py",
        "fragility.py\nhazus_params.py",
        "loss.py\nengine.py",
        "plotting.py",
    ]
    for i, (x, *_rest) in enumerate(boxes):
        ax.text(x + bw/2, y - 0.18, mod_labels[i],
                ha="center", va="top", fontsize=7, color="#AAAAAA",
                fontfamily="monospace")

    fig.tight_layout(pad=0.5)
    fig.savefig("docs/diagram_pipeline.png", dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved: docs/diagram_pipeline.png")


# =====================================================================
# Diagram 2: Module Dependency (5-Layer)
# =====================================================================
def draw_dependency():
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis("off")

    # Title
    ax.text(5, 8.2, "CAT411 Framework — Module Dependency Architecture",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1a1a1a")

    # Layer bands
    layers = [
        (7.0, "L0: DATA",     "#EDE7F6", "#B39DDB"),
        (5.6, "L1: CORE",     "#E3F2FD", "#90CAF9"),
        (4.2, "L2: DOMAIN",   "#E8F5E9", "#A5D6A7"),
        (2.8, "L3: PIPELINE", "#FFF3E0", "#FFCC80"),
        (1.4, "L4: OUTPUT",   "#FCE4EC", "#EF9A9A"),
        (0.0, "ENTRY",        "#F5F5F5", "#BDBDBD"),
    ]
    for ly, label, fill, edge in layers:
        band = FancyBboxPatch((-0.3, ly), 10.6, 1.1, boxstyle="round,pad=0.05",
                               facecolor=fill, edgecolor=edge, linewidth=1, alpha=0.4)
        ax.add_patch(band)
        ax.text(-0.1, ly + 0.55, label, ha="left", va="center",
                fontsize=9, fontweight="bold", color=edge, alpha=0.8)

    # Module boxes
    def mod_box(x, y, name, ckey):
        fill, border = COLORS[ckey]
        w, h = 2.0, 0.7
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.03",
                              facecolor=fill, edgecolor=border, linewidth=1.8)
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center",
                fontsize=9, fontweight="bold", fontfamily="monospace", color=border)
        return (x, y)

    # L0
    p_hazus = mod_box(3, 7.5, "hazus_params", "L0")
    p_config = mod_box(7, 7.5, "config", "L0")

    # L1
    p_hazard = mod_box(1.5, 6.1, "hazard", "L1")
    p_bridge = mod_box(5, 6.1, "bridge_classes", "L1")
    p_interp = mod_box(8.5, 6.1, "interpolation", "L1")

    # L2
    p_frag = mod_box(2, 4.7, "fragility", "L2")
    p_expo = mod_box(5.5, 4.7, "exposure", "L2")
    p_dload = mod_box(8.5, 4.7, "data_loader", "L2")

    # L3
    p_loss = mod_box(3, 3.3, "loss", "L3")
    p_engine = mod_box(7, 3.3, "engine", "L3")

    # L4
    p_plot = mod_box(1.5, 1.9, "plotting", "L4")
    p_north = mod_box(5, 1.9, "northridge_case", "L4")
    p_hdl = mod_box(8.5, 1.9, "hazard_download", "L4")

    # Entry
    p_main = mod_box(5, 0.5, "main.py", "entry")

    # Dependency arrows (from upstream to downstream, top to bottom)
    def dep_arrow(src, dst, label="", color="#888888"):
        ax.annotate("", xy=(dst[0], dst[1] + 0.38), xytext=(src[0], src[1] - 0.38),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                                    connectionstyle="arc3,rad=0.1", alpha=0.7))
        if label:
            mx = (src[0] + dst[0]) / 2 + 0.15
            my = (src[1] + dst[1]) / 2
            ax.text(mx, my, label, fontsize=6, color=color, ha="left", va="center",
                    style="italic")

    # hazus_params -> many
    dep_arrow(p_hazus, p_bridge, "FRAGILITY dict")
    dep_arrow(p_hazus, p_frag, "params")
    dep_arrow(p_hazus, p_loss)
    dep_arrow(p_hazus, p_plot)

    # hazard -> engine, exposure
    dep_arrow(p_hazard, p_engine, "Scenario + Sa(g)")
    dep_arrow(p_hazard, p_expo, "SiteParams")

    # bridge_classes -> data_loader
    dep_arrow(p_bridge, p_dload, "classify_bridge()")

    # fragility -> loss, plotting, northridge
    dep_arrow(p_frag, p_loss, "P(DS|IM)")
    dep_arrow(p_frag, p_plot)
    dep_arrow(p_frag, p_north)

    # exposure -> engine
    dep_arrow(p_expo, p_engine, "BridgeExposure[]")

    # loss -> engine
    dep_arrow(p_loss, p_engine, "PortfolioLossResult")

    # everything -> main
    for src in [p_config, p_dload, p_interp, p_engine, p_plot, p_north, p_hdl]:
        dep_arrow(src, p_main, color="#CCCCCC")

    fig.tight_layout(pad=0.5)
    fig.savefig("docs/diagram_dependency.png", dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Saved: docs/diagram_dependency.png")


if __name__ == "__main__":
    draw_pipeline()
    draw_dependency()
    print("\nDone! Now run: python docs/generate_task_board.py")

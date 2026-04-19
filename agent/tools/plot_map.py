"""Tool: plot_bridge_map — generate interactive map of bridge damage."""

import agent.config  # noqa: F401
from agent.config import OUTPUT_DIR, ensure_dirs
from agent.tools.registry import register_tool


def _plot_bridge_map(
    magnitude: float = 6.7,
    lat: float = 34.213,
    lon: float = -118.537,
    depth_km: float = 10.0,
    fault_type: str = "reverse",
    radius_km: float = 30.0,
    n_bridges: int = 100,
    use_nbi: bool = False,
    seed: int = 42,
) -> str:
    """Generate an interactive map showing bridge damage from an earthquake scenario."""
    import math
    import json
    import numpy as np
    import folium
    from src.hazard import EarthquakeScenario, compute_sa_at_sites
    from src.exposure import (
        generate_synthetic_portfolio, create_portfolio_from_nbi,
        portfolio_to_sites,
    )
    from src.loss import compute_bridge_loss

    ensure_dirs()

    scenario = EarthquakeScenario(
        Mw=magnitude, lat=lat, lon=lon,
        depth_km=depth_km, fault_type=fault_type,
    )

    if use_nbi:
        from src.data_loader import parse_nbi, classify_nbi_to_hazus, DATA_DIR
        import pandas as pd

        nbi_path = DATA_DIR / "CA24.txt"
        csv_path = DATA_DIR / "nbi_classified_2024.csv"
        if nbi_path.exists():
            df = parse_nbi(nbi_path)
            df = classify_nbi_to_hazus(df)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8")
        else:
            return "Error: NBI data not found"

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
        if len(df) > 300:
            df = df.sample(300, random_state=seed).reset_index(drop=True)
        portfolio = create_portfolio_from_nbi(df)
    else:
        portfolio = generate_synthetic_portfolio(
            n_bridges=n_bridges, center=(lat, lon),
            radius_km=radius_km, seed=seed,
        )

    # Compute Sa at each bridge
    sites = portfolio_to_sites(portfolio)
    median_sa, _ = compute_sa_at_sites(scenario, sites)

    # Compute loss for each bridge
    results = []
    for i, bridge in enumerate(portfolio):
        sa = float(median_sa[i])
        lr = compute_bridge_loss(
            sa=sa, hwb_class=bridge.hwb_class,
            replacement_cost=bridge.replacement_cost,
            bridge_id=bridge.bridge_id,
        )
        # Determine dominant damage state
        max_ds = max(lr.damage_probs, key=lr.damage_probs.get)
        results.append({
            "bridge": bridge,
            "sa": sa,
            "loss_result": lr,
            "dominant_ds": max_ds,
        })

    # Color by dominant damage state
    ds_colors = {
        "none": "#4CAF50",       # green
        "slight": "#2196F3",     # blue
        "moderate": "#FF9800",   # orange
        "extensive": "#F44336",  # red
        "complete": "#9C27B0",   # purple
    }

    # Create map centered on epicenter
    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="CartoDB positron")

    # Epicenter marker
    folium.Marker(
        [lat, lon],
        popup=f"<b>Epicenter</b><br>M{magnitude}<br>Depth: {depth_km}km",
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
    ).add_to(m)

    # Bridge markers
    for r in results:
        b = r["bridge"]
        lr = r["loss_result"]
        ds = r["dominant_ds"]
        color = ds_colors.get(ds, "#757575")

        popup_html = (
            f"<b>{b.bridge_id}</b><br>"
            f"Class: {b.hwb_class}<br>"
            f"Sa: {r['sa']:.3f}g<br>"
            f"Expected Loss: ${lr.expected_loss:,.0f}<br>"
            f"Loss Ratio: {lr.loss_ratio:.1%}<br>"
            f"Downtime: {lr.expected_downtime:.0f} days<br>"
            f"Dominant DS: {ds}"
        )

        folium.CircleMarker(
            location=[b.lat, b.lon],
            radius=max(4, min(15, lr.loss_ratio * 50)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:white;padding:10px;border-radius:5px;
         border:2px solid grey;font-size:12px;">
    <b>Dominant Damage State</b><br>
    <i style="background:#4CAF50;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> None<br>
    <i style="background:#2196F3;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Slight<br>
    <i style="background:#FF9800;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Moderate<br>
    <i style="background:#F44336;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Extensive<br>
    <i style="background:#9C27B0;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Complete<br>
    <b>Circle size</b> = loss ratio
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    map_path = str(OUTPUT_DIR / "bridge_damage_map.html")
    m.save(map_path)

    # Summary stats
    ds_counts = {}
    total_loss = 0
    for r in results:
        ds = r["dominant_ds"]
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
        total_loss += r["loss_result"].expected_loss

    lines = [
        f"Interactive map saved to: {map_path}",
        f"Scenario: M{magnitude} at ({lat}, {lon})",
        f"Bridges mapped: {len(results)}",
        f"Total expected loss: ${total_loss:,.0f}",
        "",
        "Damage state distribution:",
    ]
    for ds in ["none", "slight", "moderate", "extensive", "complete"]:
        cnt = ds_counts.get(ds, 0)
        pct = cnt / len(results) * 100 if results else 0
        lines.append(f"  {ds:<12s}: {cnt:>4d} ({pct:.1f}%)")

    return "\n".join(lines)


register_tool(
    name="plot_bridge_map",
    description=(
        "Generate an interactive MAP showing bridge locations on a geographic "
        "map, color-coded by damage state for an earthquake scenario. "
        "USE THIS TOOL when the user asks for a map, spatial visualization, "
        "or wants to SEE WHERE bridges are damaged. Each bridge marker shows "
        "expected loss, loss ratio, downtime, and dominant damage state on click. "
        "Circle size proportional to loss ratio. Saves an interactive HTML map file."
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
                "description": "Epicenter latitude",
                "default": 34.213,
            },
            "lon": {
                "type": "number",
                "description": "Epicenter longitude",
                "default": -118.537,
            },
            "depth_km": {"type": "number", "default": 10.0},
            "fault_type": {
                "type": "string",
                "enum": ["reverse", "strike_slip", "normal"],
                "default": "reverse",
            },
            "radius_km": {"type": "number", "default": 30.0},
            "n_bridges": {"type": "integer", "default": 100},
            "use_nbi": {
                "type": "boolean",
                "description": "Use real NBI bridges instead of synthetic",
                "default": False,
            },
            "seed": {"type": "integer", "default": 42},
        },
        "required": [],
    },
    function=_plot_bridge_map,
)

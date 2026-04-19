"""Tool: summarize_portfolio — get portfolio statistics for a region."""

import agent.config  # noqa: F401
from agent.tools.registry import register_tool


def _summarize_portfolio(
    lat_min: float = 33.8,
    lat_max: float = 34.6,
    lon_min: float = -118.9,
    lon_max: float = -118.0,
) -> str:
    """Get aggregate portfolio statistics for bridges in a region."""
    from src.data_loader import parse_nbi, classify_nbi_to_hazus, DATA_DIR
    from src.exposure import create_portfolio_from_nbi, portfolio_summary

    nbi_path = DATA_DIR / "CA24.txt"
    if not nbi_path.exists():
        import pandas as pd
        csv_path = DATA_DIR / "nbi_classified_2024.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8")
        else:
            return "Error: NBI data not found"
    else:
        df = parse_nbi(nbi_path)
        df = classify_nbi_to_hazus(df)

    mask = (
        (df["latitude"] >= lat_min) &
        (df["latitude"] <= lat_max) &
        (df["longitude"] >= lon_min) &
        (df["longitude"] <= lon_max)
    )
    df = df[mask]

    if len(df) == 0:
        return f"No bridges found in [{lat_min}–{lat_max}N, {lon_min}–{lon_max}E]"

    portfolio = create_portfolio_from_nbi(df)
    ps = portfolio_summary(portfolio)

    lines = [
        "PORTFOLIO SUMMARY",
        "=" * 55,
        f"  Region:             [{lat_min:.2f}–{lat_max:.2f}N, {lon_min:.2f}–{lon_max:.2f}E]",
        f"  Total bridges:      {ps['n_bridges']:,}",
        f"  Total replacement:  ${ps['total_replacement_cost']:,.0f}",
        f"  Avg. bridge cost:   ${ps['avg_replacement_cost']:,.0f}",
        "",
        "  HWB Class Distribution:",
    ]
    for cls, cnt in sorted(ps["class_distribution"].items()):
        pct = cnt / ps["n_bridges"] * 100
        lines.append(f"    {cls:<8s}: {cnt:>5,} ({pct:.1f}%)")

    lines.append("")
    lines.append("  Material Distribution:")
    for mat, cnt in sorted(ps["material_distribution"].items()):
        pct = cnt / ps["n_bridges"] * 100
        lines.append(f"    {mat:<22s}: {cnt:>5,} ({pct:.1f}%)")

    return "\n".join(lines)


register_tool(
    name="summarize_portfolio",
    description=(
        "Get aggregate statistics for the bridge portfolio in a region: "
        "total count, total replacement cost, class distribution, "
        "and material distribution."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat_min": {"type": "number", "description": "South boundary latitude"},
            "lat_max": {"type": "number", "description": "North boundary latitude"},
            "lon_min": {"type": "number", "description": "West boundary longitude"},
            "lon_max": {"type": "number", "description": "East boundary longitude"},
        },
        "required": ["lat_min", "lat_max", "lon_min", "lon_max"],
    },
    function=_summarize_portfolio,
)

"""Tool: query_bridges — search and filter bridge inventory."""

import agent.config  # noqa: F401 — ensures framework is on sys.path
from agent.tools.registry import register_tool


def _query_bridges(
    lat_min: float = 33.5,
    lat_max: float = 34.8,
    lon_min: float = -119.0,
    lon_max: float = -117.5,
    hwb_classes: list[str] | None = None,
    material: str | None = None,
    max_results: int = 20,
) -> str:
    """Query the NBI bridge inventory within a bounding box."""
    from src.data_loader import parse_nbi, classify_nbi_to_hazus, DATA_DIR

    nbi_path = DATA_DIR / "CA24.txt"
    if not nbi_path.exists():
        # Try classified CSV
        csv_path = DATA_DIR / "nbi_classified_2024.csv"
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path, encoding="utf-8")
        else:
            return "Error: NBI data file not found (CA24.txt or nbi_classified_2024.csv)"
    else:
        df = parse_nbi(nbi_path)
        df = classify_nbi_to_hazus(df)

    # Apply bounding box
    mask = (
        (df["latitude"] >= lat_min) &
        (df["latitude"] <= lat_max) &
        (df["longitude"] >= lon_min) &
        (df["longitude"] <= lon_max)
    )
    df = df[mask]

    # Filter by HWB class
    if hwb_classes:
        df = df[df["hwb_class"].isin(hwb_classes)]

    # Filter by material
    if material:
        df = df[df["material"].str.lower() == material.lower()]

    total = len(df)

    # Class distribution
    class_dist = df["hwb_class"].value_counts().to_dict()

    # Material distribution
    mat_dist = df["material"].value_counts().to_dict()

    # Sample bridges
    sample = df.head(max_results)
    sample_lines = []
    for _, row in sample.iterrows():
        sample_lines.append(
            f"  {row.get('structure_number', 'N/A'):>15s} | "
            f"({row['latitude']:.4f}, {row['longitude']:.4f}) | "
            f"{row['hwb_class']:<6s} | {row.get('material', 'N/A')}"
        )

    lines = [
        f"Found {total} bridges in region "
        f"[{lat_min:.2f}–{lat_max:.2f}N, {lon_min:.2f}–{lon_max:.2f}E]",
        "",
        "HWB Class Distribution:",
    ]
    for cls, cnt in sorted(class_dist.items()):
        lines.append(f"  {cls}: {cnt}")

    lines.append("")
    lines.append("Material Distribution:")
    for mat, cnt in sorted(mat_dist.items()):
        lines.append(f"  {mat}: {cnt}")

    if sample_lines:
        lines.append("")
        lines.append(f"Sample bridges (first {len(sample_lines)}):")
        lines.append("  Structure#       | (Lat, Lon)          | HWB    | Material")
        lines.extend(sample_lines)

    return "\n".join(lines)


register_tool(
    name="query_bridges",
    description=(
        "Search the California NBI bridge inventory within a geographic "
        "bounding box. Can filter by HWB fragility class and material type. "
        "Returns count, class distribution, material distribution, and sample bridges."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lat_min": {"type": "number", "description": "South boundary latitude"},
            "lat_max": {"type": "number", "description": "North boundary latitude"},
            "lon_min": {"type": "number", "description": "West boundary longitude (negative for western hemisphere)"},
            "lon_max": {"type": "number", "description": "East boundary longitude"},
            "hwb_classes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by HWB classes, e.g. ['HWB3', 'HWB5']. Omit for all classes.",
            },
            "material": {
                "type": "string",
                "description": "Filter by material: concrete, steel, prestressed_concrete, wood, other",
            },
            "max_results": {
                "type": "integer",
                "description": "Max sample bridges to show (default 20)",
                "default": 20,
            },
        },
        "required": ["lat_min", "lat_max", "lon_min", "lon_max"],
    },
    function=_query_bridges,
)

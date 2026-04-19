"""Tool: plot_results — generate visualization plots."""

from agent.config import OUTPUT_DIR, ensure_dirs
from agent.tools.registry import register_tool


def _plot_fragility_curves(
    hwb_class: str,
    im_max: float = 2.5,
) -> str:
    """Plot fragility curves for a bridge class."""
    import numpy as np
    from src.plotting import plot_single_class

    ensure_dirs()
    output_dir = str(OUTPUT_DIR)
    filepath = plot_single_class(
        hwb_class=hwb_class.upper(),
        im_values=np.linspace(0, im_max, 200),
        output_dir=output_dir,
    )
    return f"Fragility curves for {hwb_class.upper()} saved to: {filepath}"


def _plot_damage_distribution(
    hwb_class: str,
    sa_values: list[float] | None = None,
) -> str:
    """Plot damage state distribution at given intensities."""
    import numpy as np
    from src.plotting import plot_damage_distribution

    if sa_values is None:
        sa_values = [0.1, 0.3, 0.5, 0.8, 1.0]

    ensure_dirs()
    output_dir = str(OUTPUT_DIR)
    filepath = plot_damage_distribution(
        hwb_class=hwb_class.upper(),
        im_values_sample=sa_values,
        output_dir=output_dir,
    )
    return f"Damage distribution for {hwb_class.upper()} at Sa={sa_values} saved to: {filepath}"


def _plot_class_comparison(
    hwb_classes: list[str],
    damage_state: str = "extensive",
    im_max: float = 2.5,
) -> str:
    """Plot fragility comparison across multiple bridge classes."""
    import numpy as np
    from src.plotting import plot_comparison

    ensure_dirs()
    output_dir = str(OUTPUT_DIR)
    classes = [c.upper() for c in hwb_classes]
    filepath = plot_comparison(
        hwb_classes=classes,
        damage_state=damage_state,
        im_values=np.linspace(0, im_max, 200),
        output_dir=output_dir,
    )
    return f"Comparison plot ({damage_state}) for {classes} saved to: {filepath}"


register_tool(
    name="plot_fragility_curves",
    description=(
        "Plot the 4 fragility curves (slight/moderate/extensive/complete) "
        "for a given HWB bridge class. Saves a PNG file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "hwb_class": {
                "type": "string",
                "description": "Hazus bridge class, e.g. 'HWB5'",
            },
            "im_max": {
                "type": "number",
                "description": "Maximum Sa value for x-axis (default 2.5g)",
                "default": 2.5,
            },
        },
        "required": ["hwb_class"],
    },
    function=_plot_fragility_curves,
)

register_tool(
    name="plot_damage_distribution",
    description=(
        "Plot damage state probability distribution (bar chart) "
        "for a bridge class at multiple Sa intensity levels."
    ),
    parameters={
        "type": "object",
        "properties": {
            "hwb_class": {
                "type": "string",
                "description": "Hazus bridge class, e.g. 'HWB5'",
            },
            "sa_values": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of Sa values to show, e.g. [0.1, 0.3, 0.5, 1.0]",
            },
        },
        "required": ["hwb_class"],
    },
    function=_plot_damage_distribution,
)

register_tool(
    name="plot_class_comparison",
    description=(
        "Compare fragility curves across multiple HWB classes for a "
        "given damage state. Useful for showing which bridge types are more vulnerable."
    ),
    parameters={
        "type": "object",
        "properties": {
            "hwb_classes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of HWB classes to compare, e.g. ['HWB3', 'HWB5', 'HWB17']",
            },
            "damage_state": {
                "type": "string",
                "enum": ["slight", "moderate", "extensive", "complete"],
                "description": "Which damage state to compare (default 'extensive')",
                "default": "extensive",
            },
            "im_max": {
                "type": "number",
                "description": "Maximum Sa for x-axis (default 2.5g)",
                "default": 2.5,
            },
        },
        "required": ["hwb_classes"],
    },
    function=_plot_class_comparison,
)

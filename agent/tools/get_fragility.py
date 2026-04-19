"""Tool: get_fragility — query fragility parameters and damage probabilities."""

import agent.config  # noqa: F401
from agent.tools.registry import register_tool


def _get_fragility(
    hwb_class: str,
    sa_value: float | None = None,
) -> str:
    """Get fragility parameters and optionally compute damage probabilities."""
    from src.hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER
    from src.fragility import damage_state_probabilities

    hwb_class = hwb_class.upper()
    if hwb_class not in HAZUS_BRIDGE_FRAGILITY:
        available = ", ".join(sorted(HAZUS_BRIDGE_FRAGILITY.keys()))
        return f"Error: Unknown HWB class '{hwb_class}'. Available: {available}"

    params = HAZUS_BRIDGE_FRAGILITY[hwb_class]
    lines = [
        f"Fragility Parameters for {hwb_class}: {params['name']}",
        "",
        "Damage State     | Median Sa(1.0s) [g] | Beta",
        "-" * 55,
    ]
    for ds in DAMAGE_STATE_ORDER:
        p = params["damage_states"][ds]
        lines.append(f"  {ds:<14s} | {p['median']:>19.2f} | {p['beta']:.2f}")

    if sa_value is not None and sa_value > 0:
        probs = damage_state_probabilities(sa_value, hwb_class)
        lines.append("")
        lines.append(f"Damage State Probabilities at Sa = {sa_value:.3f}g:")
        lines.append("-" * 40)
        for ds in ["none"] + DAMAGE_STATE_ORDER:
            lines.append(f"  {ds:<14s}: {probs[ds]:>8.4f} ({probs[ds]*100:.2f}%)")

    return "\n".join(lines)


register_tool(
    name="get_fragility",
    description=(
        "Get Hazus 6.1 fragility curve parameters for a bridge class (HWB1-HWB28). "
        "Optionally compute damage state probabilities at a given Sa(1.0s) value. "
        "Returns median and beta for each damage state, and P(DS) if sa_value provided."
    ),
    parameters={
        "type": "object",
        "properties": {
            "hwb_class": {
                "type": "string",
                "description": "Hazus bridge class, e.g. 'HWB5', 'HWB17'",
            },
            "sa_value": {
                "type": "number",
                "description": "Optional: Sa(1.0s) in g to compute damage probabilities at this intensity",
            },
        },
        "required": ["hwb_class"],
    },
    function=_get_fragility,
)

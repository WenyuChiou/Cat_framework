"""Tool: compute_loss — compute bridge or portfolio loss."""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.tools.registry import register_tool


def _compute_single_bridge_loss(
    sa: float,
    hwb_class: str,
    replacement_cost: float = 5_000_000.0,
    bridge_id: str = "user-query",
) -> str:
    """Compute expected loss for a single bridge."""
    from src.loss import compute_bridge_loss

    hwb_class = hwb_class.upper()
    result = compute_bridge_loss(
        sa=sa,
        hwb_class=hwb_class,
        replacement_cost=replacement_cost,
        bridge_id=bridge_id,
    )

    lines = [
        f"Bridge Loss Result: {result.bridge_id}",
        f"  HWB Class:        {hwb_class}",
        f"  Sa(1.0s):         {result.sa:.4f} g",
        f"  Replacement Cost: ${result.replacement_cost:,.0f}",
        f"  Expected Loss:    ${result.expected_loss:,.0f}",
        f"  Loss Ratio:       {result.loss_ratio:.4f} ({result.loss_ratio*100:.2f}%)",
        f"  Expected Downtime:{result.expected_downtime:.1f} days",
        "",
        "  Damage Probabilities:",
    ]
    for ds, p in result.damage_probs.items():
        lines.append(f"    {ds:<14s}: {p:.4f} ({p*100:.2f}%)")

    return "\n".join(lines)


register_tool(
    name="compute_bridge_loss",
    description=(
        "Compute expected economic loss for a single bridge given "
        "Sa(1.0s) intensity, HWB class, and replacement cost. "
        "Returns expected loss, loss ratio, downtime, and damage probabilities."
    ),
    parameters={
        "type": "object",
        "properties": {
            "sa": {
                "type": "number",
                "description": "Spectral acceleration Sa(1.0s) in g at the bridge site",
            },
            "hwb_class": {
                "type": "string",
                "description": "Hazus bridge class, e.g. 'HWB5'",
            },
            "replacement_cost": {
                "type": "number",
                "description": "Bridge replacement cost in USD (default $5M)",
                "default": 5000000,
            },
            "bridge_id": {
                "type": "string",
                "description": "Bridge identifier (optional)",
                "default": "user-query",
            },
        },
        "required": ["sa", "hwb_class"],
    },
    function=_compute_single_bridge_loss,
)

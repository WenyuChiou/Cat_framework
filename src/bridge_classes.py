"""
Bridge classification definitions and Northridge-relevant inventory.

Provides dataclass-based bridge classification and a helper to look up
the Hazus bridge class from structural attributes.
"""

from dataclasses import dataclass
from typing import Optional

from .hazus_params import HAZUS_BRIDGE_FRAGILITY


@dataclass
class BridgeClassification:
    """Structural attributes used to determine Hazus bridge class."""
    hwb_class: str
    name: str
    material: str          # "concrete", "steel", "other"
    span_type: str         # "single", "multi_continuous", "multi_simply_supported"
    design_era: str        # "conventional", "seismic"
    length_category: str   # "major" (>150m), "standard"
    subtype: str = ""      # e.g. "box_girder", "frame"


# Bridge classes relevant to the 1994 Northridge earthquake
NORTHRIDGE_BRIDGE_CLASSES = {
    "HWB1": BridgeClassification(
        hwb_class="HWB1",
        name="Major Bridge - Conventional Design (>150m)",
        material="mixed",
        span_type="multi_continuous",
        design_era="conventional",
        length_category="major",
    ),
    "HWB2": BridgeClassification(
        hwb_class="HWB2",
        name="Major Bridge - Seismic Design (>150m)",
        material="mixed",
        span_type="multi_continuous",
        design_era="seismic",
        length_category="major",
    ),
    "HWB3": BridgeClassification(
        hwb_class="HWB3",
        name="Single Span - Concrete, Conventional",
        material="concrete",
        span_type="single",
        design_era="conventional",
        length_category="standard",
    ),
    "HWB4": BridgeClassification(
        hwb_class="HWB4",
        name="Single Span - Concrete, Seismic",
        material="concrete",
        span_type="single",
        design_era="seismic",
        length_category="standard",
    ),
    "HWB5": BridgeClassification(
        hwb_class="HWB5",
        name="Multi-Span Concrete Continuous - Conventional",
        material="concrete",
        span_type="multi_continuous",
        design_era="conventional",
        length_category="standard",
    ),
    "HWB6": BridgeClassification(
        hwb_class="HWB6",
        name="Multi-Span Concrete Continuous - Seismic",
        material="concrete",
        span_type="multi_continuous",
        design_era="seismic",
        length_category="standard",
    ),
    "HWB7": BridgeClassification(
        hwb_class="HWB7",
        name="Multi-Span Concrete Simply-Supported - Conventional",
        material="concrete",
        span_type="multi_simply_supported",
        design_era="conventional",
        length_category="standard",
    ),
    "HWB8": BridgeClassification(
        hwb_class="HWB8",
        name="Multi-Span Concrete Simply-Supported - Seismic",
        material="concrete",
        span_type="multi_simply_supported",
        design_era="seismic",
        length_category="standard",
    ),
    "HWB10": BridgeClassification(
        hwb_class="HWB10",
        name="Multi-Span Steel Continuous - Conventional",
        material="steel",
        span_type="multi_continuous",
        design_era="conventional",
        length_category="standard",
    ),
    "HWB11": BridgeClassification(
        hwb_class="HWB11",
        name="Multi-Span Steel Continuous - Seismic",
        material="steel",
        span_type="multi_continuous",
        design_era="seismic",
        length_category="standard",
    ),
    "HWB15": BridgeClassification(
        hwb_class="HWB15",
        name="Single Span - Steel, Conventional",
        material="steel",
        span_type="single",
        design_era="conventional",
        length_category="standard",
    ),
    "HWB16": BridgeClassification(
        hwb_class="HWB16",
        name="Single Span - Steel, Seismic",
        material="steel",
        span_type="single",
        design_era="seismic",
        length_category="standard",
    ),
    "HWB17": BridgeClassification(
        hwb_class="HWB17",
        name="Multi-Span Concrete Continuous Box Girder - Conventional",
        material="concrete",
        span_type="multi_continuous",
        design_era="conventional",
        length_category="standard",
        subtype="box_girder",
    ),
    "HWB22": BridgeClassification(
        hwb_class="HWB22",
        name="Multi-Span Concrete Continuous Frame - Conventional",
        material="concrete",
        span_type="multi_continuous",
        design_era="conventional",
        length_category="standard",
        subtype="frame",
    ),
    "HWB28": BridgeClassification(
        hwb_class="HWB28",
        name="All Others",
        material="other",
        span_type="other",
        design_era="conventional",
        length_category="standard",
    ),
}


def classify_bridge(
    material: str,
    span_type: str,
    design_era: str,
    length: float,
    subtype: Optional[str] = None,
) -> str:
    """
    Determine Hazus bridge class from structural attributes.

    Parameters
    ----------
    material : str
        "concrete" or "steel"
    span_type : str
        "single", "multi_continuous", or "multi_simply_supported"
    design_era : str
        "conventional" or "seismic"
    length : float
        Total bridge length in meters
    subtype : str, optional
        Additional type info ("box_girder", "frame")

    Returns
    -------
    str
        Hazus bridge class identifier (e.g. "HWB5")
    """
    # Major bridges (>150m)
    if length > 150:
        return "HWB2" if design_era == "seismic" else "HWB1"

    # Single span
    if span_type == "single":
        if material == "concrete":
            return "HWB4" if design_era == "seismic" else "HWB3"
        if material == "steel":
            return "HWB16" if design_era == "seismic" else "HWB15"

    # Multi-span concrete
    if material == "concrete":
        if span_type == "multi_continuous":
            if subtype == "box_girder":
                return "HWB17"
            if subtype == "frame":
                return "HWB22"
            return "HWB6" if design_era == "seismic" else "HWB5"
        if span_type == "multi_simply_supported":
            return "HWB8" if design_era == "seismic" else "HWB7"

    # Multi-span steel
    if material == "steel" and span_type == "multi_continuous":
        return "HWB11" if design_era == "seismic" else "HWB10"

    return "HWB28"


def get_bridge_params(hwb_class: str) -> dict:
    """Return Hazus fragility parameters for a bridge class."""
    if hwb_class not in HAZUS_BRIDGE_FRAGILITY:
        raise ValueError(f"Unknown bridge class: {hwb_class}")
    return HAZUS_BRIDGE_FRAGILITY[hwb_class]

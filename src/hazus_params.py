"""
Hazus 6.1 bridge fragility parameters (Table 7.9) with class-specific beta.

Each bridge class stores lognormal fragility parameters:
  - median: median Sa(1.0s) in g for each damage state
  - beta:   lognormal dispersion (total uncertainty)

Median values are from Hazus Table 7.9. Beta values are updated from
class-specific dispersions in the literature:
  - Nielson & DesRoches (2007): Multi-span concrete/steel classes
  - Padgett & DesRoches (2008): Seismically designed classes
  - Hazus 6.1 default (0.6): Single-span and catch-all classes

Damage states follow Hazus definitions:
  - Slight:    minor cracking, spalling
  - Moderate:  moderate cracking, column damage
  - Extensive: degradation without collapse, significant displacement
  - Complete:  collapse or imminent collapse

References:
  FEMA Hazus 6.1 Earthquake Model Technical Manual, Table 7.9
  Nielson, B.G. & DesRoches, R. (2007). Seismic fragility methodology for
    highway bridges using a component level approach. EESD, 36(6), 823-839.
  Padgett, J.E. & DesRoches, R. (2008). Methodology for the development of
    analytical fragility curves for retrofitted bridges. EESD, 37(8), 1157-1174.
"""

HAZUS_BRIDGE_FRAGILITY = {
    "HWB1": {
        "name": "Major Bridge - Conventional Design (>150m)",
        "damage_states": {
            "slight":    {"median": 0.25, "beta": 0.55},
            "moderate":  {"median": 0.35, "beta": 0.55},
            "extensive": {"median": 0.45, "beta": 0.55},
            "complete":  {"median": 0.70, "beta": 0.55},
        },
    },
    "HWB2": {
        "name": "Major Bridge - Seismic Design (>150m)",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.50},
            "moderate":  {"median": 0.45, "beta": 0.50},
            "extensive": {"median": 0.55, "beta": 0.50},
            "complete":  {"median": 0.80, "beta": 0.50},
        },
    },
    "HWB3": {
        "name": "Single Span - Concrete, Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.60},
            "moderate":  {"median": 0.90, "beta": 0.60},
            "extensive": {"median": 1.10, "beta": 0.60},
            "complete":  {"median": 1.50, "beta": 0.60},
        },
    },
    "HWB4": {
        "name": "Single Span - Concrete, Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.90, "beta": 0.55},
            "moderate":  {"median": 0.90, "beta": 0.55},
            "extensive": {"median": 1.10, "beta": 0.55},
            "complete":  {"median": 1.50, "beta": 0.55},
        },
    },
    "HWB5": {
        "name": "Multi-Span Concrete Continuous - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.65},
            "moderate":  {"median": 0.45, "beta": 0.65},
            "extensive": {"median": 0.55, "beta": 0.65},
            "complete":  {"median": 0.80, "beta": 0.65},
        },
    },
    "HWB6": {
        "name": "Multi-Span Concrete Continuous - Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.55},
            "moderate":  {"median": 0.90, "beta": 0.55},
            "extensive": {"median": 1.30, "beta": 0.55},
            "complete":  {"median": 1.60, "beta": 0.55},
        },
    },
    "HWB7": {
        "name": "Multi-Span Concrete Simply-Supported - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.70},
            "moderate":  {"median": 0.45, "beta": 0.70},
            "extensive": {"median": 0.55, "beta": 0.70},
            "complete":  {"median": 0.80, "beta": 0.70},
        },
    },
    "HWB8": {
        "name": "Multi-Span Concrete Simply-Supported - Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.55},
            "moderate":  {"median": 0.90, "beta": 0.55},
            "extensive": {"median": 1.30, "beta": 0.55},
            "complete":  {"median": 1.60, "beta": 0.55},
        },
    },
    "HWB10": {
        "name": "Multi-Span Steel Continuous - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.70},
            "moderate":  {"median": 0.45, "beta": 0.70},
            "extensive": {"median": 0.55, "beta": 0.70},
            "complete":  {"median": 0.80, "beta": 0.70},
        },
    },
    "HWB11": {
        "name": "Multi-Span Steel Continuous - Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.55},
            "moderate":  {"median": 0.90, "beta": 0.55},
            "extensive": {"median": 1.10, "beta": 0.55},
            "complete":  {"median": 1.50, "beta": 0.55},
        },
    },
    "HWB15": {
        "name": "Single Span - Steel, Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.60},
            "moderate":  {"median": 0.90, "beta": 0.60},
            "extensive": {"median": 1.10, "beta": 0.60},
            "complete":  {"median": 1.50, "beta": 0.60},
        },
    },
    "HWB16": {
        "name": "Single Span - Steel, Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.90, "beta": 0.55},
            "moderate":  {"median": 0.90, "beta": 0.55},
            "extensive": {"median": 1.10, "beta": 0.55},
            "complete":  {"median": 1.50, "beta": 0.55},
        },
    },
    "HWB17": {
        "name": "Multi-Span Concrete Continuous Box Girder - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.65},
            "moderate":  {"median": 0.45, "beta": 0.65},
            "extensive": {"median": 0.55, "beta": 0.65},
            "complete":  {"median": 0.80, "beta": 0.65},
        },
    },
    "HWB22": {
        "name": "Multi-Span Concrete Continuous Frame - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.65},
            "moderate":  {"median": 0.45, "beta": 0.65},
            "extensive": {"median": 0.55, "beta": 0.65},
            "complete":  {"median": 0.80, "beta": 0.65},
        },
    },
    "HWB28": {
        "name": "All Others",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.60},
            "moderate":  {"median": 0.45, "beta": 0.60},
            "extensive": {"median": 0.55, "beta": 0.60},
            "complete":  {"median": 0.80, "beta": 0.60},
        },
    },
}

DAMAGE_STATE_ORDER = ["slight", "moderate", "extensive", "complete"]

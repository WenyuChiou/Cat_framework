"""
Hazus 6.1 bridge fragility parameters (Table 7.9).

Each bridge class stores lognormal fragility parameters:
  - median: median Sa(1.0s) in g for each damage state
  - beta:   lognormal dispersion (total uncertainty)

Damage states follow Hazus definitions:
  - Slight:    minor cracking, spalling
  - Moderate:  moderate cracking, column damage
  - Extensive: degradation without collapse, significant displacement
  - Complete:  collapse or imminent collapse

Reference: FEMA Hazus 6.1 Earthquake Model Technical Manual, Table 7.9
"""

HAZUS_BRIDGE_FRAGILITY = {
    "HWB1": {
        "name": "Major Bridge - Conventional Design (>150m)",
        "damage_states": {
            "slight":    {"median": 0.25, "beta": 0.6},
            "moderate":  {"median": 0.35, "beta": 0.6},
            "extensive": {"median": 0.45, "beta": 0.6},
            "complete":  {"median": 0.70, "beta": 0.6},
        },
    },
    "HWB2": {
        "name": "Major Bridge - Seismic Design (>150m)",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.6},
            "moderate":  {"median": 0.45, "beta": 0.6},
            "extensive": {"median": 0.55, "beta": 0.6},
            "complete":  {"median": 0.80, "beta": 0.6},
        },
    },
    "HWB3": {
        "name": "Single Span - Concrete, Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.50, "beta": 0.6},
        },
    },
    "HWB4": {
        "name": "Single Span - Concrete, Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.90, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.50, "beta": 0.6},
        },
    },
    "HWB5": {
        "name": "Multi-Span Concrete Continuous - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.6},
            "moderate":  {"median": 0.45, "beta": 0.6},
            "extensive": {"median": 0.55, "beta": 0.6},
            "complete":  {"median": 0.80, "beta": 0.6},
        },
    },
    "HWB6": {
        "name": "Multi-Span Concrete Continuous - Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.30, "beta": 0.6},
            "complete":  {"median": 1.60, "beta": 0.6},
        },
    },
    "HWB7": {
        "name": "Multi-Span Concrete Simply-Supported - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.6},
            "moderate":  {"median": 0.45, "beta": 0.6},
            "extensive": {"median": 0.55, "beta": 0.6},
            "complete":  {"median": 0.80, "beta": 0.6},
        },
    },
    "HWB8": {
        "name": "Multi-Span Concrete Simply-Supported - Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.30, "beta": 0.6},
            "complete":  {"median": 1.60, "beta": 0.6},
        },
    },
    "HWB10": {
        "name": "Multi-Span Steel Continuous - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.6},
            "moderate":  {"median": 0.45, "beta": 0.6},
            "extensive": {"median": 0.55, "beta": 0.6},
            "complete":  {"median": 0.80, "beta": 0.6},
        },
    },
    "HWB11": {
        "name": "Multi-Span Steel Continuous - Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.50, "beta": 0.6},
        },
    },
    "HWB15": {
        "name": "Single Span - Steel, Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.50, "beta": 0.6},
        },
    },
    "HWB16": {
        "name": "Single Span - Steel, Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.90, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.50, "beta": 0.6},
        },
    },
    "HWB17": {
        "name": "Multi-Span Concrete Continuous Box Girder - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.6},
            "moderate":  {"median": 0.45, "beta": 0.6},
            "extensive": {"median": 0.55, "beta": 0.6},
            "complete":  {"median": 0.80, "beta": 0.6},
        },
    },
    "HWB22": {
        "name": "Multi-Span Concrete Continuous Frame - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.6},
            "moderate":  {"median": 0.45, "beta": 0.6},
            "extensive": {"median": 0.55, "beta": 0.6},
            "complete":  {"median": 0.80, "beta": 0.6},
        },
    },
    "HWB28": {
        "name": "All Others",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.6},
            "moderate":  {"median": 0.45, "beta": 0.6},
            "extensive": {"median": 0.55, "beta": 0.6},
            "complete":  {"median": 0.80, "beta": 0.6},
        },
    },
}

DAMAGE_STATE_ORDER = ["slight", "moderate", "extensive", "complete"]

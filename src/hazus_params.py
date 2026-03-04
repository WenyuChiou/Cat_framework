"""
Hazus 6.1 bridge fragility parameters (Table 7.9).

Each bridge class stores lognormal fragility parameters:
  - median: median Sa(1.0s) in g for each damage state
  - beta:   lognormal dispersion (total uncertainty), uniform 0.6 per Hazus

All 28 HWB classes (HWB1-HWB28) are included, covering both California
and non-California bridges with conventional and seismic design eras.

Damage states follow Hazus definitions:
  - Slight:    minor cracking, spalling
  - Moderate:  moderate cracking, column damage
  - Extensive: degradation without collapse, significant displacement
  - Complete:  collapse or imminent collapse

Reference: FEMA Hazus 6.1 Earthquake Model Technical Manual, Table 7.9
Cross-verified against: Sirisha Kedarsetty HazusTable.xlsx
"""

HAZUS_BRIDGE_FRAGILITY = {
    # ── Major Bridges (length > 150m) ──────────────────────────────────
    "HWB1": {
        "name": "Major Bridge - Conventional Design (>150m)",
        "damage_states": {
            "slight":    {"median": 0.40, "beta": 0.6},
            "moderate":  {"median": 0.50, "beta": 0.6},
            "extensive": {"median": 0.70, "beta": 0.6},
            "complete":  {"median": 0.90, "beta": 0.6},
        },
    },
    "HWB2": {
        "name": "Major Bridge - Seismic Design (>150m)",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.70, "beta": 0.6},
        },
    },
    # ── Single Span ────────────────────────────────────────────────────
    "HWB3": {
        "name": "Single Span - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.80, "beta": 0.6},
            "moderate":  {"median": 1.00, "beta": 0.6},
            "extensive": {"median": 1.20, "beta": 0.6},
            "complete":  {"median": 1.70, "beta": 0.6},
        },
    },
    "HWB4": {
        "name": "Single Span - Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.80, "beta": 0.6},
            "moderate":  {"median": 1.00, "beta": 0.6},
            "extensive": {"median": 1.20, "beta": 0.6},
            "complete":  {"median": 1.70, "beta": 0.6},
        },
    },
    # ── Multi-Col. Bent, Simple Support - Concrete ─────────────────────
    "HWB5": {
        "name": "Multi-Col. Bent, Simple Support - Concrete, Conventional (Non-CA)",
        "damage_states": {
            "slight":    {"median": 0.25, "beta": 0.6},
            "moderate":  {"median": 0.35, "beta": 0.6},
            "extensive": {"median": 0.45, "beta": 0.6},
            "complete":  {"median": 0.70, "beta": 0.6},
        },
    },
    "HWB6": {
        "name": "Multi-Col. Bent, Simple Support - Concrete, Conventional (CA)",
        "damage_states": {
            "slight":    {"median": 0.30, "beta": 0.6},
            "moderate":  {"median": 0.50, "beta": 0.6},
            "extensive": {"median": 0.60, "beta": 0.6},
            "complete":  {"median": 0.90, "beta": 0.6},
        },
    },
    "HWB7": {
        "name": "Multi-Col. Bent, Simple Support - Concrete, Seismic",
        "damage_states": {
            "slight":    {"median": 0.50, "beta": 0.6},
            "moderate":  {"median": 0.80, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.70, "beta": 0.6},
        },
    },
    # ── Single Col., Box Girder - Continuous Concrete (CA only) ────────
    "HWB8": {
        "name": "Single Col., Box Girder - Continuous Concrete, Conventional (CA)",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.6},
            "moderate":  {"median": 0.45, "beta": 0.6},
            "extensive": {"median": 0.55, "beta": 0.6},
            "complete":  {"median": 0.80, "beta": 0.6},
        },
    },
    "HWB9": {
        "name": "Single Col., Box Girder - Continuous Concrete, Seismic (CA)",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.30, "beta": 0.6},
            "complete":  {"median": 1.60, "beta": 0.6},
        },
    },
    # ── Continuous Concrete ────────────────────────────────────────────
    "HWB10": {
        "name": "Continuous Concrete - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.50, "beta": 0.6},
        },
    },
    "HWB11": {
        "name": "Continuous Concrete - Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.90, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.50, "beta": 0.6},
        },
    },
    # ── Multi-Col. Bent, Simple Support - Steel ────────────────────────
    "HWB12": {
        "name": "Multi-Col. Bent, Simple Support - Steel, Conventional (Non-CA)",
        "damage_states": {
            "slight":    {"median": 0.25, "beta": 0.6},
            "moderate":  {"median": 0.35, "beta": 0.6},
            "extensive": {"median": 0.45, "beta": 0.6},
            "complete":  {"median": 0.70, "beta": 0.6},
        },
    },
    "HWB13": {
        "name": "Multi-Col. Bent, Simple Support - Steel, Conventional (CA)",
        "damage_states": {
            "slight":    {"median": 0.30, "beta": 0.6},
            "moderate":  {"median": 0.50, "beta": 0.6},
            "extensive": {"median": 0.60, "beta": 0.6},
            "complete":  {"median": 0.90, "beta": 0.6},
        },
    },
    "HWB14": {
        "name": "Multi-Col. Bent, Simple Support - Steel, Seismic",
        "damage_states": {
            "slight":    {"median": 0.50, "beta": 0.6},
            "moderate":  {"median": 0.80, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.70, "beta": 0.6},
        },
    },
    # ── Continuous Steel ───────────────────────────────────────────────
    "HWB15": {
        "name": "Continuous Steel - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.75, "beta": 0.6},
            "moderate":  {"median": 0.75, "beta": 0.6},
            "extensive": {"median": 0.75, "beta": 0.6},
            "complete":  {"median": 1.10, "beta": 0.6},
        },
    },
    "HWB16": {
        "name": "Continuous Steel - Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.90, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.50, "beta": 0.6},
        },
    },
    # ── Multi-Col. Bent, Simple Support - Prestressed Concrete ─────────
    "HWB17": {
        "name": "Multi-Col. Bent, Simple Support - Prestressed Concrete, Conventional (Non-CA)",
        "damage_states": {
            "slight":    {"median": 0.25, "beta": 0.6},
            "moderate":  {"median": 0.35, "beta": 0.6},
            "extensive": {"median": 0.45, "beta": 0.6},
            "complete":  {"median": 0.70, "beta": 0.6},
        },
    },
    "HWB18": {
        "name": "Multi-Col. Bent, Simple Support - Prestressed Concrete, Conventional (CA)",
        "damage_states": {
            "slight":    {"median": 0.30, "beta": 0.6},
            "moderate":  {"median": 0.50, "beta": 0.6},
            "extensive": {"median": 0.60, "beta": 0.6},
            "complete":  {"median": 0.90, "beta": 0.6},
        },
    },
    "HWB19": {
        "name": "Multi-Col. Bent, Simple Support - Prestressed Concrete, Seismic",
        "damage_states": {
            "slight":    {"median": 0.50, "beta": 0.6},
            "moderate":  {"median": 0.80, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.70, "beta": 0.6},
        },
    },
    # ── Single Col., Box Girder - Prestressed Continuous Concrete (CA) ─
    "HWB20": {
        "name": "Single Col., Box Girder - Prestressed Continuous Concrete, Conventional (CA)",
        "damage_states": {
            "slight":    {"median": 0.35, "beta": 0.6},
            "moderate":  {"median": 0.45, "beta": 0.6},
            "extensive": {"median": 0.55, "beta": 0.6},
            "complete":  {"median": 0.80, "beta": 0.6},
        },
    },
    "HWB21": {
        "name": "Single Col., Box Girder - Prestressed Continuous Concrete, Seismic (CA)",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.30, "beta": 0.6},
            "complete":  {"median": 1.60, "beta": 0.6},
        },
    },
    # ── Continuous Prestressed Concrete ────────────────────────────────
    "HWB22": {
        "name": "Continuous Prestressed Concrete - Conventional Design",
        "damage_states": {
            "slight":    {"median": 0.60, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.50, "beta": 0.6},
        },
    },
    "HWB23": {
        "name": "Continuous Prestressed Concrete - Seismic Design",
        "damage_states": {
            "slight":    {"median": 0.90, "beta": 0.6},
            "moderate":  {"median": 0.90, "beta": 0.6},
            "extensive": {"median": 1.10, "beta": 0.6},
            "complete":  {"median": 1.50, "beta": 0.6},
        },
    },
    # ── Short Steel (length < 20m) ────────────────────────────────────
    "HWB24": {
        "name": "Multi-Col. Bent, Simple Support - Steel, Short (<20m), Conventional (Non-CA)",
        "damage_states": {
            "slight":    {"median": 0.25, "beta": 0.6},
            "moderate":  {"median": 0.35, "beta": 0.6},
            "extensive": {"median": 0.45, "beta": 0.6},
            "complete":  {"median": 0.70, "beta": 0.6},
        },
    },
    "HWB25": {
        "name": "Multi-Col. Bent, Simple Support - Steel, Short (<20m), Conventional (CA)",
        "damage_states": {
            "slight":    {"median": 0.30, "beta": 0.6},
            "moderate":  {"median": 0.50, "beta": 0.6},
            "extensive": {"median": 0.60, "beta": 0.6},
            "complete":  {"median": 0.90, "beta": 0.6},
        },
    },
    "HWB26": {
        "name": "Continuous Steel, Short (<20m), Conventional (Non-CA)",
        "damage_states": {
            "slight":    {"median": 0.75, "beta": 0.6},
            "moderate":  {"median": 0.75, "beta": 0.6},
            "extensive": {"median": 0.75, "beta": 0.6},
            "complete":  {"median": 1.10, "beta": 0.6},
        },
    },
    "HWB27": {
        "name": "Continuous Steel, Short (<20m), Conventional (CA)",
        "damage_states": {
            "slight":    {"median": 0.75, "beta": 0.6},
            "moderate":  {"median": 0.75, "beta": 0.6},
            "extensive": {"median": 0.75, "beta": 0.6},
            "complete":  {"median": 1.10, "beta": 0.6},
        },
    },
    # ── Catch-all ──────────────────────────────────────────────────────
    "HWB28": {
        "name": "All Other Bridges (unclassified)",
        "damage_states": {
            "slight":    {"median": 0.80, "beta": 0.6},
            "moderate":  {"median": 1.00, "beta": 0.6},
            "extensive": {"median": 1.20, "beta": 0.6},
            "complete":  {"median": 1.70, "beta": 0.6},
        },
    },
}

DAMAGE_STATE_ORDER = ["slight", "moderate", "extensive", "complete"]

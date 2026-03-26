"""
Bridge Inventory & Financial Exposure.

Provides the BridgeExposure dataclass, replacement cost estimation,
synthetic portfolio generation for demonstration, and conversion
from NBI data to exposure objects.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .hazard import SiteParams


# ── Replacement cost parameters ───────────────────────────────────────────

# Unit costs (USD per m² of deck area) by material, based on
# Caltrans and FHWA average bid data (simplified for demonstration).
REPLACEMENT_COST_PER_M2 = {
    "concrete":              2500.0,
    "steel":                 3200.0,
    "prestressed_concrete":  2800.0,
    "wood":                  1800.0,
    "other":                 2600.0,
}


# ── Dataclass ─────────────────────────────────────────────────────────────

@dataclass
class BridgeExposure:
    """Single bridge with location, classification, and financial data."""
    bridge_id: str
    lat: float
    lon: float
    hwb_class: str
    material: str = "concrete"
    length: float = 30.0          # meters
    deck_area: float = 300.0      # m²
    replacement_cost: float = 0.0 # USD
    vs30: float = 760.0           # m/s
    skew_angle: float = 0.0       # degrees


# ── Cost estimation ───────────────────────────────────────────────────────

def estimate_replacement_cost(
    material: str,
    deck_area: float,
    length: float,
) -> float:
    """
    Estimate bridge replacement cost from structural attributes.

    Uses unit deck-area cost with a length adjustment factor
    (longer bridges have higher per-unit costs due to foundations).

    Parameters
    ----------
    material : str
    deck_area : float  (m²)
    length : float  (m)

    Returns
    -------
    float
        Estimated replacement cost in USD.
    """
    unit_cost = REPLACEMENT_COST_PER_M2.get(material, 2600.0)
    # Length adjustment: bridges over 100m cost ~15% more per m²
    length_factor = 1.0 + 0.15 * max(0, (length - 100.0)) / 200.0
    return unit_cost * deck_area * length_factor


# ── FHWA 2024 cost estimation ─────────────────────────────────────────

def _lookup_factor(value: float, breaks: list[float], factors: list[float]) -> float:
    """Piecewise constant lookup: return factors[i] where breaks[i-1] <= value < breaks[i]."""
    if not breaks or value <= 0:
        return factors[0] if factors else 1.0
    for i, brk in enumerate(breaks):
        if value < brk:
            return factors[i]
    return factors[-1]


def estimate_replacement_cost_fhwa(
    deck_area: float,
    material: str = "other",
    max_span_length: float = 0.0,
    skew_angle: float = 0.0,
    year_built: int = 1970,
    cost_config=None,
) -> float:
    """
    Estimate bridge RCV using FHWA 2024 state-level unit costs
    with engineering adjustment factors.

    RCV = deck_area * base_unit_cost * f_material * f_span * f_skew * f_seismic * f_region

    Parameters
    ----------
    deck_area : float (m2)
    material : str
    max_span_length : float (m), 0 = unknown (factor=1.0)
    skew_angle : float (degrees), 0 = no skew
    year_built : int
    cost_config : CostConfig, optional
        If None, uses default CostConfig().

    Returns
    -------
    float : Estimated replacement cost in USD.

    References
    ----------
    Base rates: FHWA Bridge Replacement Unit Costs 2024
        (fhwa.dot.gov/bridge/nbi/sd2024.cfm)
    Material factors: Caltrans Comparative Bridge Costs 2022.
    Span/skew adjustments: Mackie & Stojadinovic (2010), EE&SD 39(3).
    """
    from .config import CostConfig
    if cost_config is None:
        cost_config = CostConfig()

    if deck_area <= 0:
        return 0.0

    base = cost_config.base_unit_cost
    f_material = cost_config.material_factors.get(material, 1.0)
    f_span = _lookup_factor(max_span_length, cost_config.span_breaks, cost_config.span_factors)
    f_skew = _lookup_factor(skew_angle, cost_config.skew_breaks, cost_config.skew_factors)

    if year_built < 1975:
        f_seismic = cost_config.seismic_era_factors.get("pre_1975", 1.0)
    elif year_built <= 1990:
        f_seismic = cost_config.seismic_era_factors.get("1975_1990", 1.10)
    else:
        f_seismic = cost_config.seismic_era_factors.get("post_1990", 1.15)

    f_region = cost_config.region_factor

    return deck_area * base * f_material * f_span * f_skew * f_seismic * f_region


# ── Synthetic portfolio ───────────────────────────────────────────────────

# Realistic class distribution for Southern California (Northridge area)
_CLASS_WEIGHTS = {
    "HWB3":  0.12, "HWB4":  0.08,
    "HWB5":  0.14, "HWB6":  0.06,
    "HWB7":  0.10, "HWB8":  0.04,
    "HWB10": 0.05, "HWB11": 0.03,
    "HWB15": 0.06, "HWB16": 0.04,
    "HWB17": 0.10, "HWB22": 0.08,
    "HWB28": 0.10,
}

_MATERIAL_FOR_CLASS = {
    "HWB3": "concrete", "HWB4": "concrete",
    "HWB5": "concrete", "HWB6": "concrete",
    "HWB7": "concrete", "HWB8": "concrete",
    "HWB10": "steel",   "HWB11": "steel",
    "HWB15": "steel",   "HWB16": "steel",
    "HWB17": "concrete", "HWB22": "concrete",
    "HWB28": "other",
}


def generate_synthetic_portfolio(
    n_bridges: int = 100,
    center: tuple[float, float] = (34.213, -118.537),
    radius_km: float = 30.0,
    seed: int = 42,
) -> list[BridgeExposure]:
    """
    Generate a synthetic bridge portfolio around a center point.

    Parameters
    ----------
    n_bridges : int
        Number of bridges to generate.
    center : (lat, lon)
        Geographic center of the portfolio.
    radius_km : float
        Approximate radius for bridge placement.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[BridgeExposure]
    """
    rng = np.random.default_rng(seed)

    classes = list(_CLASS_WEIGHTS.keys())
    weights = np.array(list(_CLASS_WEIGHTS.values()))
    weights /= weights.sum()

    portfolio = []
    for i in range(n_bridges):
        # Random class
        hwb = rng.choice(classes, p=weights)
        material = _MATERIAL_FOR_CLASS[hwb]

        # Random location (uniform in disc, ~approximate in lat/lon)
        angle = rng.uniform(0, 2 * np.pi)
        r = radius_km * np.sqrt(rng.uniform())
        dlat = r / 111.0  # ~111 km per degree latitude
        dlon = r / (111.0 * np.cos(np.radians(center[0])))
        lat = center[0] + dlat * np.sin(angle)
        lon = center[1] + dlon * np.cos(angle)

        # Random structural attributes
        length = rng.uniform(15, 200) if "HWB1" in hwb or "HWB2" in hwb else rng.uniform(15, 120)
        width = rng.uniform(8, 25)
        deck_area = length * width
        vs30 = rng.uniform(200, 800)
        skew = rng.choice([0, 0, 0, 10, 20, 30, 45])  # Most bridges not skewed

        cost = estimate_replacement_cost(material, deck_area, length)

        portfolio.append(BridgeExposure(
            bridge_id=f"SYN-{i+1:04d}",
            lat=lat,
            lon=lon,
            hwb_class=hwb,
            material=material,
            length=length,
            deck_area=deck_area,
            replacement_cost=cost,
            vs30=vs30,
            skew_angle=skew,
        ))

    return portfolio


# ── NBI conversion ────────────────────────────────────────────────────────

def create_portfolio_from_nbi(
    nbi_df,
    default_vs30: float = 360.0,
    cost_config=None,
) -> list[BridgeExposure]:
    """
    Convert a classified NBI DataFrame to a list of BridgeExposure objects.

    The NBI DataFrame must already have an 'hwb_class' column
    (from data_loader.classify_nbi_to_hazus).

    Parameters
    ----------
    nbi_df : pd.DataFrame
        Classified NBI data with columns: structure_number, latitude,
        longitude, hwb_class, material, structure_length_m, deck_width_m.
        Optional: max_span_length_m, skew_angle, year_built.
    default_vs30 : float
        Default Vs30 when site data is unavailable.
    cost_config : CostConfig, optional
        If provided, uses FHWA 2024 multi-factor cost model.
        If None, uses legacy material × deck_area model.

    Returns
    -------
    list[BridgeExposure]
    """
    import pandas as pd

    portfolio = []
    for _, row in nbi_df.iterrows():
        length = row.get("structure_length_m", 30.0)
        if pd.isna(length):
            length = 30.0
        width = row.get("deck_width_m", 10.0)
        if pd.isna(width):
            width = 10.0
        deck_area = length * width
        material = row.get("material", "other")

        # New NBI fields (safe defaults when absent or NaN)
        max_span = row.get("max_span_length_m", 0.0)
        if pd.isna(max_span):
            max_span = 0.0
        skew = row.get("skew_angle", 0.0)
        if pd.isna(skew):
            skew = 0.0
        year = row.get("year_built", 1970)
        if pd.isna(year):
            year = 1970

        if cost_config is not None:
            cost = estimate_replacement_cost_fhwa(
                deck_area, material, float(max_span),
                float(skew), int(year), cost_config,
            )
        else:
            cost = estimate_replacement_cost(material, deck_area, length)

        portfolio.append(BridgeExposure(
            bridge_id=str(row.get("structure_number", "")),
            lat=float(row["latitude"]),
            lon=float(row["longitude"]),
            hwb_class=row["hwb_class"],
            material=material,
            length=float(length),
            deck_area=float(deck_area),
            replacement_cost=cost,
            vs30=default_vs30,
            skew_angle=float(skew),
        ))

    return portfolio


# ── Portfolio utilities ───────────────────────────────────────────────────

def portfolio_summary(portfolio: list[BridgeExposure]) -> dict:
    """Aggregate statistics for a bridge portfolio."""
    if not portfolio:
        return {"n_bridges": 0}

    costs = [b.replacement_cost for b in portfolio]
    classes = {}
    materials = {}
    for b in portfolio:
        classes[b.hwb_class] = classes.get(b.hwb_class, 0) + 1
        materials[b.material] = materials.get(b.material, 0) + 1

    return {
        "n_bridges": len(portfolio),
        "total_replacement_cost": sum(costs),
        "avg_replacement_cost": np.mean(costs),
        "class_distribution": dict(sorted(classes.items())),
        "material_distribution": dict(sorted(materials.items())),
    }


def filter_portfolio(
    portfolio: list[BridgeExposure],
    hwb_classes: list[str] | None = None,
    materials: list[str] | None = None,
) -> list[BridgeExposure]:
    """
    Filter a portfolio by HWB class and/or material type.

    Parameters
    ----------
    portfolio : list[BridgeExposure]
    hwb_classes : list[str], optional
        Keep only bridges with these HWB classes.
    materials : list[str], optional
        Keep only bridges with these materials.

    Returns
    -------
    list[BridgeExposure]
    """
    result = portfolio
    if hwb_classes:
        result = [b for b in result if b.hwb_class in hwb_classes]
    if materials:
        result = [b for b in result if b.material in materials]
    return result


def portfolio_to_sites(portfolio: list[BridgeExposure]) -> list[SiteParams]:
    """Convert portfolio to list of SiteParams for hazard computation."""
    return [
        SiteParams(lat=b.lat, lon=b.lon, vs30=b.vs30)
        for b in portfolio
    ]

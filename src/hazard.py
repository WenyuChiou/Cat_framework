"""
Ground Motion Prediction — Boore & Atkinson (2008) GMPE.

Implements the BA08 ground motion prediction equation for Sa(1.0s),
spatial correlation model (Jayaram & Baker 2009), and correlated
ground motion field generation via Cholesky decomposition.

References:
  - Boore, D.M. & Atkinson, G.M. (2008). Ground-Motion Prediction
    Equations for the Average Horizontal Component of PGA, PGV, and
    5%-Damped PSA at Spectral Periods between 0.01s and 10.0s.
    Earthquake Spectra, 24(1), 99-138.
  - Jayaram, N. & Baker, J.W. (2009). Correlation model for spatially
    distributed ground-motion intensities. Earthquake Engineering &
    Structural Dynamics, 38(15), 1687-1708.
"""

import math
from dataclasses import dataclass

import numpy as np


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass
class EarthquakeScenario:
    """Earthquake rupture parameters."""
    Mw: float
    lat: float
    lon: float
    depth_km: float = 10.0
    fault_type: str = "reverse"  # "strike_slip", "normal", "reverse", "unspecified"


@dataclass
class SiteParams:
    """Single site location and soil class."""
    lat: float
    lon: float
    vs30: float = 760.0  # m/s, NEHRP B/C boundary


# ── Distance calculation ──────────────────────────────────────────────────

_EARTH_RADIUS_KM = 6371.0


def haversine_distance_km(lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth (km)."""
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2)
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# ── BA08 GMPE for Sa(1.0s) ───────────────────────────────────────────────

# Coefficients for T = 1.0s from Boore & Atkinson (2008) Table 3.
# Notation follows the paper: e1–e7, Mh, c1–c3, h, blin, b1, b2, V1, V2, Vref,
# sigma, tau, sigma_T
_BA08_T10 = {
    "e1": -0.23898,   # unspecified mechanism
    "e2": -0.28892,   # strike-slip
    "e3":  0.00000,   # normal (not given; use e1)
    "e4": -0.20608,   # reverse
    "e5":  0.09228,
    "e6": -0.06768,
    "e7":  0.01000,
    "Mh":  6.75,
    "c1": -0.68898,
    "c2":  0.21521,
    "c3": -0.00707,
    "h":   2.54,
    "blin": -0.60,
    "b1":  -0.50,
    "b2":  -0.06,
    "V1":  180.0,
    "V2":  300.0,
    "Vref": 760.0,
    "sigma": 0.502,
    "tau":   0.255,
    "sigma_T": 0.564,
}


def boore_atkinson_2008_sa10(
    Mw: float,
    R_JB: float,
    Vs30: float = 760.0,
    fault_type: str = "reverse",
) -> tuple[float, float]:
    """
    BA08 GMPE for Sa(1.0s), 5%-damped horizontal component.

    Parameters
    ----------
    Mw : float
        Moment magnitude.
    R_JB : float
        Joyner-Boore distance (km).  For point source approximation
        this equals epicentral distance.
    Vs30 : float
        Time-averaged shear-wave velocity in top 30m (m/s).
    fault_type : str
        "strike_slip", "normal", "reverse", or "unspecified".

    Returns
    -------
    (median_sa_g, sigma_ln) : tuple[float, float]
        Median Sa(1.0s) in g and total aleatory sigma (natural log).
    """
    c = _BA08_T10

    # ── Source (magnitude) term F_M ──
    if fault_type == "strike_slip":
        e_val = c["e2"]
    elif fault_type == "reverse":
        e_val = c["e4"]
    elif fault_type == "normal":
        e_val = c["e3"] if c["e3"] != 0.0 else c["e1"]
    else:
        e_val = c["e1"]

    U, SS, NS, RS = 0, 0, 0, 0
    if fault_type == "strike_slip":
        SS = 1
    elif fault_type == "normal":
        NS = 1
    elif fault_type == "reverse":
        RS = 1
    else:
        U = 1

    Mh = c["Mh"]
    if Mw <= Mh:
        F_M = (e_val * U + c["e2"] * SS + c["e3"] * NS + c["e4"] * RS
               + c["e5"] * (Mw - Mh) + c["e6"] * (Mw - Mh) ** 2)
    else:
        F_M = (e_val * U + c["e2"] * SS + c["e3"] * NS + c["e4"] * RS
               + c["e7"] * (Mw - Mh))

    # ── Distance term F_D ──
    R = math.sqrt(R_JB ** 2 + c["h"] ** 2)
    F_D = (c["c1"] + c["c2"] * (Mw - c["Mh"])) * math.log(R) + c["c3"] * (R - 1.0)

    # ── Site amplification term F_S ──
    # Linear site term
    Vref = c["Vref"]
    blin = c["blin"]
    F_lin = blin * math.log(min(Vs30, Vref) / Vref)

    # Non-linear site term (simplified: compute PGA_ref on rock first)
    # For the non-linear term we need PGA on reference rock (Vs30=760).
    # We approximate by setting F_S_nl = 0 for Vs30 >= 300 (weak non-linearity)
    # and applying full non-linear correction for softer soils.
    F_nl = 0.0
    if Vs30 < c["V2"]:
        # Estimate rock PGA from a simplified distance scaling
        pga_ref = _estimate_pga_ref(Mw, R_JB, fault_type)
        bnl = c["b1"] if Vs30 <= c["V1"] else (
            c["b1"] + (c["b2"] - c["b1"]) *
            math.log(Vs30 / c["V1"]) / math.log(c["V2"] / c["V1"])
        )
        # Transition function
        a1 = 0.03
        a2 = 0.09
        dx = math.log(a2 / a1)
        if pga_ref <= a1:
            F_nl = bnl * math.log(a1 / 0.1)
        elif pga_ref <= a2:
            f_pga = math.log(pga_ref / a1) / dx
            F_nl = bnl * math.log(a1 / 0.1) + (
                bnl * (math.log(pga_ref / 0.1) - math.log(a1 / 0.1))
            ) * f_pga
        else:
            F_nl = bnl * math.log(pga_ref / 0.1)

    F_S = F_lin + F_nl

    # ── Combine ──
    ln_sa = F_M + F_D + F_S
    median_sa_g = math.exp(ln_sa)
    sigma_ln = c["sigma_T"]

    return median_sa_g, sigma_ln


def _estimate_pga_ref(Mw: float, R_JB: float, fault_type: str) -> float:
    """Rough PGA on reference rock for non-linear site correction."""
    # Simplified BA08 PGA coefficients (T=0, Vs30=760)
    if fault_type == "reverse":
        e_val = 0.03707
    elif fault_type == "strike_slip":
        e_val = -0.03279
    else:
        e_val = -0.01231
    Mh = 6.75
    if Mw <= Mh:
        F_M = e_val + 0.09841 * (Mw - Mh) - 0.00760 * (Mw - Mh) ** 2
    else:
        F_M = e_val + 0.01000 * (Mw - Mh)
    R = math.sqrt(R_JB ** 2 + 1.35 ** 2)
    F_D = (-0.66050 + 0.11311 * (Mw - Mh)) * math.log(R) - 0.01151 * (R - 1.0)
    return math.exp(F_M + F_D)


# ── Vectorized site computation ───────────────────────────────────────────

def compute_sa_at_sites(
    scenario: EarthquakeScenario,
    sites: list[SiteParams],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute median Sa(1.0s) and sigma at each site.

    Returns
    -------
    (medians, sigmas) : tuple of 1-D arrays, length = len(sites)
    """
    n = len(sites)
    medians = np.empty(n)
    sigmas = np.empty(n)
    for i, site in enumerate(sites):
        R_JB = haversine_distance_km(scenario.lat, scenario.lon,
                                     site.lat, site.lon)
        # Minimum distance clamp (point-source simplification)
        R_JB = max(R_JB, 0.1)
        med, sig = boore_atkinson_2008_sa10(
            scenario.Mw, R_JB, site.vs30, scenario.fault_type
        )
        medians[i] = med
        sigmas[i] = sig
    return medians, sigmas


# ── Spatial correlation (Jayaram-Baker 2009) ──────────────────────────────

def spatial_correlation_matrix(
    sites: list[SiteParams],
    period: float = 1.0,
) -> np.ndarray:
    """
    Compute intra-event spatial correlation matrix (Jayaram-Baker 2009).

    For T = 1.0s: rho(h) = exp(-3h / b), where b = 40.7 km.

    Parameters
    ----------
    sites : list[SiteParams]
    period : float
        Spectral period (s).  Controls correlation range b.

    Returns
    -------
    np.ndarray
        (N, N) correlation matrix.
    """
    # Correlation range b (km) from Jayaram-Baker (2009) Eq. 16–17
    if period < 1.0:
        b = 8.5 + 17.2 * period
    else:
        b = 22.0 + 3.7 * period  # For T>=1s

    # For T=1.0 exactly: b = 22.0 + 3.7*1.0 = 25.7 km
    # (some implementations use b=40.7 for T=1.0; we follow the paper's Eq. 17)
    # Using the semi-variogram: b = 40.7 for T=1.0s from Table 1 of J-B 2009
    if abs(period - 1.0) < 0.01:
        b = 40.7

    n = len(sites)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            h = haversine_distance_km(
                sites[i].lat, sites[i].lon,
                sites[j].lat, sites[j].lon,
            )
            rho = math.exp(-3.0 * h / b)
            corr[i, j] = rho
            corr[j, i] = rho
    return corr


# ── Ground motion field generation ────────────────────────────────────────

def generate_ground_motion_field(
    scenario: EarthquakeScenario,
    sites: list[SiteParams],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a single realization of correlated Sa(1.0s) values.

    Uses inter-event and spatially-correlated intra-event variability.

    Parameters
    ----------
    scenario : EarthquakeScenario
    sites : list[SiteParams]
    rng : numpy random Generator, optional

    Returns
    -------
    np.ndarray
        Sa(1.0s) in g at each site, shape (N,).
    """
    if rng is None:
        rng = np.random.default_rng()

    medians, _ = compute_sa_at_sites(scenario, sites)
    tau = _BA08_T10["tau"]
    sigma = _BA08_T10["sigma"]

    # Inter-event residual (same for all sites)
    eta = rng.normal(0.0, 1.0) * tau

    # Spatially-correlated intra-event residuals
    corr = spatial_correlation_matrix(sites)
    L = np.linalg.cholesky(corr + 1e-10 * np.eye(len(sites)))
    z = rng.normal(0.0, 1.0, size=len(sites))
    eps = L @ z * sigma

    # Combine: ln(Sa) = ln(median) + eta + eps
    ln_sa = np.log(medians) + eta + eps
    return np.exp(ln_sa)


def generate_ground_motion_fields(
    scenario: EarthquakeScenario,
    sites: list[SiteParams],
    n_realizations: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate multiple correlated ground motion field realizations.

    Parameters
    ----------
    scenario : EarthquakeScenario
    sites : list[SiteParams]
    n_realizations : int
    seed : int, optional

    Returns
    -------
    np.ndarray
        Shape (n_realizations, n_sites), Sa(1.0s) in g.
    """
    rng = np.random.default_rng(seed)
    medians, _ = compute_sa_at_sites(scenario, sites)

    tau = _BA08_T10["tau"]
    sigma = _BA08_T10["sigma"]

    # Pre-compute Cholesky of correlation matrix
    corr = spatial_correlation_matrix(sites)
    n_sites = len(sites)
    L = np.linalg.cholesky(corr + 1e-10 * np.eye(n_sites))

    fields = np.empty((n_realizations, n_sites))
    ln_medians = np.log(medians)

    for k in range(n_realizations):
        eta = rng.normal() * tau
        z = rng.normal(size=n_sites)
        eps = L @ z * sigma
        fields[k, :] = np.exp(ln_medians + eta + eps)

    return fields

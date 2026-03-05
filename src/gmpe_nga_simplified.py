"""
Simplified NGA GMPE models for comparison and testing.

Contains 7 GMPE models with representative (approximate) coefficients,
adapted from Kubilay Albayrak's unified implementation. These are
SIMPLIFIED versions intended for:
  - Cross-model comparison and visualization
  - Sensitivity testing (Vs30 variation, distance attenuation)
  - Educational/demonstration purposes

WARNING: These use representative coefficients, NOT the official published
         values. For production risk calculations, use the full BSSA21
         implementation in gmpe_bssa21.py.

Models included:
  1. ASK14  — Abrahamson, Silva & Kamai (2014)
  2. BSSA14 — Boore, Stewart, Seyhan & Atkinson (2014)
  3. CB14   — Campbell & Bozorgnia (2014)
  4. CY14   — Chiou & Youngs (2014)
  5. ID14   — Idriss (2014)
  6. GK15   — Graizer & Kalkan (2015)
  7. NGA-East — Generic NGA-East (seed models composite)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.gmpe_base import register_gmpe


# ── Representative coefficients per model ────────────────────────────────

@dataclass(frozen=True)
class _SimplifiedCoeffs:
    """Container for simplified GMPE coefficients."""
    name: str
    sigma: float  # total sigma in ln units


# ── Base class ───────────────────────────────────────────────────────────

class SimplifiedGMPE:
    """Base for simplified GMPE models (PGA only, representative coeffs)."""

    _VREF = 760.0  # reference Vs30 (m/s)

    def __init__(self, model_name: str, sigma: float):
        self._name = model_name
        self._sigma = sigma

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_periods(self) -> list[float]:
        return [0.0]  # PGA only

    def compute(
        self,
        Mw: float,
        R_JB: float,
        Vs30: float,
        fault_type: str = "unspecified",
        period: float = 0.0,
    ) -> tuple[float, float]:
        if period != 0.0:
            raise ValueError(
                f"{self._name} simplified model supports PGA only (period=0.0), "
                f"got period={period}"
            )
        ln_y = self._compute_ln_pga(Mw, R_JB, Vs30, fault_type)
        return math.exp(ln_y), self._sigma

    def _compute_ln_pga(
        self, Mw: float, R_JB: float, Vs30: float, fault_type: str
    ) -> float:
        raise NotImplementedError


# ── 1. ASK14 ─────────────────────────────────────────────────────────────

class ASK14(SimplifiedGMPE):
    """Abrahamson, Silva & Kamai (2014) — simplified."""

    def __init__(self):
        super().__init__("ask14", sigma=0.60)

    def _compute_ln_pga(self, Mw, R_JB, Vs30, fault_type):
        a1, a2, a3 = 1.581, 0.512, -0.1
        a4, a5, a6 = -2.118, 0.17, -0.003
        b1, b2, c = -0.6, -0.3, 0.1
        h = 4.5
        Mref = 6.0

        # Fault type flags
        FRV = 1.0 if fault_type == "reverse" else 0.0
        FNM = 1.0 if fault_type == "normal" else 0.0
        a7, a8 = 0.1, -0.05

        f_mag = a1 + a2 * (Mw - Mref) + a3 * (Mw - Mref) ** 2
        f_dist = (a4 + a5 * Mw) * math.log(math.sqrt(R_JB**2 + h**2)) + a6 * R_JB
        f_flt = a7 * FRV + a8 * FNM

        Yref = math.exp(f_mag + f_dist + f_flt)
        f_site = b1 * math.log(Vs30 / self._VREF) + b2 * math.log((Yref + c) / c)

        return f_mag + f_dist + f_flt + f_site


# ── 2. BSSA14 (simplified) ───────────────────────────────────────────────

class BSSA14_Simplified(SimplifiedGMPE):
    """Boore, Stewart, Seyhan & Atkinson (2014) — simplified."""

    def __init__(self):
        super().__init__("bssa14_simplified", sigma=0.58)

    def _compute_ln_pga(self, Mw, R_JB, Vs30, fault_type):
        e0, e1, e2 = -0.5, 0.9, -0.12
        c1, c2, c3 = -1.5, 0.1, -0.002
        c4, c5 = -0.5, -0.8
        Mh = 6.0
        h = 4.5
        Vc = 1000.0

        f_E = e0 + e1 * (Mw - Mh) + e2 * (Mw - Mh) ** 2
        f_P = (c1 + c2 * (Mw - Mh)) * math.log(R_JB + h) + c3 * R_JB

        if Vs30 > Vc:
            f_S = c4 * math.log(Vs30 / self._VREF)
        else:
            f_S = c4 * math.log(Vc / self._VREF) + c5 * math.log(Vs30 / Vc)

        return f_E + f_P + f_S


# ── 3. CB14 ──────────────────────────────────────────────────────────────

class CB14(SimplifiedGMPE):
    """Campbell & Bozorgnia (2014) — simplified."""

    def __init__(self):
        super().__init__("cb14", sigma=0.62)

    def _compute_ln_pga(self, Mw, R_JB, Vs30, fault_type):
        c0, c1, c2 = -1.715, 0.5, -0.05
        c3, c4, c5 = -2.0, 0.2, -0.003
        c8, c9, k1 = -0.6, -0.4, 0.1
        h = 4.0

        f_mag = c0 + c1 * Mw + c2 * Mw ** 2
        f_dis = (c3 + c4 * Mw) * math.log(math.sqrt(R_JB**2 + h**2)) + c5 * R_JB

        Yref = math.exp(f_mag + f_dis)
        f_site = c8 * math.log(Vs30 / self._VREF) + c9 * math.log((Yref + k1) / k1)

        return f_mag + f_dis + f_site


# ── 4. CY14 ──────────────────────────────────────────────────────────────

class CY14(SimplifiedGMPE):
    """Chiou & Youngs (2014) — simplified."""

    def __init__(self):
        super().__init__("cy14", sigma=0.60)

    def _compute_ln_pga(self, Mw, R_JB, Vs30, fault_type):
        c1, c2, c3 = -1.2, 0.8, -0.1
        c4, c5 = -1.7, 0.15
        c6, c7 = 5.0, 0.3
        c8 = -0.002
        c11, c12 = -0.6, -0.3
        c_nl = 0.1

        f_main = (
            c1 + c2 * (Mw - 6) + c3 * (Mw - 6) ** 2
            + (c4 + c5 * Mw) * math.log(R_JB + c6 * math.exp(c7 * Mw))
            + c8 * R_JB
        )

        Yref = math.exp(f_main)
        f_site = c11 * math.log(Vs30 / self._VREF) + c12 * math.log((Yref + c_nl) / c_nl)

        return f_main + f_site


# ── 5. Idriss (2014) ────────────────────────────────────────────────────

class Idriss14(SimplifiedGMPE):
    """Idriss (2014) — simplified."""

    def __init__(self):
        super().__init__("idriss14", sigma=0.55)

    def _compute_ln_pga(self, Mw, R_JB, Vs30, fault_type):
        a, b, c, d = -1.715, 0.5, 0.003, -0.4
        return a + b * Mw - math.log(R_JB + 0.1) - c * R_JB + d * math.log(Vs30 / self._VREF)


# ── 6. GK15 ──────────────────────────────────────────────────────────────

class GK15(SimplifiedGMPE):
    """Graizer & Kalkan (2015) — simplified."""

    def __init__(self):
        super().__init__("gk15", sigma=0.65)

    def _compute_ln_pga(self, Mw, R_JB, Vs30, fault_type):
        a0, a1, a2 = -1.5, 0.8, -1.1
        b_coef, c_coef = 0.2, 5.0
        Q0, s = 0.002, -0.5

        return (
            a0
            + a1 * (Mw - 6)
            + a2 * math.log(R_JB + c_coef * math.exp(b_coef * Mw))
            - Q0 * R_JB
            + s * math.log(Vs30 / self._VREF)
        )


# ── 7. NGA-East ──────────────────────────────────────────────────────────

class NGAEast(SimplifiedGMPE):
    """Generic NGA-East composite — simplified."""

    def __init__(self):
        super().__init__("nga_east", sigma=0.60)

    def _compute_ln_pga(self, Mw, R_JB, Vs30, fault_type):
        c1, c2, c3 = -1.8, 0.1, -0.001
        h = 10.0

        f_R = (c1 + c2 * Mw) * math.log(R_JB + h) + c3 * R_JB
        f_S = -0.5 * math.log(Vs30 / self._VREF)

        return f_R + f_S


# ── Convenience: all simplified models ───────────────────────────────────

ALL_SIMPLIFIED_MODELS: dict[str, SimplifiedGMPE] = {}


def _init_models():
    """Instantiate and register all simplified models."""
    models = [ASK14(), BSSA14_Simplified(), CB14(), CY14(),
              Idriss14(), GK15(), NGAEast()]
    for m in models:
        ALL_SIMPLIFIED_MODELS[m.name] = m
        register_gmpe(m)


_init_models()


# ── Comparison utilities ─────────────────────────────────────────────────

def compare_models(
    Mw: float,
    R_JB: float,
    Vs30: float,
    fault_type: str = "unspecified",
) -> dict[str, dict]:
    """Compute PGA from all simplified models for one scenario.

    Returns dict[model_name] -> {"pga_g": float, "sigma": float, "ln_pga": float}
    """
    results = {}
    for name, model in ALL_SIMPLIFIED_MODELS.items():
        pga, sigma = model.compute(Mw, R_JB, Vs30, fault_type)
        results[name] = {
            "pga_g": pga,
            "sigma": sigma,
            "ln_pga": math.log(pga),
        }
    return results


def vs30_sensitivity(
    Mw: float,
    R_JB: float,
    vs30_values: list[float] | np.ndarray | None = None,
    fault_type: str = "unspecified",
) -> dict[str, np.ndarray]:
    """Compute PGA across a range of Vs30 values for all models.

    Parameters
    ----------
    vs30_values : array-like, optional
        Vs30 values to test. Default: 150 to 1500 m/s (50 points).

    Returns
    -------
    dict with keys: "vs30", plus one key per model name containing PGA array.
    """
    if vs30_values is None:
        vs30_values = np.linspace(150, 1500, 50)
    else:
        vs30_values = np.asarray(vs30_values)

    results: dict[str, np.ndarray] = {"vs30": vs30_values}
    for name, model in ALL_SIMPLIFIED_MODELS.items():
        pga_arr = np.array([
            model.compute(Mw, R_JB, float(v), fault_type)[0]
            for v in vs30_values
        ])
        results[name] = pga_arr
    return results


def attenuation_curves(
    Mw: float,
    Vs30: float,
    distances: np.ndarray | None = None,
    fault_type: str = "unspecified",
) -> dict[str, np.ndarray]:
    """Compute PGA attenuation curves for all models.

    Parameters
    ----------
    distances : array-like, optional
        R_JB values (km). Default: 1 to 200 km (100 log-spaced points).

    Returns
    -------
    dict with keys: "distance", plus one key per model name containing PGA array.
    """
    if distances is None:
        distances = np.logspace(0, np.log10(200), 100)
    else:
        distances = np.asarray(distances)

    results: dict[str, np.ndarray] = {"distance": distances}
    for name, model in ALL_SIMPLIFIED_MODELS.items():
        pga_arr = np.array([
            model.compute(Mw, float(r), Vs30, fault_type)[0]
            for r in distances
        ])
        results[name] = pga_arr
    return results


# ── CLI demo ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Simplified NGA GMPE Comparison (representative coefficients)")
    print("WARNING: For testing/comparison only — not for production use")
    print("=" * 70)

    # Test scenario
    Mw, R_JB, Vs30 = 6.7, 20.0, 360.0
    print(f"\nScenario: Mw={Mw}, R_JB={R_JB} km, Vs30={Vs30} m/s\n")
    print(f"{'Model':<20} {'PGA (g)':>10} {'sigma':>8} {'ln(PGA)':>10}")
    print("-" * 50)

    results = compare_models(Mw, R_JB, Vs30)
    for name, r in results.items():
        print(f"{name:<20} {r['pga_g']:>10.4f} {r['sigma']:>8.2f} {r['ln_pga']:>10.4f}")

    # Vs30 sensitivity
    print(f"\n{'─' * 70}")
    print("Vs30 Sensitivity (same M, R):\n")
    vs30_test = [200, 360, 500, 760, 1000]
    print(f"{'Model':<20}", end="")
    for v in vs30_test:
        print(f" {'Vs30='+str(v):>10}", end="")
    print()
    print("-" * 72)

    for name, model in ALL_SIMPLIFIED_MODELS.items():
        print(f"{name:<20}", end="")
        for v in vs30_test:
            pga, _ = model.compute(Mw, R_JB, float(v))
            print(f" {pga:>10.4f}", end="")
        print()

    # Include BSSA21 full model for comparison if available
    print(f"\n{'─' * 70}")
    print("Comparison with full BSSA21 (official coefficients):\n")
    try:
        import src.gmpe_bssa21  # noqa: F401
        from src.gmpe_base import get_gmpe
        bssa21 = get_gmpe("bssa21")
        pga_full, sigma_full = bssa21.compute(Mw, R_JB, Vs30, "unspecified", 0.0)
        pga_simp, sigma_simp = ALL_SIMPLIFIED_MODELS["bssa14_simplified"].compute(
            Mw, R_JB, Vs30
        )
        print(f"  BSSA21 (full):       PGA = {pga_full:.4f} g, sigma = {sigma_full:.4f}")
        print(f"  BSSA14 (simplified): PGA = {pga_simp:.4f} g, sigma = {sigma_simp:.2f}")
        diff_pct = abs(pga_full - pga_simp) / pga_full * 100
        print(f"  Difference: {diff_pct:.1f}%")
    except Exception as e:
        print(f"  (Could not load BSSA21: {e})")

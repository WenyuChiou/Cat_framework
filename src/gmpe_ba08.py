"""
BA08 GMPE wrapper — Boore & Atkinson (2008) as a pluggable GMPEModel.

Thin adapter around the existing BA08 implementation in src/hazard.py,
exposing it through the GMPE protocol/registry system.

Supports PGA (period=0.0) and Sa(1.0s) (period=1.0).
"""

from __future__ import annotations

import math

from src.gmpe_base import GMPEModel, register_gmpe
from src.hazard import boore_atkinson_2008_sa10, _estimate_pga_ref


class BA08:
    """Boore & Atkinson (2008) NGA-West1 GMPE wrapper."""

    @property
    def name(self) -> str:
        return "ba08"

    @property
    def supported_periods(self) -> list[float]:
        return [0.0, 1.0]

    def compute(
        self,
        Mw: float,
        R_JB: float,
        Vs30: float,
        fault_type: str,
        period: float,
    ) -> tuple[float, float]:
        """Compute median SA (g) and total sigma (ln units).

        Parameters
        ----------
        Mw, R_JB, Vs30, fault_type : see GMPEModel protocol.
        period : float
            0.0 for PGA, 1.0 for Sa(1.0s).

        Returns
        -------
        (median_g, sigma_ln)
        """
        if abs(period - 1.0) < 1e-6:
            return boore_atkinson_2008_sa10(Mw, R_JB, Vs30, fault_type)
        elif abs(period) < 1e-6:
            # PGA on reference rock (Vs30=760) — site amplification NOT included.
            # The BA08 PGA implementation in hazard.py only provides rock-site PGA.
            # For site-specific PGA predictions, use BSSA21 which has full site terms.
            pga_rock = _estimate_pga_ref(Mw, R_JB, fault_type)
            if Vs30 < 700:
                import warnings
                warnings.warn(
                    f"BA08 PGA wrapper returns rock-site PGA (Vs30=760) regardless "
                    f"of input Vs30={Vs30:.0f}. Use gmpe_model=bssa21 for "
                    f"site-specific PGA predictions.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return pga_rock, 0.564
        else:
            raise ValueError(
                f"BA08 wrapper only supports periods 0.0 (PGA) and 1.0 (Sa 1.0s), "
                f"got {period}s. Use BSSA21 for full spectral period support."
            )


# ── Auto-register on import ─────────────────────────────────────────────

register_gmpe(BA08())

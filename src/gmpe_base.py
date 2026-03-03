"""
GMPE (Ground Motion Prediction Equation) Protocol and Registry.

Provides a lightweight protocol-based interface for pluggable GMPE models,
a global registry for model lookup, and IM type-to-period mapping.

Usage:
    from src.gmpe_base import get_gmpe, IM_TYPE_TO_PERIOD

    model = get_gmpe("bssa21")
    median_g, sigma_ln = model.compute(Mw=6.7, R_JB=20.0, Vs30=360.0,
                                        fault_type="reverse", period=1.0)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


# ── IM type to spectral period mapping ────────────────────────────────────

IM_TYPE_TO_PERIOD: dict[str, float] = {
    "PGA":  0.0,    # Peak Ground Acceleration
    "SA03": 0.3,    # Sa(0.3s)
    "SA10": 1.0,    # Sa(1.0s)
    "SA30": 3.0,    # Sa(3.0s)
}


# ── GMPE Protocol ────────────────────────────────────────────────────────

@runtime_checkable
class GMPEModel(Protocol):
    """Protocol for ground motion prediction equation models.

    Any class implementing this protocol can be registered and used
    interchangeably in the analysis pipeline.
    """

    @property
    def name(self) -> str:
        """Short identifier for the model (e.g. 'ba08', 'bssa21')."""
        ...

    @property
    def supported_periods(self) -> list[float]:
        """List of spectral periods (seconds) this model supports.

        Period 0.0 represents PGA.
        """
        ...

    def compute(
        self,
        Mw: float,
        R_JB: float,
        Vs30: float,
        fault_type: str,
        period: float,
    ) -> tuple[float, float]:
        """Compute median ground motion and aleatory sigma.

        Parameters
        ----------
        Mw : float
            Moment magnitude.
        R_JB : float
            Joyner-Boore distance in km.
        Vs30 : float
            Time-averaged shear-wave velocity in top 30 m (m/s).
        fault_type : str
            One of 'strike_slip', 'normal', 'reverse', 'unspecified'.
        period : float
            Spectral period in seconds (0.0 for PGA).

        Returns
        -------
        (median_g, sigma_ln) : tuple[float, float]
            Median spectral acceleration in g and total aleatory
            standard deviation (natural log units).
        """
        ...


# ── Registry ─────────────────────────────────────────────────────────────

GMPE_REGISTRY: dict[str, GMPEModel] = {}


def register_gmpe(model: GMPEModel) -> None:
    """Register a GMPE model instance in the global registry."""
    GMPE_REGISTRY[model.name] = model


def get_gmpe(name: str) -> GMPEModel:
    """Retrieve a registered GMPE model by name.

    Raises
    ------
    KeyError
        If the model name is not registered.
    """
    if name not in GMPE_REGISTRY:
        available = list(GMPE_REGISTRY.keys()) or ["(none registered)"]
        raise KeyError(
            f"GMPE model '{name}' not found. "
            f"Available models: {available}. "
            f"Ensure the model module is imported (e.g. 'import src.gmpe_bssa21')."
        )
    return GMPE_REGISTRY[name]

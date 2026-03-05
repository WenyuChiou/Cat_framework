"""Tests for simplified NGA GMPE comparison module."""

import math

import numpy as np
import pytest

from src.gmpe_nga_simplified import (
    ALL_SIMPLIFIED_MODELS,
    ASK14,
    BSSA14_Simplified,
    CB14,
    CY14,
    GK15,
    Idriss14,
    NGAEast,
    attenuation_curves,
    compare_models,
    vs30_sensitivity,
)


# ── Model instantiation ─────────────────────────────────────────────────

class TestModelRegistry:
    def test_all_seven_models_registered(self):
        assert len(ALL_SIMPLIFIED_MODELS) == 7

    @pytest.mark.parametrize("name", [
        "ask14", "bssa14_simplified", "cb14", "cy14",
        "idriss14", "gk15", "nga_east",
    ])
    def test_model_exists(self, name):
        assert name in ALL_SIMPLIFIED_MODELS

    @pytest.mark.parametrize("name", [
        "ask14", "bssa14_simplified", "cb14", "cy14",
        "idriss14", "gk15", "nga_east",
    ])
    def test_supported_periods_pga_only(self, name):
        model = ALL_SIMPLIFIED_MODELS[name]
        assert model.supported_periods == [0.0]


# ── Basic computation ────────────────────────────────────────────────────

class TestCompute:
    """All models should return positive PGA and positive sigma."""

    @pytest.mark.parametrize("name", list(ALL_SIMPLIFIED_MODELS.keys()))
    def test_positive_pga(self, name):
        model = ALL_SIMPLIFIED_MODELS[name]
        pga, sigma = model.compute(Mw=6.5, R_JB=30.0, Vs30=500.0)
        assert pga > 0
        assert sigma > 0

    @pytest.mark.parametrize("name", list(ALL_SIMPLIFIED_MODELS.keys()))
    def test_period_not_zero_raises(self, name):
        model = ALL_SIMPLIFIED_MODELS[name]
        with pytest.raises(ValueError, match="PGA only"):
            model.compute(Mw=6.5, R_JB=30.0, Vs30=500.0, period=1.0)

    @pytest.mark.parametrize("name", list(ALL_SIMPLIFIED_MODELS.keys()))
    def test_pga_decreases_with_distance(self, name):
        model = ALL_SIMPLIFIED_MODELS[name]
        pga_near, _ = model.compute(Mw=6.5, R_JB=10.0, Vs30=500.0)
        pga_far, _ = model.compute(Mw=6.5, R_JB=100.0, Vs30=500.0)
        assert pga_near > pga_far

    @pytest.mark.parametrize("name", list(ALL_SIMPLIFIED_MODELS.keys()))
    def test_pga_increases_with_magnitude(self, name):
        model = ALL_SIMPLIFIED_MODELS[name]
        pga_small, _ = model.compute(Mw=5.0, R_JB=30.0, Vs30=500.0)
        pga_large, _ = model.compute(Mw=7.5, R_JB=30.0, Vs30=500.0)
        assert pga_large > pga_small


# ── Vs30 sensitivity ────────────────────────────────────────────────────

class TestVs30Sensitivity:
    @pytest.mark.parametrize("name", list(ALL_SIMPLIFIED_MODELS.keys()))
    def test_softer_soil_higher_pga(self, name):
        """Softer soil (lower Vs30) should amplify ground motion."""
        model = ALL_SIMPLIFIED_MODELS[name]
        pga_soft, _ = model.compute(Mw=6.5, R_JB=30.0, Vs30=200.0)
        pga_rock, _ = model.compute(Mw=6.5, R_JB=30.0, Vs30=760.0)
        assert pga_soft > pga_rock

    def test_vs30_sensitivity_function(self):
        results = vs30_sensitivity(Mw=6.5, R_JB=30.0)
        assert "vs30" in results
        assert len(results["vs30"]) == 50
        for name in ALL_SIMPLIFIED_MODELS:
            assert name in results
            assert len(results[name]) == 50

    def test_vs30_custom_values(self):
        vs30_vals = [200, 400, 800]
        results = vs30_sensitivity(Mw=6.5, R_JB=30.0, vs30_values=vs30_vals)
        assert len(results["vs30"]) == 3


# ── Comparison utilities ────────────────────────────────────────────────

class TestCompareModels:
    def test_compare_returns_all_models(self):
        results = compare_models(Mw=6.7, R_JB=20.0, Vs30=360.0)
        assert len(results) == 7
        for name, r in results.items():
            assert "pga_g" in r
            assert "sigma" in r
            assert "ln_pga" in r
            assert r["pga_g"] > 0
            assert abs(r["ln_pga"] - math.log(r["pga_g"])) < 1e-10

    def test_compare_with_fault_type(self):
        r_ss = compare_models(Mw=6.7, R_JB=20.0, Vs30=360.0, fault_type="strike_slip")
        r_rv = compare_models(Mw=6.7, R_JB=20.0, Vs30=360.0, fault_type="reverse")
        # ASK14 has fault type sensitivity
        assert r_ss["ask14"]["pga_g"] != r_rv["ask14"]["pga_g"]


class TestAttenuationCurves:
    def test_default_distances(self):
        results = attenuation_curves(Mw=6.5, Vs30=500.0)
        assert "distance" in results
        assert len(results["distance"]) == 100
        for name in ALL_SIMPLIFIED_MODELS:
            assert name in results

    def test_monotonic_decay(self):
        results = attenuation_curves(Mw=6.5, Vs30=500.0)
        for name in ALL_SIMPLIFIED_MODELS:
            pga_arr = results[name]
            # PGA should generally decrease with distance
            assert pga_arr[0] > pga_arr[-1]

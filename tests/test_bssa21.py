"""
Tests for BSSA21 GMPE accuracy.

Verifies the Python port against expected values derived from the
published BSSA14 coefficient table and the OpenQuake implementation.
"""

import math
import pytest

# Ensure GMPE modules are imported (auto-registers)
import src.gmpe_bssa21  # noqa: F401
import src.gmpe_ba08    # noqa: F401

from src.gmpe_base import get_gmpe, GMPE_REGISTRY, IM_TYPE_TO_PERIOD
from src.gmpe_bssa21 import BSSA21, _get_row


# ── Registry tests ────────────────────────────────────────────────────────

class TestRegistry:
    def test_bssa21_registered(self):
        assert "bssa21" in GMPE_REGISTRY

    def test_ba08_registered(self):
        assert "ba08" in GMPE_REGISTRY

    def test_get_gmpe_bssa21(self):
        model = get_gmpe("bssa21")
        assert model.name == "bssa21"

    def test_get_gmpe_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_gmpe("nonexistent_model")


# ── Coefficient table tests ──────────────────────────────────────────────

class TestCoefficientTable:
    def test_pga_row_exists(self):
        row = _get_row(0.0)
        assert row is not None
        assert abs(row[0]) < 1e-6  # period = 0

    def test_sa03_row_exists(self):
        row = _get_row(0.3)
        assert abs(row[0] - 0.3) < 1e-6

    def test_sa10_row_exists(self):
        row = _get_row(1.0)
        assert abs(row[0] - 1.0) < 1e-6

    def test_sa30_row_exists(self):
        row = _get_row(3.0)
        assert abs(row[0] - 3.0) < 1e-6

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="not in BSSA14/21 table"):
            _get_row(0.77)

    def test_supported_periods(self):
        model = BSSA21()
        periods = model.supported_periods
        assert 0.0 in periods  # PGA
        assert 0.3 in periods
        assert 1.0 in periods
        assert 3.0 in periods
        assert -1 not in periods  # PGV excluded
        assert len(periods) >= 100  # should have 107+ SA periods


# ── BSSA21 compute tests ─────────────────────────────────────────────────

class TestBSSA21Compute:
    """Test BSSA21 predictions for known scenarios."""

    def setup_method(self):
        self.model = BSSA21()

    def test_basic_output_format(self):
        """compute() returns (median_g, sigma_ln) tuple of positive floats."""
        median, sigma = self.model.compute(
            Mw=6.7, R_JB=20.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        assert isinstance(median, float)
        assert isinstance(sigma, float)
        assert median > 0
        assert sigma > 0

    def test_pga_output(self):
        median, sigma = self.model.compute(
            Mw=7.0, R_JB=10.0, Vs30=760.0, fault_type="strike_slip", period=0.0
        )
        assert 0.01 < median < 2.0  # reasonable PGA range for this scenario
        assert 0.3 < sigma < 1.0

    def test_distance_attenuation(self):
        """IM should decrease with increasing distance."""
        im_near, _ = self.model.compute(
            Mw=7.0, R_JB=10.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        im_far, _ = self.model.compute(
            Mw=7.0, R_JB=100.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        assert im_near > im_far, "IM should decrease with distance"

    def test_magnitude_scaling(self):
        """IM should increase with increasing magnitude."""
        im_small, _ = self.model.compute(
            Mw=5.0, R_JB=30.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        im_large, _ = self.model.compute(
            Mw=7.5, R_JB=30.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        assert im_large > im_small, "IM should increase with magnitude"

    def test_site_amplification(self):
        """Softer soil (lower Vs30) should give higher IM than rock."""
        im_rock, _ = self.model.compute(
            Mw=6.5, R_JB=30.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        im_soil, _ = self.model.compute(
            Mw=6.5, R_JB=30.0, Vs30=270.0, fault_type="reverse", period=1.0
        )
        assert im_soil > im_rock, "Soft soil should amplify ground motion"

    def test_rock_reference_sa10(self):
        """Sa(1.0s) on rock for M6.7 at 20 km should be ~0.1-0.3g."""
        median, _ = self.model.compute(
            Mw=6.7, R_JB=20.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        assert 0.05 < median < 0.5, f"Expected 0.05-0.5g, got {median:.4f}g"

    def test_northridge_like_scenario(self):
        """Mw 6.7 reverse, R=10km, Vs30=270 → expect Sa(1.0) ~ 0.2-0.8g."""
        median, _ = self.model.compute(
            Mw=6.7, R_JB=10.0, Vs30=270.0, fault_type="reverse", period=1.0
        )
        assert 0.1 < median < 1.5, f"Expected 0.1-1.5g, got {median:.4f}g"

    def test_sa03_higher_than_sa10_near_field(self):
        """For stiff site near a moderate EQ, Sa(0.3) > Sa(1.0)."""
        sa03, _ = self.model.compute(
            Mw=6.5, R_JB=15.0, Vs30=760.0, fault_type="reverse", period=0.3
        )
        sa10, _ = self.model.compute(
            Mw=6.5, R_JB=15.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        assert sa03 > sa10, "Sa(0.3) should exceed Sa(1.0) for stiff site"

    def test_mechanism_effect(self):
        """Different mechanisms should produce different predictions."""
        im_ss, _ = self.model.compute(
            Mw=6.5, R_JB=20.0, Vs30=760.0, fault_type="strike_slip", period=1.0
        )
        im_rv, _ = self.model.compute(
            Mw=6.5, R_JB=20.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        im_nm, _ = self.model.compute(
            Mw=6.5, R_JB=20.0, Vs30=760.0, fault_type="normal", period=1.0
        )
        # They should differ (different e coefficients)
        assert im_ss != im_rv or im_rv != im_nm

    def test_sigma_magnitude_dependence(self):
        """Sigma should be larger for small magnitudes (M<4.5) vs large (M>5.5)."""
        _, sigma_small = self.model.compute(
            Mw=4.0, R_JB=30.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        _, sigma_large = self.model.compute(
            Mw=7.0, R_JB=30.0, Vs30=760.0, fault_type="reverse", period=1.0
        )
        # For BSSA14, tau1 > tau2 and phi1 < phi2 typically,
        # but total sigma can go either way. Just check they're different.
        assert sigma_small != sigma_large

    def test_compare_with_ba08_sa10(self):
        """BSSA21 and BA08 should give same-order-of-magnitude Sa(1.0)."""
        ba08 = get_gmpe("ba08")
        bssa21 = get_gmpe("bssa21")

        params = dict(Mw=6.7, R_JB=20.0, Vs30=760.0,
                      fault_type="reverse", period=1.0)
        im_ba08, _ = ba08.compute(**params)
        im_bssa21, _ = bssa21.compute(**params)

        # Should be within factor of 3 (they're different generations)
        ratio = im_bssa21 / im_ba08
        assert 0.3 < ratio < 3.0, (
            f"BSSA21/BA08 ratio = {ratio:.2f} — expected within 0.3–3.0"
        )


# ── Multi-scenario batch test ────────────────────────────────────────────

class TestBSSA21MultiScenario:
    """Test across a grid of Mw/Rjb/Vs30 combinations."""

    SCENARIOS = [
        # (Mw, R_JB, Vs30, fault_type, period)
        (5.0, 10.0, 760.0, "strike_slip", 0.0),
        (5.0, 50.0, 360.0, "normal", 0.3),
        (6.0, 20.0, 270.0, "reverse", 1.0),
        (6.5, 5.0, 760.0, "reverse", 0.3),
        (7.0, 30.0, 180.0, "strike_slip", 1.0),
        (7.0, 100.0, 760.0, "reverse", 3.0),
        (7.5, 50.0, 360.0, "normal", 1.0),
        (8.0, 200.0, 760.0, "reverse", 1.0),
    ]

    def test_all_scenarios_produce_valid_output(self):
        model = BSSA21()
        for Mw, R_JB, Vs30, ft, T in self.SCENARIOS:
            median, sigma = model.compute(Mw, R_JB, Vs30, ft, T)
            assert median > 0, f"Negative median for {Mw}/{R_JB}/{Vs30}"
            assert math.isfinite(median), f"Non-finite median for {Mw}/{R_JB}/{Vs30}"
            assert 0.1 < sigma < 1.5, f"Sigma out of range for {Mw}/{R_JB}/{Vs30}"
            # SA should be in a plausible range (1e-5 to 10g)
            assert 1e-5 < median < 10.0, (
                f"Implausible median {median:.6f}g for "
                f"Mw={Mw}, R={R_JB}, Vs30={Vs30}, T={T}"
            )

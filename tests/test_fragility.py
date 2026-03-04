"""
Tests for fragility curve computation and damage state probabilities.

Covers:
  - Lognormal fragility curve correctness
  - All 28 HWB classes produce valid probability distributions
  - Edge cases: IM=0, very large IM, negative IM
  - Monotonicity of exceedance probabilities
  - Skew angle modification
"""

import math
import numpy as np
import pytest

from src.fragility import (
    fragility_curve,
    compute_all_curves,
    damage_state_probabilities,
    apply_skew_modification,
)
from src.hazus_params import HAZUS_BRIDGE_FRAGILITY, DAMAGE_STATE_ORDER


# ── Fragility curve basic tests ────────────────────────────────────────────

class TestFragilityCurve:
    def test_zero_im_gives_zero_prob(self):
        prob = fragility_curve(np.array([0.0]), median=0.5, beta=0.6)
        assert prob[0] == 0.0

    def test_negative_im_gives_zero_prob(self):
        prob = fragility_curve(np.array([-0.5]), median=0.5, beta=0.6)
        assert prob[0] == 0.0

    def test_very_large_im_gives_near_one(self):
        prob = fragility_curve(np.array([100.0]), median=0.5, beta=0.6)
        assert prob[0] > 0.999

    def test_at_median_gives_50_percent(self):
        """At IM = median, P(exceedance) should be exactly 0.5."""
        prob = fragility_curve(np.array([0.5]), median=0.5, beta=0.6)
        assert abs(prob[0] - 0.5) < 1e-10

    def test_output_shape_matches_input(self):
        im = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
        prob = fragility_curve(im, median=0.5, beta=0.6)
        assert prob.shape == im.shape

    def test_monotonically_increasing(self):
        """Exceedance probability should increase with IM."""
        im = np.linspace(0.01, 2.0, 50)
        prob = fragility_curve(im, median=0.5, beta=0.6)
        for i in range(1, len(prob)):
            assert prob[i] >= prob[i - 1]

    def test_probabilities_bounded_0_1(self):
        im = np.logspace(-3, 1, 100)
        prob = fragility_curve(im, median=0.5, beta=0.6)
        assert np.all(prob >= 0.0)
        assert np.all(prob <= 1.0)

    def test_higher_beta_gives_wider_curve(self):
        """Higher dispersion → more spread (lower prob at low IM, higher at high IM)."""
        im = np.array([0.1])  # well below median
        prob_narrow = fragility_curve(im, median=0.5, beta=0.3)
        prob_wide = fragility_curve(im, median=0.5, beta=0.9)
        # At IM << median, wider dispersion gives higher probability
        assert prob_wide[0] > prob_narrow[0]


# ── All 28 HWB classes ────────────────────────────────────────────────────

class TestAllHWBClasses:
    """Verify all 28 HWB classes produce valid damage probability distributions."""

    ALL_CLASSES = [f"HWB{i}" for i in range(1, 29)]

    @pytest.mark.parametrize("hwb_class", ALL_CLASSES)
    def test_class_exists_in_table(self, hwb_class):
        assert hwb_class in HAZUS_BRIDGE_FRAGILITY

    @pytest.mark.parametrize("hwb_class", ALL_CLASSES)
    def test_probs_sum_to_one(self, hwb_class):
        probs = damage_state_probabilities(0.5, hwb_class)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-10, f"{hwb_class}: probs sum to {total}"

    @pytest.mark.parametrize("hwb_class", ALL_CLASSES)
    def test_all_probs_non_negative(self, hwb_class):
        probs = damage_state_probabilities(0.5, hwb_class)
        for ds, p in probs.items():
            assert p >= 0.0, f"{hwb_class} {ds}: negative prob {p}"

    @pytest.mark.parametrize("hwb_class", ALL_CLASSES)
    def test_expected_keys(self, hwb_class):
        probs = damage_state_probabilities(0.5, hwb_class)
        expected = {"none", "slight", "moderate", "extensive", "complete"}
        assert set(probs.keys()) == expected

    @pytest.mark.parametrize("hwb_class", ALL_CLASSES)
    def test_zero_im_all_none(self, hwb_class):
        """At IM=0, all probability should be in 'none' state."""
        probs = damage_state_probabilities(0.0, hwb_class)
        assert probs["none"] == 1.0
        for ds in DAMAGE_STATE_ORDER:
            assert probs[ds] == 0.0

    @pytest.mark.parametrize("hwb_class", ALL_CLASSES)
    def test_very_large_im_mostly_complete(self, hwb_class):
        """At very large IM, most probability should be in 'complete'."""
        probs = damage_state_probabilities(50.0, hwb_class)
        assert probs["complete"] > 0.99

    def test_medians_are_ordered(self):
        """For each class, medians should be: slight < moderate < extensive < complete."""
        for hwb_class in self.ALL_CLASSES:
            params = HAZUS_BRIDGE_FRAGILITY[hwb_class]["damage_states"]
            medians = [params[ds]["median"] for ds in DAMAGE_STATE_ORDER]
            for i in range(len(medians) - 1):
                assert medians[i] <= medians[i + 1], (
                    f"{hwb_class}: {DAMAGE_STATE_ORDER[i]} median {medians[i]} > "
                    f"{DAMAGE_STATE_ORDER[i+1]} median {medians[i+1]}"
                )


# ── compute_all_curves ─────────────────────────────────────────────────────

class TestComputeAllCurves:
    def test_returns_four_damage_states(self):
        im = np.array([0.1, 0.5, 1.0])
        curves = compute_all_curves("HWB5", im)
        assert set(curves.keys()) == set(DAMAGE_STATE_ORDER)

    def test_exceedance_ordering(self):
        """P(slight) >= P(moderate) >= P(extensive) >= P(complete) at any IM."""
        im = np.array([0.3])
        curves = compute_all_curves("HWB5", im)
        for i in range(len(DAMAGE_STATE_ORDER) - 1):
            ds1 = DAMAGE_STATE_ORDER[i]
            ds2 = DAMAGE_STATE_ORDER[i + 1]
            assert curves[ds1][0] >= curves[ds2][0], (
                f"P({ds1})={curves[ds1][0]} < P({ds2})={curves[ds2][0]}"
            )

    def test_invalid_class_raises(self):
        with pytest.raises(KeyError):
            compute_all_curves("HWB99", np.array([0.5]))


# ── Skew modification ─────────────────────────────────────────────────────

class TestSkewModification:
    def test_zero_skew_no_change(self):
        assert apply_skew_modification(0.5, 0.0) == 0.5

    def test_90_degree_skew_gives_zero(self):
        assert apply_skew_modification(0.5, 90.0) == 0.0

    def test_45_degree_reduces_median(self):
        modified = apply_skew_modification(0.5, 45.0)
        assert 0.0 < modified < 0.5

    def test_skew_clamped_to_range(self):
        """Skew angles outside [0, 90] should be clamped."""
        assert apply_skew_modification(0.5, -10.0) == 0.5  # clamped to 0
        assert apply_skew_modification(0.5, 100.0) == 0.0   # clamped to 90

    def test_moderate_skew_value(self):
        """30-degree skew: factor = sqrt(1 - (30/90)^2) = sqrt(8/9) ≈ 0.9428."""
        modified = apply_skew_modification(1.0, 30.0)
        expected = math.sqrt(1.0 - (30.0 / 90.0) ** 2)
        assert abs(modified - expected) < 1e-10

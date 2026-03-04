"""
Tests for loss computation module.

Covers:
  - Damage ratio table correctness
  - Single bridge loss calculation
  - Portfolio loss aggregation
  - EP curve and AAL computation
  - Edge cases: zero SA, zero cost
"""

import numpy as np
import pytest

from src.loss import (
    HAZUS_DAMAGE_RATIOS,
    HAZUS_DOWNTIME_DAYS,
    compute_bridge_loss,
    compute_portfolio_loss,
    compute_ep_curve,
    compute_aal,
    loss_summary_table,
    BridgeLossResult,
    PortfolioLossResult,
)
from src.exposure import BridgeExposure, generate_synthetic_portfolio


# ── Damage ratio table ─────────────────────────────────────────────────────

class TestDamageRatios:
    def test_none_is_zero(self):
        assert HAZUS_DAMAGE_RATIOS["none"] == 0.0

    def test_complete_is_one(self):
        assert HAZUS_DAMAGE_RATIOS["complete"] == 1.0

    def test_ratios_monotonically_increasing(self):
        order = ["none", "slight", "moderate", "extensive", "complete"]
        for i in range(len(order) - 1):
            assert HAZUS_DAMAGE_RATIOS[order[i]] <= HAZUS_DAMAGE_RATIOS[order[i + 1]]

    def test_all_damage_states_present(self):
        expected = {"none", "slight", "moderate", "extensive", "complete"}
        assert set(HAZUS_DAMAGE_RATIOS.keys()) == expected

    def test_downtime_monotonically_increasing(self):
        order = ["none", "slight", "moderate", "extensive", "complete"]
        for i in range(len(order) - 1):
            assert HAZUS_DOWNTIME_DAYS[order[i]] <= HAZUS_DOWNTIME_DAYS[order[i + 1]]


# ── Single bridge loss ─────────────────────────────────────────────────────

class TestBridgeLoss:
    def test_basic_output_type(self):
        result = compute_bridge_loss(sa=0.5, hwb_class="HWB5", replacement_cost=1_000_000)
        assert isinstance(result, BridgeLossResult)

    def test_loss_between_zero_and_replacement(self):
        result = compute_bridge_loss(sa=0.5, hwb_class="HWB5", replacement_cost=1_000_000)
        assert 0.0 <= result.expected_loss <= 1_000_000

    def test_loss_ratio_bounded(self):
        result = compute_bridge_loss(sa=0.5, hwb_class="HWB5", replacement_cost=1_000_000)
        assert 0.0 <= result.loss_ratio <= 1.0

    def test_zero_sa_gives_zero_loss(self):
        result = compute_bridge_loss(sa=0.0, hwb_class="HWB5", replacement_cost=1_000_000)
        assert result.expected_loss == 0.0
        assert result.expected_downtime == 0.0

    def test_zero_cost_gives_zero_loss(self):
        result = compute_bridge_loss(sa=0.5, hwb_class="HWB5", replacement_cost=0.0)
        assert result.expected_loss == 0.0
        assert result.loss_ratio == 0.0

    def test_higher_sa_higher_loss(self):
        low = compute_bridge_loss(sa=0.1, hwb_class="HWB5", replacement_cost=1_000_000)
        high = compute_bridge_loss(sa=1.0, hwb_class="HWB5", replacement_cost=1_000_000)
        assert high.expected_loss > low.expected_loss

    def test_very_large_sa_near_total_loss(self):
        result = compute_bridge_loss(sa=50.0, hwb_class="HWB5", replacement_cost=1_000_000)
        assert result.loss_ratio > 0.95

    def test_damage_probs_sum_to_one(self):
        result = compute_bridge_loss(sa=0.5, hwb_class="HWB5", replacement_cost=1_000_000)
        total = sum(result.damage_probs.values())
        assert abs(total - 1.0) < 1e-10

    def test_downtime_non_negative(self):
        result = compute_bridge_loss(sa=0.5, hwb_class="HWB5", replacement_cost=1_000_000)
        assert result.expected_downtime >= 0.0

    def test_bridge_id_stored(self):
        result = compute_bridge_loss(sa=0.5, hwb_class="HWB5",
                                     replacement_cost=1_000_000, bridge_id="BR-001")
        assert result.bridge_id == "BR-001"


# ── Portfolio loss ─────────────────────────────────────────────────────────

class TestPortfolioLoss:
    def setup_method(self):
        self.portfolio = generate_synthetic_portfolio(n_bridges=20, seed=42)
        self.sa_values = np.full(20, 0.4)

    def test_basic_output_type(self):
        result = compute_portfolio_loss(self.portfolio, self.sa_values)
        assert isinstance(result, PortfolioLossResult)

    def test_total_loss_positive(self):
        result = compute_portfolio_loss(self.portfolio, self.sa_values)
        assert result.total_loss > 0

    def test_loss_ratio_bounded(self):
        result = compute_portfolio_loss(self.portfolio, self.sa_values)
        assert 0.0 <= result.loss_ratio <= 1.0

    def test_bridge_count_matches(self):
        result = compute_portfolio_loss(self.portfolio, self.sa_values)
        assert len(result.bridge_results) == 20

    def test_count_by_ds_sums_to_n(self):
        result = compute_portfolio_loss(self.portfolio, self.sa_values)
        total_count = sum(result.count_by_ds.values())
        assert abs(total_count - 20.0) < 1e-6

    def test_loss_by_class_sums_to_total(self):
        result = compute_portfolio_loss(self.portfolio, self.sa_values)
        class_total = sum(result.loss_by_class.values())
        assert abs(class_total - result.total_loss) < 0.01


# ── EP curve ───────────────────────────────────────────────────────────────

class TestEPCurve:
    def test_basic_output_keys(self):
        losses = np.array([100, 200, 300, 400, 500], dtype=float)
        ep = compute_ep_curve(losses)
        assert "loss_thresholds" in ep
        assert "exceedance_prob" in ep
        assert "return_period" in ep

    def test_ep_bounded_0_1(self):
        losses = np.random.default_rng(42).uniform(0, 1e6, 50)
        ep = compute_ep_curve(losses)
        assert np.all(ep["exceedance_prob"] >= 0)
        assert np.all(ep["exceedance_prob"] <= 1)

    def test_losses_sorted_descending(self):
        losses = np.array([100, 500, 200, 400, 300], dtype=float)
        ep = compute_ep_curve(losses)
        thresholds = ep["loss_thresholds"]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] >= thresholds[i + 1]

    def test_with_custom_rates(self):
        losses = np.array([1e6, 5e5, 2e5], dtype=float)
        rates = np.array([0.01, 0.05, 0.10])
        ep = compute_ep_curve(losses, rates)
        assert len(ep["loss_thresholds"]) == 3


# ── AAL ────────────────────────────────────────────────────────────────────

class TestAAL:
    def test_basic_aal(self):
        losses = np.array([1e6, 5e5, 1e5])
        rates = np.array([0.01, 0.05, 0.20])
        aal = compute_aal(losses, rates)
        expected = 1e6 * 0.01 + 5e5 * 0.05 + 1e5 * 0.20
        assert abs(aal - expected) < 0.01

    def test_zero_rates_give_zero_aal(self):
        losses = np.array([1e6, 5e5])
        rates = np.array([0.0, 0.0])
        assert compute_aal(losses, rates) == 0.0


# ── Summary table ──────────────────────────────────────────────────────────

class TestSummaryTable:
    def test_summary_produces_string(self):
        portfolio = generate_synthetic_portfolio(n_bridges=10, seed=42)
        sa = np.full(10, 0.3)
        result = compute_portfolio_loss(portfolio, sa)
        text = loss_summary_table(result)
        assert isinstance(text, str)
        assert "PORTFOLIO LOSS SUMMARY" in text
        assert "DAMAGE STATE DISTRIBUTION" in text

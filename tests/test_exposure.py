"""
Tests for bridge exposure and portfolio modules.

Covers:
  - BridgeExposure dataclass
  - Replacement cost estimation
  - Synthetic portfolio generation
  - NBI to portfolio conversion
  - Portfolio utilities (summary, filter)
"""

import numpy as np
import pytest

from src.exposure import (
    BridgeExposure,
    REPLACEMENT_COST_PER_M2,
    estimate_replacement_cost,
    generate_synthetic_portfolio,
    create_portfolio_from_nbi,
    portfolio_summary,
    filter_portfolio,
    portfolio_to_sites,
)
from src.hazard import SiteParams


# ── BridgeExposure dataclass ───────────────────────────────────────────────

class TestBridgeExposure:
    def test_create_with_defaults(self):
        b = BridgeExposure(bridge_id="B001", lat=34.0, lon=-118.0, hwb_class="HWB5")
        assert b.bridge_id == "B001"
        assert b.vs30 == 760.0
        assert b.skew_angle == 0.0
        assert b.material == "concrete"

    def test_custom_attributes(self):
        b = BridgeExposure(
            bridge_id="B002", lat=34.0, lon=-118.0, hwb_class="HWB15",
            material="steel", length=100.0, deck_area=1000.0,
            replacement_cost=5_000_000, vs30=350.0, skew_angle=30.0,
        )
        assert b.material == "steel"
        assert b.vs30 == 350.0
        assert b.replacement_cost == 5_000_000


# ── Replacement cost estimation ────────────────────────────────────────────

class TestReplacementCost:
    def test_concrete_cost(self):
        cost = estimate_replacement_cost("concrete", deck_area=300.0, length=30.0)
        assert cost > 0
        # 300 m² * 2500 $/m² * 1.0 (no length adjustment) = 750,000
        assert abs(cost - 750_000) < 1.0

    def test_steel_more_expensive(self):
        cost_concrete = estimate_replacement_cost("concrete", 300.0, 30.0)
        cost_steel = estimate_replacement_cost("steel", 300.0, 30.0)
        assert cost_steel > cost_concrete

    def test_longer_bridge_costs_more(self):
        cost_short = estimate_replacement_cost("concrete", 300.0, 50.0)
        cost_long = estimate_replacement_cost("concrete", 300.0, 200.0)
        assert cost_long > cost_short

    def test_unknown_material_uses_default(self):
        cost = estimate_replacement_cost("unknown_material", 300.0, 30.0)
        assert cost > 0  # uses 2600 default

    def test_zero_deck_area(self):
        cost = estimate_replacement_cost("concrete", 0.0, 30.0)
        assert cost == 0.0


# ── Synthetic portfolio ────────────────────────────────────────────────────

class TestSyntheticPortfolio:
    def test_correct_count(self):
        portfolio = generate_synthetic_portfolio(n_bridges=50)
        assert len(portfolio) == 50

    def test_all_have_valid_class(self):
        from src.hazus_params import HAZUS_BRIDGE_FRAGILITY
        portfolio = generate_synthetic_portfolio(n_bridges=100)
        for b in portfolio:
            assert b.hwb_class in HAZUS_BRIDGE_FRAGILITY

    def test_all_have_positive_cost(self):
        portfolio = generate_synthetic_portfolio(n_bridges=50)
        for b in portfolio:
            assert b.replacement_cost > 0

    def test_locations_near_center(self):
        center = (34.213, -118.537)
        portfolio = generate_synthetic_portfolio(n_bridges=50, center=center, radius_km=30.0)
        for b in portfolio:
            dlat = abs(b.lat - center[0])
            dlon = abs(b.lon - center[1])
            # Should be within ~0.5 degrees (~55 km)
            assert dlat < 0.5, f"Bridge too far north/south: {b.lat}"
            assert dlon < 0.5, f"Bridge too far east/west: {b.lon}"

    def test_reproducible_with_seed(self):
        p1 = generate_synthetic_portfolio(n_bridges=10, seed=123)
        p2 = generate_synthetic_portfolio(n_bridges=10, seed=123)
        for b1, b2 in zip(p1, p2):
            assert b1.lat == b2.lat
            assert b1.hwb_class == b2.hwb_class

    def test_different_seeds_differ(self):
        p1 = generate_synthetic_portfolio(n_bridges=10, seed=1)
        p2 = generate_synthetic_portfolio(n_bridges=10, seed=2)
        # At least some bridges should differ
        diffs = sum(1 for b1, b2 in zip(p1, p2) if b1.lat != b2.lat)
        assert diffs > 0


# ── NBI to portfolio conversion ────────────────────────────────────────────

class TestNBIConversion:
    def setup_method(self):
        import pandas as pd
        self.nbi_df = pd.DataFrame({
            "structure_number": ["BR001", "BR002", "BR003"],
            "latitude": [34.05, 34.10, 34.15],
            "longitude": [-118.25, -118.30, -118.35],
            "hwb_class": ["HWB5", "HWB17", "HWB3"],
            "material": ["concrete", "concrete", "steel"],
            "structure_length_m": [30.0, 50.0, 100.0],
            "deck_width_m": [10.0, 12.0, 15.0],
        })

    def test_correct_count(self):
        portfolio = create_portfolio_from_nbi(self.nbi_df)
        assert len(portfolio) == 3

    def test_bridge_ids_match(self):
        portfolio = create_portfolio_from_nbi(self.nbi_df)
        ids = [b.bridge_id for b in portfolio]
        assert ids == ["BR001", "BR002", "BR003"]

    def test_coordinates_match(self):
        portfolio = create_portfolio_from_nbi(self.nbi_df)
        assert portfolio[0].lat == 34.05
        assert portfolio[0].lon == -118.25

    def test_default_vs30_applied(self):
        portfolio = create_portfolio_from_nbi(self.nbi_df, default_vs30=400.0)
        for b in portfolio:
            assert b.vs30 == 400.0

    def test_replacement_cost_computed(self):
        portfolio = create_portfolio_from_nbi(self.nbi_df)
        for b in portfolio:
            assert b.replacement_cost > 0


# ── Portfolio utilities ────────────────────────────────────────────────────

class TestPortfolioUtilities:
    def setup_method(self):
        self.portfolio = generate_synthetic_portfolio(n_bridges=30, seed=42)

    def test_summary_keys(self):
        summary = portfolio_summary(self.portfolio)
        assert "n_bridges" in summary
        assert summary["n_bridges"] == 30
        assert "total_replacement_cost" in summary
        assert summary["total_replacement_cost"] > 0

    def test_empty_portfolio_summary(self):
        summary = portfolio_summary([])
        assert summary["n_bridges"] == 0

    def test_filter_by_class(self):
        filtered = filter_portfolio(self.portfolio, hwb_classes=["HWB5"])
        assert all(b.hwb_class == "HWB5" for b in filtered)
        assert len(filtered) <= 30

    def test_filter_by_material(self):
        filtered = filter_portfolio(self.portfolio, materials=["steel"])
        assert all(b.material == "steel" for b in filtered)

    def test_filter_no_match_returns_empty(self):
        filtered = filter_portfolio(self.portfolio, hwb_classes=["HWB99"])
        assert len(filtered) == 0

    def test_portfolio_to_sites(self):
        sites = portfolio_to_sites(self.portfolio)
        assert len(sites) == 30
        assert all(isinstance(s, SiteParams) for s in sites)
        assert sites[0].lat == self.portfolio[0].lat

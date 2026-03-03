"""
End-to-end integration tests for the GMPE pipeline.

Tests the full flow: config → GMPE model → IM computation → damage probabilities.
"""

import math
import pytest
import numpy as np
import pandas as pd

# Ensure GMPE module is registered
import src.gmpe_bssa21  # noqa: F401

from src.config import AnalysisConfig, validate_config
from src.gmpe_base import get_gmpe, IM_TYPE_TO_PERIOD
from src.hazard import haversine_distance_km, EarthquakeScenario
from src.fragility import damage_state_probabilities


# ── Config validation tests ──────────────────────────────────────────────

class TestConfigValidation:
    def test_gmpe_without_scenario_raises(self):
        cfg = AnalysisConfig(im_source="gmpe", gmpe_scenario=None)
        with pytest.raises(ValueError, match="gmpe_scenario"):
            validate_config(cfg)

    def test_gmpe_with_valid_scenario(self):
        cfg = AnalysisConfig(
            im_source="gmpe",
            gmpe_model="bssa21",
            gmpe_scenario={
                "Mw": 6.7, "lat": 34.213, "lon": -118.537,
                "fault_type": "reverse",
            },
        )
        validate_config(cfg)  # should not raise

    def test_unknown_gmpe_model_raises(self):
        cfg = AnalysisConfig(
            im_source="gmpe",
            gmpe_model="nonexistent",
            gmpe_scenario={"Mw": 6.7, "lat": 34.0, "lon": -118.0},
        )
        with pytest.raises(ValueError, match="Unknown gmpe_model"):
            validate_config(cfg)

    def test_shakemap_config_unchanged(self):
        """Default shakemap config should still validate without error."""
        cfg = AnalysisConfig()  # defaults to shakemap + SA10
        validate_config(cfg)


# ── E2E: Config → GMPE → IM → Damage ────────────────────────────────────

class TestGMPEtoDamage:
    """End-to-end test: compute IMs for synthetic bridges, then damage probs."""

    def setup_method(self):
        self.config = AnalysisConfig(
            im_source="gmpe",
            im_type="SA10",
            gmpe_model="bssa21",
            gmpe_scenario={
                "Mw": 6.7,
                "lat": 34.213,
                "lon": -118.537,
                "depth_km": 18.4,
                "fault_type": "reverse",
                "vs30": 360.0,
            },
        )
        # Synthetic bridge portfolio near epicenter
        self.bridges = pd.DataFrame({
            "latitude":  [34.25, 34.30, 34.35, 34.40, 34.50],
            "longitude": [-118.50, -118.45, -118.40, -118.35, -118.20],
            "hwb_class": ["HWB5", "HWB17", "HWB5", "HWB8", "HWB5"],
        })

    def test_compute_im_for_bridges(self):
        """Compute Sa(1.0) at each bridge using BSSA21."""
        gmpe = get_gmpe("bssa21")
        sc = self.config.gmpe_scenario
        period = IM_TYPE_TO_PERIOD["SA10"]

        for _, bridge in self.bridges.iterrows():
            R_JB = haversine_distance_km(
                sc["lat"], sc["lon"],
                bridge["latitude"], bridge["longitude"],
            )
            R_JB = max(R_JB, 0.1)
            median, sigma = gmpe.compute(
                Mw=sc["Mw"], R_JB=R_JB,
                Vs30=sc.get("vs30", 760.0),
                fault_type=sc["fault_type"],
                period=period,
            )
            assert median > 0
            assert sigma > 0

    def test_full_pipeline_im_to_damage(self):
        """Full pipeline: GMPE → IM → damage state probabilities."""
        gmpe = get_gmpe("bssa21")
        sc = self.config.gmpe_scenario
        period = IM_TYPE_TO_PERIOD["SA10"]

        for _, bridge in self.bridges.iterrows():
            R_JB = haversine_distance_km(
                sc["lat"], sc["lon"],
                bridge["latitude"], bridge["longitude"],
            )
            R_JB = max(R_JB, 0.1)
            median_g, _ = gmpe.compute(
                Mw=sc["Mw"], R_JB=R_JB,
                Vs30=sc.get("vs30", 760.0),
                fault_type=sc["fault_type"],
                period=period,
            )
            # Compute damage probabilities
            probs = damage_state_probabilities(median_g, bridge["hwb_class"])

            # Validate damage probability structure
            assert set(probs.keys()) == {"none", "slight", "moderate", "extensive", "complete"}
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-6, f"Probs sum to {total}, expected 1.0"
            for ds, p in probs.items():
                assert 0.0 <= p <= 1.0, f"{ds} probability {p} out of [0,1]"

    def test_distance_gradient_in_damage(self):
        """Bridges closer to epicenter should have higher damage probability."""
        gmpe = get_gmpe("bssa21")
        sc = self.config.gmpe_scenario
        period = IM_TYPE_TO_PERIOD["SA10"]

        complete_probs = []
        for _, bridge in self.bridges.iterrows():
            R_JB = haversine_distance_km(
                sc["lat"], sc["lon"],
                bridge["latitude"], bridge["longitude"],
            )
            R_JB = max(R_JB, 0.1)
            median_g, _ = gmpe.compute(
                Mw=sc["Mw"], R_JB=R_JB,
                Vs30=sc.get("vs30", 760.0),
                fault_type=sc["fault_type"],
                period=period,
            )
            probs = damage_state_probabilities(median_g, "HWB5")
            complete_probs.append(probs["complete"])

        # First bridge is closest, last is farthest
        # Complete damage probability should generally decrease
        assert complete_probs[0] > complete_probs[-1], (
            f"Nearest bridge P(complete)={complete_probs[0]:.4f} should exceed "
            f"farthest bridge P(complete)={complete_probs[-1]:.4f}"
        )


# ── IM_TYPE_TO_PERIOD mapping ────────────────────────────────────────────

class TestIMTypeMapping:
    def test_all_im_types_mapped(self):
        assert IM_TYPE_TO_PERIOD["PGA"] == 0.0
        assert IM_TYPE_TO_PERIOD["SA03"] == 0.3
        assert IM_TYPE_TO_PERIOD["SA10"] == 1.0
        assert IM_TYPE_TO_PERIOD["SA30"] == 3.0

"""
Integration test: end-to-end pipeline from config to loss.

Runs the full pipeline on a small subset of bridges to verify
all components work together correctly.
"""

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestPipelineIntegration:
    """End-to-end: config → load → interpolate → fragility → loss."""

    @pytest.fixture(autouse=True)
    def _check_data(self):
        if not (PROJECT_ROOT / "data" / "grid.xml").exists():
            pytest.skip("Data files not available")
        if not (PROJECT_ROOT / "data" / "CA24.txt").exists():
            pytest.skip("Data files not available")

    def test_shakemap_pipeline(self):
        """Full ShakeMap pipeline produces valid loss results."""
        from src.config import load_config, IM_COLUMN_MAP
        from src.data_loader import load_shakemap, load_nbi, classify_nbi_to_hazus
        from src.interpolation import interpolate_im
        from src.fragility import damage_state_probabilities
        from src.loss import compute_bridge_loss

        # 1. Config
        cfg = load_config(PROJECT_ROOT / "config.yaml")
        assert cfg.im_type == "SA10"

        # 2. Load data
        grid = load_shakemap(PROJECT_ROOT / "data" / "grid.xml")
        assert len(grid) > 0

        nbi = load_nbi(PROJECT_ROOT / "data" / "CA24.txt",
                       northridge_bbox=cfg.region)
        nbi = classify_nbi_to_hazus(nbi)
        assert "hwb_class" in nbi.columns

        # Use first 50 bridges for speed
        nbi_small = nbi.head(50).copy()

        # 3. Interpolate
        im_col = IM_COLUMN_MAP.get(cfg.im_type, "PSA10")
        sa_values = interpolate_im(
            grid["LAT"].values, grid["LON"].values, grid[im_col].values,
            nbi_small["latitude"].values, nbi_small["longitude"].values,
            method="nearest",
        )
        assert len(sa_values) == 50
        assert np.all(sa_values >= 0)

        # 4. Fragility
        probs = damage_state_probabilities(sa_values[0], nbi_small.iloc[0]["hwb_class"])
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert all(0 <= v <= 1 for v in probs.values())

        # 5. Loss
        result = compute_bridge_loss(
            sa=sa_values[0],
            hwb_class=nbi_small.iloc[0]["hwb_class"],
            replacement_cost=1_000_000,
        )
        assert result.expected_loss >= 0
        assert result.loss_ratio >= 0
        assert result.loss_ratio <= 1.0

    def test_gmpe_pipeline(self):
        """GMPE path produces valid Sa predictions."""
        from src.hazard import haversine_distance_km, EarthquakeScenario
        from src.gmpe_base import get_gmpe
        import src.gmpe_bssa21  # noqa: F401

        scenario = EarthquakeScenario(
            Mw=6.7, lat=34.213, lon=-118.537,
            depth_km=18.4, fault_type="reverse",
        )
        gmpe = get_gmpe("bssa21")

        # Compute at a few distances
        for dist in [5.0, 20.0, 50.0, 100.0]:
            median, sigma = gmpe.compute(
                Mw=scenario.Mw, R_JB=dist, Vs30=760.0,
                fault_type=scenario.fault_type, period=1.0,
            )
            assert median > 0, f"Negative Sa at {dist} km"
            assert sigma > 0, f"Negative sigma at {dist} km"

        # Sa should decrease with distance (attenuation)
        sa_near, _ = gmpe.compute(Mw=6.7, R_JB=5.0, Vs30=760.0,
                                   fault_type="reverse", period=1.0)
        sa_far, _ = gmpe.compute(Mw=6.7, R_JB=100.0, Vs30=760.0,
                                  fault_type="reverse", period=1.0)
        assert sa_near > sa_far, "Sa should attenuate with distance"

    def test_damage_distribution_sums_to_one(self):
        """Portfolio damage fractions must sum to 1.0."""
        from src.config import load_config, IM_COLUMN_MAP
        from src.data_loader import load_shakemap, load_nbi, classify_nbi_to_hazus
        from src.interpolation import interpolate_im
        from src.fragility import damage_state_probabilities

        cfg = load_config(PROJECT_ROOT / "config.yaml")
        grid = load_shakemap(PROJECT_ROOT / "data" / "grid.xml")
        nbi = load_nbi(PROJECT_ROOT / "data" / "CA24.txt",
                       northridge_bbox=cfg.region)
        nbi = classify_nbi_to_hazus(nbi)
        nbi_small = nbi.head(20)

        im_col = IM_COLUMN_MAP.get(cfg.im_type, "PSA10")
        sa = interpolate_im(
            grid["LAT"].values, grid["LON"].values, grid[im_col].values,
            nbi_small["latitude"].values, nbi_small["longitude"].values,
            method="nearest",
        )

        for i, (_, row) in enumerate(nbi_small.iterrows()):
            probs = damage_state_probabilities(sa[i], row["hwb_class"])
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-6, (
                f"Bridge {row['structure_number']}: probs sum to {total}"
            )

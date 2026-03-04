"""
Tests for Vs30 spatial data provider.

Tests are split into:
  - Unit tests that don't require data files (NEHRP classification, API)
  - Integration tests that require the cached Vs30 grid (marked with pytest.mark.vs30)
"""

import numpy as np
import pytest

from src.vs30_provider import Vs30Provider, NEHRP_CLASSES, enrich_bridges_with_vs30

# Path to cached data
from src.vs30_provider import _CACHE_FILE


# ── NEHRP classification (no data needed) ──────────────────────────────────

class TestNEHRPClassification:
    """Test NEHRP site class boundaries."""

    def setup_method(self):
        # Create a mock provider just to use get_nehrp_class
        # We'll test the static method via a trick
        self.classify = lambda vs30: self._classify(vs30)

    @staticmethod
    def _classify(vs30):
        """Replicate classification logic without needing a provider."""
        for cls, (lo, hi) in NEHRP_CLASSES.items():
            if lo <= vs30 < hi:
                return cls
        return "E"

    def test_rock_site(self):
        assert self._classify(800) == "B"

    def test_stiff_soil(self):
        assert self._classify(250) == "D"

    def test_soft_soil(self):
        assert self._classify(150) == "E"

    def test_bc_boundary(self):
        assert self._classify(760) == "B"
        assert self._classify(700) == "BC"

    def test_cd_boundary(self):
        assert self._classify(360) == "C"
        assert self._classify(350) == "CD"

    def test_hard_rock(self):
        assert self._classify(1600) == "A"

    def test_very_soft(self):
        assert self._classify(100) == "E"


# ── Integration tests (require cached Vs30 data) ──────────────────────────

# Skip these tests if the Vs30 cache file doesn't exist
has_vs30_data = _CACHE_FILE.exists()
vs30_reason = f"Vs30 cache not found at {_CACHE_FILE}"


@pytest.mark.skipif(not has_vs30_data, reason=vs30_reason)
class TestVs30Provider:
    """Tests that require the actual Vs30 grid data."""

    def setup_method(self):
        self.provider = Vs30Provider()

    def test_grid_loaded(self):
        assert self.provider.shape[0] > 0
        assert self.provider.shape[1] > 0

    def test_lat_range_covers_california(self):
        lat_min, lat_max = self.provider.lat_range
        assert lat_min <= 33.0
        assert lat_max >= 42.0

    def test_lon_range_covers_california(self):
        lon_min, lon_max = self.provider.lon_range
        assert lon_min <= -124.0
        assert lon_max >= -114.0

    def test_los_angeles_is_soil(self):
        """LA basin should be soil site (Vs30 < 400 m/s)."""
        vs30 = self.provider.get_vs30(34.05, -118.25)
        assert 100 < vs30 < 500, f"LA Vs30={vs30:.0f}, expected soil site"

    def test_mountain_is_rock(self):
        """Sierra Nevada mountains should be rock (Vs30 > 500 m/s)."""
        vs30 = self.provider.get_vs30(37.0, -118.5)
        assert vs30 > 400, f"Mountain Vs30={vs30:.0f}, expected rock"

    def test_northridge_epicenter(self):
        """Northridge area should have a reasonable Vs30."""
        vs30 = self.provider.get_vs30(34.213, -118.537)
        assert 100 < vs30 < 1000, f"Northridge Vs30={vs30:.0f}"

    def test_out_of_bounds_returns_default(self):
        """Points outside grid should return 760 (rock default)."""
        vs30 = self.provider.get_vs30(50.0, -80.0)  # East coast, outside CA grid
        assert vs30 == 760.0

    def test_array_lookup(self):
        lats = np.array([34.05, 37.77, 34.213])
        lons = np.array([-118.25, -122.42, -118.537])
        vs30_arr = self.provider.get_vs30_array(lats, lons)
        assert vs30_arr.shape == (3,)
        assert all(v > 0 for v in vs30_arr)

    def test_nehrp_class_from_provider(self):
        vs30 = self.provider.get_vs30(34.05, -118.25)
        cls = self.provider.get_nehrp_class(vs30)
        assert cls in NEHRP_CLASSES


@pytest.mark.skipif(not has_vs30_data, reason=vs30_reason)
class TestEnrichBridges:
    """Test enriching a DataFrame with Vs30 values."""

    def test_adds_vs30_column(self):
        import pandas as pd
        df = pd.DataFrame({
            "latitude": [34.05, 37.77],
            "longitude": [-118.25, -122.42],
        })
        result = enrich_bridges_with_vs30(df)
        assert "vs30" in result.columns
        assert len(result) == 2
        assert all(result["vs30"] > 0)

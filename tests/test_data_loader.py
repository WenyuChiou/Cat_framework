"""
Tests for data loading (ShakeMap XML, NBI text, bridge classification).

Covers:
  - ShakeMap parsing output columns and units
  - NBI loading and column extraction
  - HWB classification decision tree
  - Error handling for missing files
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import parse_shakemap_grid, load_shakemap, load_nbi, classify_nbi_to_hazus


# ── ShakeMap parser ──────────────────────────────────────────────────

class TestShakeMapParser:
    @pytest.fixture(autouse=True)
    def _load_grid(self):
        grid_path = Path(__file__).parent.parent / "data" / "grid.xml"
        if not grid_path.exists():
            pytest.skip("grid.xml not available")
        self.grid = parse_shakemap_grid(grid_path)

    def test_output_is_dataframe(self):
        assert isinstance(self.grid, pd.DataFrame)

    def test_has_required_columns(self):
        for col in ["LAT", "LON", "PGA", "PSA03", "PSA10", "PSA30"]:
            assert col in self.grid.columns, f"Missing column: {col}"

    def test_pga_in_g_not_percent(self):
        """PGA should be in g (0-2 range), not %g (0-200 range)."""
        assert self.grid["PGA"].max() < 5.0, "PGA appears to be in %g, not g"

    def test_psa10_in_g(self):
        assert self.grid["PSA10"].max() < 5.0

    def test_lat_lon_ranges(self):
        assert self.grid["LAT"].min() > 28.0
        assert self.grid["LON"].max() < -110.0

    def test_no_negative_im(self):
        assert (self.grid["PGA"] >= 0).all()
        assert (self.grid["PSA10"] >= 0).all()

    def test_grid_size(self):
        """Northridge ShakeMap should have thousands of grid points."""
        assert len(self.grid) > 1000


class TestShakeMapErrors:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_shakemap_grid("nonexistent_grid.xml")


# ── NBI loader ──────────────────────────────────────────────────────

class TestNBILoader:
    @pytest.fixture(autouse=True)
    def _load_nbi(self):
        nbi_path = Path(__file__).parent.parent / "data" / "CA24.txt"
        if not nbi_path.exists():
            pytest.skip("CA24.txt not available")
        self.nbi = load_nbi(nbi_path)

    def test_output_is_dataframe(self):
        assert isinstance(self.nbi, pd.DataFrame)

    def test_has_core_columns(self):
        for col in ["structure_number", "latitude", "longitude", "year_built"]:
            assert col in self.nbi.columns, f"Missing column: {col}"

    def test_california_only(self):
        """All bridges should be in California lat/lon range."""
        assert self.nbi["latitude"].min() > 30.0
        assert self.nbi["latitude"].max() < 43.0

    def test_bridge_count(self):
        """California has 25,000+ bridges."""
        assert len(self.nbi) > 20000

    def test_bbox_filtering(self):
        bbox = {"lat_min": 33.8, "lat_max": 34.6,
                "lon_min": -118.9, "lon_max": -118.0}
        filtered = load_nbi(
            Path(__file__).parent.parent / "data" / "CA24.txt",
            northridge_bbox=bbox,
        )
        assert len(filtered) < len(self.nbi)
        assert (filtered["latitude"] >= 33.8).all()
        assert (filtered["latitude"] <= 34.6).all()


# ── HWB classification ──────────────────────────────────────────────

class TestClassification:
    @pytest.fixture(autouse=True)
    def _load_and_classify(self):
        nbi_path = Path(__file__).parent.parent / "data" / "CA24.txt"
        if not nbi_path.exists():
            pytest.skip("CA24.txt not available")
        bbox = {"lat_min": 33.8, "lat_max": 34.6,
                "lon_min": -118.9, "lon_max": -118.0}
        nbi = load_nbi(nbi_path, northridge_bbox=bbox)
        self.classified = classify_nbi_to_hazus(nbi)

    def test_hwb_class_column_exists(self):
        assert "hwb_class" in self.classified.columns

    def test_all_hwb_classes_valid(self):
        valid_classes = {f"HWB{i}" for i in range(1, 29)}
        actual = set(self.classified["hwb_class"].unique())
        assert actual.issubset(valid_classes), f"Invalid classes: {actual - valid_classes}"

    def test_no_null_hwb(self):
        assert self.classified["hwb_class"].notna().all()

    def test_material_column(self):
        assert "material" in self.classified.columns
        assert self.classified["material"].notna().all()

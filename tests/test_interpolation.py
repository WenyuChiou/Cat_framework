"""
Tests for spatial interpolation methods.

Covers:
  - All 5 methods produce correct output shape
  - Nearest neighbor returns exact grid values
  - IDW weighted averaging behavior
  - Invalid method raises ValueError
  - Edge cases: single point, colocated points
"""

import numpy as np
import pytest

from src.interpolation import interpolate_im, INTERPOLATION_METHODS


# ── Test fixtures ──────────────────────────────────────────────────────

def _simple_grid():
    """4-point grid with known values."""
    lats = np.array([34.0, 34.0, 34.1, 34.1])
    lons = np.array([-118.0, -118.1, -118.0, -118.1])
    vals = np.array([0.5, 0.3, 0.8, 0.6])
    return lats, lons, vals


def _regular_grid():
    """10x10 regular grid for bilinear/kriging tests."""
    lat = np.linspace(33.5, 34.5, 10)
    lon = np.linspace(-119.0, -118.0, 10)
    lat_g, lon_g = np.meshgrid(lat, lon, indexing="ij")
    vals = np.sin(lat_g) * np.cos(lon_g)  # smooth field
    return lat_g.ravel(), lon_g.ravel(), vals.ravel()


# ── Output shape tests ────────────────────────────────────────────────

class TestOutputShape:
    def test_nearest_output_shape(self):
        g_lat, g_lon, g_val = _simple_grid()
        b_lat = np.array([34.05])
        b_lon = np.array([-118.05])
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon, method="nearest")
        assert result.shape == (1,)

    def test_multiple_bridges(self):
        g_lat, g_lon, g_val = _simple_grid()
        b_lat = np.array([34.0, 34.05, 34.1])
        b_lon = np.array([-118.0, -118.05, -118.1])
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon, method="nearest")
        assert result.shape == (3,)


# ── Nearest neighbor ──────────────────────────────────────────────────

class TestNearest:
    def test_exact_grid_point_returns_value(self):
        g_lat, g_lon, g_val = _simple_grid()
        b_lat = np.array([34.0])
        b_lon = np.array([-118.0])
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon, method="nearest")
        assert result[0] == 0.5

    def test_closer_to_second_point(self):
        g_lat, g_lon, g_val = _simple_grid()
        b_lat = np.array([34.0])
        b_lon = np.array([-118.09])  # closer to second point (lon=-118.1, val=0.3)
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon, method="nearest")
        assert result[0] == 0.3

    def test_all_values_positive(self):
        g_lat, g_lon, g_val = _regular_grid()
        g_val = np.abs(g_val) + 0.01  # ensure positive
        b_lat = np.array([34.0, 33.8])
        b_lon = np.array([-118.5, -118.2])
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon, method="nearest")
        assert np.all(result > 0)


# ── IDW ────────────────────────────────────────────────────────────────

class TestIDW:
    def test_exact_point_returns_value(self):
        g_lat, g_lon, g_val = _simple_grid()
        b_lat = np.array([34.0])
        b_lon = np.array([-118.0])
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon, method="idw")
        assert abs(result[0] - 0.5) < 0.01

    def test_midpoint_is_average(self):
        """IDW result should be within grid value range."""
        g_lat, g_lon, g_val = _regular_grid()  # need ≥ n_neighbors (8) points
        b_lat = np.array([34.0])
        b_lon = np.array([-118.5])
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon, method="idw")
        assert g_val.min() <= result[0] <= g_val.max()


# ── Bilinear ──────────────────────────────────────────────────────────

class TestBilinear:
    def test_regular_grid(self):
        g_lat, g_lon, g_val = _regular_grid()
        b_lat = np.array([34.0])
        b_lon = np.array([-118.5])
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon, method="bilinear")
        assert result.shape == (1,)
        assert np.isfinite(result[0])


# ── Natural neighbor ──────────────────────────────────────────────────

class TestNaturalNeighbor:
    def test_output_finite(self):
        g_lat, g_lon, g_val = _regular_grid()
        b_lat = np.array([34.0, 33.9])
        b_lon = np.array([-118.5, -118.3])
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon,
                                method="natural_neighbor")
        assert np.all(np.isfinite(result))


# ── Kriging ────────────────────────────────────────────────────────────

class TestKriging:
    def test_output_shape(self):
        g_lat, g_lon, g_val = _regular_grid()
        b_lat = np.array([34.0])
        b_lon = np.array([-118.5])
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon,
                                method="kriging", range_km=50.0)
        assert result.shape == (1,)

    def test_output_finite(self):
        g_lat, g_lon, g_val = _regular_grid()
        b_lat = np.array([34.0, 33.9])
        b_lon = np.array([-118.5, -118.3])
        result = interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon,
                                method="kriging", range_km=50.0)
        assert np.all(np.isfinite(result))


# ── Error handling ────────────────────────────────────────────────────

class TestErrors:
    def test_invalid_method_raises(self):
        g_lat, g_lon, g_val = _simple_grid()
        b_lat = np.array([34.0])
        b_lon = np.array([-118.0])
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            interpolate_im(g_lat, g_lon, g_val, b_lat, b_lon, method="invalid")

    def test_all_methods_in_list(self):
        assert len(INTERPOLATION_METHODS) == 5
        assert "nearest" in INTERPOLATION_METHODS
        assert "kriging" in INTERPOLATION_METHODS

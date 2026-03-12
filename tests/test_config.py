"""
Tests for configuration loading and validation.

Covers:
  - Default config creation
  - YAML loading with various sections
  - IM type validation (fail-fast)
  - GMPE scenario requirement
  - Config properties (im_column, bbox)
"""

import tempfile
from pathlib import Path

import pytest

from src.config import AnalysisConfig, load_config, validate_config, IM_COLUMN_MAP


# ── Default config ──────────────────────────────────────────────────────

class TestAnalysisConfigDefaults:
    def test_default_im_source(self):
        cfg = AnalysisConfig()
        assert cfg.im_source == "shakemap"

    def test_default_im_type(self):
        cfg = AnalysisConfig()
        assert cfg.im_type == "SA10"

    def test_default_interpolation(self):
        cfg = AnalysisConfig()
        assert cfg.interpolation_method == "nearest"

    def test_default_region_is_none(self):
        cfg = AnalysisConfig()
        assert cfg.region is None

    def test_default_bbox_is_none(self):
        cfg = AnalysisConfig()
        assert cfg.bbox is None

    def test_im_column_property(self):
        cfg = AnalysisConfig()
        assert cfg.im_column == "PSA10"

    def test_im_column_for_pga(self):
        cfg = AnalysisConfig(im_type="PGA")
        assert cfg.im_column == "PGA"

    def test_bbox_property(self):
        cfg = AnalysisConfig(region={
            "lat_min": 33.8, "lat_max": 34.6,
            "lon_min": -118.9, "lon_max": -118.0,
        })
        assert cfg.bbox == [33.8, 34.6, -118.9, -118.0]

    def test_bbox_dict_property(self):
        region = {"lat_min": 33.8, "lat_max": 34.6,
                  "lon_min": -118.9, "lon_max": -118.0}
        cfg = AnalysisConfig(region=region)
        assert cfg.bbox_dict == region


# ── YAML loading ──────────────────────────────────────────────────────

class TestLoadConfig:
    def test_missing_file_returns_defaults(self):
        cfg = load_config("nonexistent_config_xyz.yaml")
        assert cfg.im_type == "SA10"
        assert cfg.im_source == "shakemap"

    def test_load_minimal_yaml(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("im_type: SA10\nim_source: shakemap\n",
                               encoding="utf-8")
        cfg = load_config(config_file)
        assert cfg.im_type == "SA10"

    def test_load_region(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            "region:\n  lat_min: 33.0\n  lat_max: 35.0\n"
            "  lon_min: -119.0\n  lon_max: -117.0\n",
            encoding="utf-8",
        )
        cfg = load_config(config_file)
        assert cfg.region["lat_min"] == 33.0
        assert cfg.region["lon_max"] == -117.0

    def test_load_interpolation_method(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            "interpolation:\n  method: kriging\n  range_km: 50.0\n",
            encoding="utf-8",
        )
        cfg = load_config(config_file)
        assert cfg.interpolation_method == "kriging"
        assert cfg.interpolation_params["range_km"] == 50.0

    def test_load_hwb_filter_as_list(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("hwb_filter: [HWB3, HWB5]\n", encoding="utf-8")
        cfg = load_config(config_file)
        assert cfg.hwb_filter == ["HWB3", "HWB5"]

    def test_load_hwb_filter_as_single(self, tmp_path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("hwb_filter: HWB3\n", encoding="utf-8")
        cfg = load_config(config_file)
        assert cfg.hwb_filter == ["HWB3"]


# ── Validation ──────────────────────────────────────────────────────────

class TestValidateConfig:
    def test_valid_sa10_passes(self):
        cfg = AnalysisConfig(im_type="SA10")
        validate_config(cfg)  # should not raise

    def test_invalid_im_type_raises(self):
        cfg = AnalysisConfig(im_type="INVALID")
        with pytest.raises(ValueError, match="Unknown im_type"):
            validate_config(cfg)

    def test_non_sa10_without_overrides_raises(self):
        cfg = AnalysisConfig(im_type="PGA")
        with pytest.raises(ValueError, match="fragility_overrides"):
            validate_config(cfg)

    def test_non_sa10_with_overrides_passes(self):
        cfg = AnalysisConfig(
            im_type="PGA",
            fragility_overrides={"HWB1": {"slight": {"median": 0.3, "beta": 0.6}}},
        )
        validate_config(cfg)  # should not raise

    def test_gmpe_without_scenario_raises(self):
        cfg = AnalysisConfig(im_source="gmpe")
        with pytest.raises(ValueError, match="gmpe_scenario"):
            validate_config(cfg)

    def test_gmpe_with_scenario_passes(self):
        cfg = AnalysisConfig(
            im_source="gmpe",
            gmpe_scenario={"Mw": 6.7, "lat": 34.2, "lon": -118.5,
                           "fault_type": "reverse"},
        )
        validate_config(cfg)  # should not raise


# ── IM_COLUMN_MAP ──────────────────────────────────────────────────────

class TestIMColumnMap:
    def test_all_types_present(self):
        assert set(IM_COLUMN_MAP.keys()) == {"PGA", "SA03", "SA10", "SA30"}

    def test_sa10_maps_to_psa10(self):
        assert IM_COLUMN_MAP["SA10"] == "PSA10"

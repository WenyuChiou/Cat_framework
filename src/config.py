"""
Configuration Loader for CAT411 Framework.

Loads analysis configuration from a YAML file, providing defaults
for all optional settings. Supports bridge selection filters,
IM type selection, fragility overrides, and calibration factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── IM type mapping: config name -> ShakeMap column ──────────────────────

IM_COLUMN_MAP = {
    "PGA":  "PGA",
    "SA03": "PSA03",   # Sa(0.3s)
    "SA10": "PSA10",   # Sa(1.0s) — default
    "SA30": "PSA30",   # Sa(3.0s)
}

# ── Configuration dataclass ──────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    """Parsed analysis configuration."""

    # Region
    region: Optional[dict[str, float]] = None  # lat_min/max, lon_min/max

    # Bridge selection
    bridge_selection: dict[str, Any] = field(default_factory=dict)
    hwb_filter: Optional[list[str]] = None
    design_era: Optional[str] = None          # "conventional" or "seismic"
    material_filter: Optional[list[str]] = None

    # IM source
    im_source: str = "shakemap"        # "shakemap" or "gmpe"
    im_type: str = "SA10"

    # Spatial interpolation
    interpolation_method: str = "nearest"
    interpolation_params: dict[str, Any] = field(default_factory=dict)

    # GMPE scenario (when im_source == "gmpe")
    gmpe_scenario: Optional[dict[str, Any]] = None
    gmpe_model: str = "bssa21"          # GMPE model name (default: bssa21)

    # Fragility overrides
    fragility_overrides: dict[str, dict] = field(default_factory=dict)

    # Calibration
    global_median_factor: float = 1.0
    class_factors: dict[str, float] = field(default_factory=dict)

    # Analysis settings
    n_realizations: int = 50
    n_events: int = 50
    seed: int = 42

    # Validation settings
    validation_enabled: bool = False
    validation_data: Optional[str] = None       # path to validation CSV
    validation_im_source: str = "gmpe"          # "gmpe" or "shakemap"
    validation_levels: list[int] = field(default_factory=lambda: [1, 2, 3])
    validation_stationlist: Optional[str] = None  # path to stationlist.json
    validation_output_dir: str = "output/validation"

    @property
    def im_column(self) -> str:
        """ShakeMap column name for the selected IM type."""
        return IM_COLUMN_MAP.get(self.im_type, "PSA10")

    @property
    def bbox(self) -> Optional[list[float]]:
        """Return region as [lat_min, lat_max, lon_min, lon_max] or None."""
        if self.region is None:
            return None
        r = self.region
        return [r["lat_min"], r["lat_max"], r["lon_min"], r["lon_max"]]

    @property
    def bbox_dict(self) -> Optional[dict]:
        """Return region as dict for load_nbi(), or None."""
        if self.region is None:
            return None
        return self.region


# ── Loader ───────────────────────────────────────────────────────────────

def load_config(path: str | Path = "config.yaml") -> AnalysisConfig:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to YAML config file.

    Returns
    -------
    AnalysisConfig
    """
    path = Path(path)
    if not path.exists():
        print(f"[Config] No config file found at {path}, using defaults.")
        return AnalysisConfig()

    try:
        import yaml
    except ImportError:
        print("[Config] PyYAML not installed. Install with: pip install pyyaml")
        print("[Config] Using default configuration.")
        return AnalysisConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = AnalysisConfig()

    # Region
    if "region" in raw and isinstance(raw["region"], dict):
        cfg.region = raw["region"]

    # Bridge selection
    if "bridge_selection" in raw and isinstance(raw["bridge_selection"], dict):
        cfg.bridge_selection = raw["bridge_selection"]

    # Specific filters
    if "hwb_filter" in raw:
        val = raw["hwb_filter"]
        cfg.hwb_filter = val if isinstance(val, list) else [val]

    if "design_era" in raw:
        cfg.design_era = raw["design_era"]

    if "material_filter" in raw:
        val = raw.get("material_filter")
        if val:
            cfg.material_filter = val if isinstance(val, list) else [val]

    # IM source
    if "im_source" in raw:
        src = raw["im_source"].lower()
        if src in ("shakemap", "gmpe"):
            cfg.im_source = src
        else:
            print(f"[Config] Unknown im_source '{src}', using 'shakemap'.")

    # IM type
    if "im_type" in raw:
        im = str(raw["im_type"]).upper()
        if im in IM_COLUMN_MAP:
            cfg.im_type = im
        else:
            raise ValueError(
                f"Unknown im_type '{raw['im_type']}'. "
                f"Valid options: {list(IM_COLUMN_MAP.keys())}"
            )

    # Interpolation
    interp = raw.get("interpolation", {})
    if isinstance(interp, dict):
        cfg.interpolation_method = interp.get("method", "nearest")
        cfg.interpolation_params = {
            k: v for k, v in interp.items() if k != "method"
        }

    # GMPE scenario
    if "gmpe_scenario" in raw and isinstance(raw["gmpe_scenario"], dict):
        cfg.gmpe_scenario = raw["gmpe_scenario"]

    # GMPE model
    if "gmpe_model" in raw:
        cfg.gmpe_model = str(raw["gmpe_model"]).lower()

    # Fragility overrides
    if "fragility_overrides" in raw and isinstance(raw["fragility_overrides"], dict):
        cfg.fragility_overrides = raw["fragility_overrides"]

    # Calibration
    cal = raw.get("calibration", {})
    if isinstance(cal, dict):
        cfg.global_median_factor = cal.get("global_median_factor", 1.0)
        cfg.class_factors = cal.get("class_factors", {})

    # Analysis settings
    analysis = raw.get("analysis", {})
    if isinstance(analysis, dict):
        cfg.n_realizations = analysis.get("n_realizations", 50)
        cfg.n_events = analysis.get("n_events", 50)
        cfg.seed = analysis.get("seed", 42)

    # Validation settings
    val = raw.get("validation", {})
    if isinstance(val, dict):
        cfg.validation_enabled = bool(val.get("enabled", False))
        cfg.validation_data = val.get("data", None)
        im_src = str(val.get("im_source", "gmpe")).lower()
        if im_src in ("gmpe", "shakemap"):
            cfg.validation_im_source = im_src
        if "levels" in val:
            cfg.validation_levels = list(val["levels"])
        if "stationlist" in val:
            cfg.validation_stationlist = val["stationlist"]
        if "output_dir" in val:
            cfg.validation_output_dir = val["output_dir"]

    validate_config(cfg)

    return cfg


def validate_config(cfg: AnalysisConfig) -> None:
    """
    Validate configuration for internal consistency.

    Raises ValueError for invalid combinations. Call this after any
    mutation of the config object (e.g., CLI overrides).
    """
    if cfg.im_type not in IM_COLUMN_MAP:
        raise ValueError(
            f"Unknown im_type '{cfg.im_type}'. "
            f"Valid options: {list(IM_COLUMN_MAP.keys())}"
        )
    if cfg.im_type != "SA10" and not cfg.fragility_overrides:
        raise ValueError(
            f"Configuration error: im_type='{cfg.im_type}' but no fragility_overrides "
            f"provided. Default Hazus fragility parameters are calibrated for Sa(1.0s) "
            f"only. Either set im_type: SA10 or provide fragility_overrides with "
            f"parameters calibrated for {cfg.im_type}."
        )
    # GMPE-specific validation
    if cfg.im_source == "gmpe":
        if not cfg.gmpe_scenario:
            raise ValueError(
                "Configuration error: im_source='gmpe' requires a gmpe_scenario "
                "section with at least Mw, lat, lon, and fault_type."
            )
        # Validate gmpe_model is registered (lazy import to avoid circular deps)
        try:
            import src.gmpe_bssa21  # noqa: F401 — registers BSSA21
            import src.gmpe_nga_simplified  # noqa: F401 — registers 7 simplified models
            from src.gmpe_base import GMPE_REGISTRY
            if cfg.gmpe_model not in GMPE_REGISTRY:
                raise ValueError(
                    f"Unknown gmpe_model '{cfg.gmpe_model}'. "
                    f"Available: {list(GMPE_REGISTRY.keys())}"
                )
        except ImportError as e:
            if "numpy" in str(e).lower():
                pass  # numpy not available; skip registry check
            else:
                raise

    # Validation-specific checks
    if cfg.validation_enabled:
        if not cfg.validation_data:
            raise ValueError(
                "Configuration error: validation.enabled=true but no "
                "validation.data path specified."
            )
        # Resolve relative paths from project root
        data_path = Path(cfg.validation_data)
        if not data_path.is_absolute():
            data_path = Path(__file__).parent.parent / data_path
        if not data_path.exists():
            raise ValueError(
                f"Configuration error: validation data file not found: "
                f"'{data_path}'"
            )
        if cfg.validation_im_source not in ("gmpe", "shakemap"):
            raise ValueError(
                f"Configuration error: validation.im_source must be "
                f"'gmpe' or 'shakemap', got '{cfg.validation_im_source}'"
            )


def print_config_summary(cfg: AnalysisConfig) -> None:
    """Print a summary of the active configuration."""
    print("=" * 60)
    print("ANALYSIS CONFIGURATION")
    print("=" * 60)

    if cfg.region:
        r = cfg.region
        print(f"  Region: lat[{r['lat_min']}, {r['lat_max']}] "
              f"lon[{r['lon_min']}, {r['lon_max']}]")
    else:
        print("  Region: Default (greater LA / Northridge)")

    if cfg.bridge_selection:
        print(f"  NBI Filters: {cfg.bridge_selection}")
    if cfg.hwb_filter:
        print(f"  HWB Filter: {cfg.hwb_filter}")
    if cfg.design_era:
        print(f"  Design Era: {cfg.design_era}")
    if cfg.material_filter:
        print(f"  Material Filter: {cfg.material_filter}")

    print(f"  IM Source: {cfg.im_source}")
    print(f"  IM Type: {cfg.im_type} (ShakeMap column: {cfg.im_column})")
    print(f"  Interpolation: {cfg.interpolation_method}")
    if cfg.interpolation_params:
        print(f"    Params: {cfg.interpolation_params}")

    if cfg.im_source == "gmpe":
        print(f"  GMPE Model: {cfg.gmpe_model}")
        if cfg.gmpe_scenario:
            s = cfg.gmpe_scenario
            print(f"  GMPE Scenario: Mw={s.get('Mw')}, "
                  f"({s.get('lat')}, {s.get('lon')}), "
                  f"depth={s.get('depth_km')}km, "
                  f"fault={s.get('fault_type', 'reverse')}")
        else:
            print("  WARNING: im_source=gmpe but no gmpe_scenario defined!")

    if cfg.im_type != "SA10":
        if cfg.fragility_overrides:
            print(f"  NOTE: im_type='{cfg.im_type}' with user-supplied fragility_overrides.")
        else:
            # This should not be reachable if validate_config() was called,
            # but print a warning just in case.
            print(f"  !! WARNING: im_type='{cfg.im_type}' without fragility_overrides.")

    if cfg.fragility_overrides:
        print(f"  Fragility Overrides: {list(cfg.fragility_overrides.keys())}")

    if cfg.global_median_factor != 1.0:
        print(f"  Global Calibration Factor: {cfg.global_median_factor}")
    if cfg.class_factors:
        print(f"  Class Calibration Factors: {cfg.class_factors}")

    print(f"  Realizations: {cfg.n_realizations}, Events: {cfg.n_events}, "
          f"Seed: {cfg.seed}")

    if cfg.validation_enabled:
        print(f"  Validation: ENABLED")
        print(f"    Data: {cfg.validation_data}")
        print(f"    IM Source: {cfg.validation_im_source}")

    print("=" * 60)
    print()

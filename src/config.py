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

    # Intensity Measure
    im_type: str = "SA10"

    # Fragility overrides
    fragility_overrides: dict[str, dict] = field(default_factory=dict)

    # Calibration
    global_median_factor: float = 1.0
    class_factors: dict[str, float] = field(default_factory=dict)

    # Analysis settings
    n_realizations: int = 50
    n_events: int = 50
    seed: int = 42

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

    # IM type
    if "im_type" in raw:
        im = raw["im_type"].upper()
        if im in IM_COLUMN_MAP:
            cfg.im_type = im
        else:
            print(f"[Config] Unknown im_type '{raw['im_type']}', "
                  f"using SA10. Valid: {list(IM_COLUMN_MAP.keys())}")

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

    return cfg


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

    print(f"  IM Type: {cfg.im_type} (ShakeMap column: {cfg.im_column})")

    if cfg.im_type != "SA10":
        print("  ⚠ WARNING: Hazus fragility params are calibrated for Sa(1.0s).")
        print("    Results with other IMs may be inaccurate unless you provide")
        print("    fragility_overrides in config.yaml.")

    if cfg.fragility_overrides:
        print(f"  Fragility Overrides: {list(cfg.fragility_overrides.keys())}")

    if cfg.global_median_factor != 1.0:
        print(f"  Global Calibration Factor: {cfg.global_median_factor}")
    if cfg.class_factors:
        print(f"  Class Calibration Factors: {cfg.class_factors}")

    print(f"  Realizations: {cfg.n_realizations}, Events: {cfg.n_events}, "
          f"Seed: {cfg.seed}")
    print("=" * 60)
    print()

"""
USGS Hazard Data Download Module.

Downloads seismic hazard data from USGS APIs for the CAT411 framework:
- ShakeMap: Event-specific ground motion maps (grid.xml, shape.zip)
- NSHMP Hazard Curves: Probabilistic hazard curves for any location
- Design Maps: Spectral design values per ASCE 7

Focused on California/Northridge case study.

References:
    - USGS Earthquake API: https://earthquake.usgs.gov/fdsnws/event/1/
    - USGS NSHMP Hazard: https://earthquake.usgs.gov/nshmp/
    - USGS ShakeMap: https://earthquake.usgs.gov/data/shakemap/
"""

from __future__ import annotations

import json
import shutil
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


# ── Configuration ─────────────────────────────────────────────────────────

# Default output directories
DEFAULT_HAZARD_DIR = Path("data/hazard/usgs")
DEFAULT_SHAKEMAP_DIR = DEFAULT_HAZARD_DIR / "shakemap"
DEFAULT_HAZARD_CURVES_DIR = DEFAULT_HAZARD_DIR / "hazard_curves"

# Northridge case study defaults (1994 Northridge earthquake)
NORTHRIDGE_EVENT_ID = "ci3144585"
NORTHRIDGE_LAT = 34.213
NORTHRIDGE_LON = -118.537

# USGS API endpoints
USGS_EARTHQUAKE_API = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail"
USGS_NSHMP_HAZARD_API = "https://earthquake.usgs.gov/nshmp/ws/hazard"
USGS_DESIGN_MAPS_API = "https://earthquake.usgs.gov/ws/designmaps"


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass
class ShakeMapData:
    """Container for downloaded ShakeMap data."""
    event_id: str
    grid_path: Optional[Path] = None
    info_path: Optional[Path] = None
    shape_path: Optional[Path] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HazardCurveData:
    """Container for downloaded hazard curve data."""
    latitude: float
    longitude: float
    vs30: int
    edition: str
    imt: str
    curves: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: Dict[str, Any] = field(default_factory=dict)


# ── Utility Functions ─────────────────────────────────────────────────────

def _now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir(path: Path) -> None:
    """Create directory and parents if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)


def _fetch_json(url: str) -> Dict[str, Any]:
    """Fetch JSON from URL and parse."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        content_type = resp.headers.get("Content-Type", "")
        text = data.decode("utf-8", errors="replace").strip()

        if not text:
            raise RuntimeError("Empty response body from USGS API")

        # NSHMP endpoints sometimes return HTML/plain text on failures.
        if "json" not in content_type.lower():
            preview = text[:300].replace("\n", " ")
            raise RuntimeError(
                f"Unexpected content-type '{content_type}' from USGS API. "
                f"Response preview: {preview}"
            )

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            preview = text[:300].replace("\n", " ")
            raise RuntimeError(
                f"Invalid JSON from USGS API: {e}. Response preview: {preview}"
            ) from e


def _probe_json_endpoint(url: str) -> None:
    """
    Pre-flight probe for a JSON API endpoint.

    Raises a RuntimeError when the endpoint does not look like JSON.
    """
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if "json" not in content_type.lower():
            preview = resp.read(300).decode("utf-8", errors="replace").replace("\n", " ")
            raise RuntimeError(
                f"Endpoint is not returning JSON (content-type={content_type}). "
                f"Response preview: {preview}"
            )


def _fetch_text(url: str) -> str:
    """Fetch text content from URL."""
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _download_file(url: str, dest: Path, overwrite: bool = False) -> bool:
    """Download file from URL to destination path."""
    if dest.exists() and not overwrite:
        print(f"    [Skip] {dest.name} already exists")
        return False
    
    _ensure_dir(dest.parent)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            with dest.open("wb") as f:
                shutil.copyfileobj(resp, f)
        return True
    except Exception as e:
        print(f"    [Error] Failed to download {url}: {e}")
        return False


def _write_meta(path: Path, meta: Dict[str, Any]) -> None:
    """Write metadata JSON file."""
    _ensure_dir(path.parent)
    path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")


def _write_log(path: Path, lines: List[str]) -> None:
    """Write log file."""
    _ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── ShakeMap Download ─────────────────────────────────────────────────────

def download_shakemap(
    event_id: str = NORTHRIDGE_EVENT_ID,
    output_dir: Path = DEFAULT_SHAKEMAP_DIR,
    files: List[str] = None,
    overwrite: bool = False,
) -> ShakeMapData:
    """
    Download ShakeMap data from USGS Earthquake API.
    
    Downloads grid.xml, info.json, and shape.zip for a given earthquake event.
    
    Parameters
    ----------
    event_id : str
        USGS event ID (e.g., "ci3144585" for Northridge)
    output_dir : Path
        Output directory for downloaded files
    files : list of str, optional
        Specific files to download. Defaults to ["grid.xml", "info.json", "shape.zip"]
    overwrite : bool
        Whether to overwrite existing files
        
    Returns
    -------
    ShakeMapData
        Container with paths to downloaded files and metadata
    """
    if files is None:
        files = ["grid.xml", "info.json", "shape.zip"]
    
    print(f"[ShakeMap] Downloading event: {event_id}")
    
    # Fetch event details
    detail_url = f"{USGS_EARTHQUAKE_API}/{event_id}.geojson"
    print(f"  Fetching event detail: {detail_url}")
    
    try:
        data = _fetch_json(detail_url)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch event details: {e}")
    
    # Find shakemap product
    products = data.get("properties", {}).get("products", {}).get("shakemap", [])
    if not products:
        raise ValueError(f"No ShakeMap product found for event {event_id}")
    
    # Pick preferred product
    product = next((p for p in products if p.get("preferred")), products[0])
    contents = product.get("contents", {})
    
    # Setup directories
    raw_dir = output_dir / "raw"
    meta_dir = output_dir / "meta"
    log_dir = output_dir / "logs"
    _ensure_dir(raw_dir)
    _ensure_dir(meta_dir)
    _ensure_dir(log_dir)
    
    # Download files
    result = ShakeMapData(event_id=event_id)
    downloaded = []
    
    for fname in files:
        # Try multiple content key patterns
        keys = [fname, f"download/{fname}"]
        content_entry = None
        for key in keys:
            if key in contents:
                content_entry = contents[key]
                break
        
        if not content_entry or "url" not in content_entry:
            print(f"    [Skip] {fname} not available")
            continue
        
        url = content_entry["url"]
        dest = raw_dir / fname
        
        print(f"  Downloading {fname}...")
        if _download_file(url, dest, overwrite):
            downloaded.append(str(dest))
            
            # Store paths in result
            if fname == "grid.xml":
                result.grid_path = dest
            elif fname == "info.json":
                result.info_path = dest
            elif fname == "shape.zip":
                result.shape_path = dest
    
    # Write metadata
    result.meta = {
        "event_id": event_id,
        "downloaded_at_utc": _now_iso(),
        "detail_url": detail_url,
        "downloaded_files": downloaded,
        "product": {
            "source": product.get("source"),
            "code": product.get("code"),
            "update_time": product.get("updateTime"),
        },
    }
    meta_path = meta_dir / f"shakemap_{event_id}_meta.json"
    _write_meta(meta_path, result.meta)
    
    # Write log
    log_lines = [
        f"run_utc={_now_iso()}",
        f"event_id={event_id}",
        f"downloaded={len(downloaded)}",
        f"files={downloaded}",
    ]
    log_path = log_dir / f"shakemap_{event_id}_run.log"
    _write_log(log_path, log_lines)
    
    print(f"  Downloaded {len(downloaded)} files to {raw_dir}")

    return result


# ── NSHMP Hazard Curves Download ──────────────────────────────────────────

@dataclass
class HazardCurveData:
    """Container for downloaded hazard curve data."""
    latitude: float
    longitude: float
    vs30: int
    edition: str
    imt: str
    curves: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: Dict[str, Any] = field(default_factory=dict)


def download_hazard_curves(
    latitude: float = NORTHRIDGE_LAT,
    longitude: float = NORTHRIDGE_LON,
    vs30: int = 760,
    edition: str = "E2014",
    imt: str = "SA1P0",
    output_dir: Path = DEFAULT_HAZARD_CURVES_DIR,
    overwrite: bool = False,
) -> HazardCurveData:
    """
    Download probabilistic hazard curves from USGS NSHMP API.
    
    Parameters
    ----------
    latitude : float
        Site latitude (e.g., 34.213 for Northridge)
    longitude : float
        Site longitude (e.g., -118.537 for Northridge)
    vs30 : int
        Vs30 site class in m/s (default: 760, NEHRP B/C boundary)
    edition : str
        NSHM edition (e.g., "E2014" for 2014 CONUS model)
    imt : str
        Intensity measure type:
        - "PGA" for peak ground acceleration
        - "SA0P2" for Sa(0.2s)
        - "SA1P0" for Sa(1.0s)
    output_dir : Path
        Output directory
    overwrite : bool
        Whether to overwrite existing files
        
    Returns
    -------
    HazardCurveData
        Container with hazard curve data and metadata
    """
    print(f"[HazardCurves] Downloading for ({latitude}, {longitude})")
    print(f"  Vs30={vs30} m/s, Edition={edition}, IMT={imt}")
    
    # Build API URL
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "vs30": vs30,
        "edition": edition,
        "imt": imt,
    }
    url = f"{USGS_NSHMP_HAZARD_API}?" + urllib.parse.urlencode(params)
    print(f"  API URL: {url}")
    
    # Setup directories
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    meta_dir = output_dir / "meta"
    log_dir = output_dir / "logs"
    _ensure_dir(raw_dir)
    _ensure_dir(processed_dir)
    _ensure_dir(meta_dir)
    _ensure_dir(log_dir)
    
    # Generate filename based on location
    loc_str = f"{latitude:.3f}_{longitude:.3f}".replace("-", "m").replace(".", "p")
    json_file = raw_dir / f"hazard_{loc_str}_{imt}_{vs30}.json"
    
    # Check if already exists
    if json_file.exists() and not overwrite:
        print(f"  Loading cached data from {json_file}")
        with json_file.open("r") as f:
            data = json.load(f)
    else:
        # Fetch from API
        print("  Pre-flight endpoint check...")
        try:
            _probe_json_endpoint(url)
        except Exception as e:
            raise RuntimeError(f"Hazard API pre-flight failed: {e}")

        print("  Fetching hazard curves from USGS...")
        try:
            data = _fetch_json(url)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch hazard curves: {e}")
        
        # Save raw response
        json_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  Saved raw data to {json_file}")
    
    # Parse hazard curves
    result = HazardCurveData(
        latitude=latitude,
        longitude=longitude,
        vs30=vs30,
        edition=edition,
        imt=imt,
    )
    
    # Extract curve data from response
    try:
        curves_data = _parse_nshmp_response(data)
        result.curves = curves_data
        
        # Save processed CSV
        csv_file = processed_dir / f"hazard_{loc_str}_{imt}_{vs30}.csv"
        curves_data.to_csv(csv_file, index=False)
        print(f"  Saved processed curves to {csv_file}")
    except Exception as e:
        print(f"  [Warning] Could not parse hazard curves: {e}")
    
    # Write metadata
    result.meta = {
        "latitude": latitude,
        "longitude": longitude,
        "vs30": vs30,
        "edition": edition,
        "imt": imt,
        "downloaded_at_utc": _now_iso(),
        "api_url": url,
        "raw_file": str(json_file),
    }
    meta_path = meta_dir / f"hazard_{loc_str}_{imt}_{vs30}_meta.json"
    _write_meta(meta_path, result.meta)
    
    # Write log
    log_lines = [
        f"run_utc={_now_iso()}",
        f"latitude={latitude}",
        f"longitude={longitude}",
        f"vs30={vs30}",
        f"edition={edition}",
        f"imt={imt}",
        f"curves_rows={len(result.curves)}",
    ]
    log_path = log_dir / f"hazard_{loc_str}_{imt}_{vs30}_run.log"
    _write_log(log_path, log_lines)
    
    print(f"  Download complete: {len(result.curves)} data points")
    return result


def _parse_nshmp_response(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse NSHMP hazard API response into a DataFrame.
    
    Returns DataFrame with columns: iml (intensity measure level), 
    annual_rate (annual frequency of exceedance).
    """
    # Response structure varies by API version
    # Try common patterns
    
    # Pattern 1: response.data[0].data (newer API)
    if "response" in data:
        resp = data["response"]
        if isinstance(resp, list) and len(resp) > 0:
            curves = resp[0].get("data", [])
            if curves:
                first_curve = curves[0] if isinstance(curves, list) else curves
                xvals = first_curve.get("xvalues") or first_curve.get("xs") or first_curve.get("imls", [])
                yvals = first_curve.get("yvalues") or first_curve.get("ys") or first_curve.get("rates", [])
                if xvals and yvals:
                    return pd.DataFrame({"iml": xvals, "annual_rate": yvals})
    
    # Pattern 2: data.hazardCurves (older API)
    if "hazardCurves" in data:
        hc = data["hazardCurves"]
        if isinstance(hc, dict):
            xvals = hc.get("xValues", hc.get("imls", []))
            yvals = hc.get("yValues", hc.get("rates", []))
            if xvals and yvals:
                return pd.DataFrame({"iml": xvals, "annual_rate": yvals})
    
    # Pattern 3: Flat structure
    if "xValues" in data or "imls" in data:
        xvals = data.get("xValues", data.get("imls", []))
        yvals = data.get("yValues", data.get("rates", []))
        if xvals and yvals:
            return pd.DataFrame({"iml": xvals, "annual_rate": yvals})
    
    # Return empty if can't parse
    return pd.DataFrame(columns=["iml", "annual_rate"])


# ── Grid-based Hazard Map Download ────────────────────────────────────────

def download_hazard_grid(
    bounding_box: Dict[str, float] = None,
    grid_spacing: float = 0.1,
    vs30: int = 760,
    imt: str = "SA1P0",
    output_dir: Path = DEFAULT_HAZARD_CURVES_DIR,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Download hazard curves for a grid of locations (California focus).
    
    Parameters
    ----------
    bounding_box : dict
        Dict with keys: min_lat, max_lat, min_lon, max_lon
        Default: Northridge area (33.5-35.0, -119.5--117.5)
    grid_spacing : float
        Grid spacing in degrees (default: 0.1 ~ 10km)
    vs30 : int
        Site Vs30 in m/s
    imt : str
        Intensity measure type
    output_dir : Path
        Output directory
    overwrite : bool
        Whether to overwrite existing files
        
    Returns
    -------
    pd.DataFrame
        Grid of hazard values with columns: lat, lon, iml_X, rate_X, ...
    """
    if bounding_box is None:
        # Default: Greater Los Angeles / Northridge area
        bounding_box = {
            "min_lat": 33.5,
            "max_lat": 35.0,
            "min_lon": -119.5,
            "max_lon": -117.5,
        }
    
    print(f"[HazardGrid] Downloading grid for bounding box:")
    print(f"  Lat: {bounding_box['min_lat']} to {bounding_box['max_lat']}")
    print(f"  Lon: {bounding_box['min_lon']} to {bounding_box['max_lon']}")
    print(f"  Spacing: {grid_spacing}° (~{grid_spacing * 111:.1f} km)")
    
    # Generate grid points
    lats = np.arange(bounding_box["min_lat"], bounding_box["max_lat"] + grid_spacing/2, grid_spacing)
    lons = np.arange(bounding_box["min_lon"], bounding_box["max_lon"] + grid_spacing/2, grid_spacing)
    
    n_points = len(lats) * len(lons)
    print(f"  Grid points: {len(lats)} x {len(lons)} = {n_points}")
    
    # Download hazard curves for each point
    results = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            try:
                hc = download_hazard_curves(
                    latitude=float(lat),
                    longitude=float(lon),
                    vs30=vs30,
                    imt=imt,
                    output_dir=output_dir,
                    overwrite=overwrite,
                )
                
                # Extract key values (e.g., 2% in 50 year ~ 0.000404/yr)
                row = {"lat": lat, "lon": lon}
                if not hc.curves.empty:
                    # Find Sa at specific return periods
                    for rp, label in [(475, "10pct_50yr"), (2475, "2pct_50yr")]:
                        target_rate = 1.0 / rp
                        idx = np.abs(hc.curves["annual_rate"].values - target_rate).argmin()
                        row[f"sa_{label}"] = hc.curves["iml"].iloc[idx]
                results.append(row)
            except Exception as e:
                print(f"    [Error] ({lat}, {lon}): {e}")
                results.append({"lat": lat, "lon": lon})
    
    # Combine into DataFrame
    grid_df = pd.DataFrame(results)
    
    # Save grid
    grid_dir = output_dir / "grids"
    _ensure_dir(grid_dir)
    grid_file = grid_dir / f"hazard_grid_{imt}_{vs30}.csv"
    grid_df.to_csv(grid_file, index=False)
    print(f"  Saved grid to {grid_file}")
    
    return grid_df


# ── Integrated Download Pipeline ──────────────────────────────────────────

def download_all_hazard_data(
    event_id: str = NORTHRIDGE_EVENT_ID,
    latitude: float = NORTHRIDGE_LAT,
    longitude: float = NORTHRIDGE_LON,
    output_dir: Path = DEFAULT_HAZARD_DIR,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Download all hazard data for California/Northridge case study.
    
    Downloads:
    1. ShakeMap for the specified event
    2. Hazard curves for site location
    
    Parameters
    ----------
    event_id : str
        USGS event ID for ShakeMap
    latitude, longitude : float
        Site coordinates for hazard curves
    output_dir : Path
        Base output directory
    overwrite : bool
        Whether to overwrite existing files
        
    Returns
    -------
    dict
        Dictionary with download results
    """
    print("=" * 60)
    print("USGS Hazard Data Download Pipeline")
    print("California/Northridge Case Study")
    print("=" * 60)
    print()
    
    results = {}
    
    # 1. ShakeMap
    print("[1/2] ShakeMap Download")
    print("-" * 40)
    try:
        shakemap = download_shakemap(
            event_id=event_id,
            output_dir=output_dir / "shakemap",
            overwrite=overwrite,
        )
        results["shakemap"] = shakemap
        print()
    except Exception as e:
        print(f"  [Error] ShakeMap download failed: {e}")
        results["shakemap"] = None
    
    # 2. Hazard Curves
    print("[2/2] Hazard Curves Download")
    print("-" * 40)
    try:
        # Download for multiple IMTs
        for imt in ["PGA", "SA1P0"]:
            hc = download_hazard_curves(
                latitude=latitude,
                longitude=longitude,
                imt=imt,
                output_dir=output_dir / "hazard_curves",
                overwrite=overwrite,
            )
            results[f"hazard_curves_{imt}"] = hc
            print()
    except Exception as e:
        print(f"  [Error] Hazard curves download failed: {e}")
    
    print("=" * 60)
    print("Download complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    return results


# ── CLI Entry Point ───────────────────────────────────────────────────────

def main():
    """CLI entry point for hazard data download."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download USGS hazard data for CAT411 framework"
    )
    parser.add_argument(
        "--event", "-e",
        type=str,
        default=NORTHRIDGE_EVENT_ID,
        help=f"USGS event ID for ShakeMap (default: {NORTHRIDGE_EVENT_ID})",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=NORTHRIDGE_LAT,
        help=f"Latitude for hazard curves (default: {NORTHRIDGE_LAT})",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=NORTHRIDGE_LON,
        help=f"Longitude for hazard curves (default: {NORTHRIDGE_LON})",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_HAZARD_DIR,
        help=f"Output directory (default: {DEFAULT_HAZARD_DIR})",
    )
    parser.add_argument(
        "--shakemap-only",
        action="store_true",
        help="Download only ShakeMap data",
    )
    parser.add_argument(
        "--curves-only",
        action="store_true",
        help="Download only hazard curves",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    
    args = parser.parse_args()
    
    if args.shakemap_only:
        download_shakemap(
            event_id=args.event,
            output_dir=args.output / "shakemap",
            overwrite=args.overwrite,
        )
    elif args.curves_only:
        download_hazard_curves(
            latitude=args.lat,
            longitude=args.lon,
            output_dir=args.output / "hazard_curves",
            overwrite=args.overwrite,
        )
    else:
        download_all_hazard_data(
            event_id=args.event,
            latitude=args.lat,
            longitude=args.lon,
            output_dir=args.output,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

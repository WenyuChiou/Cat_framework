"""
Vs30 spatial data provider.

Provides site-specific Vs30 (time-averaged shear-wave velocity in the top 30m)
from the USGS Global Hybrid Vs30 Map (Heath et al. 2020).

Data source: https://earthquake.usgs.gov/data/vs30/
Resolution: 30 arc-seconds (~1 km)
Format: GMT grid (NetCDF) read via rasterio

Usage:
    provider = Vs30Provider()           # auto-loads California cache
    vs30 = provider.get_vs30(34.05, -118.25)  # Los Angeles ~340 m/s

The provider caches a California-region subset as a compressed .npz file
to avoid loading the full 600 MB global grid on every run.
"""

import os
import numpy as np
from pathlib import Path

# Default paths
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "vs30"
_CACHE_FILE = _DATA_DIR / "california_vs30.npz"
_GLOBAL_GRID = _DATA_DIR / "global_vs30.grd"

# California bounding box (generous, covers full state + margins)
CA_BOUNDS = {
    "lat_min": 32.0,
    "lat_max": 42.5,
    "lon_min": -125.0,
    "lon_max": -113.5,
}

# NEHRP site class boundaries
NEHRP_CLASSES = {
    "A": (1500, float("inf")),   # Hard rock
    "B": (760, 1500),            # Rock
    "BC": (560, 760),            # B/C boundary
    "C": (360, 560),             # Dense soil / soft rock
    "CD": (270, 360),            # C/D boundary
    "D": (180, 270),             # Stiff soil
    "E": (0, 180),               # Soft soil
}


class Vs30Provider:
    """Provides Vs30 lookups from a cached regional grid."""

    def __init__(self, cache_path=None, bounds=None):
        """
        Load Vs30 data from cache. If cache doesn't exist, attempts to
        extract from the global grid file.

        Parameters
        ----------
        cache_path : str or Path, optional
            Path to the .npz cache file. Default: data/vs30/california_vs30.npz
        bounds : dict, optional
            Region bounds with keys lat_min, lat_max, lon_min, lon_max.
            Default: California bounds.
        """
        self.cache_path = Path(cache_path) if cache_path else _CACHE_FILE
        self.bounds = bounds or CA_BOUNDS
        self._grid = None
        self._lats = None
        self._lons = None
        self._interpolator = None

        if self.cache_path.exists():
            self._load_cache()
        elif _GLOBAL_GRID.exists():
            self._extract_and_cache()
        else:
            raise FileNotFoundError(
                f"Vs30 data not found. Please either:\n"
                f"  1. Place global_vs30.grd in {_DATA_DIR}/\n"
                f"  2. Run: python -m src.vs30_provider --download\n"
                f"  3. Place california_vs30.npz in {_DATA_DIR}/\n"
                f"Download from: https://earthquake.usgs.gov/data/vs30/"
            )

    def _load_cache(self):
        """Load California Vs30 grid from compressed cache."""
        data = np.load(self.cache_path)
        self._lats = data["lats"]
        self._lons = data["lons"]
        self._grid = data["vs30"]
        self._build_interpolator()

    def _extract_and_cache(self):
        """Extract California region from global grid and save cache."""
        import rasterio
        from rasterio.windows import from_bounds

        os.makedirs(self.cache_path.parent, exist_ok=True)

        with rasterio.open(str(_GLOBAL_GRID)) as src:
            # Get window for California bounds
            window = from_bounds(
                self.bounds["lon_min"], self.bounds["lat_min"],
                self.bounds["lon_max"], self.bounds["lat_max"],
                src.transform,
            )
            vs30_data = src.read(1, window=window)

            # Build coordinate arrays from the window transform
            win_transform = src.window_transform(window)
            nrows, ncols = vs30_data.shape
            lons = np.array([win_transform.c + win_transform.a * j
                             for j in range(ncols)])
            lats = np.array([win_transform.f + win_transform.e * i
                             for i in range(nrows)])

        # Replace nodata with NaN
        vs30_float = vs30_data.astype(np.float32)
        vs30_float[vs30_float <= 0] = np.nan

        self._lats = lats
        self._lons = lons
        self._grid = vs30_float

        # Save cache
        np.savez_compressed(
            self.cache_path,
            lats=self._lats,
            lons=self._lons,
            vs30=self._grid,
        )
        print(f"[Vs30] Cached California region: {self._grid.shape} "
              f"({self.cache_path})")

        self._build_interpolator()

    def _build_interpolator(self):
        """Build a fast 2D interpolator for Vs30 lookups."""
        from scipy.interpolate import RegularGridInterpolator

        # Ensure lats are ascending for RegularGridInterpolator
        if self._lats[0] > self._lats[-1]:
            self._lats = self._lats[::-1]
            self._grid = self._grid[::-1, :]

        # Replace NaN with nearest valid for interpolation (edges)
        grid_filled = self._grid.copy()
        nan_mask = np.isnan(grid_filled)
        if nan_mask.any():
            from scipy.ndimage import generic_filter
            # Simple fill: use column/row median for ocean pixels
            col_median = np.nanmedian(grid_filled, axis=0)
            for i in range(grid_filled.shape[0]):
                for j in range(grid_filled.shape[1]):
                    if np.isnan(grid_filled[i, j]):
                        grid_filled[i, j] = col_median[j] if not np.isnan(col_median[j]) else 760.0

        self._interpolator = RegularGridInterpolator(
            (self._lats, self._lons),
            grid_filled,
            method="linear",
            bounds_error=False,
            fill_value=760.0,  # Rock default for out-of-bounds
        )

    def get_vs30(self, lat, lon):
        """
        Get Vs30 at a single point.

        Parameters
        ----------
        lat : float
            Latitude in degrees.
        lon : float
            Longitude in degrees.

        Returns
        -------
        float
            Vs30 in m/s. Returns 760.0 (rock) if outside grid bounds.
        """
        result = self._interpolator(np.array([[lat, lon]]))[0]
        return float(result)

    def get_vs30_array(self, lats, lons):
        """
        Get Vs30 for arrays of coordinates.

        Parameters
        ----------
        lats : array-like
            Latitudes in degrees.
        lons : array-like
            Longitudes in degrees.

        Returns
        -------
        np.ndarray
            Vs30 values in m/s.
        """
        points = np.column_stack([np.asarray(lats), np.asarray(lons)])
        return self._interpolator(points)

    def get_nehrp_class(self, vs30):
        """Return NEHRP site class for a given Vs30 value."""
        for cls, (lo, hi) in NEHRP_CLASSES.items():
            if lo <= vs30 < hi:
                return cls
        return "E"  # below 180

    @property
    def shape(self):
        """Grid dimensions (nrows, ncols)."""
        return self._grid.shape if self._grid is not None else (0, 0)

    @property
    def lat_range(self):
        """(min_lat, max_lat) of loaded grid."""
        if self._lats is not None:
            return (float(self._lats.min()), float(self._lats.max()))
        return (0.0, 0.0)

    @property
    def lon_range(self):
        """(min_lon, max_lon) of loaded grid."""
        if self._lons is not None:
            return (float(self._lons.min()), float(self._lons.max()))
        return (0.0, 0.0)


def download_global_vs30(dest_dir=None):
    """
    Download the USGS global Vs30 grid file.

    Parameters
    ----------
    dest_dir : str or Path, optional
        Destination directory. Default: data/vs30/
    """
    import urllib.request

    dest = Path(dest_dir) if dest_dir else _DATA_DIR
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / "global_vs30.grd"

    if out_path.exists():
        print(f"[Vs30] Global grid already exists: {out_path}")
        return out_path

    url = "https://apps.usgs.gov/shakemap_geodata/vs30/global_vs30.grd"
    print(f"[Vs30] Downloading global Vs30 grid (~600 MB)...")
    print(f"  URL: {url}")
    print(f"  Destination: {out_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "CAT411/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        with open(str(out_path), "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 / total
                    print(f"\r  {downloaded/1e6:.0f} MB / {total/1e6:.0f} MB ({pct:.0f}%)",
                          end="", flush=True)
    print(f"\n[Vs30] Download complete: {out_path}")
    return out_path


def enrich_bridges_with_vs30(nbi_df, provider=None):
    """
    Add a 'vs30' column to a bridge DataFrame using spatial Vs30 data.

    Parameters
    ----------
    nbi_df : pd.DataFrame
        Must have 'latitude' and 'longitude' columns.
    provider : Vs30Provider, optional
        If None, creates a new provider (loads cache).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with 'vs30' column added.
    """
    if provider is None:
        provider = Vs30Provider()

    vs30_values = provider.get_vs30_array(
        nbi_df["latitude"].values,
        nbi_df["longitude"].values,
    )
    nbi_df["vs30"] = vs30_values
    return nbi_df


if __name__ == "__main__":
    import sys

    if "--download" in sys.argv:
        grid_path = download_global_vs30()
        print(f"\nExtracting California region...")
        provider = Vs30Provider()
        print(f"Grid shape: {provider.shape}")
        print(f"Lat range: {provider.lat_range}")
        print(f"Lon range: {provider.lon_range}")
    elif "--test" in sys.argv:
        try:
            provider = Vs30Provider()
            # Test points
            test_points = [
                ("Los Angeles", 34.05, -118.25),
                ("San Francisco", 37.77, -122.42),
                ("Northridge epicenter", 34.213, -118.537),
                ("Sacramento (valley)", 38.58, -121.49),
                ("Death Valley", 36.46, -116.87),
            ]
            print(f"Vs30 Grid: {provider.shape}, "
                  f"lat {provider.lat_range}, lon {provider.lon_range}\n")
            for name, lat, lon in test_points:
                v = provider.get_vs30(lat, lon)
                cls = provider.get_nehrp_class(v)
                print(f"  {name:30s}  Vs30 = {v:6.0f} m/s  (NEHRP {cls})")
        except FileNotFoundError as e:
            print(f"Error: {e}")
    else:
        print("Usage:")
        print("  python -m src.vs30_provider --download   Download global Vs30 grid")
        print("  python -m src.vs30_provider --test       Test with sample points")

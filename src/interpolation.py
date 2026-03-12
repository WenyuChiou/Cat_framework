"""
Spatial Interpolation Methods for Ground Motion Fields.

Provides multiple interpolation strategies for assigning intensity
measure (IM) values from ShakeMap grid points to bridge locations.
All methods relevant to earthquake engineering are included.

Methods
-------
- nearest : Nearest-neighbor (KD-tree) — current default
- idw     : Inverse Distance Weighting
- bilinear: Bilinear grid interpolation (scipy RegularGridInterpolator)
- natural : Natural-neighbor (Voronoi-based, scipy griddata 'linear')
- kriging : Ordinary Kriging (simplified exponential variogram)

References
----------
- Worden et al. (2018), "Spatial and spectral interpolation of ground
  motion intensity measure observations", BSSA.
- ShakeMap Manual: https://usgs.github.io/shakemap/manual4_0/
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


# ── Public API ───────────────────────────────────────────────────────────

INTERPOLATION_METHODS = [
    "nearest",
    "idw",
    "bilinear",
    "natural_neighbor",
    "kriging",
]


def interpolate_im(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    grid_values: np.ndarray,
    bridge_lats: np.ndarray,
    bridge_lons: np.ndarray,
    method: str = "nearest",
    **kwargs,
) -> np.ndarray:
    """
    Interpolate IM values from ShakeMap grid to bridge locations.

    Parameters
    ----------
    grid_lats, grid_lons : (N,) arrays
        ShakeMap grid coordinates.
    grid_values : (N,) array
        IM values at grid points.
    bridge_lats, bridge_lons : (M,) arrays
        Bridge coordinates.
    method : str
        Interpolation method (see INTERPOLATION_METHODS).
    **kwargs
        Additional method-specific parameters.

    Returns
    -------
    np.ndarray
        (M,) interpolated IM values at bridge locations.
    """
    method = method.lower().replace("-", "_")

    if method == "nearest":
        return _nearest(grid_lats, grid_lons, grid_values,
                        bridge_lats, bridge_lons, **kwargs)
    elif method == "idw":
        return _idw(grid_lats, grid_lons, grid_values,
                    bridge_lats, bridge_lons, **kwargs)
    elif method == "bilinear":
        return _bilinear(grid_lats, grid_lons, grid_values,
                         bridge_lats, bridge_lons, **kwargs)
    elif method == "natural_neighbor":
        return _natural_neighbor(grid_lats, grid_lons, grid_values,
                                 bridge_lats, bridge_lons, **kwargs)
    elif method == "kriging":
        return _kriging(grid_lats, grid_lons, grid_values,
                        bridge_lats, bridge_lons, **kwargs)
    else:
        raise ValueError(
            f"Unknown interpolation method '{method}'. "
            f"Available: {INTERPOLATION_METHODS}"
        )


# ── Method 1: Nearest Neighbor (KD-tree) ─────────────────────────────────

def _nearest(
    grid_lats, grid_lons, grid_values,
    bridge_lats, bridge_lons, **kwargs,
) -> np.ndarray:
    """
    Assign the IM value of the closest ShakeMap grid point.

    Fastest method. No smoothing. Suitable when grid is dense relative
    to bridge spacing.  This is the default ShakeMap approach.
    """
    tree = cKDTree(np.column_stack([grid_lats, grid_lons]))
    _, indices = tree.query(np.column_stack([bridge_lats, bridge_lons]))
    return grid_values[indices]


# ── Method 2: Inverse Distance Weighting (IDW) ───────────────────────────

def _idw(
    grid_lats, grid_lons, grid_values,
    bridge_lats, bridge_lons,
    power: float = 2.0,
    n_neighbors: int = 8,
    **kwargs,
) -> np.ndarray:
    """
    Inverse Distance Weighting interpolation.

    w_i = 1 / d_i^p,  where d_i = distance to the i-th neighbor.

    Commonly used in earthquake engineering for its simplicity and
    smooth output.  Higher power → more local behavior.

    Parameters
    ----------
    power : float
        Distance weighting exponent (default 2.0).
    n_neighbors : int
        Number of nearest grid points to use (default 8).
    """
    tree = cKDTree(np.column_stack([grid_lats, grid_lons]))
    distances, indices = tree.query(
        np.column_stack([bridge_lats, bridge_lons]),
        k=n_neighbors,
    )

    # Handle exact matches (distance = 0)
    result = np.zeros(len(bridge_lats))
    for i in range(len(bridge_lats)):
        d = distances[i]
        idx = indices[i]

        # If an exact match exists, use it
        zero_mask = d < 1e-12
        if zero_mask.any():
            result[i] = grid_values[idx[zero_mask][0]]
        else:
            weights = 1.0 / d**power
            result[i] = np.average(grid_values[idx], weights=weights)

    return result


# ── Method 3: Bilinear Grid Interpolation ─────────────────────────────────

def _bilinear(
    grid_lats, grid_lons, grid_values,
    bridge_lats, bridge_lons, **kwargs,
) -> np.ndarray:
    """
    Bilinear interpolation on a regular grid.

    Assumes ShakeMap grid is approximately regular (which it is).
    Uses scipy RegularGridInterpolator for efficiency.
    Falls back to griddata if grid is irregular.

    Note: Regular-grid assumption holds for USGS ShakeMap products
    but may fail for user-supplied grids with non-uniform spacing.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Try to detect regular grid
    unique_lats = np.sort(np.unique(grid_lats))
    unique_lons = np.sort(np.unique(grid_lons))

    if len(unique_lats) * len(unique_lons) == len(grid_values):
        # Regular grid — reshape and use fast interpolator
        values_2d = np.full((len(unique_lats), len(unique_lons)), np.nan)
        for i, (lat, lon, val) in enumerate(
            zip(grid_lats, grid_lons, grid_values)
        ):
            lat_idx = np.searchsorted(unique_lats, lat)
            lon_idx = np.searchsorted(unique_lons, lon)
            if lat_idx < len(unique_lats) and lon_idx < len(unique_lons):
                values_2d[lat_idx, lon_idx] = val

        interp = RegularGridInterpolator(
            (unique_lats, unique_lons),
            values_2d,
            method="linear",
            bounds_error=False,
            fill_value=None,  # extrapolate
        )
        return interp(np.column_stack([bridge_lats, bridge_lons]))
    else:
        # Irregular grid — fall back to griddata
        return _natural_neighbor(
            grid_lats, grid_lons, grid_values,
            bridge_lats, bridge_lons,
        )


# ── Method 4: Natural Neighbor (Voronoi-based) ───────────────────────────

def _natural_neighbor(
    grid_lats, grid_lons, grid_values,
    bridge_lats, bridge_lons, **kwargs,
) -> np.ndarray:
    """
    Natural neighbor interpolation via Delaunay triangulation.

    Also called Sibson interpolation. Produces smooth, continuous
    surfaces without user-specified parameters.  Excellent for
    irregularly-spaced seismic station data.

    Falls back to nearest-neighbor for points outside the convex hull.
    """
    from scipy.interpolate import griddata

    result = griddata(
        np.column_stack([grid_lats, grid_lons]),
        grid_values,
        np.column_stack([bridge_lats, bridge_lons]),
        method="linear",
    )

    # Fill NaN (outside convex hull) with nearest neighbor
    nan_mask = np.isnan(result)
    if nan_mask.any():
        nn = _nearest(
            grid_lats, grid_lons, grid_values,
            bridge_lats[nan_mask], bridge_lons[nan_mask],
        )
        result[nan_mask] = nn

    return result


# ── Method 5: Ordinary Kriging ────────────────────────────────────────────

def _kriging(
    grid_lats, grid_lons, grid_values,
    bridge_lats, bridge_lons,
    n_neighbors: int = 16,
    range_km: float = 50.0,
    nugget: float = 0.01,
    **kwargs,
) -> np.ndarray:
    """
    Simplified Ordinary Kriging with exponential variogram.

    Kriging is the geostatistical gold standard for spatial interpolation,
    providing the "Best Linear Unbiased Estimate" (BLUE). The exponential
    variogram model is standard for ground motion spatial correlation
    (Jayaram & Baker, 2009).

    γ(h) = sill * (1 - exp(-3h/range)) + nugget

    Parameters
    ----------
    n_neighbors : int
        Number of nearest grid points for local kriging (default 16).
    range_km : float
        Variogram range in km (default 50 km, typical for Sa 1.0s).
    nugget : float
        Variogram nugget (measurement noise, default 0.01).

    Note
    ----
    This is a simplified implementation using an assumed variogram model.
    For production use, consider fitting the variogram to the data.
    """
    tree = cKDTree(np.column_stack([grid_lats, grid_lons]))
    result = np.zeros(len(bridge_lats))

    # Approximate degree-to-km conversion at study area latitude
    mean_lat = np.mean(grid_lats)
    deg_to_km_lat = 111.0
    deg_to_km_lon = 111.0 * np.cos(np.radians(mean_lat))

    # Sill estimated from data variance
    sill = np.var(grid_values) if np.var(grid_values) > 0 else 1.0

    def variogram(h_km):
        """Exponential variogram model."""
        return nugget + sill * (1.0 - np.exp(-3.0 * h_km / range_km))

    for i in range(len(bridge_lats)):
        distances, indices = tree.query(
            [bridge_lats[i], bridge_lons[i]], k=n_neighbors
        )

        # Convert distances from degrees to km
        d_km = np.sqrt(
            ((grid_lats[indices] - bridge_lats[i]) * deg_to_km_lat) ** 2
            + ((grid_lons[indices] - bridge_lons[i]) * deg_to_km_lon) ** 2
        )

        # Build kriging system
        n = len(indices)
        K = np.zeros((n + 1, n + 1))
        k = np.zeros(n + 1)

        # Fill covariance matrix
        for a in range(n):
            for b in range(n):
                h = np.sqrt(
                    ((grid_lats[indices[a]] - grid_lats[indices[b]]) * deg_to_km_lat) ** 2
                    + ((grid_lons[indices[a]] - grid_lons[indices[b]]) * deg_to_km_lon) ** 2
                )
                K[a, b] = variogram(h)
            K[a, n] = 1.0
            K[n, a] = 1.0

        # Target point covariance
        for a in range(n):
            k[a] = variogram(d_km[a])
        k[n] = 1.0

        # Solve kriging system
        try:
            weights = np.linalg.solve(K, k)
            result[i] = np.dot(weights[:n], grid_values[indices])
        except np.linalg.LinAlgError:
            # Singular matrix fallback to IDW
            if np.min(d_km) < 1e-6:
                result[i] = grid_values[indices[np.argmin(d_km)]]
            else:
                w = 1.0 / d_km**2
                result[i] = np.average(grid_values[indices], weights=w)

    return result

"""
Data download and parsing for USGS ShakeMap and FHWA NBI bridge inventory.

Provides two main workflows:
  1. Download raw files from USGS / FHWA to data/ directory
  2. Parse local files into pandas DataFrames for analysis

ShakeMap: Northridge earthquake (ci3144585) grid.xml
  - Contains PGA, PGV, Sa(0.3s), Sa(1.0s), Sa(3.0s) at grid points
  - Values in grid.xml are in %g; parser converts to g

NBI: National Bridge Inventory delimited text (California)
  - Annual bridge inventory with structural attributes
  - Parser extracts fields needed for Hazus classification
"""

import os
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default paths and URLs
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"

SHAKEMAP_URL = (
    "https://earthquake.usgs.gov/product/shakemap/ci3144585/"
    "atlas/1594159786829/download/grid.xml"
)
SHAKEMAP_STATION_URL = (
    "https://earthquake.usgs.gov/product/shakemap/ci3144585/"
    "atlas/1594159786829/download/stationlist.json"
)
NBI_CA_URL = (
    "https://www.fhwa.dot.gov/bridge/nbi/2024/delimited/CA24.txt"
)

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _ensure_data_dir() -> Path:
    """Create the data/ directory if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def download_shakemap(
    url: str = SHAKEMAP_URL,
    dest: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Download USGS ShakeMap grid.xml for the Northridge earthquake.

    Parameters
    ----------
    url : str
        ShakeMap grid.xml download URL.
    dest : path-like, optional
        Destination file path.  Defaults to data/grid.xml.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    import requests

    dest = Path(dest) if dest else _ensure_data_dir() / "grid.xml"
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ShakeMap grid.xml → {dest}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  Downloaded {len(resp.content):,} bytes")
    return dest


def download_shakemap_stations(
    url: str = SHAKEMAP_STATION_URL,
    dest: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Download ShakeMap station list (JSON) for recorded ground motions.

    Parameters
    ----------
    url : str
        Station list JSON URL.
    dest : path-like, optional
        Destination file path.  Defaults to data/stationlist.json.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    import requests

    dest = Path(dest) if dest else _ensure_data_dir() / "stationlist.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ShakeMap stationlist.json → {dest}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  Downloaded {len(resp.content):,} bytes")
    return dest


def download_nbi(
    url: str = NBI_CA_URL,
    dest: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Download National Bridge Inventory data for California.

    Parameters
    ----------
    url : str
        NBI delimited file URL.
    dest : path-like, optional
        Destination file path.  Defaults to data/CA24.txt.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    import requests

    filename = url.split("/")[-1]
    dest = Path(dest) if dest else _ensure_data_dir() / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading NBI data → {dest}")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  Downloaded {len(resp.content):,} bytes")
    return dest


def download_all() -> dict[str, Path]:
    """Download all data files.  Returns dict of {name: path}."""
    return {
        "shakemap_grid": download_shakemap(),
        "shakemap_stations": download_shakemap_stations(),
        "nbi": download_nbi(),
    }


# ---------------------------------------------------------------------------
# ShakeMap grid.xml parser
# ---------------------------------------------------------------------------

def parse_shakemap_grid(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Parse a USGS ShakeMap grid.xml into a DataFrame.

    The grid.xml format contains space-separated values inside a
    <grid_data> element, with column definitions in <grid_field> elements.

    Intensity values (PGA, PGV, PSA) are stored in %g in the file;
    this parser converts PGA and PSA columns to units of g (divides by 100).

    Parameters
    ----------
    filepath : path-like
        Path to the grid.xml file.

    Returns
    -------
    pd.DataFrame
        Columns: LON, LAT, PGA, PGV, MMI, PSA03, PSA10, PSA30, STDPGA, URAT, SVEL
        PGA / PSA values are in g.  PGV remains in cm/s.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"ShakeMap file not found: {filepath}\n"
            f"Run download_shakemap() first or place grid.xml in {DATA_DIR}"
        )

    tree = ET.parse(filepath)
    root = tree.getroot()

    # Handle XML namespace
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    # Extract column names from <grid_field> elements
    columns = []
    for field in root.findall(f"{ns}grid_field"):
        columns.append(field.attrib["name"])

    # Extract grid data text
    grid_data_el = root.find(f"{ns}grid_data")
    if grid_data_el is None:
        raise ValueError("No <grid_data> element found in ShakeMap XML")

    raw_text = grid_data_el.text.strip()
    df = pd.read_csv(
        io.StringIO(raw_text),
        sep=r"\s+",
        header=None,
        names=columns,
    )

    # Convert %g to g for acceleration columns
    pct_g_cols = [c for c in df.columns if c in ("PGA", "PSA03", "PSA10", "PSA30")]
    for col in pct_g_cols:
        df[col] = df[col] / 100.0

    # Extract event metadata from root attributes
    event_attrs = root.attrib
    df.attrs["event_id"] = event_attrs.get("event_id", "ci3144585")
    df.attrs["magnitude"] = float(event_attrs.get("magnitude", 6.7))
    df.attrs["lat"] = float(event_attrs.get("lat", 34.213))
    df.attrs["lon"] = float(event_attrs.get("lon", -118.537))
    df.attrs["event_description"] = event_attrs.get(
        "event_description", "Northridge, California"
    )

    return df


def parse_shakemap_stations(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Parse ShakeMap stationlist.json into a DataFrame of station recordings.

    Parameters
    ----------
    filepath : path-like
        Path to stationlist.json.

    Returns
    -------
    pd.DataFrame
        Columns include: station_code, name, lat, lon, distance_km,
        pga, pgv, psa03, psa10, psa30 (all in g or cm/s as noted).
    """
    import json

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Station list not found: {filepath}\n"
            f"Run download_shakemap_stations() first."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    features = data if isinstance(data, list) else data.get("features", [])

    for feat in features:
        props = feat.get("properties", feat) if isinstance(feat, dict) else {}
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [None, None])

        # Extract peak ground motion channels
        channels = props.get("channels", [])
        pga_val = pgv_val = psa03_val = psa10_val = psa30_val = np.nan

        for ch in channels:
            amplitudes = ch.get("amplitudes", [])
            for amp in amplitudes:
                name = amp.get("name", "").lower()
                val = amp.get("value", None)
                if val is None or val == "null":
                    continue
                val = float(val)
                if name == "pga":
                    pga_val = val / 100.0  # %g → g
                elif name == "pgv":
                    pgv_val = val  # cm/s
                elif name == "psa03" or name == "sa(0.3)":
                    psa03_val = val / 100.0
                elif name == "psa10" or name == "sa(1.0)":
                    psa10_val = val / 100.0
                elif name == "psa30" or name == "sa(3.0)":
                    psa30_val = val / 100.0

        records.append({
            "station_code": props.get("code", ""),
            "name": props.get("name", ""),
            "lat": coords[1] if len(coords) > 1 else None,
            "lon": coords[0] if len(coords) > 0 else None,
            "distance_km": props.get("distance", np.nan),
            "intensity": props.get("intensity", np.nan),
            "pga": pga_val,
            "pgv": pgv_val,
            "psa03": psa03_val,
            "psa10": psa10_val,
            "psa30": psa30_val,
            "source": props.get("source", ""),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# NBI parser
# ---------------------------------------------------------------------------

# NBI Item 43A: Kind of material / design (main span)
NBI_MATERIAL_MAP = {
    "1": "concrete",
    "2": "concrete",          # concrete continuous
    "3": "steel",
    "4": "steel",             # steel continuous
    "5": "prestressed_concrete",
    "6": "prestressed_concrete",  # prestressed continuous
    "7": "wood",
    "8": "masonry",
    "9": "aluminum_iron",
    "0": "other",
}

# NBI Item 43B: Type of design (main span)
NBI_DESIGN_MAP = {
    "01": "slab",
    "02": "stringer",
    "03": "girder_floorbeam",
    "04": "tee_beam",
    "05": "box_girder_multi",
    "06": "box_girder_single",
    "07": "frame",
    "08": "orthotropic",
    "09": "truss_deck",
    "10": "truss_thru",
    "11": "arch_deck",
    "12": "arch_thru",
    "13": "suspension",
    "14": "stayed_girder",
    "15": "movable_lift",
    "16": "movable_bascule",
    "17": "movable_swing",
    "18": "tunnel",
    "19": "culvert",
    "20": "mixed",
    "21": "segmental",
    "22": "channel",
    "00": "other",
}


def parse_nbi(
    filepath: Union[str, Path],
    northridge_bbox: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Parse NBI delimited text file into a DataFrame with bridge attributes.

    Parameters
    ----------
    filepath : path-like
        Path to NBI delimited file (e.g., CA24.txt).
    northridge_bbox : dict, optional
        Bounding box to filter bridges near Northridge.
        Keys: lat_min, lat_max, lon_min, lon_max.
        Defaults to the greater LA area.

    Returns
    -------
    pd.DataFrame
        Parsed bridge inventory with columns:
        structure_number, state, county, latitude, longitude,
        year_built, material_code, material, design_code, design_type,
        num_spans, structure_length_m, deck_width_m, condition_rating,
        owner, service_on, service_under
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"NBI file not found: {filepath}\n"
            f"Run download_nbi() first or place the file in {DATA_DIR}"
        )

    # Read the full delimited file — NBI uses comma separation with
    # single-quote text qualifier
    df_raw = pd.read_csv(
        filepath,
        dtype=str,
        low_memory=False,
        quotechar="'",
        on_bad_lines="skip",
    )

    # Standardize column names (strip whitespace)
    df_raw.columns = df_raw.columns.str.strip()

    # Map NBI column names to our fields
    # NBI delimited format uses descriptive headers:
    col_map = _detect_nbi_columns(df_raw.columns.tolist())

    df = pd.DataFrame()
    df["structure_number"] = df_raw[col_map["structure_number"]].str.strip()
    df["state"] = df_raw[col_map["state"]].str.strip()
    df["county"] = df_raw[col_map["county"]].str.strip()

    # Latitude / Longitude (in NBI: DDMMSS.SS format or decimal degrees)
    df["latitude"] = _parse_nbi_coord(df_raw[col_map["latitude"]])
    df["longitude"] = -_parse_nbi_coord(df_raw[col_map["longitude"]]).abs()

    df["year_built"] = pd.to_numeric(
        df_raw[col_map["year_built"]], errors="coerce"
    )
    df["material_code"] = df_raw[col_map["material_code"]].str.strip()
    df["material"] = df["material_code"].map(NBI_MATERIAL_MAP).fillna("other")

    df["design_code"] = df_raw[col_map["design_code"]].str.strip()
    df["design_type"] = df["design_code"].map(NBI_DESIGN_MAP).fillna("other")

    df["num_spans"] = pd.to_numeric(
        df_raw[col_map["num_spans"]], errors="coerce"
    )
    # Structure length in NBI is in meters (×0.1 in some formats)
    length_raw = pd.to_numeric(df_raw[col_map["length"]], errors="coerce")
    df["structure_length_m"] = length_raw

    width_raw = pd.to_numeric(df_raw[col_map["width"]], errors="coerce")
    df["deck_width_m"] = width_raw

    if col_map.get("condition"):
        df["condition_rating"] = pd.to_numeric(
            df_raw[col_map["condition"]], errors="coerce"
        )

    # Filter to bounding box if requested
    if northridge_bbox is None:
        northridge_bbox = {
            "lat_min": 33.7,
            "lat_max": 34.8,
            "lon_min": -119.0,
            "lon_max": -117.5,
        }

    mask = (
        (df["latitude"] >= northridge_bbox["lat_min"])
        & (df["latitude"] <= northridge_bbox["lat_max"])
        & (df["longitude"] >= northridge_bbox["lon_min"])
        & (df["longitude"] <= northridge_bbox["lon_max"])
    )
    df = df.loc[mask].copy()
    df.reset_index(drop=True, inplace=True)

    return df


def _detect_nbi_columns(columns: list[str]) -> dict[str, str]:
    """
    Map NBI column headers to standardized field names.

    NBI delimited files have varied header names across years.
    This function uses pattern matching to identify key columns.
    """
    col_map = {}
    col_lower = {c: c.lower().replace("_", " ").strip() for c in columns}

    patterns = {
        "structure_number": ["structure number", "structurenumber", "struc"],
        "state": ["state code", "state", "statecd"],
        "county": ["county code", "county", "countycd"],
        "latitude": ["lat", "latitude", "degrees"],
        "longitude": ["long", "longitude"],
        "year_built": ["year built", "yearbuilt", "year_built"],
        "material_code": [
            "kind of material", "material", "main span material",
            "kind hwy str", "structure kind",
        ],
        "design_code": [
            "type of design", "design", "main span design",
            "type hwy str", "structure type",
        ],
        "num_spans": ["main unit spans", "number of spans", "spans"],
        "length": [
            "structure length", "length", "structure len",
            "max span length",
        ],
        "width": [
            "deck width", "width", "curb to curb",
            "deck structure", "roadway width",
        ],
        "condition": [
            "deck cond", "superstructure cond", "substructure cond",
            "condition", "lowest rating",
        ],
    }

    for field, keywords in patterns.items():
        matched = False
        for col_orig, col_low in col_lower.items():
            for kw in keywords:
                if kw in col_low:
                    col_map[field] = col_orig
                    matched = True
                    break
            if matched:
                break
        if not matched:
            # Fallback: try column index position for known NBI order
            col_map[field] = _fallback_nbi_column(field, columns)

    return col_map


def _fallback_nbi_column(field: str, columns: list[str]) -> str:
    """Provide fallback column by index for standard NBI file ordering."""
    # These indices are approximate for common NBI delimited formats
    fallback_idx = {
        "structure_number": 1,
        "state": 0,
        "county": 2,
        "latitude": 19,
        "longitude": 20,
        "year_built": 26,
        "material_code": 42,
        "design_code": 43,
        "num_spans": 44,
        "length": 48,
        "width": 51,
        "condition": 58,
    }
    idx = fallback_idx.get(field, 0)
    if idx < len(columns):
        return columns[idx]
    return columns[0]


def _parse_nbi_coord(series: pd.Series) -> pd.Series:
    """
    Parse NBI coordinate values.

    NBI stores coordinates as either:
      - DDMMSS.SS (integer-encoded) -> convert to decimal degrees
      - Already decimal degrees
    """
    values = pd.to_numeric(series, errors="coerce")

    # If values are large (> 200), they're in DDMMSS.SS format
    # Otherwise they're already decimal degrees
    valid = values.notna()
    large = valid & (values.abs() > 200)
    if large.any():
        # DDMMSS.SS → DD + MM/60 + SS.SS/3600
        sign = np.sign(values[large])
        absval = values[large].abs()
        dd = np.floor(absval / 1_000_000)
        mm = np.floor((absval - dd * 1_000_000) / 10_000)
        ss = (absval - dd * 1_000_000 - mm * 10_000) / 100
        decimal = dd + mm / 60.0 + ss / 3600.0
        values.loc[large] = sign * decimal

    return values


# ---------------------------------------------------------------------------
# Hazus classification from NBI data
# ---------------------------------------------------------------------------


def classify_nbi_to_hazus(nbi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Hazus bridge class (HWB) column to NBI DataFrame.

    Uses material code, design type, number of spans, structure length,
    and year built to assign each bridge a Hazus classification.

    Parameters
    ----------
    nbi_df : pd.DataFrame
        Parsed NBI data from parse_nbi().

    Returns
    -------
    pd.DataFrame
        Same DataFrame with added 'hwb_class' column.
    """
    from .bridge_classes import classify_bridge

    def _row_to_hwb(row):
        # Determine material category
        mat_code = str(row.get("material_code", "0")).strip()
        if mat_code in ("1", "2", "5", "6"):
            material = "concrete"
        elif mat_code in ("3", "4"):
            material = "steel"
        else:
            material = "other"

        # Determine span type
        n_spans = row.get("num_spans", 1)
        if pd.isna(n_spans):
            n_spans = 1
        n_spans = int(n_spans)

        continuity = mat_code in ("2", "4", "6")  # continuous types

        if n_spans <= 1:
            span_type = "single"
        elif continuity:
            span_type = "multi_continuous"
        else:
            span_type = "multi_simply_supported"

        # Determine design era (California seismic codes improved after 1975)
        year = row.get("year_built", 1960)
        if pd.isna(year):
            year = 1960
        design_era = "seismic" if int(year) >= 1975 else "conventional"

        # Structure length
        length = row.get("structure_length_m", 30)
        if pd.isna(length):
            length = 30

        # Subtype from design code
        design_code = str(row.get("design_code", "00")).strip()
        subtype = ""
        if design_code in ("05", "06"):
            subtype = "box_girder"
        elif design_code == "07":
            subtype = "frame"

        return classify_bridge(material, span_type, design_era, length, subtype)

    nbi_df = nbi_df.copy()
    nbi_df["hwb_class"] = nbi_df.apply(_row_to_hwb, axis=1)
    return nbi_df


# ---------------------------------------------------------------------------
# Convenience: load all data
# ---------------------------------------------------------------------------


def load_shakemap(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load ShakeMap grid data from local file.

    Parameters
    ----------
    filepath : path-like, optional
        Path to grid.xml.  Defaults to data/grid.xml.

    Returns
    -------
    pd.DataFrame
    """
    if filepath is None:
        filepath = DATA_DIR / "grid.xml"
    return parse_shakemap_grid(filepath)


def load_stations(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load ShakeMap station data from local file.

    Parameters
    ----------
    filepath : path-like, optional
        Path to stationlist.json.  Defaults to data/stationlist.json.

    Returns
    -------
    pd.DataFrame
    """
    if filepath is None:
        filepath = DATA_DIR / "stationlist.json"
    return parse_shakemap_stations(filepath)


def load_nbi(
    filepath: Optional[Union[str, Path]] = None,
    northridge_bbox: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Load NBI bridge inventory from local file.

    Parameters
    ----------
    filepath : path-like, optional
        Path to NBI delimited file.  Defaults to data/CA24.txt.
    northridge_bbox : dict, optional
        Bounding box filter for Northridge area.

    Returns
    -------
    pd.DataFrame
    """
    if filepath is None:
        filepath = DATA_DIR / "CA24.txt"
    return parse_nbi(filepath, northridge_bbox)


def load_all(
    shakemap_path: Optional[Union[str, Path]] = None,
    stations_path: Optional[Union[str, Path]] = None,
    nbi_path: Optional[Union[str, Path]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load all data files and return as a dict of DataFrames.

    Returns
    -------
    dict
        Keys: "shakemap", "stations", "nbi"
    """
    return {
        "shakemap": load_shakemap(shakemap_path),
        "stations": load_stations(stations_path),
        "nbi": load_nbi(nbi_path),
    }

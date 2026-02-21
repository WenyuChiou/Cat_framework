"""
Data parsing for USGS ShakeMap and FHWA NBI bridge inventory.

This module parses local files into pandas DataFrames for analysis.

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
# Default paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data"

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
            f"Run: python main.py --download-pipeline"
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

    # Extract event metadata from root attributes (no silent defaults)
    event_attrs = root.attrib
    df.attrs["event_id"] = event_attrs.get("event_id", "unknown")
    df.attrs["magnitude"] = float(event_attrs.get("magnitude", 0.0))
    df.attrs["lat"] = float(event_attrs.get("lat", 0.0))
    df.attrs["lon"] = float(event_attrs.get("lon", 0.0))
    df.attrs["event_description"] = event_attrs.get(
        "event_description", ""
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
            f"Run: python main.py --download-pipeline"
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
            f"Run: python main.py --download-pipeline"
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

    # Filter to bounding box if provided; otherwise return all bridges
    if northridge_bbox is not None:
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


def classify_nbi_to_hazus(
    nbi_df: pd.DataFrame,
    hwb_filter: list[str] | None = None,
    design_era_filter: str | None = None,
    material_filter: list[str] | None = None,
    nbi_filters: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Add Hazus bridge class (HWB) column to NBI DataFrame.

    Uses material code, design type, number of spans, structure length,
    and year built to assign each bridge a Hazus classification.

    Parameters
    ----------
    nbi_df : pd.DataFrame
        Parsed NBI data from parse_nbi().
    hwb_filter : list[str], optional
        If provided, keep only bridges matching these HWB classes
        (e.g. ["HWB5", "HWB17"]).
    design_era_filter : str, optional
        "conventional" (pre-1975) or "seismic" (post-1975).
    material_filter : list[str], optional
        If provided, keep only bridges matching these materials
        (e.g. ["concrete", "steel"]).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with added 'hwb_class' column, filtered as requested.
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

    # ── Apply optional filters ────────────────────────────────────────
    n_before = len(nbi_df)

    if hwb_filter:
        nbi_df = nbi_df[nbi_df["hwb_class"].isin(hwb_filter)].copy()

    if design_era_filter == "conventional":
        nbi_df = nbi_df[nbi_df["year_built"] < 1975].copy()
    elif design_era_filter == "seismic":
        nbi_df = nbi_df[nbi_df["year_built"] >= 1975].copy()

    if material_filter:
        nbi_df = nbi_df[nbi_df["material"].isin(material_filter)].copy()

    # Generic NBI column filters (key=value, key>value, etc.)
    if nbi_filters:
        for col, condition in nbi_filters.items():
            if col not in nbi_df.columns:
                print(f"  [Filter] Warning: column '{col}' not found, skipping")
                continue
            # List match
            if isinstance(condition, list):
                nbi_df = nbi_df[nbi_df[col].astype(str).isin([str(v) for v in condition])].copy()
            # Numeric comparisons
            elif isinstance(condition, str) and condition[:2] in (">=", "<="):
                op = condition[:2]
                val = float(condition[2:])
                num_col = pd.to_numeric(nbi_df[col], errors="coerce")
                if op == ">=":
                    nbi_df = nbi_df[num_col >= val].copy()
                else:
                    nbi_df = nbi_df[num_col <= val].copy()
            elif isinstance(condition, str) and condition[0] in (">", "<"):
                op = condition[0]
                val = float(condition[1:])
                num_col = pd.to_numeric(nbi_df[col], errors="coerce")
                if op == ">":
                    nbi_df = nbi_df[num_col > val].copy()
                else:
                    nbi_df = nbi_df[num_col < val].copy()
            else:
                # Exact match
                nbi_df = nbi_df[nbi_df[col].astype(str) == str(condition)].copy()

    if n_before != len(nbi_df):
        print(f"  [Filter] {n_before} → {len(nbi_df)} bridges after filtering")

    nbi_df.reset_index(drop=True, inplace=True)
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

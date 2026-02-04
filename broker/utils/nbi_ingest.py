from __future__ import annotations

import argparse
import warnings
import html
import json
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_RAW_DIR = Path("data/nbi/raw")
DEFAULT_OUT_DIR = Path("data/nbi")
DEFAULT_HAZARD_DIR = Path("data/hazard/usgs/shakemap")


@dataclass
class RawSource:
    path: Path
    source_type: str  # "csv" or "zip"
    inner_csv: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
        return data.decode("utf-8", errors="replace")


def _download_file(url: str, dest: Path, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        return

    _ensure_dir(dest.parent)
    with urllib.request.urlopen(url) as resp:
        with dest.open("wb") as f:
            shutil.copyfileobj(resp, f)


def _extract_zip(zip_path: Path, dest_dir: Path, overwrite: bool) -> None:
    _ensure_dir(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            out_path = dest_dir / member.filename
            if out_path.exists() and not overwrite:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _list_raw_sources(raw_dir: Path) -> List[RawSource]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    candidates = [p for p in raw_dir.iterdir() if p.suffix.lower() in {".csv", ".txt", ".zip"}]
    if not candidates:
        raise FileNotFoundError(f"No .csv, .txt, or .zip found in {raw_dir}")

    sources: List[RawSource] = []
    for path in candidates:
        if path.suffix.lower() in {".csv", ".txt"}:
            sources.append(RawSource(path=path, source_type="csv"))
            continue

        with zipfile.ZipFile(path, "r") as zf:
            data_names = [
                n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))
            ]
            if not data_names:
                raise ValueError(f"Zip has no CSV/TXT files: {path}")
            largest = max(data_names, key=lambda n: zf.getinfo(n).file_size)
            sources.append(RawSource(path=path, source_type="zip", inner_csv=largest))

    return sources


def _latest_source(sources: List[RawSource]) -> RawSource:
    return max(sources, key=lambda s: s.path.stat().st_mtime)


def _read_csv_from_zip(
    zpath: Path,
    inner_name: str,
    delimiter: str,
    quotechar: str,
) -> Tuple[pd.DataFrame, int]:
    with zipfile.ZipFile(zpath, "r") as zf:
        with zf.open(inner_name) as f:
            warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
            bad_lines = {"count": 0}

            def _count_bad_line(_: List[str]) -> None:
                bad_lines["count"] += 1
                return None

            df = pd.read_csv(
                f,
                sep=delimiter,
                quotechar=quotechar,
                engine="python",
                on_bad_lines=_count_bad_line,
            )
            return df, bad_lines["count"]


def _read_csv(path: Path, delimiter: str, quotechar: str) -> Tuple[pd.DataFrame, int]:
    warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)
    bad_lines = {"count": 0}

    def _count_bad_line(_: List[str]) -> None:
        bad_lines["count"] += 1
        return None

    df = pd.read_csv(
        path,
        sep=delimiter,
        quotechar=quotechar,
        engine="python",
        on_bad_lines=_count_bad_line,
    )
    return df, bad_lines["count"]


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    # Keep original column names to preserve official NBI headers
    df = df.copy()

    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        series = df[col].astype("string")
        series = series.str.strip()
        series = series.replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "NULL": pd.NA})
        df[col] = series

    return df


def _write_meta(
    meta_path: Path,
    source: RawSource,
    df: pd.DataFrame,
    config_path: Optional[Path],
    tag: str,
    bad_lines: int,
    state_filter: Optional[str],
) -> None:
    meta = {
        "tag": tag,
        "generated_at_utc": _now_iso(),
        "source": {
            "path": str(source.path),
            "type": source.source_type,
            "inner_csv": source.inner_csv,
        },
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "bad_lines_skipped": int(bad_lines),
        "state_filter": state_filter,
        "column_names": list(df.columns),
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
        "pipeline_config": str(config_path) if config_path else None,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_pipeline_config(meta_dir: Path) -> Optional[Dict[str, object]]:
    config_path = meta_dir / "nbi_pipeline.json"
    if not config_path.exists():
        return None
    # Handle UTF-8 BOM if present (common on Windows-edited JSON files).
    return json.loads(config_path.read_text(encoding="utf-8-sig"))


def _apply_curated_rules(df: pd.DataFrame, config: Optional[Dict[str, object]]) -> pd.DataFrame:
    if not config:
        return df

    curated = df

    rename = config.get("rename") if isinstance(config, dict) else None
    if isinstance(rename, dict):
        curated = curated.rename(columns=rename)

    keep = config.get("curated_columns") if isinstance(config, dict) else None
    if isinstance(keep, list):
        keep_set = [c for c in keep if c in curated.columns]
        curated = curated[keep_set]

    return curated


def _apply_state_filter(df: pd.DataFrame, config: Optional[Dict[str, object]]) -> Tuple[pd.DataFrame, Optional[str]]:
    if not config:
        return df, None

    state_code = None
    state_column = None
    if isinstance(config, dict):
        state_code = config.get("state_code_filter")
        state_column = config.get("state_code_column")

    if not state_code or not state_column:
        return df, None

    if state_column not in df.columns:
        return df, f"{state_column} (missing)"

    series = df[state_column].astype("string").str.strip()
    series = series.str.extract(r"(\d+)")[0].str.zfill(2)
    target = str(state_code).zfill(2)
    filtered = df[series == target]
    return filtered, f"{state_column}={target}"


def _write_log(log_path: Path, lines: Iterable[str]) -> None:
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _slug_for_source(source: RawSource) -> str:
    if source.inner_csv:
        return Path(source.inner_csv).stem.replace(" ", "_")
    return source.path.stem.replace(" ", "_")


def _run_one(
    source: RawSource,
    out_dir: Path,
    delimiter: str,
    quotechar: str,
    tag: str,
) -> Tuple[Path, Path, Path]:
    meta_dir = out_dir / "meta"
    log_dir = out_dir / "logs"
    clean_dir = out_dir / "clean"
    curated_dir = out_dir / "curated"

    _ensure_dir(out_dir)
    _ensure_dir(meta_dir)
    _ensure_dir(log_dir)
    _ensure_dir(clean_dir)
    _ensure_dir(curated_dir)

    if source.source_type == "csv":
        df_raw, bad_lines = _read_csv(source.path, delimiter, quotechar)
    else:
        assert source.inner_csv
        df_raw, bad_lines = _read_csv_from_zip(
            source.path, source.inner_csv, delimiter, quotechar
        )

    config = _load_pipeline_config(meta_dir)
    df_filtered, state_filter = _apply_state_filter(df_raw, config)
    df_clean = _clean_strings(df_filtered)
    df_curated = _apply_curated_rules(df_clean, config)

    clean_path = clean_dir / f"nbi_{tag}_clean.csv"
    curated_path = curated_dir / f"nbi_{tag}_curated.csv"
    meta_path = meta_dir / f"nbi_{tag}_meta.json"
    log_path = log_dir / f"nbi_{tag}_run.log"

    df_clean.to_csv(clean_path, index=False)
    df_curated.to_csv(curated_path, index=False)
    _write_meta(
        meta_path,
        source,
        df_clean,
        (meta_dir / "nbi_pipeline.json") if config else None,
        tag,
        bad_lines,
        state_filter,
    )

    log_lines = [
        f"run_utc={_now_iso()}",
        f"raw_source={source.path}",
        f"source_type={source.source_type}",
        f"inner_csv={source.inner_csv}",
        f"bad_lines_skipped={bad_lines}",
        f"state_filter={state_filter}",
        f"rows={df_clean.shape[0]}",
        f"cols={df_clean.shape[1]}",
        f"clean_out={clean_path}",
        f"curated_out={curated_path}",
        f"meta_out={meta_path}",
    ]
    _write_log(log_path, log_lines)

    return clean_path, curated_path, meta_path


def run_ingest(
    raw_dir: Path,
    out_dir: Path,
    delimiter: str,
    quotechar: str,
    process_all: bool,
) -> List[Tuple[Path, Path, Path]]:
    sources = _list_raw_sources(raw_dir)
    outputs: List[Tuple[Path, Path, Path]] = []

    if process_all:
        for source in sources:
            tag = _slug_for_source(source)
            print(f"[NBI] Ingesting source: {source.path}")
            outputs.append(_run_one(source, out_dir, delimiter, quotechar, tag))
        return outputs

    latest = _latest_source(sources)
    print(f"[NBI] Ingesting latest source: {latest.path}")
    outputs.append(_run_one(latest, out_dir, delimiter, quotechar, "latest"))
    return outputs


def _scope_key(scope: str) -> str:
    if scope == "all_states_individual":
        return "hwybr"
    return "hwybronlyonefile"


def _find_fhwa_disclaimer_url(year: int, scope_key: str) -> str:
    page_url = f"https://www.fhwa.dot.gov/bridge/nbi/ascii{year}.cfm"
    page_html = _fetch_text(page_url)

    pattern = rf'href="([^"]*disclaim\.cfm\?nbiYear={year}{scope_key}[^\"]*)"'
    match = re.search(pattern, page_html, re.IGNORECASE)
    if match:
        return urllib.parse.urljoin(page_url, html.unescape(match.group(1)))

    return f"https://www.fhwa.dot.gov/bridge/nbi/disclaim.cfm?nbiYear={year}{scope_key}&nbiZip=zip"


def _resolve_fhwa_zip_url(disclaimer_url: str, base_url: str) -> str:
    html = _fetch_text(disclaimer_url)
    zip_match = re.search(r'href="([^"]+\.zip)"', html, re.IGNORECASE)
    if zip_match:
        # Zip links are often relative to the disclaimer page path.
        return urllib.parse.urljoin(disclaimer_url, zip_match.group(1))

    return disclaimer_url


def _download_nbi(year: int, raw_dir: Path, overwrite: bool, scope: str, unzip: bool) -> Path:
    scope_key = _scope_key(scope)
    print(f"[NBI] Resolving FHWA download link (year={year}, scope={scope})...")
    disclaimer_url = _find_fhwa_disclaimer_url(year, scope_key)
    print(f"[NBI] Disclaimer URL: {disclaimer_url}")
    final_url = _resolve_fhwa_zip_url(disclaimer_url, "https://www.fhwa.dot.gov/")
    print(f"[NBI] Downloading ZIP: {final_url}")

    dest = raw_dir / f"nbi_{year}_delimited_{scope}.zip"
    _download_file(final_url, dest, overwrite)
    if unzip:
        print(f"[NBI] Extracting ZIP to {raw_dir}...")
        _extract_zip(dest, raw_dir, overwrite)
    print(f"[NBI] Download complete: {dest}")
    return dest


def _pick_preferred_product(products: List[Dict[str, object]]) -> Dict[str, object]:
    for product in products:
        if product.get("preferred"):
            return product
    return products[0]


def _download_usgs_shakemap(event_id: str, hazard_dir: Path, files: List[str], overwrite: bool) -> List[Path]:
    detail_url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/detail/{event_id}.geojson"
    print(f"[ShakeMap] Fetching event detail: {detail_url}")
    data = json.loads(_fetch_text(detail_url))

    products = data.get("properties", {}).get("products", {}).get("shakemap", [])
    if not products:
        raise ValueError(f"No shakemap product found for event {event_id}")

    product = _pick_preferred_product(products)
    contents = product.get("contents", {})

    raw_dir = hazard_dir / "raw"
    meta_dir = hazard_dir / "meta"
    log_dir = hazard_dir / "logs"
    _ensure_dir(raw_dir)
    _ensure_dir(meta_dir)
    _ensure_dir(log_dir)

    def _resolve_content_key(name: str) -> Optional[str]:
        if name in contents:
            return name
        prefixed = f"download/{name}"
        if prefixed in contents:
            return prefixed
        if name.startswith("download/"):
            unprefixed = name[len("download/") :]
            if unprefixed in contents:
                return unprefixed
        legacy_map = {
            "grid.xml.zip": "download/grid.xml",
            "grid.xml": "download/grid.xml",
            "shape.zip": "download/shape.zip",
            "info.json": "download/info.json",
        }
        mapped = legacy_map.get(name)
        if mapped and mapped in contents:
            return mapped
        return None

    downloaded: List[Path] = []
    print(f"[ShakeMap] Downloading files: {files}")
    for fname in files:
        key = _resolve_content_key(fname)
        if not key:
            continue
        entry = contents.get(key)
        if not entry or "url" not in entry:
            continue
        url = entry["url"]
        dest = raw_dir / Path(key).name
        print(f"[ShakeMap] Downloading {Path(key).name} -> {dest}")
        _download_file(url, dest, overwrite)
        downloaded.append(dest)

    meta_path = meta_dir / f"shakemap_{event_id}_meta.json"
    meta = {
        "event_id": event_id,
        "generated_at_utc": _now_iso(),
        "detail_url": detail_url,
        "downloaded_files": [str(p) for p in downloaded],
        "product": {
            "source": product.get("source"),
            "code": product.get("code"),
            "update_time": product.get("updateTime"),
            "preferred": product.get("preferred"),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")

    log_path = log_dir / f"shakemap_{event_id}_run.log"
    log_lines = [
        f"run_utc={_now_iso()}",
        f"event_id={event_id}",
        f"detail_url={detail_url}",
        f"downloaded={len(downloaded)}",
    ]
    _write_log(log_path, log_lines)

    return downloaded


def run_downloads(
    meta_dir: Path,
    raw_dir: Path,
    hazard_dir: Path,
    nbi_year_override: Optional[int],
    usgs_event_override: Optional[str],
) -> None:
    config = _load_pipeline_config(meta_dir) or {}
    download_cfg = config.get("download", {}) if isinstance(config, dict) else {}

    nbi_cfg = download_cfg.get("nbi", {}) if isinstance(download_cfg, dict) else {}
    usgs_cfg = download_cfg.get("usgs_shakemap", {}) if isinstance(download_cfg, dict) else {}

    nbi_year = nbi_year_override or nbi_cfg.get("year")
    nbi_overwrite = bool(nbi_cfg.get("overwrite", False))
    nbi_scope = nbi_cfg.get("scope", "all_states_single")
    nbi_format = nbi_cfg.get("format", "delimited")
    nbi_unzip = bool(nbi_cfg.get("unzip", True))

    if nbi_year:
        if nbi_format != "delimited":
            raise ValueError("Only delimited NBI downloads are supported.")
        print("[Pipeline] NBI download starting...")
        _download_nbi(int(nbi_year), raw_dir, nbi_overwrite, str(nbi_scope), nbi_unzip)
        print("[Pipeline] NBI download done.")

    usgs_event = usgs_event_override or usgs_cfg.get("event_id")
    usgs_files = usgs_cfg.get("files") if isinstance(usgs_cfg.get("files"), list) else None
    usgs_overwrite = bool(usgs_cfg.get("overwrite", False))
    if usgs_event:
        files = usgs_files or ["grid.xml.zip", "shape.zip", "info.json"]
        print("[Pipeline] ShakeMap download starting...")
        _download_usgs_shakemap(str(usgs_event), hazard_dir, files, usgs_overwrite)
        print("[Pipeline] ShakeMap download done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBI raw CSV/ZIP and produce clean/curated outputs.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_DIR, help="Raw NBI directory")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output base directory")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter (default: ,)")
    parser.add_argument("--quotechar", type=str, default="'", help="CSV quote character (default: ')")
    parser.add_argument("--all", action="store_true", help="Process all raw files instead of latest only")
    parser.add_argument("--download", action="store_true", help="Download NBI and USGS ShakeMap before ingest")
    parser.add_argument("--download-only", action="store_true", help="Only download data, skip ingest")
    parser.add_argument("--nbi-year", type=int, help="Override NBI year for download")
    parser.add_argument("--usgs-event", type=str, help="Override USGS event id for ShakeMap download")

    args = parser.parse_args()

    if args.download:
        meta_dir = args.out / "meta"
        run_downloads(meta_dir, args.raw, DEFAULT_HAZARD_DIR, args.nbi_year, args.usgs_event)
        if args.download_only:
            return

    run_ingest(args.raw, args.out, args.delimiter, args.quotechar, args.all)


if __name__ == "__main__":
    main()

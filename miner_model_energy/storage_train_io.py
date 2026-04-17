from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .data_io import TARGET_COLUMN, TIMESTAMP_COLUMN
from .ml_config import ModelConfig

_EASTERN_TZ = ZoneInfo("America/New_York")


def _parquet_supported() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        return False


def storage_cache_paths(config: ModelConfig) -> Tuple[Path, Path]:
    data_cfg = config.data
    cache_dir = Path(data_cfg["storage_cache_dir"])
    cache_name = str(data_cfg.get("storage_cache_parquet_name", "train_merged.parquet"))

    if _parquet_supported() and cache_name.endswith(".parquet"):
        cache_path = cache_dir / cache_name
    else:
        # Fallback to CSV cache if pyarrow isn't installed.
        if cache_name.endswith(".parquet"):
            cache_path = cache_dir / (cache_name[: -len(".parquet")] + ".csv")
        else:
            cache_path = cache_dir / f"{cache_name}.csv"

    manifest_path = cache_dir / "manifest.json"
    return cache_path, manifest_path


def storage_cache_exists(config: ModelConfig) -> bool:
    cache_path, _manifest_path = storage_cache_paths(config)
    return cache_path.exists()


def _parse_manifest_downloaded_at(manifest_path: Path) -> datetime | None:
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    downloaded_at = payload.get("downloaded_at")
    if not isinstance(downloaded_at, str) or not downloaded_at.strip():
        return None
    try:
        parsed = datetime.fromisoformat(downloaded_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def storage_cache_last_updated_label(config: ModelConfig) -> str:
    cache_path, manifest_path = storage_cache_paths(config)
    parsed = _parse_manifest_downloaded_at(manifest_path)
    if parsed is not None:
        return parsed.astimezone(_EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S ET")
    if cache_path.exists():
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
        return mtime.astimezone(_EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S ET")
    return "unknown"


def _ensure_normalized_train_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "total_load" in out.columns and TARGET_COLUMN not in out.columns:
        out = out.rename(columns={"total_load": TARGET_COLUMN})

    if TARGET_COLUMN not in out.columns:
        raise ValueError(f"Missing `{TARGET_COLUMN}` (or `total_load`) in training data.")

    if TIMESTAMP_COLUMN not in out.columns:
        raise ValueError(f"Missing `{TIMESTAMP_COLUMN}` in training data.")

    out[TIMESTAMP_COLUMN] = pd.to_datetime(out[TIMESTAMP_COLUMN], utc=True, errors="raise")
    out[TIMESTAMP_COLUMN] = out[TIMESTAMP_COLUMN].dt.tz_convert("UTC").dt.tz_localize(None)
    out = out.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return out


def _read_cached_train_frame(cache_path: Path) -> pd.DataFrame:
    if cache_path.suffix == ".parquet":
        df = pd.read_parquet(cache_path)
    else:
        df = pd.read_csv(cache_path)
    return _ensure_normalized_train_frame(df)


def _write_cached_train_frame(cache_path: Path, frame: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix == ".parquet":
        frame.to_parquet(cache_path, index=False)
    else:
        frame.to_csv(cache_path, index=False)


def _download_train_part_csv(url: str) -> pd.DataFrame:
    # Storage dumps are public; use simple GET downloads.
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    return pd.read_csv(BytesIO(resp.content))


def _build_train_cache_from_storage(config: ModelConfig) -> Tuple[pd.DataFrame, dict]:
    data_cfg = config.data
    base_url = str(data_cfg["storage_train_base_url"])
    if not base_url.endswith("/"):
        base_url += "/"
    parts: List[str] = list(data_cfg["storage_train_parts"])

    frames: list[pd.DataFrame] = []
    for part in parts:
        url = base_url + str(part)
        frames.append(_download_train_part_csv(url))

    merged = pd.concat(frames, ignore_index=True)
    merged = _ensure_normalized_train_frame(merged)

    manifest = {
        "parts": parts,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(merged)),
    }
    return merged, manifest


def load_train_from_storage_parts(config: ModelConfig, force_refresh: bool) -> pd.DataFrame:
    cache_path, manifest_path = storage_cache_paths(config)
    cache_exists = cache_path.exists()

    if cache_exists and not force_refresh:
        return _read_cached_train_frame(cache_path)

    try:
        print(
            "  [train]     Fetching training data from Supabase Storage and building local cache…",
            flush=True,
        )
        print("  [train]     This update usually takes 2-3 minutes.", flush=True)
        merged, manifest = _build_train_cache_from_storage(config)
        _write_cached_train_frame(cache_path, merged)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return merged
    except Exception:
        if cache_exists and force_refresh:
            # Manual refresh failed; keep the system running with the last good cache.
            # Caller will log this decision at a higher level.
            print(
                "  [train]     Supabase Storage cache rebuild failed; falling back to existing cache.",
                flush=True,
            )
            return _read_cached_train_frame(cache_path)
        raise


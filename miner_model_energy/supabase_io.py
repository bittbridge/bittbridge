from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

import pandas as pd

from .data_io import TARGET_COLUMN, TIMESTAMP_COLUMN

try:
    from supabase import create_client
except Exception:  # pragma: no cover - handled at runtime
    create_client = None


def create_supabase_data_client(url: str, key: str):
    if create_client is None:
        raise RuntimeError(
            "supabase package is not installed. Install dependencies from requirements.txt first."
        )
    return create_client(url, key)


def _normalize_dt_column(frame: pd.DataFrame) -> pd.DataFrame:
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError(f"Missing `{TIMESTAMP_COLUMN}` column in Supabase payload.")
    out = frame.copy()
    out[TIMESTAMP_COLUMN] = pd.to_datetime(out[TIMESTAMP_COLUMN], utc=True, errors="raise")
    out[TIMESTAMP_COLUMN] = out[TIMESTAMP_COLUMN].dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def normalize_supabase_train_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "total_load" in out.columns and TARGET_COLUMN not in out.columns:
        out = out.rename(columns={"total_load": TARGET_COLUMN})
    if TARGET_COLUMN not in out.columns:
        raise ValueError(
            f"Missing `{TARGET_COLUMN}` (or `total_load`) in Supabase train data."
        )
    out = _normalize_dt_column(out)
    out = out.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return out


def normalize_supabase_test_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_dt_column(frame)
    for col in ("horizon_min", "fetched_at"):
        if col in out.columns:
            out = out.drop(columns=[col])
    out = out.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return out


def parse_timestamp_for_supabase(timestamp_str: str) -> pd.Timestamp:
    ts = pd.to_datetime(timestamp_str, utc=True, errors="raise")
    return ts.tz_convert("UTC").tz_localize(None)


def timestamp_candidates_for_supabase(timestamp_str: str) -> list[pd.Timestamp]:
    """
    Build timestamp candidates for querying forecast rows.

    Primary candidate is UTC-normalized naive timestamp (the canonical path).
    Secondary candidate keeps local wall-clock time (naive) for compatibility
    with rows written with an unintended timezone conversion.
    """
    raw = pd.to_datetime(timestamp_str, errors="raise")
    candidates: list[pd.Timestamp] = []

    if raw.tzinfo is None:
        candidates.append(raw)
    else:
        candidates.append(raw.tz_convert("UTC").tz_localize(None))
        candidates.append(raw.tz_localize(None))

    unique: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for ts in candidates:
        if ts in seen:
            continue
        seen.add(ts)
        unique.append(ts)
    return unique


def format_timestamp_for_supabase(timestamp_str: str) -> str:
    ts = timestamp_candidates_for_supabase(timestamp_str)[0]
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def fetch_supabase_train_all(client, schema: str, table: str, page_size: int = 1000) -> pd.DataFrame:
    offset = 0
    rows: list[Dict[str, Any]] = []
    while True:
        response = (
            client.schema(schema)
            .table(table)
            .select("*")
            .order(TIMESTAMP_COLUMN, desc=False)
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    if not rows:
        raise ValueError(f"Supabase `{schema}.{table}` returned no training rows.")
    return normalize_supabase_train_frame(pd.DataFrame(rows))


def pick_forecast_row_for_horizon(
    candidates: list[Dict[str, Any]], horizon_min: int
) -> Dict[str, Any] | None:
    """
    Pick one test/forecast row from candidates matching horizon_min when that column exists.
    Same rules as fetch_supabase_test_row (strict horizon when column present).
    """
    if not candidates:
        return None
    has_horizon = any((row.get("horizon_min") is not None) for row in candidates)
    if has_horizon:
        for row in candidates:
            value = row.get("horizon_min")
            if value is None:
                continue
            if int(value) == int(horizon_min):
                return row
        return None
    return candidates[0]


def fetch_supabase_train_tail(client, schema: str, table: str, n_rows: int) -> pd.DataFrame:
    response = (
        client.schema(schema)
        .table(table)
        .select("*")
        .order(TIMESTAMP_COLUMN, desc=True)
        .limit(int(n_rows))
        .execute()
    )
    rows = response.data or []
    if not rows:
        raise ValueError(f"Supabase `{schema}.{table}` returned no rows for recent history.")
    frame = normalize_supabase_train_frame(pd.DataFrame(rows))
    return frame.tail(int(n_rows)).reset_index(drop=True)


def fetch_supabase_test_row(
    client,
    schema: str,
    table: str,
    dt_target: str,
    horizon_min: int,
    nearest_fallback_minutes: int | None = None,
) -> Dict[str, Any] | None:
    candidate_ts = timestamp_candidates_for_supabase(dt_target)
    candidate_exact = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in candidate_ts]

    for dt_exact in candidate_exact:
        response = client.schema(schema).table(table).select("*").eq(TIMESTAMP_COLUMN, dt_exact).execute()
        rows = response.data or []
        picked = pick_forecast_row_for_horizon(rows, horizon_min)
        if picked is not None:
            return picked

    if nearest_fallback_minutes is None or nearest_fallback_minutes <= 0:
        return None

    for target in candidate_ts:
        start = (target - timedelta(minutes=int(nearest_fallback_minutes))).strftime("%Y-%m-%d %H:%M:%S")
        end = (target + timedelta(minutes=int(nearest_fallback_minutes))).strftime("%Y-%m-%d %H:%M:%S")
        around = (
            client.schema(schema)
            .table(table)
            .select("*")
            .gte(TIMESTAMP_COLUMN, start)
            .lte(TIMESTAMP_COLUMN, end)
            .order(TIMESTAMP_COLUMN, desc=False)
            .execute()
        )
        nearby_rows = around.data or []
        picked = pick_forecast_row_for_horizon(nearby_rows, horizon_min)
        if picked is not None:
            return picked
    return None


def fetch_latest_forecast_row_matching_horizon(
    client,
    schema: str,
    table: str,
    horizon_min: int,
    limit: int = 2000,
) -> Dict[str, Any] | None:
    """
    Newest rows first; return the first row matching horizon_min (or legacy row without horizon).
    Used when no forecast exists near wall-clock time (e.g. deploy compatibility probe).
    """
    response = (
        client.schema(schema)
        .table(table)
        .select("*")
        .order(TIMESTAMP_COLUMN, desc=True)
        .limit(int(limit))
        .execute()
    )
    rows = response.data or []
    return pick_forecast_row_for_horizon(rows, horizon_min)


def fetch_supabase_test_row_for_probe(
    client,
    schema: str,
    table: str,
    dt_target: str,
    horizon_min: int,
) -> Dict[str, Any] | None:
    """
    Resolve a forecast row for offline compatibility checks: try tight windows around dt_target,
    then fall back to the latest stored row for this horizon. Validator-facing paths should keep
    using fetch_supabase_test_row with a small nearest_fallback_minutes.
    """
    for fb in (5, 60, 360, 1440, 10080):
        row = fetch_supabase_test_row(
            client,
            schema,
            table,
            dt_target,
            horizon_min,
            nearest_fallback_minutes=fb,
        )
        if row is not None:
            return row
    return fetch_latest_forecast_row_matching_horizon(
        client, schema, table, horizon_min, limit=2000
    )

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_io import TARGET_COLUMN, TIMESTAMP_COLUMN

# Raw ISO-NE-style columns are named "<station>-<suffix>". Used for optional whitelist filtering.
KNOWN_WEATHER_SUFFIXES = frozenset({"tmpf", "dwpf", "relh", "sped", "drct"})

DEFAULT_LAGS = [1, 2]
DEFAULT_ROLLING_WINDOWS = [3, 6]
DEFAULT_EXCLUDE_COLS = {
    TIMESTAMP_COLUMN,
    TARGET_COLUMN,
    "Native Load",
    "Asset Related Load",
    "Total Load With Estimated Solar",
    "Native Load With Estimated Solar",
    "load_change_1h",
}


def filter_weather_suffix_columns(
    df: pd.DataFrame, include_suffixes: Optional[Sequence[str]]
) -> pd.DataFrame:
    """
    Raw ISO columns look like ``BDL-tmpf`` (suffix in :data:`KNOWN_WEATHER_SUFFIXES`).

    * **Empty or None** (default in YAML): drop **all** such raw weather columns so only
      ``dt``, ``Total Load``, and other non-weather columns remain unless you enable
      engineered features (time, lags, etc.).
    * **Non-empty list**: keep only the listed suffix groups (whitelist); drop other
      known weather suffixes.
    """
    if include_suffixes is None:
        include_suffixes = []
    allowed = frozenset(str(s).strip().lower() for s in include_suffixes if str(s).strip())

    if not allowed:
        to_drop: List[str] = []
        for col in df.columns:
            name = str(col)
            if "-" not in name:
                continue
            suffix = name.rsplit("-", 1)[-1].lower()
            if suffix in KNOWN_WEATHER_SUFFIXES:
                to_drop.append(col)
        return df.drop(columns=to_drop) if to_drop else df

    unknown = allowed - KNOWN_WEATHER_SUFFIXES
    if unknown:
        raise ValueError(
            f"Unknown weather column suffix(es): {sorted(unknown)}. "
            f"Allowed: {sorted(KNOWN_WEATHER_SUFFIXES)}."
        )
    to_drop = []
    for col in df.columns:
        name = str(col)
        if "-" not in name:
            continue
        suffix = name.rsplit("-", 1)[-1].lower()
        if suffix in KNOWN_WEATHER_SUFFIXES and suffix not in allowed:
            to_drop.append(col)
    if not to_drop:
        return df
    return df.drop(columns=to_drop)


def _weather_column_groups(columns: Sequence[str]) -> Tuple[List[str], ...]:
    cols = [str(c) for c in columns]
    tmpf = [c for c in cols if c.endswith("-tmpf")]
    dwpf = [c for c in cols if c.endswith("-dwpf")]
    relh = [c for c in cols if c.endswith("-relh")]
    sped = [c for c in cols if c.endswith("-sped")]
    drct = [c for c in cols if c.endswith("-drct")]
    return tmpf, dwpf, relh, sped, drct


def _row_std_across_stations(frame: pd.DataFrame) -> pd.Series:
    """
    Std across station columns per row. With a single column, pandas std(ddof=1) is NaN;
    real deployments often have one RELH/SPED column, so use 0.0 in that case.
    """
    if frame.shape[1] < 2:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return frame.std(axis=1, ddof=0)


def add_engineered_features(df: pd.DataFrame, feature_cfg: Dict) -> pd.DataFrame:
    """
    Feature families mirror manual notebook setup.
    All groups are gated by feature_cfg booleans (defaults are False in YAML).
    """
    out = df.copy()
    ts = out[TIMESTAMP_COLUMN]

    if feature_cfg.get("use_time_features", False):
        out["hour"] = ts.dt.hour
        out["minute"] = ts.dt.minute
        out["dayofweek"] = ts.dt.dayofweek
        out["month"] = ts.dt.month

    if feature_cfg.get("use_cyclical_features", False):
        out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
        minute_of_day = ts.dt.hour * 60 + ts.dt.minute
        out["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
        out["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

    tmpf_cols, dwpf_cols, relh_cols, sped_cols, drct_cols = _weather_column_groups(out.columns)

    if feature_cfg.get("use_station_agg_features", False):
        if tmpf_cols:
            out["tmpf_mean"] = out[tmpf_cols].mean(axis=1)
            out["tmpf_std"] = _row_std_across_stations(out[tmpf_cols])
            out["tmpf_min"] = out[tmpf_cols].min(axis=1)
            out["tmpf_max"] = out[tmpf_cols].max(axis=1)
        if relh_cols:
            out["relh_mean"] = out[relh_cols].mean(axis=1)
            out["relh_std"] = _row_std_across_stations(out[relh_cols])
        if sped_cols:
            out["sped_mean"] = out[sped_cols].mean(axis=1)
            out["sped_std"] = _row_std_across_stations(out[sped_cols])
            out["sped_max"] = out[sped_cols].max(axis=1)

    if feature_cfg.get("use_temp_dew_gap", False) and tmpf_cols and dwpf_cols:
        station_prefixes = sorted({c.split("-")[0] for c in tmpf_cols})
        gap_cols: List[str] = []
        for station in station_prefixes:
            tmp_col = f"{station}-tmpf"
            dwp_col = f"{station}-dwpf"
            if tmp_col in out.columns and dwp_col in out.columns:
                name = f"{station}_temp_dew_gap"
                out[name] = out[tmp_col] - out[dwp_col]
                gap_cols.append(name)
        if gap_cols:
            out["temp_dew_gap_mean"] = out[gap_cols].mean(axis=1)
            out["temp_dew_gap_std"] = _row_std_across_stations(out[gap_cols])

    if feature_cfg.get("use_wind_vector_features", False) and drct_cols:
        wind_x_cols: List[str] = []
        wind_y_cols: List[str] = []
        station_prefixes = sorted({c.split("-")[0] for c in drct_cols})
        for station in station_prefixes:
            drct_col = f"{station}-drct"
            sped_col = f"{station}-sped"
            if drct_col in out.columns and sped_col in out.columns:
                radians = np.deg2rad(pd.to_numeric(out[drct_col], errors="coerce").astype(float))
                sped = pd.to_numeric(out[sped_col], errors="coerce").astype(float)
                x_col = f"{station}_wind_x"
                y_col = f"{station}_wind_y"
                out[x_col] = sped * np.cos(radians)
                out[y_col] = sped * np.sin(radians)
                wind_x_cols.append(x_col)
                wind_y_cols.append(y_col)
        if wind_x_cols:
            out["wind_x_mean"] = out[wind_x_cols].mean(axis=1)
            out["wind_y_mean"] = out[wind_y_cols].mean(axis=1)

    if TARGET_COLUMN in out.columns:
        lag_steps = feature_cfg.get("load_lag_steps", DEFAULT_LAGS)
        if feature_cfg.get("use_load_lags", False):
            for lag in lag_steps:
                out[f"load_lag_{int(lag)}"] = out[TARGET_COLUMN].shift(int(lag))

        if feature_cfg.get("use_load_rolling", False):
            windows = feature_cfg.get("rolling_load_windows", DEFAULT_ROLLING_WINDOWS)
            shifted_load = out[TARGET_COLUMN].shift(1)
            for window in windows:
                w = int(window)
                roll = shifted_load.rolling(window=w)
                out[f"load_roll_mean_{w}"] = roll.mean()
                out[f"load_roll_std_{w}"] = roll.std()
                out[f"load_roll_min_{w}"] = roll.min()
                out[f"load_roll_max_{w}"] = roll.max()
            out["load_delta_1"] = out[TARGET_COLUMN].shift(1) - out[TARGET_COLUMN].shift(2)
            out["load_delta_3"] = out[TARGET_COLUMN].shift(1) - out[TARGET_COLUMN].shift(4)
            out["load_delta_12"] = out[TARGET_COLUMN].shift(1) - out[TARGET_COLUMN].shift(13)
        elif feature_cfg.get("use_load_delta", False):
            out["load_delta_1"] = out[TARGET_COLUMN].shift(1) - out[TARGET_COLUMN].shift(2)
            out["load_delta_12"] = out[TARGET_COLUMN].shift(1) - out[TARGET_COLUMN].shift(13)

    _drop_features_disabled_by_config(out, feature_cfg)
    return out


def _drop_features_disabled_by_config(out: pd.DataFrame, feature_cfg: Dict) -> None:
    """
    Remove engineered columns for toggles that are off so train/test schemas match intent.
    Also strips legacy column names from older runs if present.
    """
    if not feature_cfg.get("use_time_features", False):
        for col in ("hour", "minute", "dayofweek", "month", "day_of_week"):
            if col in out.columns:
                out.drop(columns=[col], inplace=True)

    if not feature_cfg.get("use_cyclical_features", False):
        for col in (
            "hour_sin",
            "hour_cos",
            "minute_of_day_sin",
            "minute_of_day_cos",
            "dow_sin",
            "dow_cos",
        ):
            if col in out.columns:
                out.drop(columns=[col], inplace=True)

    if not feature_cfg.get("use_station_agg_features", False):
        for col in (
            "tmpf_mean",
            "tmpf_std",
            "tmpf_min",
            "tmpf_max",
            "relh_mean",
            "relh_std",
            "sped_mean",
            "sped_std",
            "sped_max",
        ):
            if col in out.columns:
                out.drop(columns=[col], inplace=True)

    if not feature_cfg.get("use_temp_dew_gap", False):
        to_drop = [
            c
            for c in out.columns
            if str(c).endswith("_temp_dew_gap")
            or str(c) in ("temp_dew_gap_mean", "temp_dew_gap_std")
        ]
        if to_drop:
            out.drop(columns=to_drop, inplace=True)

    if not feature_cfg.get("use_wind_vector_features", False):
        to_drop = [
            c
            for c in out.columns
            if str(c).endswith("_wind_x")
            or str(c).endswith("_wind_y")
            or str(c) in ("wind_x_mean", "wind_y_mean")
        ]
        if to_drop:
            out.drop(columns=to_drop, inplace=True)

    if not feature_cfg.get("use_load_lags", False):
        lag_drop = [c for c in out.columns if str(c).startswith("load_lag_")]
        if lag_drop:
            out.drop(columns=lag_drop, inplace=True)

    if not feature_cfg.get("use_load_rolling", False):
        roll_prefixes = ("load_roll_mean_", "load_roll_std_", "load_roll_min_", "load_roll_max_")
        roll_drop = [c for c in out.columns if any(str(c).startswith(p) for p in roll_prefixes)]
        if roll_drop:
            out.drop(columns=roll_drop, inplace=True)

    want_deltas_from_delta = feature_cfg.get("use_load_delta", False) and not feature_cfg.get(
        "use_load_rolling", False
    )
    want_deltas_from_rolling = feature_cfg.get("use_load_rolling", False)
    if not want_deltas_from_delta and not want_deltas_from_rolling:
        for col in ("load_delta_1", "load_delta_3", "load_delta_12"):
            if col in out.columns:
                out.drop(columns=[col], inplace=True)
    elif want_deltas_from_delta and not want_deltas_from_rolling:
        if "load_delta_3" in out.columns:
            out.drop(columns=["load_delta_3"], inplace=True)


def add_test_load_features_from_history(
    test_df: pd.DataFrame, train_df: pd.DataFrame, feature_cfg: Dict
) -> pd.DataFrame:
    """
    Fill autoregressive load features on test from the training load history (notebook pattern).
    Broadcasts the same history-derived values to every test row when test has multiple rows.
    """
    out = test_df.copy()
    load_hist = train_df[TARGET_COLUMN].reset_index(drop=True)
    n_hist = len(load_hist)

    lag_steps = feature_cfg.get("load_lag_steps", DEFAULT_LAGS)
    if feature_cfg.get("use_load_lags", False):
        for lag in lag_steps:
            lag = int(lag)
            if n_hist < lag:
                raise ValueError(f"Not enough history for load_lag_{lag}.")
            val = float(load_hist.iloc[-lag])
            out[f"load_lag_{lag}"] = val

    windows = feature_cfg.get("rolling_load_windows", DEFAULT_ROLLING_WINDOWS)
    if feature_cfg.get("use_load_rolling", False):
        for window in windows:
            window = int(window)
            if n_hist < window:
                raise ValueError(f"Not enough history for rolling window {window}.")
            hist_window = load_hist.iloc[-window:]
            out[f"load_roll_mean_{window}"] = float(hist_window.mean())
            out[f"load_roll_std_{window}"] = float(hist_window.std())
            out[f"load_roll_min_{window}"] = float(hist_window.min())
            out[f"load_roll_max_{window}"] = float(hist_window.max())
        if n_hist < 13:
            raise ValueError("Need at least 13 rows of history for load_delta features.")
        out["load_delta_1"] = float(load_hist.iloc[-1] - load_hist.iloc[-2])
        out["load_delta_3"] = float(load_hist.iloc[-1] - load_hist.iloc[-4])
        out["load_delta_12"] = float(load_hist.iloc[-1] - load_hist.iloc[-13])
    elif feature_cfg.get("use_load_delta", False):
        if n_hist < 13:
            raise ValueError("Need at least 13 rows of history for load_delta_12.")
        out["load_delta_1"] = float(load_hist.iloc[-1] - load_hist.iloc[-2])
        out["load_delta_12"] = float(load_hist.iloc[-1] - load_hist.iloc[-13])

    return out


def build_feature_columns(train: pd.DataFrame, test: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Feature list = non-excluded columns on train. If test is given, keep only columns
    that exist on both train and test so inference rows never require missing columns.
    """
    excluded = set(DEFAULT_EXCLUDE_COLS)
    feats = [c for c in train.columns if c not in excluded]
    if test is not None:
        test_cols = set(test.columns)
        feats = [c for c in feats if c in test_cols]
    return feats

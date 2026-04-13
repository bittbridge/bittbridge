from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .data_io import TARGET_COLUMN, TIMESTAMP_COLUMN


DEFAULT_LAGS = [1, 2, 3, 6, 12]
DEFAULT_EXCLUDE_COLS = {
    TIMESTAMP_COLUMN,
    TARGET_COLUMN,
    "Native Load",
    "Asset Related Load",
    "Total Load With Estimated Solar",
    "Native Load With Estimated Solar",
    "load_change_1h",
}


def add_engineered_features(df: pd.DataFrame, feature_cfg: Dict) -> pd.DataFrame:
    out = df.copy()
    ts = out[TIMESTAMP_COLUMN]

    if feature_cfg.get("use_time_features", True):
        out["hour"] = ts.dt.hour
        out["day_of_week"] = ts.dt.dayofweek
        out["month"] = ts.dt.month

    if feature_cfg.get("use_cyclical_features", True):
        out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24.0)
        out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)

    lag_steps = feature_cfg.get("load_lag_steps", DEFAULT_LAGS)
    if feature_cfg.get("use_load_lags", True) and TARGET_COLUMN in out.columns:
        for lag in lag_steps:
            out[f"load_lag_{lag}"] = out[TARGET_COLUMN].shift(int(lag))

    if feature_cfg.get("use_load_delta", True) and TARGET_COLUMN in out.columns:
        # Use only historical information (no current-target leakage).
        out["load_delta_1"] = out[TARGET_COLUMN].shift(1) - out[TARGET_COLUMN].shift(2)
        out["load_delta_12"] = out[TARGET_COLUMN].shift(1) - out[TARGET_COLUMN].shift(13)

    _drop_features_disabled_by_config(out, feature_cfg)
    return out


def _drop_features_disabled_by_config(out: pd.DataFrame, feature_cfg: Dict) -> None:
    """
    TRAIN CSV may include raw columns (e.g. hour, month) while TEST may not.
    If a toggle is off, remove those columns so train/test schemas stay aligned
    with what we actually intend to use (same as notebook intent when flags are off).
    """
    if not feature_cfg.get("use_time_features", True):
        for col in ("hour", "month", "day_of_week"):
            if col in out.columns:
                out.drop(columns=[col], inplace=True)

    if not feature_cfg.get("use_cyclical_features", True):
        for col in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
            if col in out.columns:
                out.drop(columns=[col], inplace=True)

    if not feature_cfg.get("use_load_lags", True):
        to_drop = [c for c in out.columns if str(c).startswith("load_lag_")]
        if to_drop:
            out.drop(columns=to_drop, inplace=True)

    if not feature_cfg.get("use_load_delta", True):
        for col in ("load_delta_1", "load_delta_12"):
            if col in out.columns:
                out.drop(columns=[col], inplace=True)


def add_test_load_features_from_history(
    test_df: pd.DataFrame, train_df: pd.DataFrame, lag_steps: Iterable[int]
) -> pd.DataFrame:
    out = test_df.copy()
    history = train_df[TARGET_COLUMN]
    for lag in lag_steps:
        lag = int(lag)
        if len(history) < lag:
            raise ValueError(f"Not enough history for load_lag_{lag}.")
        out[f"load_lag_{lag}"] = float(history.iloc[-lag])
    if len(history) >= 12:
        out["load_delta_1"] = float(history.iloc[-1] - history.iloc[-2])
        out["load_delta_12"] = float(history.iloc[-1] - history.iloc[-12])
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


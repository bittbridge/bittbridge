from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .data_io import TARGET_COLUMN, TIMESTAMP_COLUMN


DEFAULT_LAGS = [1, 2, 3, 6, 12]


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
        out["load_delta_1"] = out[TARGET_COLUMN].diff(1)
        out["load_delta_12"] = out[TARGET_COLUMN].diff(12)

    return out


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


def build_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {TIMESTAMP_COLUMN, TARGET_COLUMN}
    return [c for c in df.columns if c not in excluded]


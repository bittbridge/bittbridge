from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import bittensor as bt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .artifacts import (
    feature_signature,
    prepare_artifact_dir,
    write_config_snapshot,
    write_manifest,
)
from .data_io import TARGET_COLUMN, TARGET_COLUMN_HORIZON, TIMESTAMP_COLUMN, load_train_test
from .features import (
    KNOWN_WEATHER_SUFFIXES,
    add_engineered_features,
    add_test_load_features_from_history,
    build_feature_columns,
    filter_weather_suffix_columns,
)
from .ml_config import ModelConfig
from .models_cart import predict_cart, save_cart, train_cart
from .models_cart import load_cart
from .models_linear import LinearBundle, load_linear, predict_linear, save_linear, train_linear
from .models_lstm import LSTM_SCALER_FILENAME, load_lstm, make_sequences, predict_lstm, save_lstm, train_lstm
from .models_rnn import RNN_SCALER_FILENAME, load_rnn, predict_rnn, save_rnn, train_rnn
from .split import temporal_train_val_split
from .supabase_io import (
    create_supabase_data_client,
    fetch_supabase_test_row,
    fetch_supabase_test_row_for_probe,
    fetch_supabase_train_all,
    fetch_supabase_train_tail,
    normalize_supabase_test_frame,
)
from .storage_train_io import load_train_from_storage_parts


@dataclass
class TrainingResult:
    model_type: str
    model_bundle: Any
    metrics: Dict[str, Dict[str, float]]
    features: List[str]
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    shapes: Dict[str, Tuple[int, ...]]
    y_train: np.ndarray = field(default_factory=lambda: np.array([]))
    train_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_val: np.ndarray = field(default_factory=lambda: np.array([]))
    val_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    durations_sec: Dict[str, float] = field(default_factory=dict)


def _as_numpy(frame: pd.DataFrame, features: List[str]) -> np.ndarray:
    return frame[features].astype(float).to_numpy()


def _forecast_horizon_steps(train: pd.DataFrame, horizon_min: int) -> int:
    """
    Map forecast_horizon_min to a row offset using median dt spacing.
    ISO-NE-style data is often 5-minute; irregular gaps make shift(-k) approximate.
    """
    if horizon_min <= 0:
        return 0
    if len(train) < 2:
        raise ValueError(
            "Need at least 2 training rows to infer timestep spacing for forecast horizon."
        )
    diffs = train[TIMESTAMP_COLUMN].diff().dt.total_seconds() / 60.0
    median_minutes = float(diffs.median(skipna=True))
    if not np.isfinite(median_minutes) or median_minutes <= 0:
        raise ValueError(
            "Could not infer a positive median `dt` spacing from training data; "
            "check `dt` column and sort order."
        )
    steps = int(round(float(horizon_min) / median_minutes))
    if steps < 1:
        raise ValueError(
            f"forecast_horizon_min={horizon_min} with median row spacing {median_minutes:.4f} minutes "
            "implies < 1 step; increase horizon or fix data cadence."
        )
    return steps


def _weather_feature_columns(frame: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in frame.columns:
        if not isinstance(col, str):
            continue
        if col in {TIMESTAMP_COLUMN, TARGET_COLUMN, TARGET_COLUMN_HORIZON}:
            continue
        if "-" not in col:
            continue
        suffix = col.rsplit("-", 1)[-1].strip().lower()
        if suffix in KNOWN_WEATHER_SUFFIXES:
            cols.append(col)
    return cols


def _apply_supabase_storage_feature_shift(
    train: pd.DataFrame,
    config: ModelConfig,
    show_progress: bool,
) -> tuple[pd.DataFrame, bool, int]:
    source = config.data.get("source", "csv")
    shift_min = int(config.data.get("train_feature_time_shift_min", 0))
    if source != "supabase_storage" or shift_min <= 0:
        return train, False, 0

    steps = _forecast_horizon_steps(train, shift_min)
    cols = _weather_feature_columns(train)
    if not cols:
        return train, False, 0

    shifted = train.copy()
    shifted[cols] = shifted[cols].shift(-steps)
    if show_progress:
        print(
            f"  [train]     Feature shift active: source=supabase_storage, "
            f"train_feature_time_shift_min={shift_min}, steps={steps}, shifted_cols={len(cols)}",
            flush=True,
        )
    return shifted, True, steps


def _shifted_timestamp_source_for_engineered_features(
    train: pd.DataFrame,
    config: ModelConfig,
    feature_shift_active: bool,
    feature_shift_steps: int,
) -> pd.Series | None:
    if not feature_shift_active or feature_shift_steps <= 0:
        return None
    if not (
        config.features.get("use_time_features", False)
        or config.features.get("use_cyclical_features", False)
    ):
        return None
    return train[TIMESTAMP_COLUMN].shift(-feature_shift_steps)


def _fmt_sec(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m2 = divmod(m, 60)
    return f"{h}h {m2}m {s}s"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


ACTUAL_VS_PREDICTED_CSV = "actual_vs_predicted.csv"


def _subsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def build_actual_vs_predicted_dataframe(result: TrainingResult) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for split, y_true, y_pred in (
        ("train", result.y_train, result.train_pred),
        ("validation", result.y_val, result.val_pred),
    ):
        if y_true.size == 0 or y_pred.size == 0:
            continue
        n = min(len(y_true), len(y_pred))
        for i in range(n):
            a = float(y_true[i])
            p = float(y_pred[i])
            rows.append({"split": split, "actual": a, "predicted": p, "error": p - a})
    return pd.DataFrame(rows)


def print_actual_vs_predicted_plotext(result: TrainingResult, model_label: str) -> None:
    try:
        import plotext as plt
    except ImportError:
        print("  Warning: plotext is not installed; skipping actual vs predicted plots.")
        return
    max_points = 50000
    try:
        for y_true, y_pred, title_suffix in (
            (result.y_train, result.train_pred, "training"),
            (result.y_val, result.val_pred, "validation"),
        ):
            if y_true.size == 0 or y_pred.size == 0:
                continue
            n = min(len(y_true), len(y_pred))
            idx = _subsample_indices(n, max_points)
            xa = y_true[idx].astype(float)
            yb = y_pred[idx].astype(float)
            plt.clear_data()
            lo = float(np.nanmin(np.concatenate([xa, yb])))
            hi = float(np.nanmax(np.concatenate([xa, yb])))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if hi - lo < 1e-12:
                pad = 1.0 if lo == 0.0 else abs(lo) * 0.01
                lo -= pad
                hi += pad
            # Identity line y = x (perfect prediction); draw first so points sit on top.
            plt.plot([lo, hi], [lo, hi], label="y = x (perfect)")
            plt.scatter(xa, yb, label="model")
            plt.xlabel("actual")
            plt.ylabel("predicted")
            plt.title(f"{model_label} — Actual vs predicted ({title_suffix})")
            plt.show()
    except Exception as exc:
        print(f"  Warning: could not render plotext charts ({exc}).")


def _load_supabase_train_test(config: ModelConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = config.data
    schema = data_cfg["supabase_schema"]
    train_table = data_cfg["supabase_train_table"]
    test_table = data_cfg["supabase_test_table"]
    try:
        client = create_supabase_data_client(data_cfg["supabase_url"], data_cfg["supabase_key"])
        train = fetch_supabase_train_all(
            client,
            schema=schema,
            table=train_table,
            page_size=int(data_cfg.get("supabase_page_size", 1000)),
        )
    except Exception as exc:
        raise ValueError(
            "Supabase training load failed "
            f"(schema={schema}, train_table={train_table}, test_table={test_table}): {exc}"
        ) from exc
    test_csv = data_cfg.get("test_csv")
    if test_csv:
        test = pd.read_csv(test_csv)
        test[TIMESTAMP_COLUMN] = pd.to_datetime(test[TIMESTAMP_COLUMN], errors="raise")
        test = test.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
        return train, test

    if len(train) < 2:
        raise ValueError("Supabase train data needs at least 2 rows to derive a fallback test row.")
    test = train.tail(1).drop(columns=[TARGET_COLUMN], errors="ignore").copy()
    return train, test


def _load_train_test_by_source(config: ModelConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    source = config.data.get("source", "csv")
    if source == "supabase":
        return _load_supabase_train_test(config)
    if source == "supabase_storage":
        data_cfg = config.data
        force_refresh = bool(data_cfg.get("storage_force_refresh", False))
        train = load_train_from_storage_parts(config=config, force_refresh=force_refresh)

        test_csv = data_cfg.get("test_csv")
        if test_csv:
            test = pd.read_csv(test_csv)
            test[TIMESTAMP_COLUMN] = pd.to_datetime(test[TIMESTAMP_COLUMN], errors="raise")
            test = test.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
            return train, test

        if len(train) < 2:
            raise ValueError("Supabase Storage train data needs at least 2 rows for a fallback test row.")
        test = train.tail(1).drop(columns=[TARGET_COLUMN], errors="ignore").copy()
        return train, test
    return load_train_test(config.data["train_csv"], config.data["test_csv"])


def prepare_training_data(
    config: ModelConfig,
    show_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    source = config.data.get("source", "csv")
    if source in {"supabase", "supabase_storage"} and show_progress:
        print(
            f"  [train]     Data source: {source.upper()} "
            f"(schema={config.data.get('supabase_schema')}, "
            f"train_table={config.data.get('supabase_train_table')}, "
            f"test_table={config.data.get('supabase_test_table')})",
            flush=True,
        )
    train, test = _load_train_test_by_source(config)
    train = train.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    test = test.sort_values(TIMESTAMP_COLUMN, kind="mergesort").reset_index(drop=True)
    if source in {"supabase", "supabase_storage"} and show_progress:
        print(
            f"  [train]     Supabase data pulled: train_shape={train.shape}, test_shape={test.shape}",
            flush=True,
        )
    train, feature_shift_active, feature_shift_steps = _apply_supabase_storage_feature_shift(
        train, config, show_progress=show_progress
    )
    shifted_time_source = _shifted_timestamp_source_for_engineered_features(
        train,
        config,
        feature_shift_active=feature_shift_active,
        feature_shift_steps=feature_shift_steps,
    )

    suffix_whitelist = config.features.get("include_weather_suffix_groups")
    train = filter_weather_suffix_columns(train, suffix_whitelist)
    test = filter_weather_suffix_columns(test, suffix_whitelist)
    train = add_engineered_features(train, config.features, timestamp_source=shifted_time_source)
    test = add_engineered_features(test, config.features)

    feats_cfg = config.features
    if (
        feats_cfg.get("use_load_lags", False)
        or feats_cfg.get("use_load_rolling", False)
        or feats_cfg.get("use_load_delta", False)
    ):
        test = add_test_load_features_from_history(test, train, feats_cfg)

    train_only = build_feature_columns(train, test=None)
    features = build_feature_columns(train, test=test)
    dropped = sorted(set(train_only) - set(features))
    if dropped:
        warnings.warn(
            "Train CSV has columns that TEST does not; those columns are excluded so "
            "train and inference use the same feature set. Omitted: "
            + ", ".join(dropped[:25])
            + (" ..." if len(dropped) > 25 else ""),
            UserWarning,
            stacklevel=2,
        )
    if not features:
        raise ValueError(
            "No shared feature columns between train and test after exclusions. "
            "With include_weather_suffix_groups empty, raw *-tmpf / *-dwpf / *-relh / *-sped / *-drct "
            "columns are removed; enable engineered features (e.g. use_time_features) and/or "
            "list suffixes under include_weather_suffix_groups, and ensure the test CSV can "
            "produce the same columns. Check model_params.yaml toggles and CSV schemas."
        )
    horizon_min = int(config.data.get("forecast_horizon_min", 0))
    disable_horizon_label_shift = bool(
        config.data.get("train_disable_horizon_label_shift_when_feature_shifted", False)
    )
    steps = 0
    if feature_shift_active and disable_horizon_label_shift:
        if show_progress:
            print(
                f"  [train]     Horizon label shift disabled: "
                f"feature_shift_steps={feature_shift_steps}, using `{TARGET_COLUMN}` as target.",
                flush=True,
            )
    else:
        steps = _forecast_horizon_steps(train, horizon_min)
        if steps > 0:
            train[TARGET_COLUMN_HORIZON] = train[TARGET_COLUMN].shift(-steps)
            if show_progress:
                median_m = float(
                    train[TIMESTAMP_COLUMN].diff().dt.total_seconds().div(60.0).median(skipna=True)
                )
                print(
                    f"  [train]     Horizon target: forecast_horizon_min={horizon_min}, "
                    f"median_dt_min={median_m:.4f}, steps={steps} → `{TARGET_COLUMN_HORIZON}`",
                    flush=True,
                )

    required_train_cols = features + [TARGET_COLUMN]
    if steps > 0:
        required_train_cols = features + [TARGET_COLUMN, TARGET_COLUMN_HORIZON]
    train_model = train.dropna(subset=required_train_cols).reset_index(drop=True)
    if train_model.empty:
        raise ValueError("Training frame is empty after feature engineering and dropna.")
    if source == "supabase" and show_progress:
        print(
            f"  [train]     Features applied: train_shape={train_model.shape}, "
            f"test_shape={test.shape}, n_features={len(features)}",
            flush=True,
        )
    return train_model, test, features


def train_model(model_type: str, config: ModelConfig) -> TrainingResult:
    show_progress = bool(config.training.get("show_training_progress", True))
    lstm_fit_verbose = int(config.models.get("lstm", {}).get("fit_verbose", 1))
    rnn_fit_verbose = int(config.models.get("rnn", {}).get("fit_verbose", 1))

    t0 = time.perf_counter()
    if show_progress:
        print("  [train] (1/4) Loading CSVs and building features…", flush=True)
    train_model, test, features = prepare_training_data(config, show_progress=show_progress)
    t1 = time.perf_counter()
    if show_progress:
        print(
            f"  [train]     ✓ {_fmt_sec(t1 - t0)} — {len(train_model):,} rows, {len(features)} features",
            flush=True,
        )
        print("  [train] (2/4) Building arrays and temporal train/val split…", flush=True)
    train_split, val_split = temporal_train_val_split(
        train_model, validation_split=config.training["validation_split"]
    )
    y_col = TARGET_COLUMN_HORIZON if TARGET_COLUMN_HORIZON in train_model.columns else TARGET_COLUMN
    if show_progress:
        print(f"  [train]     Target column selected: `{y_col}`", flush=True)
    X_train = _as_numpy(train_split, features)
    y_train = train_split[y_col].to_numpy()
    X_val = _as_numpy(val_split, features)
    y_val = val_split[y_col].to_numpy()
    X_test = test[features].astype(float).to_numpy()
    t2 = time.perf_counter()
    if show_progress:
        print(
            f"  [train]     ✓ {_fmt_sec(t2 - t1)} — train {X_train.shape}, val {X_val.shape}",
            flush=True,
        )

    rs = int(config.training.get("random_state", 42))
    if show_progress:
        print(f"  [train] (3/4) Training {model_type} (fit + train/val predictions)…", flush=True)

    t_fit_start = time.perf_counter()
    if model_type == "linear":
        bundle: LinearBundle = train_linear(X_train, y_train, features, config.models.get("linear", {}))
        train_pred = predict_linear(bundle, X_train)
        val_pred = predict_linear(bundle, X_val)
    elif model_type == "cart":
        cart_cfg = dict(config.models.get("cart", {}))
        cart_cfg.setdefault("random_state", rs)
        bundle = train_cart(X_train, y_train, features, cart_cfg)
        train_pred = predict_cart(bundle, X_train)
        val_pred = predict_cart(bundle, X_val)
    elif model_type == "lstm":
        bundle = train_lstm(
            X_train,
            y_train,
            features,
            config.models.get("lstm", {}),
            random_state=rs,
            fit_verbose=lstm_fit_verbose,
            X_val=X_val,
            y_val=y_val,
        )
        n_steps = bundle.n_steps
        X_train_seq, y_train_seq = make_sequences(X_train, y_train, n_steps=n_steps)
        X_val_seq, y_val_seq = make_sequences(X_val, y_val, n_steps=n_steps)
        if len(X_train_seq) == 0:
            raise ValueError("Training split too short for LSTM sequence evaluation.")
        if len(X_val_seq) == 0:
            raise ValueError("Validation split too short for LSTM sequence evaluation.")
        train_pred = predict_lstm(bundle, X_train_seq)
        val_pred = predict_lstm(bundle, X_val_seq)
        y_train = y_train_seq
        y_val = y_val_seq
    elif model_type == "rnn":
        bundle = train_rnn(
            X_train,
            y_train,
            features,
            config.models.get("rnn", {}),
            random_state=rs,
            fit_verbose=rnn_fit_verbose,
            X_val=X_val,
            y_val=y_val,
        )
        n_steps = bundle.n_steps
        X_train_seq, y_train_seq = make_sequences(X_train, y_train, n_steps=n_steps)
        X_val_seq, y_val_seq = make_sequences(X_val, y_val, n_steps=n_steps)
        if len(X_train_seq) == 0:
            raise ValueError("Training split too short for RNN sequence evaluation.")
        if len(X_val_seq) == 0:
            raise ValueError("Validation split too short for RNN sequence evaluation.")
        train_pred = predict_rnn(bundle, X_train_seq)
        val_pred = predict_rnn(bundle, X_val_seq)
        y_train = y_train_seq
        y_val = y_val_seq
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    t_fit_end = time.perf_counter()
    if show_progress:
        print(
            f"  [train]     ✓ step (3/4) done in {_fmt_sec(t_fit_end - t_fit_start)}",
            flush=True,
        )
        print("  [train] (4/4) Aggregating train/validation metrics…", flush=True)

    metrics = {
        "train": _metrics(y_train, train_pred),
        "validation": _metrics(y_val, val_pred),
    }
    t3 = time.perf_counter()
    durations_sec = {
        "prepare_data_sec": t1 - t0,
        "split_arrays_sec": t2 - t1,
        "fit_sec": t_fit_end - t_fit_start,
        "metrics_sec": t3 - t_fit_end,
        "split_and_fit_sec": t3 - t1,
        "total_sec": t3 - t0,
    }
    if show_progress:
        print(f"  [train]     ✓ done — total {_fmt_sec(t3 - t0)}", flush=True)

    return TrainingResult(
        model_type=model_type,
        model_bundle=bundle,
        metrics=metrics,
        features=features,
        train_frame=train_model,
        test_frame=test,
        shapes={
            "X_train": tuple(X_train.shape),
            "X_val": tuple(X_val.shape),
            "X_test": tuple(X_test.shape),
            "y_train": tuple(y_train.shape),
            "y_val": tuple(y_val.shape),
        },
        y_train=y_train,
        train_pred=train_pred,
        y_val=y_val,
        val_pred=val_pred,
        durations_sec=durations_sec,
    )


def build_sequence_inference_matrix(result: TrainingResult) -> np.ndarray:
    """
    LSTM / RNN need shape (n_steps, n_features). TEST often has one row; prepend the last
    (n_steps - 1) rows from train so the window matches training-time sequences.
    """
    bundle = result.model_bundle
    n_steps = int(bundle.n_steps)
    feats = result.features
    train_x = result.train_frame[feats].astype(float).to_numpy()
    test_x = result.test_frame[feats].astype(float).to_numpy()

    if test_x.shape[0] >= n_steps:
        return test_x[-n_steps:]

    need_prior = n_steps - test_x.shape[0]
    if train_x.shape[0] < need_prior:
        raise ValueError(
            f"Sequence model inference needs {need_prior} prior timestep(s) from training history; "
            f"train_frame has only {train_x.shape[0]} row(s)."
        )
    prior = train_x[-need_prior:]
    return np.vstack([prior, test_x])


def build_lstm_inference_matrix(result: TrainingResult) -> np.ndarray:
    """Backward-compatible alias for :func:`build_sequence_inference_matrix`."""
    return build_sequence_inference_matrix(result)


def predict_single_test_row(result: TrainingResult) -> float:
    pred, _ = predict_single_test_row_with_context(result)
    return pred


def predict_single_test_row_with_context(result: TrainingResult) -> tuple[float, Dict[str, Any]]:
    X_test = result.test_frame[result.features].astype(float).to_numpy()
    if result.model_type == "linear":
        pred = predict_linear(result.model_bundle, X_test)[0]
    elif result.model_type == "cart":
        pred = predict_cart(result.model_bundle, X_test)[0]
    elif result.model_type == "lstm":
        seq_2d = build_sequence_inference_matrix(result)
        pred = predict_lstm(result.model_bundle, seq_2d)[0]
    elif result.model_type == "rnn":
        seq_2d = build_sequence_inference_matrix(result)
        pred = predict_rnn(result.model_bundle, seq_2d)[0]
    else:
        raise ValueError(f"Unsupported model type: {result.model_type}")
    input_row = (
        result.test_frame[result.features].iloc[0].to_dict()
        if not result.test_frame.empty
        else {}
    )
    context: Dict[str, Any] = {
        "source": "local_test_frame",
        "model_type": result.model_type,
        "model_input_row": input_row,
    }
    return float(pred), context


def _required_history_rows_for_live(result: TrainingResult, config: ModelConfig) -> int:
    feats_cfg = config.features
    needed = 32
    n_steps_seq: int | None = None
    if result.model_type in {"lstm", "rnn"}:
        n_steps_seq = int(getattr(result.model_bundle, "n_steps", 12))
        needed = max(needed, n_steps_seq)
    if feats_cfg.get("use_load_lags", False):
        lag_steps = [int(v) for v in feats_cfg.get("load_lag_steps", [])]
        max_lag = max(lag_steps, default=1)
        needed = max(needed, max_lag + 2)
        # shift(max_lag) leaves the first max_lag rows NaN in load_lag_*; dropna() then
        # needs enough tail rows for (n_steps - 1) prior timesteps for LSTM/RNN live inference.
        if n_steps_seq is not None:
            live_seq_buffer = 8
            needed = max(needed, max_lag + (n_steps_seq - 1) + live_seq_buffer)
    if feats_cfg.get("use_load_rolling", False):
        rolling = [int(v) for v in feats_cfg.get("rolling_load_windows", [])]
        needed = max(needed, max(rolling, default=1) + 2, 16)
    if feats_cfg.get("use_load_delta", False) or feats_cfg.get("use_load_rolling", False):
        needed = max(needed, 16)
    return int(needed)


def required_history_rows_for_probe(config: ModelConfig, sequence_n_steps: int | None) -> int:
    """
    Row count for Supabase train-tail fetch when probing live features for custom models.
    Mirrors _required_history_rows_for_live without requiring a full TrainingResult.
    """
    feats_cfg = config.features
    needed = 32
    n_steps_seq: int | None = None
    if sequence_n_steps is not None and int(sequence_n_steps) > 1:
        n_steps_seq = int(sequence_n_steps)
        needed = max(needed, n_steps_seq)
    if feats_cfg.get("use_load_lags", False):
        lag_steps = [int(v) for v in feats_cfg.get("load_lag_steps", [])]
        max_lag = max(lag_steps, default=1)
        needed = max(needed, max_lag + 2)
        # shift(max_lag) leaves the first max_lag rows NaN in load_lag_*; dropna() then
        # needs enough tail rows for (n_steps - 1) prior timesteps for LSTM/RNN live inference.
        if n_steps_seq is not None:
            live_seq_buffer = 8
            needed = max(needed, max_lag + (n_steps_seq - 1) + live_seq_buffer)
    if feats_cfg.get("use_load_rolling", False):
        rolling = [int(v) for v in feats_cfg.get("rolling_load_windows", [])]
        needed = max(needed, max(rolling, default=1) + 2, 16)
    if feats_cfg.get("use_load_delta", False) or feats_cfg.get("use_load_rolling", False):
        needed = max(needed, 16)
    return int(needed)


def _build_live_sequence_matrix(
    history_features: pd.DataFrame,
    test_row: pd.DataFrame,
    features: List[str],
    n_steps: int,
) -> np.ndarray:
    prior = history_features[features].dropna().astype(float).to_numpy()
    need_prior = n_steps - 1
    if prior.shape[0] < need_prior:
        raise ValueError(
            f"Live sequence inference requires {need_prior} prior feature rows, got {prior.shape[0]}."
        )
    x_test = test_row[features].astype(float).to_numpy()
    return np.vstack([prior[-need_prior:], x_test])


def live_probe_feature_matrix_for_custom(
    config: ModelConfig,
    timestamp_str: str,
    feature_list: List[str],
    sequence_n_steps: int | None,
    *,
    use_resilient_forecast_fetch: bool = False,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Build the same engineered feature matrix used at inference time for a custom plugin model.
    Returns X with shape (1, n_features) for dense / sklearn models, or (1, n_steps, n_features) for sequence Keras.
    """
    source = config.data.get("source", "csv")
    if source not in {"supabase", "supabase_storage"}:
        train_model, test, _ = prepare_training_data(config, show_progress=False)
        if test.empty:
            raise ValueError("CSV probe: test frame is empty.")
        missing = [c for c in feature_list if c not in test.columns]
        if missing:
            raise ValueError(
                "CSV probe: test frame missing feature columns: " + ", ".join(missing[:30])
            )
        ts = pd.to_datetime(timestamp_str, errors="coerce")
        if pd.isna(ts):
            row_df = test.iloc[[-1]]
        else:
            test_sorted = test.sort_values(TIMESTAMP_COLUMN)
            dt_series = test_sorted[TIMESTAMP_COLUMN]
            pos = int((dt_series - ts).abs().values.argmin())
            row_df = test_sorted.iloc[[pos]]
        missing_hist = [c for c in feature_list if c not in train_model.columns]
        if missing_hist:
            raise ValueError(
                "CSV probe: train frame missing feature columns: " + ", ".join(missing_hist[:30])
            )
        n_seq = int(sequence_n_steps or 0)
        if n_seq > 1:
            seq = _build_live_sequence_matrix(train_model, row_df, feature_list, n_seq)
            X = np.asarray(seq[np.newaxis, :, :], dtype=np.float32)
        else:
            X = row_df[feature_list].astype(float).to_numpy(dtype=np.float32)
            if X.ndim == 1:
                X = X[np.newaxis, :]
        ctx: Dict[str, Any] = {
            "source": source,
            "requested_timestamp": timestamp_str,
            "model_input_row": row_df[feature_list].iloc[0].to_dict(),
        }
        return X, ctx

    data_cfg = config.data
    schema = data_cfg["supabase_schema"]
    train_table = data_cfg["supabase_train_table"]
    test_table = data_cfg["supabase_test_table"]
    horizon = int(data_cfg.get("forecast_horizon_min", 5))
    page_size = int(data_cfg.get("supabase_page_size", 1000))
    needed_rows = required_history_rows_for_probe(config, sequence_n_steps)

    try:
        client = create_supabase_data_client(data_cfg["supabase_url"], data_cfg["supabase_key"])
        history = fetch_supabase_train_tail(client, schema=schema, table=train_table, n_rows=needed_rows)
        if use_resilient_forecast_fetch:
            forecast_row = fetch_supabase_test_row_for_probe(
                client,
                schema=schema,
                table=test_table,
                dt_target=timestamp_str,
                horizon_min=horizon,
            )
        else:
            forecast_row = fetch_supabase_test_row(
                client,
                schema=schema,
                table=test_table,
                dt_target=timestamp_str,
                horizon_min=horizon,
                nearest_fallback_minutes=5,
            )
    except Exception as exc:
        raise ValueError(
            "Supabase live probe failed while fetching data "
            f"(schema={schema}, train_table={train_table}, test_table={test_table}, "
            f"timestamp={timestamp_str}, horizon_min={horizon}, page_size={page_size}): {exc}"
        ) from exc

    if forecast_row is None:
        raise ValueError(
            "Supabase live probe found no forecast row "
            f"(schema={schema}, table={test_table}, timestamp={timestamp_str}, horizon_min={horizon})."
        )

    forecast_frame = normalize_supabase_test_frame(pd.DataFrame([forecast_row]))
    suffix_whitelist = config.features.get("include_weather_suffix_groups")
    history_filtered = filter_weather_suffix_columns(history, suffix_whitelist)
    forecast_filtered = filter_weather_suffix_columns(forecast_frame, suffix_whitelist)

    history_features = add_engineered_features(history_filtered, config.features)
    test_features = add_engineered_features(forecast_filtered, config.features)
    feats_cfg = config.features
    if (
        feats_cfg.get("use_load_lags", False)
        or feats_cfg.get("use_load_rolling", False)
        or feats_cfg.get("use_load_delta", False)
    ):
        test_features = add_test_load_features_from_history(test_features, history_filtered, feats_cfg)

    missing = [c for c in feature_list if c not in test_features.columns]
    if missing:
        raise ValueError(
            "Live probe: forecast row missing feature columns: " + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )
    missing_history = [c for c in feature_list if c not in history_features.columns]
    if missing_history:
        raise ValueError(
            "Live probe: history rows missing feature columns: " + ", ".join(missing_history[:20])
            + (" ..." if len(missing_history) > 20 else "")
        )

    n_seq = int(sequence_n_steps or 0)
    if n_seq > 1:
        seq = _build_live_sequence_matrix(
            history_features=history_features,
            test_row=test_features,
            features=feature_list,
            n_steps=n_seq,
        )
        X = np.asarray(seq[np.newaxis, :, :], dtype=np.float32)
    else:
        X = test_features[feature_list].astype(float).to_numpy(dtype=np.float32)
        if X.ndim == 1:
            X = X[np.newaxis, :]

    model_input_row = (
        test_features[feature_list].iloc[0].to_dict()
        if not test_features.empty
        else {}
    )
    latest_train_row = history.iloc[-1].to_dict() if not history.empty else {}
    ctx = {
        "source": source,
        "requested_timestamp": timestamp_str,
        "forecast_row_raw": forecast_row,
        "train_row_latest_raw": latest_train_row,
        "model_input_row": model_input_row,
    }
    return X, ctx


def predict_for_timestamp(result: TrainingResult, config: ModelConfig, timestamp_str: str) -> float:
    pred, _ = predict_for_timestamp_with_context(result, config, timestamp_str)
    return pred


def predict_for_timestamp_with_context(
    result: TrainingResult, config: ModelConfig, timestamp_str: str
) -> tuple[float, Dict[str, Any]]:
    source = config.data.get("source", "csv")
    if source not in {"supabase", "supabase_storage"}:
        return predict_single_test_row_with_context(result)

    data_cfg = config.data
    schema = data_cfg["supabase_schema"]
    train_table = data_cfg["supabase_train_table"]
    test_table = data_cfg["supabase_test_table"]
    horizon = int(data_cfg.get("forecast_horizon_min", 5))
    page_size = int(data_cfg.get("supabase_page_size", 1000))

    try:
        client = create_supabase_data_client(data_cfg["supabase_url"], data_cfg["supabase_key"])
        needed_rows = _required_history_rows_for_live(result, config)
        history = fetch_supabase_train_tail(client, schema=schema, table=train_table, n_rows=needed_rows)
        forecast_row = fetch_supabase_test_row(
            client,
            schema=schema,
            table=test_table,
            dt_target=timestamp_str,
            horizon_min=horizon,
            nearest_fallback_minutes=5,
        )
    except Exception as exc:
        raise ValueError(
            "Supabase live inference failed while fetching data "
            f"(schema={schema}, train_table={train_table}, test_table={test_table}, "
            f"timestamp={timestamp_str}, horizon_min={horizon}, page_size={page_size}): {exc}"
        ) from exc

    if forecast_row is None:
        raise ValueError(
            "Supabase live inference found no forecast row "
            f"(schema={schema}, table={test_table}, timestamp={timestamp_str}, horizon_min={horizon})."
        )
    bt.logging.info(
        "[LIVE_ROW] "
        f"requested_timestamp={timestamp_str} "
        f"requested_horizon_min={horizon} "
        f"selected_dt={forecast_row.get(TIMESTAMP_COLUMN)} "
        f"selected_horizon_min={forecast_row.get('horizon_min')}"
    )

    forecast_frame = normalize_supabase_test_frame(pd.DataFrame([forecast_row]))
    suffix_whitelist = config.features.get("include_weather_suffix_groups")
    history_filtered = filter_weather_suffix_columns(history, suffix_whitelist)
    forecast_filtered = filter_weather_suffix_columns(forecast_frame, suffix_whitelist)

    history_features = add_engineered_features(history_filtered, config.features)
    test_features = add_engineered_features(forecast_filtered, config.features)
    feats_cfg = config.features
    if (
        feats_cfg.get("use_load_lags", False)
        or feats_cfg.get("use_load_rolling", False)
        or feats_cfg.get("use_load_delta", False)
    ):
        test_features = add_test_load_features_from_history(test_features, history_filtered, feats_cfg)

    missing = [c for c in result.features if c not in test_features.columns]
    if missing:
        raise ValueError(
            "Live forecast row is missing trained feature columns: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )
    missing_history = [c for c in result.features if c not in history_features.columns]
    if missing_history:
        raise ValueError(
            "Live history rows are missing trained feature columns: "
            + ", ".join(missing_history[:20])
            + (" ..." if len(missing_history) > 20 else "")
        )

    if result.model_type == "linear":
        pred = predict_linear(result.model_bundle, test_features[result.features].astype(float).to_numpy())[0]
    elif result.model_type == "cart":
        pred = predict_cart(result.model_bundle, test_features[result.features].astype(float).to_numpy())[0]
    elif result.model_type == "lstm":
        n_steps = int(getattr(result.model_bundle, "n_steps", 12))
        seq = _build_live_sequence_matrix(
            history_features=history_features,
            test_row=test_features,
            features=result.features,
            n_steps=n_steps,
        )
        pred = predict_lstm(result.model_bundle, seq)[0]
    elif result.model_type == "rnn":
        n_steps = int(getattr(result.model_bundle, "n_steps", 12))
        seq = _build_live_sequence_matrix(
            history_features=history_features,
            test_row=test_features,
            features=result.features,
            n_steps=n_steps,
        )
        pred = predict_rnn(result.model_bundle, seq)[0]
    else:
        raise ValueError(f"Unsupported model type: {result.model_type}")

    model_input_row = (
        test_features[result.features].iloc[0].to_dict()
        if not test_features.empty
        else {}
    )
    latest_train_row = history.iloc[-1].to_dict() if not history.empty else {}
    context: Dict[str, Any] = {
        "source": source,
        "model_type": result.model_type,
        "requested_timestamp": timestamp_str,
        "forecast_row_raw": forecast_row,
        "train_row_latest_raw": latest_train_row,
        "model_input_row": model_input_row,
    }
    return float(pred), context


def persist_training_result(
    result: TrainingResult,
    config: ModelConfig,
    run_id: str | None = None,
    dump_full_training_dataset: bool = False,
) -> Dict[str, str]:
    artifact_root = config.persistence["artifact_dir"]
    out_dir = prepare_artifact_dir(artifact_root, result.model_type, run_id=run_id)

    model_rel = ""
    if result.model_type == "linear":
        model_rel = "model_linear.joblib"
        save_linear(result.model_bundle, str(out_dir / model_rel))
    elif result.model_type == "cart":
        model_rel = "model_cart.joblib"
        save_cart(result.model_bundle, str(out_dir / model_rel))
    elif result.model_type == "lstm":
        model_rel = "model_lstm.keras"
        save_lstm(result.model_bundle, str(out_dir / model_rel))
    elif result.model_type == "rnn":
        model_rel = "model_rnn.keras"
        save_rnn(result.model_bundle, str(out_dir / model_rel))

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(pd.Series(result.metrics).to_json(indent=2), encoding="utf-8")

    avp_df = build_actual_vs_predicted_dataframe(result)
    avp_path = out_dir / ACTUAL_VS_PREDICTED_CSV
    avp_df.to_csv(avp_path, index=False)

    full_train_dataset_rel: str | None = None
    full_train_dataset_columns: List[str] | None = None
    if dump_full_training_dataset:
        full_train_dataset_rel = "training_dataset_full.csv"
        full_train_dataset_columns = [str(c) for c in result.train_frame.columns.tolist()]
        result.train_frame.to_csv(out_dir / full_train_dataset_rel, index=False)

    cfg_snapshot = {
        "data": config.data,
        "features": config.features,
        "training": config.training,
        "models": config.models,
        "persistence": {k: v for k, v in config.persistence.items() if k != "config_file"},
    }
    write_config_snapshot(out_dir, cfg_snapshot)

    manifest = {
        "model_type": result.model_type,
        "model_path": model_rel,
        "metrics_path": metrics_path.name,
        "actual_vs_predicted_path": ACTUAL_VS_PREDICTED_CSV,
        "features": result.features,
        "features_count": len(result.features),
        "feature_signature": feature_signature(result.features),
        "train_rows": int(len(result.train_frame)),
        "metrics": result.metrics,
        "durations_sec": result.durations_sec,
        "full_training_dataset_dumped": bool(dump_full_training_dataset),
        "full_training_dataset_path": full_train_dataset_rel,
        "full_training_dataset_rows": int(len(result.train_frame)) if dump_full_training_dataset else None,
        "full_training_dataset_columns_count": (
            int(len(full_train_dataset_columns)) if full_train_dataset_columns is not None else None
        ),
        "full_training_dataset_columns": full_train_dataset_columns,
        "lstm_n_steps": getattr(result.model_bundle, "n_steps", None)
        if result.model_type == "lstm"
        else None,
        "rnn_n_steps": getattr(result.model_bundle, "n_steps", None)
        if result.model_type == "rnn"
        else None,
    }
    if result.model_type == "lstm":
        lstm_bundle = result.model_bundle
        lstm_std = lstm_bundle.scaler is not None
        manifest["lstm_standardize_inputs"] = lstm_std
        manifest["lstm_scaler_path"] = LSTM_SCALER_FILENAME if lstm_std else None
    if result.model_type == "rnn":
        rnn_bundle = result.model_bundle
        rnn_std = rnn_bundle.scaler is not None
        manifest["rnn_standardize_inputs"] = rnn_std
        manifest["rnn_scaler_path"] = RNN_SCALER_FILENAME if rnn_std else None
    manifest_path = write_manifest(out_dir, manifest)

    return {
        "artifact_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "model_path": str(out_dir / model_rel),
    }


def load_training_bundle_from_manifest(manifest_path: str):
    manifest_path_obj = Path(manifest_path)
    manifest = json.loads(manifest_path_obj.read_text(encoding="utf-8"))
    artifact_dir = manifest_path_obj.parent
    model_type = manifest["model_type"]
    model_path = str(artifact_dir / manifest["model_path"])
    features = manifest.get("features", [])

    if model_type == "linear":
        return load_linear(model_path)
    if model_type == "cart":
        return load_cart(model_path)
    if model_type == "lstm":
        n_steps_lstm = int(manifest.get("lstm_n_steps", 12))
        scaler_path: str | None = None
        if manifest.get("lstm_standardize_inputs"):
            rel = manifest.get("lstm_scaler_path") or LSTM_SCALER_FILENAME
            scaler_path = str(artifact_dir / rel)
        return load_lstm(model_path, features=features, n_steps=n_steps_lstm, scaler_path=scaler_path)
    if model_type == "rnn":
        n_steps_rnn = int(manifest.get("rnn_n_steps", 12))
        scaler_path_rnn: str | None = None
        if manifest.get("rnn_standardize_inputs"):
            rel = manifest.get("rnn_scaler_path") or RNN_SCALER_FILENAME
            scaler_path_rnn = str(artifact_dir / rel)
        return load_rnn(model_path, features=features, n_steps=n_steps_rnn, scaler_path=scaler_path_rnn)
    raise ValueError(f"Unsupported model type in manifest: {model_type}")


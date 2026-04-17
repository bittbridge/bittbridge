from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .features import KNOWN_WEATHER_SUFFIXES


@dataclass(frozen=True)
class ModelConfig:
    data: Dict[str, Any]
    features: Dict[str, Any]
    training: Dict[str, Any]
    models: Dict[str, Any]
    persistence: Dict[str, Any]


FEATURE_BOOL_KEYS = (
    "use_time_features",
    "use_cyclical_features",
    "use_station_agg_features",
    "use_temp_dew_gap",
    "use_load_lags",
    "use_load_rolling",
    "use_load_delta",
)


def _require_path(path_value: str, key: str) -> str:
    path = Path(path_value)
    if not path.exists():
        raise ValueError(f"Config `{key}` points to missing path: {path}")
    return str(path)


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_include_weather_suffix_groups(value: Any) -> List[str]:
    """Empty list = strip all raw *-tmpf / *-dwpf / … columns. Non-empty = whitelist those suffixes only."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("`features.include_weather_suffix_groups` must be a list of strings (or empty).")
    out: List[str] = []
    for item in value:
        s = str(item).strip().lower()
        if not s:
            continue
        if s not in KNOWN_WEATHER_SUFFIXES:
            raise ValueError(
                f"`features.include_weather_suffix_groups` unknown suffix {s!r}. "
                f"Allowed: {sorted(KNOWN_WEATHER_SUFFIXES)}."
            )
        if s not in out:
            out.append(s)
    return out


def _as_int_list(value: Any, key: str, default: List[int]) -> List[int]:
    if value is None:
        return list(default)
    if not isinstance(value, list) or not value:
        raise ValueError(f"`{key}` must be a non-empty list of integers.")
    out: List[int] = []
    for item in value:
        out.append(int(item))
    return out


def _as_str_list(value: Any, key: str) -> List[str]:
    if value is None:
        raise ValueError(f"`{key}` is required.")
    if isinstance(value, str):
        # Allow comma-separated values as a convenience.
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"`{key}` must contain at least one filename.")
        return parts
    if not isinstance(value, list) or not value:
        raise ValueError(f"`{key}` must be a non-empty list of strings.")
    out: List[str] = []
    for item in value:
        s = str(item).strip()
        if not s:
            continue
        out.append(s)
    if not out:
        raise ValueError(f"`{key}` must contain at least one non-empty filename string.")
    return out


def _normalize_keras_sequence_model(models: Dict[str, Any], yaml_key: str) -> None:
    """Shared defaults for LSTM and vanilla RNN (SimpleRNN) blocks in YAML."""
    cfg = models.setdefault(yaml_key, {})
    fit_verbose = int(cfg.get("fit_verbose", 1))
    if fit_verbose not in (0, 1, 2):
        raise ValueError(f"`models.{yaml_key}.fit_verbose` must be 0, 1, or 2 (Keras fit verbosity).")
    cfg["fit_verbose"] = fit_verbose
    cfg["standardize_inputs"] = bool(cfg.get("standardize_inputs", False))
    cfg["learning_rate"] = float(cfg.get("learning_rate", 0.001))
    cfg["dense_units"] = int(cfg.get("dense_units", 16))
    if cfg["dense_units"] < 0:
        raise ValueError(
            f"`models.{yaml_key}.dense_units` must be >= 0 "
            f"(0 = no hidden Dense, only recurrent → Dropout → Dense(1))."
        )
    cfg["use_early_stopping"] = bool(cfg.get("use_early_stopping", True))
    cfg["early_stopping_patience"] = int(cfg.get("early_stopping_patience", 5))
    if cfg["early_stopping_patience"] < 0:
        raise ValueError(
            f"`models.{yaml_key}.early_stopping_patience` must be >= 0 (0 disables early stopping)."
        )


def load_model_config(path: str) -> ModelConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ValueError(f"Missing model params file: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data = raw.get("data", {})
    features = raw.get("features", {})
    training = raw.get("training", {})
    models = raw.get("models", {})
    persistence = raw.get("persistence", {})

    source = str(data.get("source", "csv")).strip().lower()
    if source not in {"csv", "supabase", "supabase_storage"}:
        raise ValueError("`data.source` must be one of: csv, supabase, supabase_storage.")
    data["source"] = source

    train_csv = _clean_optional_str(data.get("train_csv"))
    test_csv = _clean_optional_str(data.get("test_csv"))
    if source == "csv":
        data["train_csv"] = _require_path(train_csv or "", "data.train_csv")
        data["test_csv"] = _require_path(test_csv or "", "data.test_csv")
    else:
        if train_csv:
            data["train_csv"] = _require_path(train_csv, "data.train_csv")
        else:
            data["train_csv"] = None
        if test_csv:
            data["test_csv"] = _require_path(test_csv, "data.test_csv")
        else:
            data["test_csv"] = None

        required_supabase_keys = (
            "supabase_url",
            "supabase_key",
            "supabase_schema",
            "supabase_train_table",
            "supabase_test_table",
        )
        for key in required_supabase_keys:
            value = _clean_optional_str(data.get(key))
            if not value:
                raise ValueError(f"`data.{key}` is required when `data.source: supabase/supabase_storage`.")
            data[key] = value
        data["forecast_horizon_min"] = int(data.get("forecast_horizon_min", 5))
        data["supabase_page_size"] = int(data.get("supabase_page_size", 1000))
        if data["supabase_page_size"] <= 0:
            raise ValueError("`data.supabase_page_size` must be > 0.")

        if source == "supabase_storage":
            data["storage_train_base_url"] = _clean_optional_str(data.get("storage_train_base_url"))
            if not data["storage_train_base_url"]:
                raise ValueError("`data.storage_train_base_url` is required when `data.source: supabase_storage`.")

            data["storage_train_parts"] = _as_str_list(
                data.get("storage_train_parts"), "data.storage_train_parts"
            )

            data["storage_cache_dir"] = _clean_optional_str(data.get("storage_cache_dir"))
            if not data["storage_cache_dir"]:
                raise ValueError("`data.storage_cache_dir` is required when `data.source: supabase_storage`.")

            data["storage_cache_parquet_name"] = _clean_optional_str(
                data.get("storage_cache_parquet_name")
            ) or "train_merged.parquet"

            data["storage_force_refresh"] = bool(data.get("storage_force_refresh", False))

    validation_split = float(training.get("validation_split", 0.2))
    if validation_split <= 0.0 or validation_split >= 0.5:
        raise ValueError("`training.validation_split` must be between 0 and 0.5.")
    training["validation_split"] = validation_split

    training["random_state"] = int(training.get("random_state", 42))
    training["show_training_progress"] = bool(training.get("show_training_progress", True))

    _normalize_keras_sequence_model(models, "lstm")
    _normalize_keras_sequence_model(models, "rnn")

    for key in FEATURE_BOOL_KEYS:
        features[key] = bool(features.get(key, False))

    default_lags = [1, 2, 3, 6, 12]
    default_rolling = [3, 6, 12, 24]
    features["load_lag_steps"] = _as_int_list(
        features.get("load_lag_steps"), "features.load_lag_steps", default_lags
    )
    features["rolling_load_windows"] = _as_int_list(
        features.get("rolling_load_windows"),
        "features.rolling_load_windows",
        default_rolling,
    )
    features["include_weather_suffix_groups"] = _normalize_include_weather_suffix_groups(
        features.get("include_weather_suffix_groups")
    )

    artifact_dir = persistence.get("artifact_dir", "miner_model_energy/artifacts")
    persistence["artifact_dir"] = str(Path(artifact_dir))
    persistence["save_on_deploy"] = bool(persistence.get("save_on_deploy", True))

    return ModelConfig(
        data=data,
        features=features,
        training=training,
        models=models,
        persistence=persistence,
    )

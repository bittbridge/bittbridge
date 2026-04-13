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
    "use_wind_vector_features",
    "use_load_lags",
    "use_load_rolling",
    "use_load_delta",
)


def _require_path(path_value: str, key: str) -> str:
    path = Path(path_value)
    if not path.exists():
        raise ValueError(f"Config `{key}` points to missing path: {path}")
    return str(path)


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

    train_path = _require_path(data.get("train_csv", ""), "data.train_csv")
    test_path = _require_path(data.get("test_csv", ""), "data.test_csv")
    data["train_csv"] = train_path
    data["test_csv"] = test_path

    validation_split = float(training.get("validation_split", 0.2))
    if validation_split <= 0.0 or validation_split >= 0.5:
        raise ValueError("`training.validation_split` must be between 0 and 0.5.")
    training["validation_split"] = validation_split

    training["random_state"] = int(training.get("random_state", 42))
    training["show_training_progress"] = bool(training.get("show_training_progress", True))

    lstm_cfg = models.setdefault("lstm", {})
    fit_verbose = lstm_cfg.get("fit_verbose", 1)
    fit_verbose = int(fit_verbose)
    if fit_verbose not in (0, 1, 2):
        raise ValueError("`models.lstm.fit_verbose` must be 0, 1, or 2 (Keras fit verbosity).")
    lstm_cfg["fit_verbose"] = fit_verbose

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

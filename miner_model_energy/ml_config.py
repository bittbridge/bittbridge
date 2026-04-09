from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


ALLOWED_MODELS = {"linear", "cart", "lstm"}


@dataclass(frozen=True)
class ModelConfig:
    data: Dict[str, Any]
    features: Dict[str, Any]
    training: Dict[str, Any]
    models: Dict[str, Any]
    persistence: Dict[str, Any]


def _require_path(path_value: str, key: str) -> str:
    path = Path(path_value)
    if not path.exists():
        raise ValueError(f"Config `{key}` points to missing path: {path}")
    return str(path)


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

    selected_model = str(training.get("selected_model", "linear")).lower()
    if selected_model not in ALLOWED_MODELS:
        raise ValueError(
            "`training.selected_model` must be one of: linear, cart, lstm."
        )
    training["selected_model"] = selected_model

    enabled = models.get("enabled", {})
    for model_name in ALLOWED_MODELS:
        enabled[model_name] = bool(enabled.get(model_name, model_name != "lstm"))
    models["enabled"] = enabled

    for key in (
        "use_time_features",
        "use_cyclical_features",
        "use_load_lags",
        "use_load_delta",
    ):
        features[key] = bool(features.get(key, True))

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


from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .artifacts import (
    feature_signature,
    prepare_artifact_dir,
    write_config_snapshot,
    write_manifest,
)
from .data_io import TARGET_COLUMN, load_train_test
from .features import (
    add_engineered_features,
    add_test_load_features_from_history,
    build_feature_columns,
)
from .ml_config import ModelConfig
from .models_cart import predict_cart, save_cart, train_cart
from .models_cart import load_cart
from .models_linear import LinearBundle, load_linear, predict_linear, save_linear, train_linear
from .models_lstm import load_lstm, make_sequences, predict_lstm, save_lstm, train_lstm
from .split import temporal_train_val_split


@dataclass
class TrainingResult:
    model_type: str
    model_bundle: Any
    metrics: Dict[str, float]
    features: List[str]
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


def _as_numpy(frame: pd.DataFrame, features: List[str]) -> np.ndarray:
    return frame[features].astype(float).to_numpy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def prepare_training_data(config: ModelConfig) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train, test = load_train_test(config.data["train_csv"], config.data["test_csv"])
    train = add_engineered_features(train, config.features)
    test = add_engineered_features(test, config.features)

    lag_steps = config.features.get("load_lag_steps", [1, 2, 3, 6, 12])
    if config.features.get("use_load_lags", True):
        test = add_test_load_features_from_history(test, train, lag_steps=lag_steps)

    train_model = train.dropna().copy()
    if train_model.empty:
        raise ValueError("Training frame is empty after feature engineering and dropna.")

    features = build_feature_columns(train_model)
    missing = [c for c in features if c not in test.columns]
    for col in missing:
        test[col] = 0.0
    return train_model, test, features


def train_model(model_type: str, config: ModelConfig) -> TrainingResult:
    train_model, test, features = prepare_training_data(config)
    train_split, val_split = temporal_train_val_split(
        train_model, validation_split=config.training["validation_split"]
    )
    X_train = _as_numpy(train_split, features)
    y_train = train_split[TARGET_COLUMN].to_numpy()
    X_val = _as_numpy(val_split, features)
    y_val = val_split[TARGET_COLUMN].to_numpy()

    if model_type == "linear":
        bundle: LinearBundle = train_linear(X_train, y_train, features, config.models.get("linear", {}))
        val_pred = predict_linear(bundle, X_val)
    elif model_type == "cart":
        bundle = train_cart(X_train, y_train, features, config.models.get("cart", {}))
        val_pred = predict_cart(bundle, X_val)
    elif model_type == "lstm":
        bundle = train_lstm(X_train, y_train, features, config.models.get("lstm", {}))
        n_steps = bundle.n_steps
        X_val_seq, y_val_seq = make_sequences(X_val, y_val, n_steps=n_steps)
        if len(X_val_seq) == 0:
            raise ValueError("Validation split too short for LSTM sequence evaluation.")
        val_pred = predict_lstm(bundle, X_val_seq)
        y_val = y_val_seq
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return TrainingResult(
        model_type=model_type,
        model_bundle=bundle,
        metrics=_metrics(y_val, val_pred),
        features=features,
        train_frame=train_model,
        test_frame=test,
    )


def predict_single_test_row(result: TrainingResult) -> float:
    X_test = result.test_frame[result.features].astype(float).to_numpy()
    if result.model_type == "linear":
        pred = predict_linear(result.model_bundle, X_test)[0]
    elif result.model_type == "cart":
        pred = predict_cart(result.model_bundle, X_test)[0]
    elif result.model_type == "lstm":
        pred = predict_lstm(result.model_bundle, X_test)[0]
    else:
        raise ValueError(f"Unsupported model type: {result.model_type}")
    return float(pred)


def persist_training_result(result: TrainingResult, config: ModelConfig, run_id: str | None = None) -> Dict[str, str]:
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

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(pd.Series(result.metrics).to_json(indent=2), encoding="utf-8")

    cfg_snapshot = {
        "data": config.data,
        "features": config.features,
        "training": config.training,
        "models": config.models,
        "persistence": config.persistence,
    }
    write_config_snapshot(out_dir, cfg_snapshot)

    manifest = {
        "model_type": result.model_type,
        "model_path": model_rel,
        "metrics_path": metrics_path.name,
        "features": result.features,
        "features_count": len(result.features),
        "feature_signature": feature_signature(result.features),
        "train_rows": int(len(result.train_frame)),
        "metrics": result.metrics,
        "lstm_n_steps": getattr(result.model_bundle, "n_steps", None),
    }
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
        return load_lstm(model_path, features=features, n_steps=int(manifest.get("lstm_n_steps", 12)))
    raise ValueError(f"Unsupported model type in manifest: {model_type}")


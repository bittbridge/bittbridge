from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import yaml

from miner_model_energy.artifacts import load_manifest
from miner_model_energy.inference_runtime import AdvancedModelPredictor, PredictorRouter
from miner_model_energy.ml_config import load_model_config
from miner_model_energy.pipeline import persist_training_result, train_model


def _write_small_dataset(tmp_path):
    start = datetime(2025, 1, 1, 0, 0, 0)
    rows = []
    for i in range(96):
        dt = start + timedelta(minutes=5 * i)
        rows.append(
            {
                "dt": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Total Load": 10000 + i * 2 + (i % 6),
                "4B8-tmpf": 35 + (i % 10),
            }
        )
    train = pd.DataFrame(rows)
    test = train.tail(1).drop(columns=["Total Load"]).copy()
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def _write_config(tmp_path, train_path, test_path):
    cfg = {
        "data": {"train_csv": str(train_path), "test_csv": str(test_path)},
        "features": {
            "use_time_features": True,
            "use_cyclical_features": True,
            "use_load_lags": True,
            "use_load_delta": True,
            "load_lag_steps": [1, 2, 3],
        },
        "training": {"validation_split": 0.2},
        "models": {
            "linear": {"fit_intercept": True},
            "cart": {"max_depth": 4, "min_samples_split": 4, "min_samples_leaf": 2},
            "lstm": {"n_steps": 12, "units": 8, "dropout": 0.1, "epochs": 1, "batch_size": 16},
        },
        "persistence": {"artifact_dir": str(tmp_path / "artifacts"), "save_on_deploy": True},
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_linear_training_and_persistence(tmp_path):
    train_path, test_path = _write_small_dataset(tmp_path)
    cfg_path = _write_config(tmp_path, train_path, test_path)
    cfg = load_model_config(str(cfg_path))
    result = train_model("linear", cfg)
    assert result.metrics["validation"]["rmse"] >= 0.0
    assert result.durations_sec.get("total_sec", 0) >= 0
    pred = AdvancedModelPredictor(result).predict("2025-01-01 12:00:00")
    assert isinstance(pred, float)

    saved = persist_training_result(result, cfg, run_id="pytest")
    manifest = load_manifest(saved["manifest_path"])
    assert manifest["model_type"] == "linear"


def test_cart_training(tmp_path):
    train_path, test_path = _write_small_dataset(tmp_path)
    cfg_path = _write_config(tmp_path, train_path, test_path)
    cfg = load_model_config(str(cfg_path))
    result = train_model("cart", cfg)
    assert result.metrics["validation"]["mae"] >= 0.0


def test_predictor_router_switch(tmp_path):
    train_path, test_path = _write_small_dataset(tmp_path)
    cfg_path = _write_config(tmp_path, train_path, test_path)
    cfg = load_model_config(str(cfg_path))
    result = train_model("linear", cfg)
    router = PredictorRouter(AdvancedModelPredictor(result))
    router.set_predictor(AdvancedModelPredictor(result), mode="advanced:linear")
    value = router.predict("2025-01-01 12:00:00")
    assert isinstance(value, float)


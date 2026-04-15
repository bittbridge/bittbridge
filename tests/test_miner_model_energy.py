from __future__ import annotations

import sys
from pathlib import Path

# Allow `python tests/test_miner_model_energy.py` and pytest without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datetime import datetime, timedelta

import pandas as pd
import pytest
import yaml

from miner_model_energy.artifacts import load_manifest
from miner_model_energy.features import KNOWN_WEATHER_SUFFIXES
from miner_model_energy.inference_runtime import AdvancedModelPredictor, PredictorRouter
from miner_model_energy.ml_config import load_model_config
from miner_model_energy.models_lstm import LSTM_SCALER_FILENAME
from miner_model_energy.pipeline import (
    load_training_bundle_from_manifest,
    persist_training_result,
    predict_single_test_row,
    train_model,
)


def _weather_row(i: int, start: datetime) -> dict:
    dt = start + timedelta(minutes=5 * i)
    return {
        "dt": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Total Load": 10000 + i * 2 + (i % 6),
        "4B8-tmpf": 35 + (i % 10),
        "4B8-dwpf": 30 + (i % 8),
        "4B8-relh": 50 + (i % 5),
        "4B8-sped": 5.0 + (i % 3),
        "4B8-drct": float((i * 17) % 360),
        "BDL-tmpf": 32 + (i % 7),
    }


def _write_dataset(tmp_path):
    start = datetime(2025, 1, 1, 0, 0, 0)
    rows = [_weather_row(i, start) for i in range(96)]
    train = pd.DataFrame(rows)
    test = train.tail(1).drop(columns=["Total Load"]).copy()
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, test_path


def _default_features():
    return {
        "use_time_features": False,
        "use_cyclical_features": False,
        "use_station_agg_features": False,
        "use_temp_dew_gap": False,
        "use_load_lags": False,
        "use_load_rolling": False,
        "use_load_delta": False,
        "load_lag_steps": [1, 2, 3],
        "rolling_load_windows": [3, 6, 12],
        # Whitelist all raw weather columns so parametrize cases match pre-filter behavior.
        "include_weather_suffix_groups": sorted(KNOWN_WEATHER_SUFFIXES),
    }


def _write_config(tmp_path, train_path, test_path, feature_patch: dict | None = None):
    features = _default_features()
    if feature_patch:
        features.update(feature_patch)
    cfg = {
        "data": {"train_csv": str(train_path), "test_csv": str(test_path)},
        "features": features,
        "training": {
            "validation_split": 0.2,
            "random_state": 123,
            "show_training_progress": False,
        },
        "models": {
            "linear": {"fit_intercept": True},
            "cart": {"max_depth": 4, "min_samples_split": 4, "min_samples_leaf": 2},
            "lstm": {
                "n_steps": 12,
                "units": 8,
                "dropout": 0.1,
                "epochs": 1,
                "batch_size": 16,
                "fit_verbose": 0,
            },
            "rnn": {
                "n_steps": 12,
                "units": 8,
                "dropout": 0.1,
                "epochs": 1,
                "batch_size": 16,
                "use_early_stopping": False,
                "fit_verbose": 0,
            },
        },
        "persistence": {"artifact_dir": str(tmp_path / "artifacts"), "save_on_deploy": True},
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


FEATURE_COMBOS = [
    pytest.param({}, id="all_off_raw_only"),
    pytest.param({"use_time_features": True}, id="time"),
    pytest.param({"use_cyclical_features": True}, id="cyclical"),
    pytest.param({"use_station_agg_features": True}, id="station_agg"),
    pytest.param({"use_temp_dew_gap": True}, id="temp_dew_gap"),
    pytest.param({"use_load_lags": True, "load_lag_steps": [1, 3]}, id="lags"),
    pytest.param({"use_load_delta": True}, id="delta_only"),
    pytest.param({"use_load_rolling": True, "rolling_load_windows": [3, 6, 12]}, id="rolling"),
    pytest.param(
        {"use_load_lags": True, "use_load_rolling": True, "load_lag_steps": [1, 2]},
        id="lags_and_rolling",
    ),
    pytest.param(
        {
            "use_time_features": True,
            "use_station_agg_features": True,
            "use_load_lags": True,
        },
        id="stacked_time_station_lags",
    ),
    pytest.param(
        {"use_load_rolling": True, "use_load_delta": True, "rolling_load_windows": [3, 6]},
        id="rolling_plus_delta_flag_rolling_wins",
    ),
]


@pytest.mark.parametrize("feature_patch", FEATURE_COMBOS)
def test_linear_feature_toggle_combinations(tmp_path, feature_patch):
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(tmp_path, train_path, test_path, feature_patch)
    cfg = load_model_config(str(cfg_path))
    result = train_model("linear", cfg)
    assert result.metrics["validation"]["rmse"] >= 0.0
    pred = predict_single_test_row(result)
    assert isinstance(pred, float)


@pytest.mark.parametrize("feature_patch", FEATURE_COMBOS)
def test_cart_feature_toggle_combinations(tmp_path, feature_patch):
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(tmp_path, train_path, test_path, feature_patch)
    cfg = load_model_config(str(cfg_path))
    result = train_model("cart", cfg)
    assert result.metrics["validation"]["mae"] >= 0.0


def test_linear_training_and_persistence(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(
        tmp_path,
        train_path,
        test_path,
        {"use_load_lags": True, "use_load_delta": True},
    )
    cfg = load_model_config(str(cfg_path))
    result = train_model("linear", cfg)
    assert result.durations_sec.get("total_sec", 0) >= 0
    pred = AdvancedModelPredictor(result).predict("2025-01-01 12:00:00")
    assert isinstance(pred, float)

    saved = persist_training_result(result, cfg, run_id="pytest")
    manifest = load_manifest(saved["manifest_path"])
    assert manifest["model_type"] == "linear"


def test_cart_training(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(tmp_path, train_path, test_path, {"use_load_lags": True})
    cfg = load_model_config(str(cfg_path))
    result = train_model("cart", cfg)
    assert result.metrics["validation"]["mae"] >= 0.0


def test_predictor_router_switch(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(tmp_path, train_path, test_path, {"use_time_features": True})
    cfg = load_model_config(str(cfg_path))
    result = train_model("linear", cfg)
    router = PredictorRouter(AdvancedModelPredictor(result))
    router.set_predictor(AdvancedModelPredictor(result), mode="advanced:linear")
    value = router.predict("2025-01-01 12:00:00")
    assert isinstance(value, float)


def test_empty_weather_whitelist_drops_raw_columns(tmp_path):
    """Default YAML semantics: [] removes *-tmpf etc.; need engineered features to train."""
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(
        tmp_path,
        train_path,
        test_path,
        {"include_weather_suffix_groups": [], "use_time_features": True},
    )
    cfg = load_model_config(str(cfg_path))
    result = train_model("linear", cfg)
    assert "4B8-tmpf" not in result.train_frame.columns
    assert "hour" in result.train_frame.columns
    assert result.metrics["validation"]["rmse"] >= 0.0


def test_linear_with_weather_suffix_whitelist(tmp_path):
    """Only *-tmpf and *-dwpf raw columns kept; training still completes."""
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(
        tmp_path,
        train_path,
        test_path,
        {"include_weather_suffix_groups": ["tmpf", "dwpf"], "use_station_agg_features": True},
    )
    cfg = load_model_config(str(cfg_path))
    result = train_model("linear", cfg)
    assert "4B8-drct" not in result.train_frame.columns
    assert result.metrics["validation"]["rmse"] >= 0.0


def test_load_config_rejects_unknown_weather_suffix(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    bad = _default_features()
    bad["include_weather_suffix_groups"] = ["tmpf", "not_a_real_suffix"]
    cfg = {
        "data": {"train_csv": str(train_path), "test_csv": str(test_path)},
        "features": bad,
        "training": {"validation_split": 0.2, "random_state": 0},
        "models": {"linear": {"fit_intercept": True}},
        "persistence": {"artifact_dir": str(tmp_path / "a"), "save_on_deploy": True},
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown suffix"):
        load_model_config(str(path))


def test_rnn_runs_with_feature_patch(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(
        tmp_path,
        train_path,
        test_path,
        {"use_time_features": True, "use_load_lags": True},
    )
    cfg = load_model_config(str(cfg_path))
    result = train_model("rnn", cfg)
    assert result.metrics["validation"]["rmse"] >= 0.0
    assert result.model_bundle.scaler is None


def test_lstm_runs_with_feature_patch(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(
        tmp_path,
        train_path,
        test_path,
        {"use_time_features": True, "use_load_lags": True},
    )
    cfg = load_model_config(str(cfg_path))
    result = train_model("lstm", cfg)
    assert result.metrics["validation"]["rmse"] >= 0.0
    assert result.model_bundle.scaler is None


def test_lstm_standardize_inputs_and_reload(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(
        tmp_path,
        train_path,
        test_path,
        {"use_time_features": True, "use_load_lags": True},
    )
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["models"]["lstm"]["standardize_inputs"] = True
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    cfg = load_model_config(str(cfg_path))
    assert cfg.models["lstm"]["standardize_inputs"] is True
    result = train_model("lstm", cfg)
    assert result.model_bundle.scaler is not None
    pred = predict_single_test_row(result)
    assert isinstance(pred, float)

    saved = persist_training_result(result, cfg, run_id="pytest_lstm_std")
    scaler_fp = Path(saved["artifact_dir"]) / LSTM_SCALER_FILENAME
    assert scaler_fp.is_file()
    manifest = load_manifest(saved["manifest_path"])
    assert manifest.get("lstm_standardize_inputs") is True
    assert manifest.get("lstm_scaler_path") == LSTM_SCALER_FILENAME

    reloaded = load_training_bundle_from_manifest(saved["manifest_path"])
    assert reloaded.scaler is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

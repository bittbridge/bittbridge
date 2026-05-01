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

from neurons import miner as miner_module
from miner_model_energy.artifacts import load_manifest
from miner_model_energy.features import KNOWN_WEATHER_SUFFIXES
from miner_model_energy.inference_runtime import AdvancedModelPredictor, PredictorRouter
from miner_model_energy.ml_config import ModelConfig, load_model_config
from miner_model_energy.models_lstm import LSTM_SCALER_FILENAME
from miner_model_energy.data_io import TARGET_COLUMN, TARGET_COLUMN_HORIZON
from miner_model_energy.pipeline import (
    TrainingResult,
    _required_history_rows_for_live,
    load_training_bundle_from_manifest,
    prepare_training_data,
    persist_training_result,
    predict_single_test_row,
    train_model,
)
from miner_model_energy.supabase_io import fetch_supabase_test_row, fetch_supabase_train_all
from miner_model_energy.storage_train_io import (
    load_train_from_storage_parts,
    storage_cache_exists,
    storage_cache_last_updated_label,
    storage_cache_paths,
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


def _write_config(
    tmp_path,
    train_path,
    test_path,
    feature_patch: dict | None = None,
    data_patch: dict | None = None,
):
    features = _default_features()
    if feature_patch:
        features.update(feature_patch)
    data_cfg = {"train_csv": str(train_path), "test_csv": str(test_path)}
    if data_patch:
        data_cfg.update(data_patch)
    cfg = {
        "data": data_cfg,
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
    assert cfg.persistence.get("config_file") == str(cfg_path.resolve())
    result = train_model("linear", cfg)
    assert result.durations_sec.get("total_sec", 0) >= 0
    pred = AdvancedModelPredictor(result).predict("2025-01-01 12:00:00")
    assert isinstance(pred, float)

    saved = persist_training_result(result, cfg, run_id="pytest")
    manifest = load_manifest(saved["manifest_path"])
    assert manifest["model_type"] == "linear"
    assert manifest.get("actual_vs_predicted_path") == "actual_vs_predicted.csv"
    assert manifest.get("full_training_dataset_dumped") is False
    assert manifest.get("full_training_dataset_path") is None
    avp = Path(saved["artifact_dir"]) / "actual_vs_predicted.csv"
    assert avp.is_file()
    assert len(pd.read_csv(avp)) > 0


def test_linear_training_and_persistence_with_full_dataset_dump(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(
        tmp_path,
        train_path,
        test_path,
        {"use_load_lags": True, "use_load_delta": True},
    )
    cfg = load_model_config(str(cfg_path))
    result = train_model("linear", cfg)

    saved = persist_training_result(
        result,
        cfg,
        run_id="pytest_full_dump",
        dump_full_training_dataset=True,
    )
    artifact_dir = Path(saved["artifact_dir"])
    manifest = load_manifest(saved["manifest_path"])
    dumped_path = artifact_dir / "training_dataset_full.csv"
    dumped_df = pd.read_csv(dumped_path)

    assert dumped_path.is_file()
    assert manifest.get("full_training_dataset_dumped") is True
    assert manifest.get("full_training_dataset_path") == "training_dataset_full.csv"
    assert manifest.get("full_training_dataset_rows") == len(result.train_frame)
    assert manifest.get("full_training_dataset_columns_count") == len(result.train_frame.columns)
    assert manifest.get("full_training_dataset_columns") == [str(c) for c in result.train_frame.columns]
    assert len(dumped_df) == len(result.train_frame)


def test_relative_artifact_dir_resolves_next_to_config_file(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    nest = tmp_path / "confdir"
    nest.mkdir(parents=True, exist_ok=True)
    cfg_path = nest / "params.yaml"
    features = _default_features()
    cfg = {
        "data": {"train_csv": str(train_path), "test_csv": str(test_path)},
        "features": features,
        "training": {"validation_split": 0.2, "random_state": 123, "show_training_progress": False},
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
        "persistence": {"artifact_dir": "artifacts", "save_on_deploy": True},
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    loaded = load_model_config(str(cfg_path))
    assert loaded.persistence["config_file"] == str(cfg_path.resolve())
    assert loaded.persistence["artifact_dir"] == str((nest / "artifacts").resolve())
    assert loaded.data["forecast_horizon_min"] == 0


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
    assert (Path(saved["artifact_dir"]) / "actual_vs_predicted.csv").is_file()
    manifest = load_manifest(saved["manifest_path"])
    assert manifest.get("actual_vs_predicted_path") == "actual_vs_predicted.csv"
    assert manifest.get("lstm_standardize_inputs") is True
    assert manifest.get("lstm_scaler_path") == LSTM_SCALER_FILENAME

    reloaded = load_training_bundle_from_manifest(saved["manifest_path"])
    assert reloaded.scaler is not None


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._predicates = []
        self._order_desc = False
        self._range = None
        self._limit = None

    def select(self, *_args, **_kwargs):
        return self

    def order(self, _col, desc=False):
        self._order_desc = bool(desc)
        return self

    def range(self, start, end):
        self._range = (int(start), int(end))
        return self

    def limit(self, n):
        self._limit = int(n)
        return self

    def eq(self, key, value):
        self._predicates.append(("eq", key, value))
        return self

    def gte(self, key, value):
        self._predicates.append(("gte", key, value))
        return self

    def lte(self, key, value):
        self._predicates.append(("lte", key, value))
        return self

    def execute(self):
        rows = list(self._rows)
        for op, key, value in self._predicates:
            if op == "eq":
                rows = [r for r in rows if str(r.get(key)) == str(value)]
            elif op == "gte":
                rows = [r for r in rows if str(r.get(key)) >= str(value)]
            elif op == "lte":
                rows = [r for r in rows if str(r.get(key)) <= str(value)]
        rows = sorted(rows, key=lambda r: str(r.get("dt")), reverse=self._order_desc)
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        if self._limit is not None:
            rows = rows[: self._limit]
        return _FakeResponse(rows)


class _FakeSchemaClient:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return _FakeQuery(self._tables[name])


class _FakeSupabaseClient:
    def __init__(self, tables):
        self._tables = tables

    def schema(self, _schema_name):
        return _FakeSchemaClient(self._tables)


def test_fetch_supabase_train_all_paginates_and_normalizes():
    train_rows = [
        {"dt": "2026-04-13 20:45:00+00:00", "total_load": 1000.0, "4B8-tmpf": 55.0},
        {"dt": "2026-04-13 20:50:00+00:00", "total_load": 1001.0, "4B8-tmpf": 56.0},
        {"dt": "2026-04-13 20:55:00+00:00", "total_load": 1002.0, "4B8-tmpf": 57.0},
    ]
    client = _FakeSupabaseClient({"train_table": train_rows})
    frame = fetch_supabase_train_all(client, schema="hackathon", table="train_table", page_size=2)
    assert list(frame.columns)[0] == "dt"
    assert "Total Load" in frame.columns
    assert "total_load" not in frame.columns
    assert len(frame) == 3
    assert str(frame["dt"].iloc[0]) == "2026-04-13 20:45:00"


def test_fetch_supabase_test_row_prefers_requested_horizon():
    rows = [
        {"dt": "2026-04-13 20:45:00", "horizon_min": 10, "4B8-tmpf": 50.0},
        {"dt": "2026-04-13 20:45:00", "horizon_min": 5, "4B8-tmpf": 51.0},
    ]
    client = _FakeSupabaseClient({"test_table": rows})
    row = fetch_supabase_test_row(
        client,
        schema="hackathon",
        table="test_table",
        dt_target="2026-04-13 20:45:00+00:00",
        horizon_min=5,
    )
    assert row is not None
    assert int(row["horizon_min"]) == 5


def test_fetch_supabase_test_row_falls_back_to_local_wall_clock_exact():
    rows = [
        {"dt": "2026-04-13 10:35:00", "horizon_min": 5, "4B8-tmpf": 51.0},
    ]
    client = _FakeSupabaseClient({"test_table": rows})
    row = fetch_supabase_test_row(
        client,
        schema="hackathon",
        table="test_table",
        dt_target="2026-04-13T10:35:00-04:00",
        horizon_min=5,
    )
    assert row is not None
    assert row["dt"] == "2026-04-13 10:35:00"
    assert int(row["horizon_min"]) == 5


def test_fetch_supabase_test_row_returns_none_on_horizon_mismatch():
    rows = [
        {"dt": "2026-04-13 20:45:00", "horizon_min": 10, "4B8-tmpf": 50.0},
        {"dt": "2026-04-13 20:45:00", "horizon_min": 15, "4B8-tmpf": 51.0},
    ]
    client = _FakeSupabaseClient({"test_table": rows})
    row = fetch_supabase_test_row(
        client,
        schema="hackathon",
        table="test_table",
        dt_target="2026-04-13 20:45:00+00:00",
        horizon_min=5,
    )
    assert row is None


def test_fetch_supabase_test_row_for_probe_falls_back_to_latest_matching_horizon():
    """When wall-clock has no row, probe path still finds latest row for that horizon."""
    from miner_model_energy.supabase_io import fetch_supabase_test_row_for_probe

    rows = [
        {"dt": "2026-01-01 12:00:00", "horizon_min": 360, "4B8-tmpf": 40.0},
        {"dt": "2026-03-15 18:00:00", "horizon_min": 360, "4B8-tmpf": 41.0},
    ]
    client = _FakeSupabaseClient({"test_table": rows})
    row = fetch_supabase_test_row_for_probe(
        client,
        schema="hackathon",
        table="test_table",
        dt_target="2026-04-30T12:00:00-04:00",
        horizon_min=360,
    )
    assert row is not None
    assert int(row["horizon_min"]) == 360
    assert row["dt"] == "2026-03-15 18:00:00"


def test_required_history_rows_for_live_rnn_load_lag_24_covers_sequence_window():
    """After shift(max_lag), only tail rows are non-NaN in load_lag_*; fetch enough for RNN window."""

    class _Bundle:
        n_steps = 12

    result = TrainingResult(
        model_type="rnn",
        model_bundle=_Bundle(),
        metrics={},
        features=[],
        train_frame=pd.DataFrame(),
        test_frame=pd.DataFrame(),
        shapes={},
    )
    cfg = ModelConfig(
        data={},
        features={
            "use_load_lags": True,
            "load_lag_steps": [12, 24],
        },
        training={},
        models={},
        persistence={},
    )
    needed = _required_history_rows_for_live(result, cfg)
    assert needed >= 24 + 11 + 8  # max_lag + (n_steps - 1) + live_seq_buffer


def test_required_history_rows_for_live_linear_with_load_lags_no_sequence_tail():
    class _LinearBundle:
        pass

    result = TrainingResult(
        model_type="linear",
        model_bundle=_LinearBundle(),
        metrics={},
        features=[],
        train_frame=pd.DataFrame(),
        test_frame=pd.DataFrame(),
        shapes={},
    )
    cfg = ModelConfig(
        data={},
        features={
            "use_load_lags": True,
            "load_lag_steps": [12, 24],
        },
        training={},
        models={},
        persistence={},
    )
    needed = _required_history_rows_for_live(result, cfg)
    assert needed == 32


def test_prepare_training_data_csv_has_no_horizon_target_by_default(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    cfg_path = _write_config(tmp_path, train_path, test_path)
    cfg = load_model_config(str(cfg_path))
    train_model_frame, _, _ = prepare_training_data(cfg, show_progress=False)
    assert cfg.data["forecast_horizon_min"] == 0
    assert cfg.data["train_feature_time_shift_min"] == 0
    assert cfg.data["train_disable_horizon_label_shift_when_feature_shifted"] is False
    assert TARGET_COLUMN_HORIZON not in train_model_frame.columns


def test_prepare_training_data_supabase_storage_shift_mode_keeps_current_label(tmp_path, monkeypatch):
    train_path, test_path = _write_dataset(tmp_path)
    train_df = pd.read_csv(train_path)
    train_df["dt"] = pd.to_datetime(train_df["dt"], errors="raise")

    def _fake_load_train_from_storage_parts(config, force_refresh):
        assert config.data["source"] == "supabase_storage"
        assert force_refresh is False
        return train_df.copy()

    monkeypatch.setattr(
        "miner_model_energy.pipeline.load_train_from_storage_parts",
        _fake_load_train_from_storage_parts,
    )

    cfg_path = _write_config(
        tmp_path,
        train_path=train_path,
        test_path=test_path,
        feature_patch={"use_time_features": True},
        data_patch={
            "source": "supabase_storage",
            "forecast_horizon_min": 10,
            "train_feature_time_shift_min": 10,
            "train_disable_horizon_label_shift_when_feature_shifted": True,
            "storage_train_base_url": "https://example.supabase.co/storage/v1/object/public/public-dumps/hackathon-train-data/",
            "storage_train_parts": ["part-2024-01.csv"],
            "storage_cache_dir": str(tmp_path / "cache"),
            "storage_cache_parquet_name": "train_merged.parquet",
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "sb_test",
            "supabase_schema": "hackathon",
            "supabase_train_table": "hackathon-train-data",
            "supabase_test_table": "hackathon-test-data",
        },
    )
    cfg = load_model_config(str(cfg_path))
    train_model_frame, _, _ = prepare_training_data(cfg, show_progress=False)

    assert TARGET_COLUMN_HORIZON not in train_model_frame.columns
    assert len(train_model_frame) == len(train_df) - 2  # 10 min on 5-min cadence => 2 shifted tail rows dropped
    assert float(train_model_frame["4B8-tmpf"].iloc[0]) == float(train_df["4B8-tmpf"].iloc[2])
    assert int(train_model_frame["hour"].iloc[0]) == int(train_df["dt"].iloc[2].hour)
    assert float(train_model_frame[TARGET_COLUMN].iloc[0]) == float(train_df[TARGET_COLUMN].iloc[0])


def test_prepare_training_data_uses_supabase_branch(tmp_path, monkeypatch):
    train_rows = []
    start = datetime(2026, 4, 13, 20, 0, 0)
    for i in range(50):
        dt = start + timedelta(minutes=5 * i)
        train_rows.append(
            {
                "dt": dt.strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "total_load": 1000 + i,
                "4B8-tmpf": 55 + (i % 3),
                "4B8-dwpf": 45 + (i % 2),
            }
        )

    def _fake_create_client(url, key):
        assert url == "https://example.supabase.co"
        assert key == "sb_publishable_test"
        return object()

    def _fake_fetch_train_all(client, schema, table, page_size):
        assert schema == "hackathon"
        assert table == "hackathon-train-data"
        assert page_size == 1000
        assert client is not None
        return pd.DataFrame(train_rows).rename(columns={"total_load": "Total Load"}).assign(
            dt=lambda d: pd.to_datetime(d["dt"], utc=True).dt.tz_localize(None)
        )

    monkeypatch.setattr("miner_model_energy.pipeline.create_supabase_data_client", _fake_create_client)
    monkeypatch.setattr("miner_model_energy.pipeline.fetch_supabase_train_all", _fake_fetch_train_all)

    cfg_path = _write_config(
        tmp_path,
        train_path=tmp_path / "unused-train.csv",
        test_path=tmp_path / "unused-test.csv",
        feature_patch={"include_weather_suffix_groups": ["tmpf", "dwpf"]},
        data_patch={
            "source": "supabase",
            "train_csv": None,
            "test_csv": None,
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "sb_publishable_test",
            "supabase_schema": "hackathon",
            "supabase_train_table": "hackathon-train-data",
            "supabase_test_table": "hackathon-test-data",
            "forecast_horizon_min": 5,
            "supabase_page_size": 1000,
        },
    )
    cfg = load_model_config(str(cfg_path))
    train_model_frame, test_frame, features = prepare_training_data(cfg, show_progress=False)

    assert len(train_model_frame) > 0
    assert len(test_frame) == 1
    assert "Total Load" in train_model_frame.columns
    assert "4B8-tmpf" in features
    assert TARGET_COLUMN_HORIZON in train_model_frame.columns
    # 5 min spacing + forecast_horizon_min=5 => 1 step ahead
    assert len(train_model_frame) == 49
    assert int(train_model_frame[TARGET_COLUMN_HORIZON].iloc[0]) == int(
        train_model_frame["Total Load"].iloc[0] + 1
    )


def test_storage_cache_miss_downloads_and_writes_cache(tmp_path, monkeypatch):
    train_path, test_path = _write_dataset(tmp_path)

    part1 = pd.DataFrame(
        [
            {
                "dt": "2026-04-13 20:45:00+00:00",
                "total_load": 1000,
                "4B8-tmpf": 55,
            },
            {
                "dt": "2026-04-13 20:50:00+00:00",
                "total_load": 1001,
                "4B8-tmpf": 56,
            },
        ]
    )
    part2 = pd.DataFrame(
        [
            {
                "dt": "2026-04-13 20:55:00+00:00",
                "total_load": 1002,
                "4B8-tmpf": 57,
            }
        ]
    )

    base_url = "https://example.supabase.co/storage/v1/object/public/public-dumps/hackathon-train-data/"
    part_map = {
        base_url + "part-2024-01.csv": part1.to_csv(index=False),
        base_url + "part-2024-02.csv": part2.to_csv(index=False),
    }

    class _FakeResp:
        def __init__(self, content: str):
            self.content = content.encode("utf-8")

        def raise_for_status(self):
            return None

    def _fake_get(url, timeout=0):
        if url not in part_map:
            raise AssertionError(f"Unexpected download URL: {url}")
        return _FakeResp(part_map[url])

    monkeypatch.setattr("miner_model_energy.storage_train_io.requests.get", _fake_get)

    cache_dir = tmp_path / "cache"
    cfg_path = _write_config(
        tmp_path,
        train_path=train_path,
        test_path=test_path,
        feature_patch={"use_time_features": False, "use_station_agg_features": False},
        data_patch={
            "source": "supabase_storage",
            "storage_train_base_url": base_url,
            "storage_train_parts": ["part-2024-01.csv", "part-2024-02.csv"],
            "storage_cache_dir": str(cache_dir),
            "storage_cache_parquet_name": "train_merged.parquet",
            # API keys (not used by this test; storage loader is public-download only)
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "sb_test",
            "supabase_schema": "hackathon",
            "supabase_train_table": "hackathon-train-data",
            "supabase_test_table": "hackathon-test-data",
        },
    )
    cfg = load_model_config(str(cfg_path))

    assert storage_cache_exists(cfg) is False
    df = load_train_from_storage_parts(cfg, force_refresh=False)
    assert len(df) == 3
    assert storage_cache_exists(cfg) is True


def test_storage_cache_last_updated_label_uses_eastern_time(tmp_path):
    train_path, test_path = _write_dataset(tmp_path)
    cache_dir = tmp_path / "cache"
    cfg_path = _write_config(
        tmp_path,
        train_path=train_path,
        test_path=test_path,
        data_patch={
            "source": "supabase_storage",
            "storage_train_base_url": "https://example.supabase.co/storage/v1/object/public/public-dumps/hackathon-train-data/",
            "storage_train_parts": ["part-2024-01.csv"],
            "storage_cache_dir": str(cache_dir),
            "storage_cache_parquet_name": "train_merged.parquet",
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "sb_test",
            "supabase_schema": "hackathon",
            "supabase_train_table": "hackathon-train-data",
            "supabase_test_table": "hackathon-test-data",
        },
    )
    cfg = load_model_config(str(cfg_path))
    _cache_path, manifest_path = storage_cache_paths(cfg)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"downloaded_at":"2026-04-16T14:15:21+00:00"}', encoding="utf-8")

    assert storage_cache_last_updated_label(cfg) == "2026-04-16 10:15:21 ET"


def test_storage_cache_hit_avoids_download(tmp_path, monkeypatch):
    train_path, test_path = _write_dataset(tmp_path)

    part_df = pd.DataFrame(
        [
            {"dt": "2026-04-13 20:45:00+00:00", "total_load": 1000, "4B8-tmpf": 55},
            {"dt": "2026-04-13 20:50:00+00:00", "total_load": 1001, "4B8-tmpf": 56},
        ]
    )
    base_url = "https://example.supabase.co/storage/v1/object/public/public-dumps/hackathon-train-data/"
    part_map = {base_url + "part-2024-01.csv": part_df.to_csv(index=False)}

    class _FakeResp:
        def __init__(self, content: str):
            self.content = content.encode("utf-8")

        def raise_for_status(self):
            return None

    def _fake_get(url, timeout=0):
        return _FakeResp(part_map[url])

    monkeypatch.setattr("miner_model_energy.storage_train_io.requests.get", _fake_get)

    cache_dir = tmp_path / "cache"
    cfg_path = _write_config(
        tmp_path,
        train_path=train_path,
        test_path=test_path,
        data_patch={
            "source": "supabase_storage",
            "storage_train_base_url": base_url,
            "storage_train_parts": ["part-2024-01.csv"],
            "storage_cache_dir": str(cache_dir),
            "storage_cache_parquet_name": "train_merged.parquet",
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "sb_test",
            "supabase_schema": "hackathon",
            "supabase_train_table": "hackathon-train-data",
            "supabase_test_table": "hackathon-test-data",
        },
    )
    cfg = load_model_config(str(cfg_path))

    # First load -> download.
    df1 = load_train_from_storage_parts(cfg, force_refresh=False)
    assert len(df1) == 2
    assert storage_cache_exists(cfg) is True

    # Second load -> must not download again.
    def _fake_get_should_not_be_called(_url, timeout=0):
        raise AssertionError("Storage download should not be called on cache hit.")

    monkeypatch.setattr("miner_model_energy.storage_train_io.requests.get", _fake_get_should_not_be_called)
    df2 = load_train_from_storage_parts(cfg, force_refresh=False)
    assert df2.shape == df1.shape


def test_storage_cache_rebuild_failure_falls_back_to_existing_cache(tmp_path, monkeypatch):
    train_path, test_path = _write_dataset(tmp_path)

    part_df = pd.DataFrame(
        [
            {"dt": "2026-04-13 20:45:00+00:00", "total_load": 1000, "4B8-tmpf": 55},
            {"dt": "2026-04-13 20:50:00+00:00", "total_load": 1001, "4B8-tmpf": 56},
        ]
    )
    base_url = "https://example.supabase.co/storage/v1/object/public/public-dumps/hackathon-train-data/"
    part_map = {base_url + "part-2024-01.csv": part_df.to_csv(index=False)}

    class _FakeResp:
        def __init__(self, content: str):
            self.content = content.encode("utf-8")

        def raise_for_status(self):
            return None

    def _fake_get(url, timeout=0):
        return _FakeResp(part_map[url])

    monkeypatch.setattr("miner_model_energy.storage_train_io.requests.get", _fake_get)

    cache_dir = tmp_path / "cache"
    cfg_path = _write_config(
        tmp_path,
        train_path=train_path,
        test_path=test_path,
        data_patch={
            "source": "supabase_storage",
            "storage_train_base_url": base_url,
            "storage_train_parts": ["part-2024-01.csv"],
            "storage_cache_dir": str(cache_dir),
            "storage_cache_parquet_name": "train_merged.parquet",
            "supabase_url": "https://example.supabase.co",
            "supabase_key": "sb_test",
            "supabase_schema": "hackathon",
            "supabase_train_table": "hackathon-train-data",
            "supabase_test_table": "hackathon-test-data",
        },
    )
    cfg = load_model_config(str(cfg_path))

    df1 = load_train_from_storage_parts(cfg, force_refresh=False)
    assert len(df1) == 2
    assert storage_cache_exists(cfg) is True

    def _fake_get_fails(_url, timeout=0):
        raise RuntimeError("network fail during rebuild")

    monkeypatch.setattr("miner_model_energy.storage_train_io.requests.get", _fake_get_fails)
    # Force refresh rebuild -> should fall back to cached df1 without raising.
    df2 = load_train_from_storage_parts(cfg, force_refresh=True)
    assert df2.shape == df1.shape


def test_ask_model_type_preflight_accepts_exit(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "3")
    with pytest.raises(miner_module.PreflightExitRequested):
        miner_module._ask_model_type_preflight()


def test_ask_after_deploy_decline_accepts_exit(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "3")
    choice = miner_module._ask_after_deploy_decline()
    assert choice == "exit"


def test_run_preflight_returns_exit_mode(monkeypatch):
    def _raise_exit(*_args, **_kwargs):
        raise miner_module.PreflightExitRequested()

    monkeypatch.setattr(miner_module, "_ask_yes_no_preflight", _raise_exit)
    result = miner_module.run_preflight(model_params_path="unused.yaml", non_interactive=False)
    assert result.mode == "exit"


def test_run_preflight_deploy_yes_prompts_dataset_dump_and_passes_flag(monkeypatch):
    cfg = ModelConfig(data={"source": "csv"}, features={}, training={}, models={}, persistence={})
    fake_result = object()
    captured = {}
    prompts: list[str] = []
    yes_no_answers = iter([False, True, True])

    def _fake_yes_no(prompt: str, default_yes: bool):
        prompts.append(prompt)
        return next(yes_no_answers)

    def _fake_persist(result, _cfg, run_id=None, dump_full_training_dataset=False):
        captured["result"] = result
        captured["run_id"] = run_id
        captured["dump"] = dump_full_training_dataset
        return {"artifact_dir": "fake_artifacts", "manifest_path": "fake_manifest", "model_path": "fake_model"}

    monkeypatch.setattr(miner_module, "_ask_yes_no_preflight", _fake_yes_no)
    monkeypatch.setattr(miner_module, "_ask_model_type_preflight", lambda: "linear")
    monkeypatch.setattr(miner_module, "_print_ml_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(miner_module, "load_model_config", lambda _path: cfg)
    monkeypatch.setattr(miner_module, "train_model", lambda _model, _cfg: fake_result)
    monkeypatch.setattr(miner_module, "persist_training_result", _fake_persist)

    out = miner_module.run_preflight(model_params_path="unused.yaml", non_interactive=False)

    assert out.mode == "advanced:linear"
    assert out.training_result is fake_result
    assert captured["result"] is fake_result
    assert captured["run_id"] == "miner"
    assert captured["dump"] is True
    assert "Deploy this trained model?" in prompts
    assert "Dump full training dataset with engineered features?" in prompts


def test_run_preflight_deploy_no_skips_dataset_dump_prompt(monkeypatch):
    cfg = ModelConfig(data={"source": "csv"}, features={}, training={}, models={}, persistence={})
    fake_result = object()
    prompts: list[str] = []
    yes_no_answers = iter([False, False])

    def _fake_yes_no(prompt: str, default_yes: bool):
        prompts.append(prompt)
        return next(yes_no_answers)

    monkeypatch.setattr(miner_module, "_ask_yes_no_preflight", _fake_yes_no)
    monkeypatch.setattr(miner_module, "_ask_model_type_preflight", lambda: "linear")
    monkeypatch.setattr(miner_module, "_print_ml_report", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(miner_module, "load_model_config", lambda _path: cfg)
    monkeypatch.setattr(miner_module, "train_model", lambda _model, _cfg: fake_result)
    monkeypatch.setattr(
        miner_module,
        "persist_training_result",
        lambda *_args, **_kwargs: {
            "artifact_dir": "fake_artifacts",
            "manifest_path": "fake_manifest",
            "model_path": "fake_model",
        },
    )
    monkeypatch.setattr(miner_module, "_ask_after_deploy_decline", lambda: "baseline")

    out = miner_module.run_preflight(model_params_path="unused.yaml", non_interactive=False)

    assert out.mode == "baseline"
    assert "Deploy this trained model?" in prompts
    assert "Dump full training dataset with engineered features?" not in prompts


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

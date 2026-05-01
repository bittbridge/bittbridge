from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional

from bittbridge.utils.iso_ne_api import fetch_fiveminute_system_load
from bittbridge.utils.timestamp import get_now

from .custom_plugin_runtime import CustomModelWrapper
from .ml_config import ModelConfig
from .pipeline import (
    TrainingResult,
    live_probe_feature_matrix_for_custom,
    predict_for_timestamp_with_context,
    predict_single_test_row_with_context,
)


def _get_latest_load_values(n_steps: int) -> Optional[list]:
    now = get_now()
    today = now.strftime("%Y%m%d")
    data = fetch_fiveminute_system_load(today, use_cache=False)
    if now.hour < 1 and now.minute < 30:
        yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")
        data = fetch_fiveminute_system_load(yesterday, use_cache=False) + data
    if not data:
        return None
    load_values = [load_mw for _, load_mw in data]
    return load_values[-n_steps:] if len(load_values) >= n_steps else None


class BaselineMovingAveragePredictor:
    def __init__(self, n_steps: int = 12):
        self.n_steps = n_steps
        self.last_prediction_context: dict[str, Any] = {}

    def predict(self, timestamp: str) -> Optional[float]:
        del timestamp
        values = _get_latest_load_values(self.n_steps)
        if not values:
            self.last_prediction_context = {}
            return None
        self.last_prediction_context = {
            "source": "iso_ne_api",
            "n_steps": self.n_steps,
            "model_input_row": {"recent_load_values": values},
        }
        return float(sum(values) / len(values))


@dataclass
class AdvancedModelPredictor:
    result: TrainingResult
    last_prediction_context: dict[str, Any] = None

    def predict(self, timestamp: str) -> Optional[float]:
        del timestamp
        pred, ctx = predict_single_test_row_with_context(self.result)
        self.last_prediction_context = ctx
        return pred


@dataclass
class SupabaseLiveAdvancedPredictor:
    result: TrainingResult
    config: ModelConfig
    last_prediction_context: dict[str, Any] = None

    def predict(self, timestamp: str) -> Optional[float]:
        pred, ctx = predict_for_timestamp_with_context(self.result, self.config, timestamp)
        self.last_prediction_context = ctx
        return pred


@dataclass
class CustomModelPredictor:
    """
    User-provided sklearn or Keras regression model; features match plugin feature_contract.json.
    """

    wrapper: CustomModelWrapper
    config: ModelConfig
    features: list[str]
    sequence_n_steps: int | None = None
    last_prediction_context: dict[str, Any] = None

    def predict(self, timestamp: str) -> Optional[float]:
        X, ctx = live_probe_feature_matrix_for_custom(
            self.config,
            timestamp,
            self.features,
            self.sequence_n_steps,
        )
        vals = self.wrapper.predict_values(X)
        pred = float(vals.ravel()[0])
        self.last_prediction_context = {
            **ctx,
            "custom_model_input_shape": list(X.shape),
            "custom_model_kind": self.wrapper.kind,
        }
        return pred


class PredictorRouter:
    def __init__(self, predictor):
        self._predictor = predictor
        self.mode = "baseline"
        self.last_prediction_context: dict[str, Any] = {}

    def set_predictor(self, predictor, mode: str):
        self._predictor = predictor
        self.mode = mode
        self.last_prediction_context = {}

    def predict(self, timestamp: str) -> Optional[float]:
        pred = self._predictor.predict(timestamp)
        self.last_prediction_context = getattr(self._predictor, "last_prediction_context", {}) or {}
        return pred


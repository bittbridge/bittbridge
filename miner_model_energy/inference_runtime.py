from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from bittbridge.utils.iso_ne_api import fetch_fiveminute_system_load
from bittbridge.utils.timestamp import get_now

from .ml_config import ModelConfig
from .pipeline import TrainingResult, predict_for_timestamp, predict_single_test_row


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

    def predict(self, timestamp: str) -> Optional[float]:
        del timestamp
        values = _get_latest_load_values(self.n_steps)
        if not values:
            return None
        return float(sum(values) / len(values))


@dataclass
class AdvancedModelPredictor:
    result: TrainingResult

    def predict(self, timestamp: str) -> Optional[float]:
        del timestamp
        return predict_single_test_row(self.result)


@dataclass
class SupabaseLiveAdvancedPredictor:
    result: TrainingResult
    config: ModelConfig

    def predict(self, timestamp: str) -> Optional[float]:
        return predict_for_timestamp(self.result, self.config, timestamp)


class PredictorRouter:
    def __init__(self, predictor):
        self._predictor = predictor
        self.mode = "baseline"

    def set_predictor(self, predictor, mode: str):
        self._predictor = predictor
        self.mode = mode

    def predict(self, timestamp: str) -> Optional[float]:
        return self._predictor.predict(timestamp)


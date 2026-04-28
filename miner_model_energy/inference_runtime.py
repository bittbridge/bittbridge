from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from .ml_config import ModelConfig
from .pipeline import TrainingResult, predict_for_timestamp, predict_single_test_row

try:
    import bittensor as bt
except ModuleNotFoundError:  # pragma: no cover - local ML tests can run without miner deps.
    class _NoopLogging:
        def __getattr__(self, _name):
            def _log(*_args, **_kwargs):
                return None

            return _log

    class _BittensorShim:
        logging = _NoopLogging()

    bt = _BittensorShim()

DEFAULT_FALLBACK_LOAD_MW = 15000.0


def _get_latest_load_values(n_steps: int) -> Optional[list]:
    try:
        from bittbridge.utils.iso_ne_api import fetch_fiveminute_system_load
        from bittbridge.utils.timestamp import get_now
    except ModuleNotFoundError:
        return None

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
    def __init__(self, predictor, fallback_predictor=None, default_prediction: float = DEFAULT_FALLBACK_LOAD_MW):
        self._predictor = predictor
        self._fallback_predictor = fallback_predictor or BaselineMovingAveragePredictor()
        self._default_prediction = float(default_prediction)
        self.mode = "baseline"

    def set_predictor(self, predictor, mode: str):
        self._predictor = predictor
        self.mode = mode

    def predict(self, timestamp: str) -> Optional[float]:
        try:
            value = self._predictor.predict(timestamp)
            if value is not None:
                return float(value)
            bt.logging.warning(
                f"[{self.mode}] Predictor returned no value for timestamp={timestamp}; using fallback."
            )
        except Exception as exc:
            bt.logging.error(
                f"[{self.mode}] Predictor failed for timestamp={timestamp}; using fallback: {exc}"
            )

        try:
            fallback_value = self._fallback_predictor.predict(timestamp)
            if fallback_value is not None:
                return float(fallback_value)
            bt.logging.warning(
                f"Fallback predictor returned no value for timestamp={timestamp}; "
                f"using default {self._default_prediction:.1f}."
            )
        except Exception as exc:
            bt.logging.error(
                f"Fallback predictor failed for timestamp={timestamp}; "
                f"using default {self._default_prediction:.1f}: {exc}"
            )
        return self._default_prediction


import os
import threading
import time
import pickle
from typing import Tuple, Optional

import numpy as np


class ModelService:
    def __init__(self, artifact_path: str, backend: str = "sklearn", refresh_on_start: bool = True):
        self.artifact_path = artifact_path
        self.backend = backend
        self.refresh_on_start = refresh_on_start
        self._model = None
        self._model_mtime: Optional[float] = None
        self._lock = threading.RLock()

    def load(self):
        if not os.path.exists(self.artifact_path):
            raise FileNotFoundError(f"Model artifact not found at {self.artifact_path}")
        with open(self.artifact_path, "rb") as f:
            model = pickle.load(f)
        with self._lock:
            self._model = model
            self._model_mtime = os.path.getmtime(self.artifact_path)

    def maybe_update(self):
        if not self.refresh_on_start:
            return
        if not os.path.exists(self.artifact_path):
            return
        current_mtime = os.path.getmtime(self.artifact_path)
        with self._lock:
            if self._model is None or self._model_mtime != current_mtime:
                self.load()

    def predict_with_interval(self, features: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not loaded")
            # Expect sklearn-like API
            y_hat = float(self._model.predict(features.reshape(1, -1))[0])
            if hasattr(self._model, "predict_interval"):
                lo, hi = self._model.predict_interval(features.reshape(1, -1))
                return y_hat, (float(lo), float(hi))
            if hasattr(self._model, "predict_std"):
                std = float(self._model.predict_std(features.reshape(1, -1))[0])
                z = 1.64
                return y_hat, (y_hat - z * std, y_hat + z * std)
            # If the trained baseline encodes an interval attribute
            if hasattr(self._model, "interval_width"):
                w = float(getattr(self._model, "interval_width"))
                return y_hat, (y_hat - w, y_hat + w)
        # If the model does not provide an interval, raise to signal caller
        raise RuntimeError("Model does not provide interval; please implement it in your model")



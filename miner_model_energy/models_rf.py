from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class RandomForestBundle:
    model: RandomForestRegressor
    features: List[str]


def train_rf(X_train: np.ndarray, y_train: np.ndarray, features: List[str], cfg: Dict) -> RandomForestBundle:
    model = RandomForestRegressor(
        n_estimators=int(cfg.get("n_estimators", 200)),
        max_depth=None if cfg.get("max_depth") is None else int(cfg.get("max_depth", 12)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 5)),
        random_state=int(cfg.get("random_state", 42)),
        n_jobs=int(cfg.get("n_jobs", -1)),
    )
    try:
        model.fit(X_train, y_train)
    except PermissionError:
        if int(cfg.get("n_jobs", -1)) == 1:
            raise
        model.set_params(n_jobs=1)
        model.fit(X_train, y_train)
    return RandomForestBundle(model=model, features=features)


def predict_rf(bundle: RandomForestBundle, X: np.ndarray) -> np.ndarray:
    return bundle.model.predict(X)


def save_rf(bundle: RandomForestBundle, out_path: str) -> str:
    payload = {"model": bundle.model, "features": bundle.features}
    joblib.dump(payload, out_path)
    return out_path


def load_rf(path: str) -> RandomForestBundle:
    payload = joblib.load(path)
    return RandomForestBundle(model=payload["model"], features=payload["features"])

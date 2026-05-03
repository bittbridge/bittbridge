from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


@dataclass
class HistGradientBoostingBundle:
    model: HistGradientBoostingRegressor
    features: List[str]


def train_hgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    features: List[str],
    cfg: Dict,
) -> HistGradientBoostingBundle:
    model = HistGradientBoostingRegressor(
        max_iter=int(cfg.get("max_iter", 500)),
        learning_rate=float(cfg.get("learning_rate", 0.035)),
        max_leaf_nodes=int(cfg.get("max_leaf_nodes", 31)),
        l2_regularization=float(cfg.get("l2_regularization", 0.05)),
        random_state=int(cfg.get("random_state", 42)),
    )
    with threadpool_limits(limits=1):
        model.fit(X_train, y_train)
    return HistGradientBoostingBundle(model=model, features=features)


def predict_hgb(bundle: HistGradientBoostingBundle, X: np.ndarray) -> np.ndarray:
    return bundle.model.predict(X)


def save_hgb(bundle: HistGradientBoostingBundle, out_path: str) -> str:
    payload = {"model": bundle.model, "features": bundle.features}
    joblib.dump(payload, out_path)
    return out_path


def load_hgb(path: str) -> HistGradientBoostingBundle:
    payload = joblib.load(path)
    return HistGradientBoostingBundle(model=payload["model"], features=payload["features"])

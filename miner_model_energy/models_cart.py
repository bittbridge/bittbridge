from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor


@dataclass
class CartBundle:
    model: DecisionTreeRegressor
    features: List[str]


def train_cart(X_train: np.ndarray, y_train: np.ndarray, features: List[str], cfg: Dict) -> CartBundle:
    model = DecisionTreeRegressor(
        max_depth=cfg.get("max_depth", 6),
        min_samples_split=cfg.get("min_samples_split", 10),
        min_samples_leaf=cfg.get("min_samples_leaf", 5),
        random_state=42,
    )
    model.fit(X_train, y_train)
    return CartBundle(model=model, features=features)


def predict_cart(bundle: CartBundle, X: np.ndarray) -> np.ndarray:
    return bundle.model.predict(X)


def save_cart(bundle: CartBundle, out_path: str) -> str:
    payload = {"model": bundle.model, "features": bundle.features}
    joblib.dump(payload, out_path)
    return out_path


def load_cart(path: str) -> CartBundle:
    payload = joblib.load(path)
    return CartBundle(model=payload["model"], features=payload["features"])


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class LinearBundle:
    model: LinearRegression
    scaler: StandardScaler
    features: List[str]


def train_linear(X_train: np.ndarray, y_train: np.ndarray, features: List[str], cfg: Dict) -> LinearBundle:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LinearRegression(fit_intercept=bool(cfg.get("fit_intercept", True)))
    model.fit(X_scaled, y_train)
    return LinearBundle(model=model, scaler=scaler, features=features)


def predict_linear(bundle: LinearBundle, X: np.ndarray) -> np.ndarray:
    return bundle.model.predict(bundle.scaler.transform(X))


def save_linear(bundle: LinearBundle, out_path: str) -> str:
    payload = {
        "model": bundle.model,
        "scaler": bundle.scaler,
        "features": bundle.features,
    }
    joblib.dump(payload, out_path)
    return out_path


def load_linear(path: str) -> LinearBundle:
    payload = joblib.load(path)
    return LinearBundle(
        model=payload["model"], scaler=payload["scaler"], features=payload["features"]
    )


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class LstmBundle:
    model: object
    features: List[str]
    n_steps: int


def _require_keras():
    try:
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Dropout, LSTM
    except Exception as exc:
        raise RuntimeError(
            "LSTM requested but TensorFlow/Keras is unavailable. Install tensorflow."
        ) from exc
    return Sequential, LSTM, Dense, Dropout


def make_sequences(X: np.ndarray, y: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    for idx in range(n_steps, len(X)):
        X_seq.append(X[idx - n_steps : idx])
        y_seq.append(y[idx])
    return np.asarray(X_seq), np.asarray(y_seq)


def _set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    features: List[str],
    cfg: Dict,
    random_state: int = 42,
) -> LstmBundle:
    _set_random_seeds(int(random_state))
    Sequential, LSTM, Dense, Dropout = _require_keras()
    n_steps = int(cfg.get("n_steps", 12))
    X_seq, y_seq = make_sequences(X_train, y_train, n_steps=n_steps)
    if len(X_seq) == 0:
        raise ValueError("Not enough rows to train LSTM sequence model.")

    units = int(cfg.get("units", 32))
    epochs = int(cfg.get("epochs", 5))
    batch_size = int(cfg.get("batch_size", 64))
    dropout = float(cfg.get("dropout", 0.1))

    model = Sequential(
        [
            LSTM(units, input_shape=(n_steps, X_seq.shape[2])),
            Dropout(dropout),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=0)
    return LstmBundle(model=model, features=features, n_steps=n_steps)


def predict_lstm(bundle: LstmBundle, X: np.ndarray) -> np.ndarray:
    """X is (n_steps, n_features) or longer 2D; or already (1, n_steps, n_features)."""
    if X.ndim == 2:
        if len(X) < bundle.n_steps:
            raise ValueError(
                "Need at least n_steps rows to build LSTM inference sequence. "
                "For a single test row, build a window from train history + test row."
            )
        X = X[-bundle.n_steps :][np.newaxis, :, :]
    preds = bundle.model.predict(X, verbose=0)
    return preds.reshape(-1)


def save_lstm(bundle: LstmBundle, out_path: str) -> str:
    bundle.model.save(out_path)
    return out_path


def load_lstm(path: str, features: List[str], n_steps: int) -> LstmBundle:
    try:
        from tensorflow.keras.models import load_model
    except Exception as exc:
        raise RuntimeError("TensorFlow/Keras is required to load saved LSTM model.") from exc
    model = load_model(path)
    return LstmBundle(model=model, features=features, n_steps=n_steps)


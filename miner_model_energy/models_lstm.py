from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np


@dataclass
class LstmBundle:
    model: object
    features: List[str]
    n_steps: int
    scaler: Optional[object] = None  # sklearn StandardScaler when standardize_inputs is enabled


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
    fit_verbose: int = 0,
) -> LstmBundle:
    _set_random_seeds(int(random_state))
    Sequential, LSTM, Dense, Dropout = _require_keras()
    n_steps = int(cfg.get("n_steps", 12))
    standardize_inputs = bool(cfg.get("standardize_inputs", False))
    scaler: Optional[object] = None
    X_work = np.asarray(X_train, dtype=float)
    if standardize_inputs:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_work = scaler.fit_transform(X_work)
    X_seq, y_seq = make_sequences(X_work, y_train, n_steps=n_steps)
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
    v_fit = int(fit_verbose)
    if v_fit not in (0, 1, 2):
        v_fit = 0
    model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, verbose=v_fit)
    return LstmBundle(model=model, features=features, n_steps=n_steps, scaler=scaler)


def _apply_input_scaler(bundle: LstmBundle, X: np.ndarray) -> np.ndarray:
    if bundle.scaler is None:
        return np.asarray(X, dtype=float)
    Xf = np.asarray(X, dtype=float)
    if Xf.ndim == 2:
        return bundle.scaler.transform(Xf)
    if Xf.ndim == 3:
        n, t, f = Xf.shape
        flat = Xf.reshape(-1, f)
        scaled = bundle.scaler.transform(flat)
        return scaled.reshape(n, t, f)
    raise ValueError(f"LSTM input must be 2D or 3D; got shape {Xf.shape}")


def predict_lstm(bundle: LstmBundle, X: np.ndarray) -> np.ndarray:
    """X is (n_steps, n_features) or longer 2D; or already (1, n_steps, n_features)."""
    X = _apply_input_scaler(bundle, X)
    if X.ndim == 2:
        if len(X) < bundle.n_steps:
            raise ValueError(
                "Need at least n_steps rows to build LSTM inference sequence. "
                "For a single test row, build a window from train history + test row."
            )
        X = X[-bundle.n_steps :][np.newaxis, :, :]
    preds = bundle.model.predict(X, verbose=0)
    return preds.reshape(-1)


LSTM_SCALER_FILENAME = "lstm_input_scaler.joblib"


def save_lstm(bundle: LstmBundle, out_path: str) -> str:
    bundle.model.save(out_path)
    if bundle.scaler is not None:
        joblib.dump(bundle.scaler, Path(out_path).parent / LSTM_SCALER_FILENAME)
    return out_path


def load_lstm(
    path: str,
    features: List[str],
    n_steps: int,
    scaler_path: str | None = None,
) -> LstmBundle:
    try:
        from tensorflow.keras.models import load_model
    except Exception as exc:
        raise RuntimeError("TensorFlow/Keras is required to load saved LSTM model.") from exc
    model = load_model(path)
    scaler = None
    if scaler_path is not None:
        sp = Path(scaler_path)
        if not sp.is_file():
            raise FileNotFoundError(f"LSTM input scaler missing at {scaler_path}")
        scaler = joblib.load(sp)
    return LstmBundle(model=model, features=features, n_steps=n_steps, scaler=scaler)


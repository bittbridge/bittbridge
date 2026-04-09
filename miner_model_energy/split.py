from __future__ import annotations

from typing import Tuple

import pandas as pd


def temporal_train_val_split(
    frame: pd.DataFrame, validation_split: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ValueError("Cannot split empty frame.")
    if validation_split <= 0 or validation_split >= 1:
        raise ValueError("validation_split must be in (0, 1).")

    split_idx = int(len(frame) * (1.0 - validation_split))
    split_idx = max(1, min(split_idx, len(frame) - 1))
    train = frame.iloc[:split_idx].copy()
    val = frame.iloc[split_idx:].copy()
    return train, val


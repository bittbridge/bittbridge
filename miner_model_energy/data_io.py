from __future__ import annotations

from typing import Tuple

import pandas as pd


TARGET_COLUMN = "Total Load"
TIMESTAMP_COLUMN = "dt"


def load_train_test(train_csv: str, test_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    if TIMESTAMP_COLUMN not in train.columns:
        raise ValueError(f"Missing `{TIMESTAMP_COLUMN}` in train data.")
    if TARGET_COLUMN not in train.columns:
        raise ValueError(f"Missing `{TARGET_COLUMN}` in train data.")
    if TIMESTAMP_COLUMN not in test.columns:
        raise ValueError(f"Missing `{TIMESTAMP_COLUMN}` in test data.")

    train[TIMESTAMP_COLUMN] = pd.to_datetime(train[TIMESTAMP_COLUMN])
    test[TIMESTAMP_COLUMN] = pd.to_datetime(test[TIMESTAMP_COLUMN])

    train = train.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    test = test.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return train, test


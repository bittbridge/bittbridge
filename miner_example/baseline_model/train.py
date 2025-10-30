import argparse
import json
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["datetime"], utc=True)
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.weekday
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    return df


def train(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, skiprows=[1])
    df = make_features(df)

    # Target is next hour close (shift -1) to simulate a simple forecast
    df["target"] = df["close"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["hour", "hour_sin", "hour_cos", "dow", "dow_sin", "dow_cos"]
    X = df[feature_cols].values
    y = df["target"].values

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    # Store a simple fixed interval width based on residual std
    residuals = y - model.predict(X)
    interval_width = float(np.std(residuals) * 1.64)  # 90% band approx
    setattr(model, "interval_width", interval_width)

    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    schema = {
        "features": feature_cols,
        "target": "next_close",
        "interval": "fixed_width_derived_from_residuals",
    }
    with open(os.path.join(out_dir, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    train(args.csv, args.out_dir)



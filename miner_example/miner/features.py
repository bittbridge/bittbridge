import datetime as dt
import numpy as np


def features_from_timestamp(timestamp_str: str) -> np.ndarray:
    # Expect ISO-like string
    ts = dt.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
    hour = ts.hour
    dow = ts.weekday()
    # Cyclical encodings
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    return np.array([hour, hour_sin, hour_cos, dow, dow_sin, dow_cos], dtype=float)



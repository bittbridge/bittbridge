"""
Helper Functions for Student Models

These functions handle the standard 1-hour-ahead prediction pattern.
Students can use these instead of writing the predict function from scratch.
"""

import pandas as pd
import numpy as np


def get_recent_prices(data, timestamp, n_steps=12):
    """
    Get the last n_steps values before the given timestamp.
    
    Works with any DataFrame that has a datetime index and a target column
    ('close_price' or 'load_mw'). Supports both price and LoadMw (energy demand) data.
    
    Args:
        data: DataFrame with datetime index and 'close_price' or 'load_mw' column
        timestamp: ISO format timestamp string (e.g., "2024-01-15T10:30:00+00:00")
        n_steps: Number of timesteps to retrieve (default: 12 = 1 hour for 5-min data)
    
    Returns:
        Array of shape (1, n_steps, 1) ready for model input, or None if insufficient data
    
    Example:
        X = get_recent_prices(data, "2024-01-15T10:30:00+00:00", n_steps=12)
        if X is not None:
            prediction = model.predict(X)[0, 0]
    """
    target_time = pd.to_datetime(timestamp)
    available_data = data[data.index < target_time]
    
    if len(available_data) < n_steps:
        return None
    
    # Prefer load_mw for energy demand, fall back to close_price
    value_col = 'load_mw' if 'load_mw' in data.columns else 'close_price'
    recent_values = available_data[value_col].tail(n_steps).values
    return recent_values.reshape(1, n_steps, 1)


def calculate_interval(prediction, method='fixed', std_error=None, percentage=0.01):
    """
    Calculate 90% confidence interval for a prediction.
    
    Works for both price and LoadMw (load in MW) predictions.
    
    Args:
        prediction: The predicted value (float) - price or LoadMw
        method: 'fixed' (use percentage) or 'std' (use standard error)
        std_error: Standard error/standard deviation (for 'std' method)
        percentage: Percentage of prediction to use as margin (for 'fixed' method, default: 1%)
    
    Returns:
        List [lower_bound, upper_bound] for 90% confidence interval
    
    Example:
        # Using fixed percentage (works for LoadMw ~12000)
        interval = calculate_interval(12000.0, method='fixed', percentage=0.01)
    """
    z_score = 1.64  # Z-score for 90% confidence interval (two-tailed)
    
    if method == 'std' and std_error is not None:
        margin = z_score * std_error
    else:
        # Fixed percentage method
        margin = percentage * prediction
    
    lower = float(prediction - margin)
    upper = float(prediction + margin)
    
    return [lower, upper]


def prepare_dataframe(df, time_col=None, price_col=None):
    """
    Prepare a DataFrame for time series prediction.
    
    Automatically detects time and target columns. Supports both price data
    (Close, close, PRICE) and energy demand (Total Load, LoadMw, load_mw).
    Output uses 'load_mw' for energy data, 'close_price' for price data.
    
    Args:
        df: Raw DataFrame from CSV
        time_col: Name of time column (auto-detected if None)
        price_col: Name of target column (auto-detected if None)
    
    Returns:
        DataFrame with datetime index and 'load_mw' or 'close_price' column
    
    Example:
        df = pd.read_csv('energydata.csv')
        data = prepare_dataframe(df)  # Uses dt, Total Load
    """
    # Auto-detect time column
    if time_col is None:
        for col in ['timestamp_utc', 'timestamp_local', 'time', 'date', 'dt', df.columns[0]]:
            if col in df.columns:
                time_col = col
                break
    
    # Auto-detect target column: energy (LoadMw, Total Load) or price
    if price_col is None:
        for col in ['LoadMw', 'Total Load', 'load_mw', 'Close', 'close', 'PRICE', 'price']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None and len(df.columns) > 1:
            price_col = df.columns[1]
    
    if time_col is None or price_col is None:
        raise ValueError(f"Could not detect time/target columns. Available: {df.columns.tolist()}")
    
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    time_series_df = df[[time_col, price_col]].copy()
    # Use load_mw for energy data, close_price for price data
    target_name = 'load_mw' if price_col in ('LoadMw', 'Total Load', 'load_mw') else 'close_price'
    time_series_df.columns = ['datetime', target_name]
    time_series_df = time_series_df.set_index('datetime')
    time_series_df = time_series_df.sort_index()
    time_series_df = time_series_df.dropna()
    
    return time_series_df


def predict_1hour_ahead(model, data, timestamp, n_steps=12, interval_method='fixed', interval_std=None):
    """
    Standard 1-hour-ahead prediction function.
    
    This is the complete standard prediction function. Students can use this
    directly or customize it for their needs.
    
    Args:
        model: Trained model (must have .predict() method)
        data: DataFrame with datetime index and 'close_price' or 'load_mw' column
        timestamp: ISO format timestamp string
        n_steps: Number of timesteps to use (default: 12 = 1 hour for 5-min data)
        interval_method: 'fixed' or 'std' for confidence interval calculation
        interval_std: Standard error for 'std' method
    
    Returns:
        Tuple of (prediction, interval) or (None, None) if prediction fails
    
    Example:
        prediction, interval = predict_1hour_ahead(
            model, data, "2024-01-15T10:30:00+00:00",
            n_steps=12,
            interval_method='std',
            interval_std=0.002586
        )
    """
    # Get recent prices
    X = get_recent_prices(data, timestamp, n_steps=n_steps)
    
    if X is None:
        return None, None
    
    # Make prediction
    try:
        prediction = model.predict(X, verbose=0)[0, 0]
        prediction = float(prediction)
    except Exception:
        return None, None
    
    # Calculate interval
    interval = calculate_interval(
        prediction,
        method=interval_method,
        std_error=interval_std
    )
    
    return prediction, interval


import time

import bittensor as bt
import yfinance as yf

from snp_oracle.protocol import Challenge


def get_point_estimate() -> float:
    """Make a naive forecast of the next price by predicting the most recent price

    Args:
        None

    Returns:
        (float): The current S&P 500 price
    """

    # Query Yahoo Finance for the most recent price
    last_price: float = yf.Ticker("^GSPC").fast_info.last_price

    # Return the current price of S&P 500 as our point estimate
    return last_price


def get_direction() -> bool:
    """Make a naive forecast that the price will always go up

    Args:
        None

    Returns:
        (bool): True to predict that the S&P 500 will be higher in 30 minutes
    """

    return True


def forward(synapse: Challenge) -> Challenge:
    total_start_time = time.perf_counter()
    bt.logging.info(
        f"👈 Received prediction request from: {synapse.dendrite.hotkey} for timestamp: {synapse.timestamp}"
    )

    # Get the naive point estimate
    # Time the point estimate calculation
    point_estimate_start = time.perf_counter()
    point_estimate: float = get_point_estimate()
    point_estimate_time = time.perf_counter() - point_estimate_start
    bt.logging.debug(f"⏱️ Point estimate function took: {point_estimate_time:.3f} seconds")

    # Get the naive direction prediction
    # Time the direction prediction calculation
    direction_start = time.perf_counter()
    direction: bool = get_direction()
    direction_time = time.perf_counter() - direction_start
    bt.logging.debug(f"⏱️ Direction prediction function took: {direction_time:.3f} seconds")

    # Capture the predicted values
    synapse.prediction = point_estimate
    synapse.direction = direction

    # Capture the total time
    total_time = time.perf_counter() - total_start_time
    bt.logging.debug(f"⏱️ Total forward call took: {total_time:.3f} seconds")

    # Verify successful predictions by logging
    if synapse.prediction is not None:
        bt.logging.success(f"Predicted Price: {synapse.prediction} | Predicted Direction: {synapse.direction}")
    else:
        bt.logging.info("No prediction for this request.")

    return synapse

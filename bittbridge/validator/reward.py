# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import numpy as np
import bittensor as bt
from typing import List, Dict, Tuple, Optional

from bittbridge.protocol import Challenge
from bittbridge.utils.iso_ne_api import get_load_mw_for_timestamp


def get_actual_load_mw(timestamp: str) -> Optional[float]:
    """
    Fetches the actual LoadMw for the 5-minute slot matching the given timestamp
    from the ISO-NE fiveminutesystemload API.

    Args:
        timestamp: ISO format timestamp string (e.g. "2024-01-15T10:30:00.000Z")

    Returns:
        Actual LoadMw for that 5-min slot, or None if fetch fails
    """
    return get_load_mw_for_timestamp(timestamp)


def calculate_point_forecast_scores(actual_load_mw: float, predictions: List[float]) -> Dict[int, float]:
    """
    Calculate point forecast scores using incentive mechanism.
    Only miners with valid predictions receive rewards.

    Args:
        actual_load_mw: Ground truth LoadMw
        predictions: List of miner predictions

    Returns:
        Dict mapping miner index to share (0.9^rank)
    """
    if actual_load_mw is None or not predictions:
        return {}

    valid_predictions = []
    valid_indices = []

    for i, pred in enumerate(predictions):
        if pred is not None:
            valid_predictions.append(pred)
            valid_indices.append(i)

    if not valid_predictions:
        bt.logging.warning("No valid predictions found - all miners will receive zero reward")
        return {}

    errors = []
    for pred in valid_predictions:
        rel_error = abs(pred - actual_load_mw) / actual_load_mw
        errors.append(rel_error)

    ranked_valid_indices = sorted(range(len(errors)), key=lambda i: errors[i])

    shares = {}
    for rank, error_idx in enumerate(ranked_valid_indices):
        miner_idx = valid_indices[error_idx]
        shares[miner_idx] = 0.9 ** rank

    bt.logging.info(f"Point forecast scores: {shares}")
    return shares


def calculate_interval_forecast_scores(actual_load_mw: float, intervals: List[List[float]]) -> Dict[int, float]:
    """
    Calculate interval forecast scores using incentive mechanism.
    Only miners with valid intervals receive rewards.

    Args:
        actual_load_mw: Ground truth LoadMw
        intervals: List of [low, high] prediction intervals

    Returns:
        Dict mapping miner index to share (0.9^rank)
    """
    if actual_load_mw is None or not intervals:
        return {}

    valid_intervals = []
    valid_indices = []

    for i, interval in enumerate(intervals):
        if interval is not None and len(interval) == 2:
            valid_intervals.append(interval)
            valid_indices.append(i)

    if not valid_intervals:
        bt.logging.warning("No valid intervals found - all miners will receive zero interval reward")
        return {}

    scores = []
    for interval in valid_intervals:
        low, high = interval

        inclusion = 1.0 if (low <= actual_load_mw <= high) else 0.0

        width = high - low
        # Scale-invariant: use relative width for MW-scale intervals
        relative_width = width / max(actual_load_mw, 1)
        width_factor = 1 / (1 + relative_width)

        interval_score = width_factor * inclusion
        scores.append(interval_score)

    ranked_valid_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    shares = {}
    for rank, score_idx in enumerate(ranked_valid_indices):
        miner_idx = valid_indices[score_idx]
        shares[miner_idx] = 0.9 ** rank

    bt.logging.info(f"Interval forecast scores: {shares}")
    return shares


def calculate_combined_shares(point_shares: Dict[int, float], interval_shares: Dict[int, float]) -> Dict[int, float]:
    all_miners = set(point_shares.keys()) | set(interval_shares.keys())
    combined_shares = {}

    for miner_idx in all_miners:
        point_share = point_shares.get(miner_idx, 0.0)
        interval_share = interval_shares.get(miner_idx, 0.0)
        combined_shares[miner_idx] = (point_share + interval_share) / 2

    bt.logging.info(f"Combined shares: {combined_shares}")
    return combined_shares


def apply_exponential_moving_average(
    current_shares: Dict[int, float],
    previous_weights: Dict[int, float],
    alpha: float = 0.00958
) -> Dict[int, float]:
    final_weights = {}

    for miner_idx in current_shares.keys():
        current_share = current_shares[miner_idx]
        prev_weight = previous_weights.get(miner_idx, 0.0)
        final_weights[miner_idx] = alpha * current_share + (1 - alpha) * prev_weight

    for miner_idx in previous_weights.keys():
        if miner_idx not in current_shares:
            prev_weight = previous_weights[miner_idx]
            final_weights[miner_idx] = (1 - alpha) * prev_weight

    bt.logging.info(f"Final EMA weights: {final_weights}")
    return final_weights


def get_incentive_mechanism_rewards(
    actual_load_mw: float,
    responses: List[Challenge],
    previous_weights: Optional[Dict[int, float]] = None,
    alpha: float = 0.00958
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Generate rewards using the complete incentive mechanism.
    Only miners with valid submissions receive rewards.

    Args:
        actual_load_mw: Ground truth LoadMw
        responses: List of Challenge synapse responses from miners
        previous_weights: Previous epoch weights for EMA
        alpha: EMA smoothing factor

    Returns:
        Tuple of (reward_array, updated_weights)
    """
    if actual_load_mw is None:
        return np.zeros(len(responses)), {}

    predictions = [r.prediction for r in responses]
    intervals = [r.interval for r in responses]

    point_shares = calculate_point_forecast_scores(actual_load_mw, predictions)
    interval_shares = calculate_interval_forecast_scores(actual_load_mw, intervals)
    combined_shares = calculate_combined_shares(point_shares, interval_shares)

    if previous_weights is not None:
        final_weights = apply_exponential_moving_average(combined_shares, previous_weights, alpha)
    else:
        final_weights = combined_shares

    rewards = np.array([final_weights.get(i, 0.0) for i in range(len(responses))])

    active_miners = len([r for r in rewards if r > 0])
    bt.logging.info(f"Reward distribution: {active_miners}/{len(responses)} miners received rewards")

    return rewards, final_weights


def reward(actual_load_mw: float, predicted_load_mw: float) -> float:
    """
    Legacy reward function for backward compatibility.
    Now uses the incentive mechanism internally.
    """
    if actual_load_mw is None or predicted_load_mw is None:
        return 0.0

    point_shares = calculate_point_forecast_scores(actual_load_mw, [predicted_load_mw])
    return point_shares.get(0, 0.0)

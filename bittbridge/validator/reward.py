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

"""
Incentive mechanism (see docs/guide/incentive-mechanism.md):
- Absolute percentage error: error_i = |prediction_i - actual| / |actual|
- Score: score_i = exp(-error_i / T), T = median baseline error (calibrated)
- Per-round weights: weight_i = score_i / sum_j score_j
"""

import math
import numpy as np
import bittensor as bt
from typing import List, Dict, Tuple, Optional

from bittbridge.protocol import Challenge
from bittbridge.utils.iso_ne_api import get_load_mw_for_timestamp

# Median percentage error of the 5-hour MA baseline (~9.76% as a fraction). See docs/guide/incentive-mechanism.md
INCENTIVE_T = 0.097634


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
    Point-forecast weights: score_i = exp(-error_i / T), then normalize to sum to 1.
    Only miners with valid predictions participate.

    Args:
        actual_load_mw: Ground truth LoadMw
        predictions: List of miner predictions (None = invalid)

    Returns:
        Dict mapping response index to normalized weight
    """
    if actual_load_mw is None or not predictions:
        return {}

    if abs(actual_load_mw) < 1e-12:
        bt.logging.warning(
            "actual_load_mw is zero or near-zero - cannot compute percentage error; skipping round"
        )
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

    raw_scores = []
    for pred in valid_predictions:
        error_i = abs(pred - actual_load_mw) / abs(actual_load_mw)
        raw_scores.append(math.exp(-error_i / INCENTIVE_T))

    total = sum(raw_scores)
    if total <= 0:
        return {}

    weights = {}
    for k, miner_idx in enumerate(valid_indices):
        weights[miner_idx] = raw_scores[k] / total

    bt.logging.info(f"Point forecast normalized weights: {weights}")
    return weights


def get_incentive_mechanism_rewards(
    actual_load_mw: float,
    responses: List[Challenge],
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Per-round rewards from the documented incentive mechanism (no EMA).

    Returns:
        Tuple of (reward_array aligned with responses, dict index -> weight for logging)
    """
    if actual_load_mw is None:
        return np.zeros(len(responses)), {}

    predictions = [r.prediction for r in responses]
    final_weights = calculate_point_forecast_scores(actual_load_mw, predictions)

    rewards = np.array([final_weights.get(i, 0.0) for i in range(len(responses))])

    active_miners = int(np.sum(rewards > 0))
    bt.logging.info(f"Reward distribution: {active_miners}/{len(responses)} miners received rewards")

    return rewards, final_weights

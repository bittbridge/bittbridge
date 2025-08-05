# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import requests
import numpy as np
import bittensor as bt
from typing import List

from bittbridge.protocol import Challenge


def get_actual_usdt_cny() -> float:
    try:
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=cny")
        response.raise_for_status()
        return response.json()["tether"]["cny"]
    except Exception as e:
        bt.logging.warning(f"Failed to fetch ground truth price: {e}")
        return None


def reward(actual_price: float, predicted_price: float) -> float:
    """
    Computes reward based on closeness of predicted price to actual price.

    Args:
        actual_price: Current ground truth USDT/CNY price
        predicted_price: Miner’s prediction

    Returns:
        float: Reward value (closer = higher, max 1.0)
    """
    error = abs(predicted_price - actual_price)
    reward_val = max(0.0, 1.0 - error / actual_price)  # Linear inverse error
    bt.logging.info(f"Prediction: {predicted_price}, Actual: {actual_price}, Error: {error}, Reward: {reward_val:.4f}")
    return reward_val


def get_rewards(self, query, responses: List[Challenge]) -> np.ndarray:
    """
    Generate a reward for each miner response to the Challenge synapse.

    Args:
        query: Placeholder for future use (e.g., timestamp)
        responses: List of Challenge synapse responses from miners

    Returns:
        np.ndarray of reward floats
    """
    actual_price = get_actual_usdt_cny()
    if actual_price is None:
        return np.zeros(len(responses))  # No rewards if we can’t validate

    return np.array([
        reward(actual_price, synapse.prediction) if synapse.prediction is not None else 0.0
        for synapse in responses
    ])
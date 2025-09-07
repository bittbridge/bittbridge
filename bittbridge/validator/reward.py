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
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import requests
import numpy as np
import bittensor as bt
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone

from bittbridge.protocol import Challenge


def get_actual_usdt_cny() -> float:
    """
    Fetches the current USDT/CNY price from CoinGecko API.
    
    Returns:
        float: Current USDT/CNY price or None if fetch fails
    """
    try:
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=cny&precision=4&x_cg_demo_api_key=CG-1UpuR3vjuAqQWJTQo3EPdUmR")
        response.raise_for_status()
        return response.json()["tether"]["cny"]
    except Exception as e:
        bt.logging.warning(f"Failed to fetch ground truth price: {e}")
        return None


def calculate_point_forecast_scores(actual_price: float, predictions: List[float]) -> Dict[int, float]:
    """
    Calculate point forecast scores using Precog methodology.
    
    Args:
        actual_price: Ground truth USDT/CNY price
        predictions: List of miner predictions
        
    Returns:
        Dict mapping miner index to share (0.9^rank)
    """
    if actual_price is None or not predictions:
        return {}
    
    # Calculate relative errors
    errors = []
    for pred in predictions:
        if pred is not None:
            rel_error = abs(pred - actual_price) / actual_price
            errors.append(rel_error)
        else:
            errors.append(float('inf'))  # Invalid prediction gets worst rank
    
    # Rank miners by error (lower error = better rank)
    ranked_indices = sorted(range(len(errors)), key=lambda i: errors[i])
    
    # Assign shares using 0.9^rank formula
    shares = {}
    for rank, miner_idx in enumerate(ranked_indices):
        shares[miner_idx] = 0.9 ** rank
    
    bt.logging.info(f"Point forecast scores: {shares}")
    return shares


def calculate_interval_forecast_scores(actual_price: float, intervals: List[List[float]]) -> Dict[int, float]:
    """
    Calculate interval forecast scores using Precog methodology.
    
    Args:
        actual_price: Ground truth USDT/CNY price
        intervals: List of [low, high] prediction intervals
        
    Returns:
        Dict mapping miner index to share (0.9^rank)
    """
    if actual_price is None or not intervals:
        return {}
    
    scores = []
    for i, interval in enumerate(intervals):
        if interval is None or len(interval) != 2:
            scores.append(0.0)  # Invalid interval gets worst score
            continue
            
        low, high = interval
        
        # Inclusion factor (1 if actual price is within interval, else 0)
        inclusion = 1.0 if (low <= actual_price <= high) else 0.0
        
        # Width factor (smaller intervals are better)
        width = high - low
        width_factor = 1 / (1 + width * 1000)  # Scale so narrower = closer to 1
        
        # Final interval score
        interval_score = width_factor * inclusion
        scores.append(interval_score)
    
    # Rank miners by score (higher score = better rank)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    # Assign shares using 0.9^rank formula
    shares = {}
    for rank, miner_idx in enumerate(ranked_indices):
        shares[miner_idx] = 0.9 ** rank
    
    bt.logging.info(f"Interval forecast scores: {shares}")
    return shares


def calculate_combined_shares(point_shares: Dict[int, float], interval_shares: Dict[int, float]) -> Dict[int, float]:
    """
    Combine point and interval shares using Precog methodology.
    
    Args:
        point_shares: Point forecast shares
        interval_shares: Interval forecast shares
        
    Returns:
        Dict mapping miner index to combined share
    """
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
    """
    Apply exponential moving average to smooth weights over time.
    
    Args:
        current_shares: Current epoch combined shares
        previous_weights: Previous epoch weights
        alpha: EMA smoothing factor (default from Precog methodology)
        
    Returns:
        Dict mapping miner index to final weight
    """
    all_miners = set(current_shares.keys()) | set(previous_weights.keys())
    final_weights = {}
    
    for miner_idx in all_miners:
        current_share = current_shares.get(miner_idx, 0.0)
        prev_weight = previous_weights.get(miner_idx, 0.5)  # Default to 0.5 for new miners
        final_weights[miner_idx] = alpha * current_share + (1 - alpha) * prev_weight
    
    bt.logging.info(f"Final EMA weights: {final_weights}")
    return final_weights


def get_precog_rewards(
    actual_price: float, 
    responses: List[Challenge], 
    previous_weights: Optional[Dict[int, float]] = None,
    alpha: float = 0.00958
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Generate rewards using the complete Precog methodology.
    
    Args:
        actual_price: Ground truth USDT/CNY price
        responses: List of Challenge synapse responses from miners
        previous_weights: Previous epoch weights for EMA
        alpha: EMA smoothing factor
        
    Returns:
        Tuple of (reward_array, updated_weights)
    """
    if actual_price is None:
        return np.zeros(len(responses)), {}
    
    # Extract predictions and intervals
    predictions = [r.prediction for r in responses]
    intervals = [r.interval for r in responses]
    
    # Calculate point forecast scores
    point_shares = calculate_point_forecast_scores(actual_price, predictions)
    
    # Calculate interval forecast scores
    interval_shares = calculate_interval_forecast_scores(actual_price, intervals)
    
    # Combine shares
    combined_shares = calculate_combined_shares(point_shares, interval_shares)
    
    # Apply EMA if previous weights are provided
    if previous_weights is not None:
        final_weights = apply_exponential_moving_average(combined_shares, previous_weights, alpha)
    else:
        final_weights = combined_shares
    
    # Convert to numpy array for compatibility
    rewards = np.array([final_weights.get(i, 0.0) for i in range(len(responses))])
    
    return rewards, final_weights


# Legacy function for backward compatibility
def reward(actual_price: float, predicted_price: float) -> float:
    """
    Legacy reward function for backward compatibility.
    Now uses the Precog methodology internally.
    """
    if actual_price is None or predicted_price is None:
        return 0.0
    
    # Use point forecast scoring for single prediction
    point_shares = calculate_point_forecast_scores(actual_price, [predicted_price])
    return point_shares.get(0, 0.0)
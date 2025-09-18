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

import os
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
    coingecko_api_key = os.getenv("COINGECKO_API_KEY")
    if coingecko_api_key is None:
        bt.logging.error("COINGECKO_API_KEY not found in environment variables.")
        return None
    
    try:
        response = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=cny&precision=4&x_cg_demo_api_key={coingecko_api_key}")
        response.raise_for_status()
        return response.json()["tether"]["cny"]
    except Exception as e:
        bt.logging.warning(f"Failed to fetch ground truth price: {e}")
        return None


def calculate_point_forecast_scores(actual_price: float, predictions: List[float]) -> Dict[int, float]:
    """
    Calculate point forecast scores using incentive mechanism.
    Only miners with valid predictions receive rewards.
    
    Args:
        actual_price: Ground truth USDT/CNY price
        predictions: List of miner predictions
        
    Returns:
        Dict mapping miner index to share (0.9^rank)
    """
    if actual_price is None or not predictions:
        return {}
    
    # Filter out miners with no valid predictions
    valid_predictions = []
    valid_indices = []
    
    for i, pred in enumerate(predictions):
        if pred is not None:
            valid_predictions.append(pred)
            valid_indices.append(i)
    
    # If no valid predictions, return empty dict
    if not valid_predictions:
        bt.logging.warning("No valid predictions found - all miners will receive zero reward")
        return {}
    
    # Calculate relative errors for valid predictions only
    errors = []
    for pred in valid_predictions:
        rel_error = abs(pred - actual_price) / actual_price
        errors.append(rel_error)
    
    # Rank miners by error (lower error = better rank)
    ranked_valid_indices = sorted(range(len(errors)), key=lambda i: errors[i])
    
    # Assign shares using 0.9^rank formula - only to miners with valid predictions
    shares = {}
    for rank, error_idx in enumerate(ranked_valid_indices):
        miner_idx = valid_indices[error_idx]
        shares[miner_idx] = 0.9 ** rank
    
    bt.logging.info(f"Point forecast scores: {shares}")
    return shares


def calculate_interval_forecast_scores(actual_price: float, intervals: List[List[float]]) -> Dict[int, float]:
    """
    Calculate interval forecast scores using incentive mechanism.
    Only miners with valid intervals receive rewards.
    
    Args:
        actual_price: Ground truth USDT/CNY price
        intervals: List of [low, high] prediction intervals
        
    Returns:
        Dict mapping miner index to share (0.9^rank)
    """
    if actual_price is None or not intervals:
        return {}
    
    # Filter out miners with no valid intervals
    valid_intervals = []
    valid_indices = []
    
    for i, interval in enumerate(intervals):
        if interval is not None and len(interval) == 2:
            valid_intervals.append(interval)
            valid_indices.append(i)
    
    # If no valid intervals, return empty dict
    if not valid_intervals:
        bt.logging.warning("No valid intervals found - all miners will receive zero interval reward")
        return {}
    
    # Calculate scores for valid intervals only
    scores = []
    for interval in valid_intervals:
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
    ranked_valid_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    # Assign shares using 0.9^rank formula - only to miners with valid intervals
    shares = {}
    for rank, score_idx in enumerate(ranked_valid_indices):
        miner_idx = valid_indices[score_idx]
        shares[miner_idx] = 0.9 ** rank
    
    bt.logging.info(f"Interval forecast scores: {shares}")
    return shares


def calculate_combined_shares(point_shares: Dict[int, float], interval_shares: Dict[int, float]) -> Dict[int, float]:
    """
    Combine point and interval shares using incentive mechanism.
    
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
    Only miners with current submissions get updated weights.
    
    Args:
        current_shares: Current epoch combined shares (only includes miners who submitted)
        previous_weights: Previous epoch weights
        alpha: EMA smoothing factor (default from incentive mechanism)
        
    Returns:
        Dict mapping miner index to final weight
    """
    final_weights = {}
    
    # Only update weights for miners who submitted in current epoch
    for miner_idx in current_shares.keys():
        current_share = current_shares[miner_idx]
        prev_weight = previous_weights.get(miner_idx, 0.0)  # Default to 0.0 for new miners (no free rewards)
        final_weights[miner_idx] = alpha * current_share + (1 - alpha) * prev_weight
    
    # Miners who didn't submit in current epoch keep their previous weights (or get 0.0 if new)
    for miner_idx in previous_weights.keys():
        if miner_idx not in current_shares:
            # Miner didn't submit this epoch - they keep previous weight but it decays over time
            prev_weight = previous_weights[miner_idx]
            final_weights[miner_idx] = (1 - alpha) * prev_weight  # Decay inactive miners
    
    bt.logging.info(f"Final EMA weights: {final_weights}")
    return final_weights


def get_incentive_mechanism_rewards(
    actual_price: float, 
    responses: List[Challenge], 
    previous_weights: Optional[Dict[int, float]] = None,
    alpha: float = 0.00958
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Generate rewards using the complete incentive mechanism.
    Only miners with valid submissions receive rewards.
    
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
    
    # Calculate point forecast scores (only for miners with valid predictions)
    point_shares = calculate_point_forecast_scores(actual_price, predictions)
    
    # Calculate interval forecast scores (only for miners with valid intervals)
    interval_shares = calculate_interval_forecast_scores(actual_price, intervals)
    
    # Combine shares (only miners with valid submissions will have shares)
    combined_shares = calculate_combined_shares(point_shares, interval_shares)
    
    # Apply EMA if previous weights are provided
    if previous_weights is not None:
        final_weights = apply_exponential_moving_average(combined_shares, previous_weights, alpha)
    else:
        final_weights = combined_shares
    
    # Convert to numpy array for compatibility
    # Miners with no valid submissions get 0.0 reward
    rewards = np.array([final_weights.get(i, 0.0) for i in range(len(responses))])
    
    # Log reward distribution
    active_miners = len([r for r in rewards if r > 0])
    bt.logging.info(f"Reward distribution: {active_miners}/{len(responses)} miners received rewards")
    
    return rewards, final_weights


# Legacy function for backward compatibility
def reward(actual_price: float, predicted_price: float) -> float:
    """
    Legacy reward function for backward compatibility.
    Now uses the incentive mechanism internally.
    """
    if actual_price is None or predicted_price is None:
        return 0.0
    
    # Use point forecast scoring for single prediction
    point_shares = calculate_point_forecast_scores(actual_price, [predicted_price])
    return point_shares.get(0, 0.0)
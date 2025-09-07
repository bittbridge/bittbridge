# Precog Methodology Implementation

This document describes the implementation of the [Coin Metrics Precog Methodology](https://docs.coinmetrics.io/bittensor/precog-methodology) in the Bittbridge validator.

## Overview

The Precog methodology implements a sophisticated scoring system for cryptocurrency price prediction that evaluates both point forecasts and interval forecasts, then combines them using an exponential moving average (EMA) to reward consistent performance over time.

## Key Components

### 1. Point Forecast Evaluation

**Purpose**: Evaluates how close miners' point predictions are to the actual USDT/CNY price.

**Methodology**:
- Calculate relative error: `|prediction - actual| / actual`
- Rank miners by error (lower error = better rank)
- Assign shares using formula: `share = 0.9^rank`
- Rank 0 (best) gets share of 1.0, Rank 1 gets 0.9, Rank 2 gets 0.81, etc.

### 2. Interval Forecast Evaluation

**Purpose**: Evaluates how well miners' prediction intervals capture the actual price movement.

**Methodology**:
- **Inclusion Factor**: 1.0 if actual price falls within predicted interval, 0.0 otherwise
- **Width Factor**: `1 / (1 + width * 1000)` where width = high - low
- **Final Score**: `Inclusion Factor × Width Factor`
- Rank miners by score (higher score = better rank)
- Assign shares using formula: `share = 0.9^rank`

### 3. Combined Scoring

**Purpose**: Combines point and interval forecast performance.

**Methodology**:
- Calculate combined share: `(point_share + interval_share) / 2`
- This gives equal weight to both prediction types

### 4. Exponential Moving Average (EMA)

**Purpose**: Smooths rewards over time to reward consistent performance.

**Methodology**:
- Formula: `new_weight = α × current_share + (1-α) × previous_weight`
- Default α = 0.00958 (from Precog methodology)
- Rewards miners who maintain good performance across multiple epochs

## Implementation Details

### Files Modified

1. **`bittbridge/validator/reward.py`**: Core Precog methodology implementation
2. **`neurons/validator.py`**: Updated validator to use Precog scoring
3. **`bittbridge/base/validator.py`**: Added state persistence for EMA weights
4. **`bittbridge/validator/forward.py`**: Updated to collect interval predictions

### Key Functions

#### `get_precog_rewards()`
Main function that orchestrates the entire Precog scoring process:
- Calculates point forecast scores
- Calculates interval forecast scores  
- Combines scores
- Applies EMA smoothing
- Returns final rewards and updated weights

#### `calculate_point_forecast_scores()`
Implements point forecast evaluation using relative error ranking.

#### `calculate_interval_forecast_scores()`
Implements interval forecast evaluation using inclusion and width factors.

#### `apply_exponential_moving_average()`
Applies EMA smoothing to combine current performance with historical performance.

### State Persistence

The validator now persists:
- Previous epoch weights for EMA calculation
- EMA smoothing factor (α)
- Standard validator state (scores, hotkeys, step)

State is saved to:
- `state.npz`: Standard validator state
- `precog_state.json`: Precog-specific state

## Usage

The enhanced validator automatically uses the Precog methodology when:
1. Miners submit both point predictions and interval predictions
2. The validator evaluates predictions after the evaluation delay
3. Rewards are calculated using the complete Precog framework

### Example Log Output

```
[PRECOG_EVAL] UID=1, Prediction=1.745, Interval=[1.740, 1.750], Actual=1.743, Reward=0.9000
[PRECOG_EVAL] UID=2, Prediction=1.748, Interval=[1.742, 1.755], Actual=1.743, Reward=0.8100
```

## Benefits

1. **Comprehensive Evaluation**: Considers both point accuracy and uncertainty quantification
2. **Fair Ranking**: Uses proven ranking methodology from Precog subnet
3. **Temporal Consistency**: EMA rewards consistent performance over time
4. **Robust Scoring**: Handles invalid predictions and missing intervals gracefully
5. **State Persistence**: Maintains performance history across validator restarts

## Configuration

Key parameters can be adjusted in the validator:

```python
self.alpha = 0.00958  # EMA smoothing factor
evaluation_delay = 15  # Seconds to wait before evaluating predictions
```

## Compatibility

The implementation maintains backward compatibility:
- Legacy `reward()` function still works for simple point predictions
- Existing miners without interval predictions will still be scored
- State loading handles missing Precog state gracefully

## References

- [Coin Metrics Precog Methodology](https://docs.coinmetrics.io/bittensor/precog-methodology)
- Original implementation based on Colab notebook analysis
- Bittensor subnet template architecture

import bittensor as bt
import numpy as np
import yfinance as yf

from snp_oracle import constants
from snp_oracle.utils.general import rank
from snp_oracle.utils.timestamp import get_before, to_datetime, to_str


################################################################################
def calc_rewards(
    self,
) -> np.ndarray:
    prediction_future_hours = constants.PREDICTION_FUTURE_HOURS

    # preallocate
    point_errors = []
    direction_errors = []

    # the current timestamp
    timestamp = self.score_timestamp
    bt.logging.debug(f"Calculating rewards for timestamp: {timestamp}")

    # get the timestamp when the forecast was made
    reference_timestamp = to_str(get_before(to_datetime(timestamp), hours=1, minutes=0))
    bt.logging.debug(f"Maturing predictions for timestamp: {reference_timestamp}")

    # Get most recent S&P Data at 5m intervals
    yfinance_data = yf.download(tickers="^GSPC", period="5d", interval="5m", progress=False, ignore_tz=True)

    # The most recent record is the most recent price
    actual_price = yfinance_data.iloc[-1]["Close", "^GSPC"]

    # Go backwards to know the price of the S&P at the time the prediction was made
    price_at_prediction_time = yfinance_data.iloc[-1 - 12 * prediction_future_hours]["Close", "^GSPC"]

    for uid in self.available_uids:
        current_miner = self.MinerHistory[int(uid)]

        # Get prediction from the evaluation window that has matured
        predicted_price, predicted_direction = current_miner.get_prediction(reference_timestamp)

        bt.logging.debug(
            f"UID: {uid} | Actual (Price, Direction): ({actual_price}, {actual_price > price_at_prediction_time}) | Prediction (Price, Direction): ({predicted_price}, {predicted_direction})"
        )

        # Penalize miners with missing predictions by increasing their point error
        if not predicted_price:
            point_errors.append(np.inf)  # Maximum penalty for no predictions
        else:
            # Calculate error as normal
            base_point_error = point_error(predicted_price, actual_price)

            point_errors.append(base_point_error)

        if not predicted_direction:
            direction_errors.append(False)
        else:
            base_direction_error = direction_error(
                predicted_direction, actual_direction=actual_price > price_at_prediction_time
            )
            direction_errors.append(base_direction_error)

        bt.logging.debug(f"UID: {uid} | point_errors: {point_errors[-1]} | direction_errors: {direction_errors[-1]}")

    # First rank on the absolute error
    point_ranks = rank(np.array(point_errors))

    # Then, adjust the ranking based on direction
    # Lower rank is better. Correct direction gets better rank than bad direction
    final_ranks = rank(
        [rank + len(point_ranks) if not direction else rank for direction, rank in zip(point_ranks, direction_errors)]
    )

    base_rewards = np.exp(-0.05 * np.array(final_ranks))

    # Simply divide the base rewards by the sum
    # Such that the sum of rewards is equal to 100%
    final_rewards = base_rewards / np.sum(base_rewards)

    return dict(zip(self.available_uids, map(float, final_rewards)))


def direction_error(prediction, actual_direction) -> int:
    if prediction is None:
        return False
    else:
        return bool(prediction == actual_direction)


def point_error(prediction, actual_price) -> float:
    if prediction is None:
        point_error = np.inf
    else:
        point_error = float(abs(prediction - actual_price))
    return point_error

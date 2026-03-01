import argparse
import os
import random
import time
from typing import Tuple, Optional
import typing
import pandas as pd
import bittensor as bt

# Bittensor Miner Template:
import bittbridge

# import base miner class which takes care of most of the boilerplate
from bittbridge.base.miner import BaseMinerNeuron

# ---------------------------
# Miner Forward Logic for USDT/CNY Prediction (Moving Average)
# ---------------------------
# This implementation is used inside the `forward()` method of the miner neuron.
# When a validator sends a Challenge synapse, this code:
#   1. Loads price data from neurons/data.csv.
#   2. Computes a simple moving average of recent Close prices.
#   3. Uses the MA as the predicted next price.
#   4. Estimates a 90% confidence interval using a naive volatility model.
#   5. Attaches the prediction and interval to the synapse and returns it.
#
# Validators will later use this to score the miner's accuracy.

# Number of 5-minute steps for moving average (12 = 1 hour)
N_STEPS = 12


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a DataFrame for time series prediction.
    Uses timestamp_utc and Close columns from neurons/data.csv.
    """
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    time_series_df = df[["timestamp_utc", "Close"]].copy()
    time_series_df.columns = ["datetime", "close_price"]
    time_series_df = time_series_df.set_index("datetime")
    time_series_df = time_series_df.sort_index()
    # Drop rows with missing close_price
    time_series_df = time_series_df.dropna()
    return time_series_df


def predict_next_price_ma(
    data: pd.DataFrame, n_steps: int = N_STEPS
) -> Optional[float]:
    """
    Predict next price using simple moving average of last n_steps Close prices.
    Validator always requests a future prediction, so dataset is always older than
    the request; we simply use the most recent n_steps prices.
    """
    if len(data) == 0:
        return None

    recent_prices = data["close_price"].tail(n_steps)
    prediction = float(recent_prices.mean())
    return prediction


def estimate_interval(prediction: float) -> Tuple[float, float]:
    """Estimate a naive 90% confidence interval based on a fixed 1% volatility assumption."""
    std_dev = 0.01  # Assume 1% standard deviation in price
    # Use 1.64 as z-score for 90% confidence in a normal distribution
    lower = prediction - 1.64 * std_dev * prediction
    upper = prediction + 1.64 * std_dev * prediction
    return lower, upper


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument(
            "--test",
            action="store_true",
            help="[Testing only] Add random noise to each prediction so multiple miners produce different values (e.g. for dashboard development).",
            default=False,
        )

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Load price data from neurons/data.csv for moving average prediction
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")
        if not os.path.exists(data_path):
            bt.logging.error(f"Data file not found: {data_path}")
            self._price_data = None
        else:
            try:
                df = pd.read_csv(data_path)
                self._price_data = prepare_dataframe(df)
                bt.logging.info(
                    f"Loaded {len(self._price_data)} rows from {data_path} for MA prediction"
                )
            except Exception as e:
                bt.logging.error(f"Failed to load data: {e}")
                self._price_data = None

        # [Testing only] When --test is set, add noise to predictions so multiple miners produce different values (e.g. for dashboard development)
        self._add_test_noise = getattr(self.config, "test", False)

    async def forward(self, synapse: bittbridge.protocol.Challenge) -> bittbridge.protocol.Challenge:
        """
        Responds to the Challenge synapse from the validator by submitting:
        - a USDT/CNY price prediction (moving average of recent Close prices)
        - a 90% confidence interval based on fixed volatility assumption
        """
        # Step 1: If data wasn't loaded, skip this round
        if self._price_data is None:
            return synapse  # prediction and interval remain None (validator will ignore)

        # Step 2: Predict next price using moving average
        prediction = predict_next_price_ma(self._price_data, n_steps=N_STEPS)

        # Step 3: If prediction failed (insufficient data), skip this round
        if prediction is None:
            return synapse

        # Step 4: [Testing only] Add noise so multiple miners produce different predictions (e.g. for dashboard development)
        if self._add_test_noise:
            prediction += random.uniform(-0.5, 0.5)

        # Step 5: Assign point prediction
        synapse.prediction = prediction

        # Step 6: Estimate and assign 90% confidence interval
        synapse.interval = list(estimate_interval(prediction))

        # Step 7: Log successful prediction
        if self._add_test_noise:
            bt.logging.success(f"Predicted: {prediction}, Interval: {synapse.interval} (with noise)")
        else:
            bt.logging.success(f"Predicted: {prediction}, Interval: {synapse.interval}")
        return synapse

    async def blacklist(self, synapse: bittbridge.protocol.Challenge) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bittbridge.protocol.Challenge) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)

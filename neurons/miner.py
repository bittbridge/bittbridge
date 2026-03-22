import argparse
import os
import random
import time
from datetime import datetime, timedelta
from typing import Optional
import typing
import bittensor as bt

# Bittensor Miner Template:
import bittbridge

# import base miner class which takes care of most of the boilerplate
from bittbridge.base.miner import BaseMinerNeuron
from bittbridge.utils.iso_ne_api import fetch_fiveminute_system_load
from bittbridge.utils.timestamp import get_now

# ---------------------------
# Miner Forward Logic for New England Energy Demand (LoadMw) Prediction
# ---------------------------
# This implementation is used inside the `forward()` method of the miner neuron.
# When a validator sends a Challenge synapse, this code:
#   1. Fetches latest LoadMw data from ISO-NE API (fiveminutesystemload/day/{day}).
#   2. Computes a simple moving average of the last N LoadMw values.
#   3. Uses the MA as the predicted next LoadMw (point forecast for the target timestamp).
#   4. Attaches the prediction to the synapse and returns it.
#
# Validators score the miner's point forecast against actual demand.

# Number of 5-minute steps for moving average (12 = 1 hour)
N_STEPS = 12


def get_latest_load_mw_values(n_steps: int = N_STEPS) -> Optional[list]:
    """
    Fetch latest N LoadMw values from ISO-NE API.
    get_now() is Eastern; API day is Eastern.
    """
    now = get_now()
    today = now.strftime("%Y%m%d")
    data = fetch_fiveminute_system_load(today, use_cache=False)
    # If first hour of Eastern day, also fetch yesterday for enough data
    if now.hour < 1 and now.minute < 30:
        yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")
        data_yesterday = fetch_fiveminute_system_load(yesterday, use_cache=False)
        data = data_yesterday + data
    if not data:
        return None
    load_values = [load_mw for _, load_mw in data]
    if len(load_values) < n_steps:
        return None
    return load_values[-n_steps:]


def predict_next_load_ma(load_values: list, n_steps: int = N_STEPS) -> Optional[float]:
    """
    Predict next LoadMw using simple moving average of last n_steps values.
    """
    if not load_values or len(load_values) < n_steps:
        return None
    recent = load_values[-n_steps:]
    return float(sum(recent) / len(recent))


class Miner(BaseMinerNeuron):
    """
    Miner neuron for New England energy demand (LoadMw) prediction.
    Uses ISO-NE API for latest 5-minute system load data.
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
        self._add_test_noise = getattr(self.config, "test", False)

    async def forward(self, synapse: bittbridge.protocol.Challenge) -> bittbridge.protocol.Challenge:
        """
        Responds to the Challenge synapse from the validator with a LoadMw point prediction
        (moving average of recent 5-min system load).
        """
        # Step 1: Fetch latest LoadMw data from API
        load_values = get_latest_load_mw_values(n_steps=N_STEPS)
        if load_values is None:
            bt.logging.warning("Failed to fetch LoadMw data from API - skipping this round")
            return synapse

        # Step 2: Predict next LoadMw using moving average
        prediction = predict_next_load_ma(load_values, n_steps=N_STEPS)
        if prediction is None:
            return synapse

        # Step 3: [Testing only] Add noise scaled to load
        if self._add_test_noise:
            prediction += random.uniform(-50, 50)

        # Step 4: Assign point prediction
        synapse.prediction = prediction

        # Step 5: Log successful prediction
        if self._add_test_noise:
            bt.logging.success(
                f"Predicting LoadMw for timestamp={synapse.timestamp}: "
                f"{prediction:.1f} (with noise)"
            )
        else:
            bt.logging.success(
                f"Predicting LoadMw for timestamp={synapse.timestamp}: {prediction:.1f}"
            )
        return synapse

    async def blacklist(self, synapse: bittbridge.protocol.Challenge) -> typing.Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
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
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )
        priority = float(
            self.metagraph.S[caller_uid]
        )
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)

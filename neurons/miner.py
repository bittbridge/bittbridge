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

import time
import requests
import random
from typing import Tuple
import typing
import bittensor as bt

# Bittensor Miner Template:
import bittbridge

# import base miner class which takes care of most of the boilerplate
from bittbridge.base.miner import BaseMinerNeuron

# ---------------------------
# Miner Forward Logic for USDT/CNY Prediction
# ---------------------------
# This implementation is used inside the `forward()` method of the miner neuron.
# When a validator sends a Challenge synapse, this code:
#   1. Fetches the current USDT/CNY price from CoinGecko API.
#   2. Sets that price as the prediction.
#   3. Estimates a 90% confidence interval using a naive volatility model.
#   4. Attaches the prediction and interval to the synapse and returns it.
#
# Validators will later use this to score the miner's accuracy.

# Fetch the current price of USDT in CNY using CoinGecko's public API.

def fetch_current_usdt_cny() -> float:
    try:
        response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=cny&precision=4&x_cg_demo_api_key=CG-1UpuR3vjuAqQWJTQo3EPdUmR")
        response.raise_for_status()  # Raise exception for HTTP errors (4xx/5xx)
        return response.json()["tether"]["cny"]
    except Exception as e:
        # Log a warning and return None if API call fails
        bt.logging.warning(f"API error: {e}")
        return None

# Estimate a naive 90% confidence interval based on a fixed 1% volatility assumption
def estimate_interval(prediction: float) -> Tuple[float, float]:
    std_dev = 0.01  # Assume 1% standard deviation in price
    # Use 1.64 as z-score for 90% confidence in a normal distribution
    lower = prediction - 1.64 * std_dev * prediction
    upper = prediction + 1.64 * std_dev * prediction
    return lower, upper

# This function is triggered when a validator queries this miner.
# It processes the synapse by providing a prediction and interval.

class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # TODO(developer): Anything specific to your use case you can do here

    async def forward(self, synapse: bittbridge.protocol.Challenge) -> bittbridge.protocol.Challenge:
        """
        Responds to the Challenge synapse from the validator by submitting:
        - a USDT/CNY price prediction (with added randomness for testing predictions)
        - a naive confidence interval based on fixed volatility assumption
        """
        
        # Step 1: Fetch the most recent USDT/CNY price
        price = fetch_current_usdt_cny()
        
        # Step 2: If the price couldn't be fetched, skip this round
        if price is None:
            return synapse  # prediction and interval remain None (validator will ignore)

        # Step 3: Add random variation (-0.5 to +0.5)
        price += random.uniform(-0.5, 0.5)

        # Step 4: Assign point prediction
        synapse.prediction = price

        # Step 4: Estimate and assign confidence interval
        synapse.interval = list(estimate_interval(price))

        # Step 5: Log successful prediction
        bt.logging.success(f"Predicted: {price}, Interval: {synapse.interval}")
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

# The MIT License (MIT)

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

import asyncio
import time

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from bittbridge.base.validator import BaseValidatorNeuron

# Bittensor Validator Template:
from bittbridge.validator import forward

# Reward calculation utilities
from bittbridge.validator.reward import get_actual_usdt_cny, reward, get_incentive_mechanism_rewards

# Protocol imports
from bittbridge.protocol import Challenge

# Timestamp utilities
from bittbridge.utils.timestamp import (
    get_now,
    to_str,
    to_datetime,
    is_query_time,
    round_to_interval,
    elapsed_seconds,
    get_before,
)

# --- NEW: W&B helper imports (setup + logging) ---
from bittbridge.utils.wandb import setup_wandb, log_wandb


class Validator(BaseValidatorNeuron):

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        # initialize W&B
        try:
            setup_wandb(self)
            self._wandb_ok = True
        except Exception as e:
            bt.logging.error(f"W&B setup failed: {e}")
            self._wandb_ok = False

        # cache hotkeys for W&B logging context
        try:
            self.hotkeys = {uid: hk for uid, hk in enumerate(self.metagraph.hotkeys)}
        except Exception:
            self.hotkeys = {}

        self.prediction_queue = []  # Store pending predictions here
        
        # Initialize incentive mechanism parameters
        self.alpha = 0.00958  # EMA smoothing factor from incentive mechanism
        self.previous_weights = {}  # Store previous epoch weights for EMA

    async def forward(self):
        """
        The forward pass for the validator. Delegates logic to bittbridge.validator.forward.forward().
        """
        return await forward(self)
    # Evaluation loop to process predictions after a delay and assign rewards using incentive mechanism
    # evaluation_delay: the delay in seconds after which the predictions are processed
    async def evaluation_loop(self, evaluation_delay=3600, check_interval=10):
        while True:
            now = time.time()
            ready = [p for p in self.prediction_queue if now - p["request_time"] >= evaluation_delay]
            
            # Initialize W&B data collection
            wb_responses = []
            wb_rewards = []
            wb_uids = []
            
            if ready:
                # Group predictions by timestamp for batch evaluation
                timestamp_groups = {}
                for pred in ready:
                    timestamp = pred["timestamp"]
                    if timestamp not in timestamp_groups:
                        timestamp_groups[timestamp] = []
                    timestamp_groups[timestamp].append(pred)
                
                # Process each timestamp group
                for timestamp, predictions in timestamp_groups.items():
                    actual = get_actual_usdt_cny()
                    if actual is not None:
                        # Convert predictions to Challenge objects for incentive mechanism scoring
                        responses = []
                        miner_uids = []
                        for pred in predictions:
                            # Create a mock Challenge response
                            response = Challenge(timestamp=timestamp)
                            response.prediction = pred["prediction"]
                            response.interval = pred.get("interval")
                            responses.append(response)
                            miner_uids.append(pred["miner_uid"])
                        
                        # Use incentive mechanism for scoring
                        rewards, updated_weights = get_incentive_mechanism_rewards(
                            actual_price=actual,
                            responses=responses,
                            previous_weights=self.previous_weights,
                            alpha=self.alpha
                        )
                        
                        # Update scores using the new reward system
                        self.update_scores(rewards, miner_uids)
                        
                        # Update previous weights for next epoch
                        self.previous_weights = updated_weights
                        
                        # Collect data for W&B logging
                        for i, pred in enumerate(predictions):
                            # Add to W&B data collection
                            wb_responses.append(responses[i])
                            wb_rewards.append(rewards[i])
                            wb_uids.append(pred["miner_uid"])
                            
                            # Log detailed results
                            bt.logging.info(
                                f"[INCENTIVE_MECHANISM_EVAL] UID={pred['miner_uid']}, "
                                f"Prediction={pred['prediction']}, "
                                f"Interval={pred.get('interval')}, "
                                f"Actual={actual}, "
                                f"Reward={rewards[i]:.4f}"
                            )
                
                # Remove processed predictions
                for pred in ready:
                    self.prediction_queue.remove(pred)
            
            # Log to W&B if we have data and W&B is available
            if getattr(self, "_wandb_ok", False) and wb_uids:
                try:
                    moving_avgs = getattr(self, "moving_average_scores", {})
                    if not isinstance(moving_avgs, dict):
                        moving_avgs = {}
                    log_wandb(
                        responses=wb_responses,
                        rewards=wb_rewards,
                        miner_uids=wb_uids,
                        hotkeys=getattr(self, "hotkeys", {}),
                        moving_average_scores=moving_avgs,
                    )
                except Exception as e:
                    bt.logging.error(f"W&B log failed: {e}")
            
            await asyncio.sleep(check_interval)


# Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph.
# Ensure that validator see new miners that have joined the network.
async def metagraph_resync_scheduler(validator, resync_interval=600):
    while True:
        validator.resync_metagraph()

        # refresh cached hotkeys after every resync (for W&B context)
        # This block ensures our hotkeys dict is always in sync with the latest metagraph.
        try:
            validator.hotkeys = {uid: hk for uid, hk in enumerate(validator.metagraph.hotkeys)}
        except Exception:
            pass

        bt.logging.info("Metagraph resynced.")
        await asyncio.sleep(resync_interval)


async def prediction_scheduler(validator):
    # Set your prediction interval (in minutes)
    prediction_interval = 5  # Query miners every 5 minutes
    # Initialize timestamp to current time, rounded to interval
    timestamp = to_str(round_to_interval(get_now(), interval_minutes=prediction_interval))
    while True:
        query_lag = elapsed_seconds(get_now(), to_datetime(timestamp))
        # Only query if it's the start of a new epoch or lag is too large
        if is_query_time(prediction_interval, timestamp) or query_lag >= 60 * prediction_interval:
            await validator.forward()
            # Update timestamp to the current rounded interval
            timestamp = to_str(round_to_interval(get_now(), interval_minutes=prediction_interval))
        await asyncio.sleep(10)  # Check every 10 seconds


async def main():
    validator = Validator()
    eval_task = asyncio.create_task(validator.evaluation_loop(evaluation_delay=3600, check_interval=10))
    pred_task = asyncio.create_task(prediction_scheduler(validator))
    resync_task = asyncio.create_task(metagraph_resync_scheduler(validator, resync_interval=600))
    await asyncio.gather(eval_task, pred_task, resync_task)


if __name__ == "__main__":
    asyncio.run(main())

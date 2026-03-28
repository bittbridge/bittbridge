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
import traceback

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from bittbridge.base.validator import BaseValidatorNeuron

# Bittensor Validator Template:
from bittbridge.validator import forward

# Reward calculation utilities
from bittbridge.validator.reward import get_actual_load_mw, get_incentive_mechanism_rewards

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

    async def forward(self):
        """
        The forward pass for the validator. Delegates logic to bittbridge.validator.forward.forward().
        """
        return await forward(self)
    # Evaluation loop to process predictions after a delay and assign rewards using incentive mechanism
    # evaluation_delay: the delay in seconds after which the predictions are processed
    async def evaluation_loop(self, evaluation_delay=600, check_interval=10):
        while True:
            now = time.time()
            ready = [p for p in self.prediction_queue if now - p["request_time"] >= evaluation_delay]
            
            # Initialize W&B data collection
            wb_responses = []
            wb_rewards = []
            wb_uids = []
            # wb_actuals = []
            # wb_timestamps = []
            
            if ready:
                # Group predictions by timestamp for batch evaluation
                timestamp_groups = {}
                for pred in ready:
                    timestamp = pred["timestamp"]
                    if timestamp not in timestamp_groups:
                        timestamp_groups[timestamp] = []
                    timestamp_groups[timestamp].append(pred)
                
                # Track predictions that were actually evaluated (so we only remove those)
                processed_preds = []
                
                # Process each timestamp group
                for timestamp, predictions in timestamp_groups.items():
                    actual = get_actual_load_mw(timestamp)
                    if actual is not None:
                        # Convert predictions to Challenge objects for incentive mechanism scoring
                        responses = []
                        miner_uids = []
                        for pred in predictions:
                            # Create a mock Challenge response
                            response = Challenge(timestamp=timestamp)
                            response.prediction = pred["prediction"]
                            responses.append(response)
                            miner_uids.append(pred["miner_uid"])
                        
                        # Use incentive mechanism for scoring
                        rewards, updated_weights = get_incentive_mechanism_rewards(
                            ground_truth=actual,
                            responses=responses,
                        )

                        # Map index-keyed weights (aligned with responses) to real UIDs for W&B
                        if isinstance(updated_weights, dict):
                            self.last_round_weights = {
                                int(miner_uids[int(i)]): float(v)
                                for i, v in updated_weights.items()
                                if 0 <= int(i) < len(miner_uids)
                            }
                        elif isinstance(updated_weights, (list, tuple)):
                            self.last_round_weights = {
                                int(miner_uids[i]): float(v)
                                for i, v in enumerate(updated_weights)
                                if 0 <= i < len(miner_uids)
                            }
                        else:
                            self.last_round_weights = {}

                        self.update_scores(rewards, miner_uids)
                        
                        # Collect data for W&B logging
                        for i, pred in enumerate(predictions):
                            # Add to W&B data collection
                            wb_responses.append(responses[i])
                            wb_rewards.append(rewards[i])
                            wb_uids.append(pred["miner_uid"])
                            # wb_actuals.append(actual)
                            # wb_timestamps.append(timestamp)
                            
                            # Log detailed results
                            bt.logging.info(
                                f"[INCENTIVE_MECHANISM_EVAL] UID={pred['miner_uid']}, "
                                f"Prediction={pred['prediction']}, "
                                f"Actual LoadMw={actual}, "
                                f"Reward={rewards[i]:.4f}"
                            )
                        processed_preds.extend(predictions)
                    else:
                        bt.logging.info(
                            f"Actual load not yet available for timestamp={timestamp} - will retry"
                        )
                
                # Remove only predictions that were actually evaluated
                for pred in processed_preds:
                    self.prediction_queue.remove(pred)
            
            # Log to W&B if we have data and W&B is available
            if getattr(self, "_wandb_ok", False) and wb_uids:
                try:
                    last_w = getattr(self, "last_round_weights", {})
                    if not isinstance(last_w, dict):
                        last_w = {}
                    log_wandb(
                        responses=wb_responses,
                        rewards=wb_rewards,
                        miner_uids=wb_uids,
                        hotkeys=getattr(self, "hotkeys", {}),
                        moving_average_scores=moving_avgs,
                        ground_truth=actual,
                        timestamp=timestamp,
                        # ground_truth=wb_actuals,
                        # timestamp=wb_timestamps,
                    )
                except Exception as e:
                    bt.logging.error(f"W&B log failed: {e}")
            
            await asyncio.sleep(check_interval)


# Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph.
# Ensure that validator see new miners that have joined the network.
async def metagraph_resync_scheduler(validator, resync_interval=60):
    while True:
        try:
            if validator.resync_metagraph():
                # refresh cached hotkeys after every resync (for W&B context)
                # This block ensures our hotkeys dict is always in sync with the latest metagraph.
                try:
                    validator.hotkeys = {uid: hk for uid, hk in enumerate(validator.metagraph.hotkeys)}
                except Exception:
                    pass
                bt.logging.info("Metagraph resynced.")
        except Exception as e:
            bt.logging.error(
                f"Metagraph resync scheduler error (will retry next interval): {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
        await asyncio.sleep(resync_interval)


async def prediction_scheduler(validator):
    # Set your prediction interval (in minutes)
    prediction_interval = 5  # Query miners every 5 minute
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
    eval_task = asyncio.create_task(validator.evaluation_loop(evaluation_delay=600, check_interval=10))
    pred_task = asyncio.create_task(prediction_scheduler(validator))
    resync_task = asyncio.create_task(metagraph_resync_scheduler(validator, resync_interval=60))
    await asyncio.gather(eval_task, pred_task, resync_task)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())

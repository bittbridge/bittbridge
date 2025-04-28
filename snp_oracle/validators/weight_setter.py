import asyncio
import os
import pickle

import bittensor as bt
from numpy import array

from snp_oracle import __spec_version__, constants
from snp_oracle.protocol import Challenge
from snp_oracle.utils.bittensor import check_uid_availability, print_info, setup_bittensor_objects
from snp_oracle.utils.classes import MinerHistory
from snp_oracle.utils.general import func_with_retry, loop_handler
from snp_oracle.utils.timestamp import (
    get_before,
    get_now,
    get_timezone,
    is_query_time,
    is_scoring_time,
    round_to_interval,
    to_str,
)
from snp_oracle.utils.wandb import log_wandb, setup_wandb
from snp_oracle.validators.reward import calc_rewards


class weight_setter:
    def __init__(self, config=None, loop=None):
        self.config = config
        self.loop = loop
        self.lock = asyncio.Lock()

    @classmethod
    async def create(cls, config=None, loop=None):
        self = cls(config, loop)
        await self.initialize()
        return self

    async def initialize(self):
        setup_bittensor_objects(self)
        self.timezone = get_timezone()
        self.prediction_interval = constants.PREDICTION_INTERVAL_MINUTES
        self.hyperparameters = func_with_retry(self.subtensor.get_subnet_hyperparameters, netuid=self.config.netuid)
        self.resync_metagraph_rate = 600  # in seconds
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network}"
        )
        self.available_uids = await self.get_available_uids()
        self.hotkeys = {uid: value for uid, value in enumerate(self.metagraph.hotkeys)}
        if self.config.reset_state:
            self.scores = {int(uid): 0 for uid in self.metagraph.uids}
            self.MinerHistory = {
                int(uid): MinerHistory(int(uid), timezone=self.timezone) for uid in self.available_uids
            }
            self.save_state()
        else:
            self.load_state()
        self.current_block = self.subtensor.get_current_block()
        self.blocks_since_last_update = self.subtensor.blocks_since_last_update(
            netuid=self.config.netuid, uid=self.my_uid
        )
        if not self.config.wandb.off:
            setup_wandb(self)
        self.stop_event = asyncio.Event()
        bt.logging.info("Setup complete, starting loop")
        self.loop.create_task(
            loop_handler(self, self.scheduled_prediction_request, sleep_time=self.config.print_cadence)
        )
        self.loop.create_task(loop_handler(self, self.resync_metagraph, sleep_time=self.resync_metagraph_rate))
        self.loop.create_task(loop_handler(self, self.set_weights, sleep_time=self.hyperparameters.weights_rate_limit))
        self.loop.create_task(loop_handler(self, self.clear_old_miner_histories, sleep_time=3600))

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_state()
        try:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
        except Exception as e:
            bt.logging.error(f"Error on __exit__ function: {e}")
        finally:
            asyncio.gather(*pending, return_exceptions=True)
            self.loop.stop()

    def __reset_instance__(self):
        self.__exit__(None, None, None)
        self.__init__(self.config, self.loop)

    async def get_available_uids(self):
        miner_uids = []
        for uid in range(len(self.metagraph.S)):
            uid_is_available = check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
            if uid_is_available:
                miner_uids.append(int(uid))
        return miner_uids

    async def clear_old_miner_histories(self):
        """Periodically clears old predictions from all MinerHistory objects."""
        bt.logging.info("Clearing old predictions (>24h) from MinerHistory...")

        try:
            for uid in self.MinerHistory:
                # Call the clear_old_predictions method on each MinerHistory object
                self.MinerHistory[int(uid)].clear_old_predictions()

            # Save the updated state after clearing
            self.save_state()
            bt.logging.success("Successfully cleared old predictions from all miners")
        except Exception as e:
            bt.logging.error(f"Error clearing old predictions: {e}")

    async def resync_metagraph(self):  # noqa: C901
        """Resyncs the metagraph and updates the hotkeys, available UIDs, and MinerHistory.
        Ensures all data structures remain in sync."""
        bt.logging.info("Syncing Metagraph...")
        try:
            self.metagraph.sync(subtensor=self.subtensor)
        except Exception as e:
            bt.logging.debug(f"Failed to sync metagraph: {e}")
            bt.logging.debug("Instantiating new subtensor")
            self.subtensor = bt.subtensor(config=self.config, network=self.config.subtensor.chain_endpoint)
            self.metagraph.sync(subtensor=self.subtensor)

        bt.logging.info("Metagraph updated, re-syncing hotkeys, dendrite pool and scores")

        # Get current state for logging
        old_uids = set(self.available_uids)
        old_history = set(self.MinerHistory.keys())
        bt.logging.debug(f"Before sync - Available UIDs: {old_uids}")
        bt.logging.debug(f"Before sync - MinerHistory keys: {old_history}")

        # Update available UIDs
        self.available_uids = await self.get_available_uids()
        new_uids = set(map(int, self.available_uids))

        # Update hotkeys dictionary
        self.hotkeys = {int(uid): value for uid, value in enumerate(self.metagraph.hotkeys)}

        # Process hotkey changes
        for uid, hotkey in enumerate(self.metagraph.hotkeys):
            new_miner = uid in new_uids and uid not in old_uids
            if not new_miner:
                replaced_miner = self.hotkeys.get(int(uid), "") != hotkey
            else:
                replaced_miner = False

            if new_miner or replaced_miner:
                bt.logging.info(f"Replacing hotkey on {uid} with {self.metagraph.hotkeys[uid]}")
                self.scores[int(uid)] = 0
                if uid in new_uids:  # Only create history for available UIDs
                    self.MinerHistory[int(uid)] = MinerHistory(int(uid), timezone=self.timezone)

        # Ensure all available UIDs have MinerHistory entries
        for uid in self.available_uids:
            if int(uid) not in self.MinerHistory:
                bt.logging.info(f"Creating new MinerHistory for available UID {uid}")
                self.MinerHistory[int(uid)] = MinerHistory(int(uid), timezone=self.timezone)

        # Clean up old MinerHistory entries
        for uid in list(self.MinerHistory.keys()):
            if uid not in new_uids:
                bt.logging.info(f"Removing MinerHistory for inactive UID {int(uid)}")
                del self.MinerHistory[int(uid)]

        bt.logging.debug(f"After sync - Available UIDs: {new_uids}")
        bt.logging.debug(f"After sync - MinerHistory keys: {set(self.MinerHistory.keys())}")

        # Save updated state
        self.save_state()

    async def query_miners(self, timestamp):
        synapse = Challenge(timestamp=to_str(timestamp))
        responses = await self.dendrite.forward(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in self.available_uids],
            synapse=synapse,
            deserialize=False,
            timeout=self.config.neuron.timeout,
        )
        return responses

    async def set_weights(self):
        try:
            self.blocks_since_last_update = func_with_retry(
                self.subtensor.blocks_since_last_update, netuid=self.config.netuid, uid=self.my_uid
            )
            self.current_block = func_with_retry(self.subtensor.get_current_block)
        except Exception as e:
            bt.logging.error(f"Failed to get current block with error {e}, skipping block update")
            return

        if self.blocks_since_last_update >= self.hyperparameters.weights_rate_limit:
            for uid in self.available_uids:
                if uid not in self.scores:
                    bt.logging.debug(f"Initializing score for new UID: {uid}")
                    self.scores[int(uid)] = 0.0

            uids = array(self.available_uids)
            weights = [self.scores[int(uid)] for uid in self.available_uids]
            if not weights:
                bt.logging.error("No weights to set, skipping")
                return
            for i, j in zip(weights, self.available_uids):
                bt.logging.debug(f"UID: {j}  |  Weight: {i}")
            if sum(weights) == 0:
                weights = [1] * len(weights)
            # Convert to uint16 weights and uids.
            (
                uint_uids,
                uint_weights,
            ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(uids=uids, weights=array(weights))
            # Update the incentive mechanism on the Bittensor blockchain.
            result, msg = self.subtensor.set_weights(
                netuid=self.config.netuid,
                wallet=self.wallet,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_inclusion=True,
                version_key=__spec_version__,
            )
            if result:
                bt.logging.success("✅ Set Weights on chain successfully!")
                self.blocks_since_last_update = 0
            else:
                bt.logging.debug(
                    "Failed to set weights this iteration with message:",
                    msg,
                )

    async def scheduled_prediction_request(self):  # noqa: C901
        if not hasattr(self, "timestamp"):
            self.timestamp = get_before(minutes=self.prediction_interval)

        if not hasattr(self, "score_timestamp"):
            self.score_timestamp = get_before(minutes=self.prediction_interval)

        if len(self.available_uids) == 0:
            bt.logging.info("No miners available. Sleeping for 10 minutes...")
            print_info(self)
            await asyncio.sleep(600)

        else:
            query_time = is_query_time(self.timestamp)
            scoring_time = is_scoring_time(self.score_timestamp)

            responses = None
            # If market is open with more than an hour remaining
            if query_time:
                bt.logging.debug("IS QUERY TIME")
                self.timestamp = round_to_interval(get_now(), interval_minutes=5)
                responses = await self.query_miners(self.timestamp)
                bt.logging.debug(f"Processing responses for UIDs: {self.available_uids}")
                bt.logging.debug(f"Number of responses: {len(responses)}")
                for uid, response in zip(self.available_uids, responses):
                    bt.logging.debug(f"Response from UID {uid}: {response}")
                    current_miner = self.MinerHistory[int(uid)]
                    current_miner.add_prediction(response.timestamp, response.prediction, response.direction)

            # If market is open and has been open for at least an hour
            if scoring_time:
                bt.logging.debug("IS SCORING TIME")
                try:
                    self.score_timestamp = round_to_interval(get_now(), interval_minutes=5)

                    # Adjust the scores based on responses from miners
                    self.scores = calc_rewards(self)

                    bt.logging.debug(f"New Scores: `{self.scores}`")

                except Exception as e:
                    import traceback

                    bt.logging.error(f"Failed to calculate rewards with error: {str(e)}")
                    bt.logging.error(f"Error type: {type(e)}")
                    bt.logging.error("Full traceback:")
                    bt.logging.error(traceback.format_exc())

            # If nothing is happening, log something
            if not (query_time or scoring_time):
                print_info(self)

            # If something happened we want wandb logging
            elif not self.config.wandb.off:
                log_wandb(responses, self.scores, self.available_uids, self.hotkeys)

    def save_state(self):
        """Saves the state of the validator to a file."""

        state_path = os.path.join(self.config.full_path, "state.pt")
        state = {
            "scores": self.scores,
            "MinerHistory": self.MinerHistory,
        }
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        bt.logging.info(f"Saved {self.config.neuron.name} state.")

    def load_state(self) -> None:
        """Initialize or load the current state of the validator from a file."""

        state_path = os.path.join(self.config.full_path, "state.pt")

        bt.logging.info("Loading validator state.")
        bt.logging.info(f"State path: {state_path}")

        # Attempt to load validator state
        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)

        # If we fail to load the state, initialize the state
        except Exception as e:
            bt.logging.error(f"Failed to load state with error: {e}")

            self.scores = {int(uid): 0 for uid in self.metagraph.uids}
            self.MinerHistory = {
                int(uid): MinerHistory(int(uid), timezone=self.timezone) for uid in self.metagraph.uids
            }

        # If we successfully loaded the file, then load the state
        else:
            self.scores = state["scores"]
            self.MinerHistory = state["MinerHistory"]

        # Regardless log this text
        finally:
            bt.logging.info("State has been created successfully")

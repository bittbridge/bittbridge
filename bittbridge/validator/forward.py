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

import asyncio
import time
from datetime import datetime, timedelta, timezone

import bittensor as bt
from bittbridge.protocol import Challenge
from bittbridge.validator.reward import get_actual_usdt_cny, reward
from bittbridge.utils.uids import get_random_uids
import numpy as np

VERIFY_DELAY_SEC = 60
VERIFY_BUFFER_SEC = 3  # small cushion to ensure target time has passed

async def forward(self):
    """
    Called by the validator every cycle.
    Steps:
    1. Build a Challenge for target_timestamp = now+60s
    2. Select miners to query
    3. Query miners using dendrite with the Challenge synapse
    4. Store responses in `self.pending`
    5. Schedule a verifier task that scores at T+60s
    6. Score responses
    7. Uodate scores
    """
    # Step 1: Generate timestamp
    now = datetime.now(timezone.utc)
    target_dt = now + timedelta(seconds=VERIFY_DELAY_SEC)
    challenge = Challenge(target_timestamp=target_dt.isoformat())

    # Step 2: Select miners (k comes from self.config.neuron.sample_size)
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    selected_axons = [self.metagraph.axons[uid] for uid in miner_uids]

    # Step 3: Query miners (raw synapses back; we keep deserialize=False to retain fields)
    responses = await self.dendrite(axons=selected_axons, synapse=challenge, deserialize=False)

    bt.logging.info(f"[VALIDATOR] Challenge {challenge.challenge_id} -> miners {miner_uids}")
    for i, r in enumerate(responses):
        bt.logging.info(f"[RESP] ch={challenge.challenge_id} uid={miner_uids[i]} "
                        f"pred={r.prediction} interval={r.interval}")

    # Step 4: # Store in pending
    async with self.pending_lock:
        self.pending[challenge.challenge_id] = {
            "target_dt": target_dt,
            "miner_uids": miner_uids,
            "responses": responses,
        }

    # Step 5: schedule verification
    asyncio.create_task(_verify_later(self, challenge.challenge_id))

async def _verify_later(self, challenge_id: str):
    # Sleep until target + buffer
    async with self.pending_lock:
        item = self.pending.get(challenge_id)
    if not item:
        return

    target_dt = item["target_dt"]
    now = datetime.now(timezone.utc)
    delay = (target_dt - now).total_seconds() + VERIFY_BUFFER_SEC
    if delay > 0:
        await asyncio.sleep(delay)

    # Step 6: Score responses. Fetch ground truth at ~T+60s
    actual = get_actual_usdt_cny()
    if actual is None:
        bt.logging.warning(f"[VERIFY] {challenge_id}: no ground truth -> zero rewards this round")
        rewards = np.zeros(len(item["responses"]), dtype=float)
    else:
        # Score each response
        rewards = []
        for syn in item["responses"]:
            if syn.prediction is None:
                rewards.append(0.0)
            else:
                rewards.append(reward(actual, syn.prediction))
        rewards = np.array(rewards, dtype=float)

    # Step 7: Update scores
    self.update_scores(rewards, item["miner_uids"])

    # Cleanup
    async with self.pending_lock:
        self.pending.pop(challenge_id, None)

    bt.logging.info(f"[VERIFY] {challenge_id} done. Rewards: {rewards.tolist()}")
time.sleep(5)

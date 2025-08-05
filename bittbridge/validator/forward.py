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
from datetime import datetime

import bittensor as bt
from bittbridge.protocol import Challenge
from bittbridge.validator.reward import get_rewards
from bittbridge.utils.uids import get_random_uids


async def forward(self):
    """
    Called by the validator every cycle.
    Steps:
    1. Generate a current timestamp for prediction.
    2. Select miners to query.
    3. Query miners using dendrite with the Challenge synapse.
    4. Score responses based on how close predictions are to the current price.
    5. Update miner scores.
    """
    # Step 1: Generate timestamp
    timestamp = datetime.utcnow().isoformat()

    # Step 2: Select miners (k comes from self.config.neuron.sample_size)
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    selected_axons = [self.metagraph.axons[uid] for uid in miner_uids]

    # Step 3: Build challenge synapse
    challenge = Challenge(timestamp=timestamp)

    # Step 4: Query miners
    responses = await self.dendrite(
        axons=selected_axons,
        synapse=challenge,
        deserialize=True
    )

    bt.logging.info(f"[VALIDATOR] Queried miners: {[uid for uid in miner_uids]}")
    bt.logging.info(f"[VALIDATOR] Received {len(responses)} responses")

    for i, response in enumerate(responses):
        bt.logging.info(f"[RESPONSE {i}] UID={miner_uids[i]}, Prediction={response.prediction}, Interval={response.interval}")

    # Step 5: Score responses (immediate scoring for now)
    rewards = get_rewards(self, timestamp, responses)

    # Step 6: Update scores
    self.update_scores(rewards, miner_uids)
    time.sleep(5)

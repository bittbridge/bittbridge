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
import asyncio
from datetime import datetime, timedelta, timezone

# import bittensor
import bittensor as bt
# import base validator class which takes care of most of the boilerplate
from bittbridge.base.validator import BaseValidatorNeuron
from bittbridge.validator import forward

CYCLE_PERIOD_SEC = 60  # <-- one cycle every 1 minute

class Validator(BaseValidatorNeuron):

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.pending = {} # challenge_id -> dict with uids, responses, target_dt
        self.pending_lock = asyncio.Lock()
        self._next_fire_at = 0.0          # monotonic timestamp when next cycle is allowed
        
        bt.logging.info("load_state()")
        self.load_state()

    async def forward(self):
        """
        The forward pass for the validator
        Delegates logic to bittbridge.validator.forward().
        Only run a full cycle when the gate is open
        """
        now = time.monotonic()
        if now < self._next_fire_at:
            # Tiny sleep avoids hot loop while waiting for the next window.
            await asyncio.sleep(0.2)
            return

        # Open the window for this cycle and schedule the next one.
        self._next_fire_at = now + CYCLE_PERIOD_SEC
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)

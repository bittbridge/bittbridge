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

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from bittbridge.base.validator import BaseValidatorNeuron

# Bittensor Validator Template:
from bittbridge.validator import forward

# Reward calculation utilities
from bittbridge.validator.reward import get_actual_usdt_cny, reward


class Validator(BaseValidatorNeuron):

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        self.prediction_queue = []  # Store pending predictions here

    async def forward(self):
        """
        The forward pass for the validator. Delegates logic to bittbridge.validator.forward.forward().
        """
        return await forward(self)
  
    async def evaluation_loop(self, evaluation_delay=60, check_interval=5):
        while True:
            now = time.time()
            ready = [p for p in self.prediction_queue if now - p["request_time"] >= evaluation_delay]
            for pred in ready:
                actual = get_actual_usdt_cny()
                if actual is not None and pred["prediction"] is not None:
                    reward_val = reward(actual, pred["prediction"])
                    self.update_scores([reward_val], [pred["miner_uid"]])
                    bt.logging.info(f"[EVAL] UID={pred['miner_uid']}, Prediction={pred['prediction']}, Actual={actual}, Reward={reward_val}")
                self.prediction_queue.remove(pred)
            await asyncio.sleep(check_interval)


async def prediction_scheduler(validator):
    while True:
        await validator.forward()
        await asyncio.sleep(30)  # 30 seconds between predictions

async def main():
    validator = Validator()
    eval_task = asyncio.create_task(validator.evaluation_loop(evaluation_delay=60, check_interval=5))
    pred_task = asyncio.create_task(prediction_scheduler(validator))
    await asyncio.gather(eval_task, pred_task)

if __name__ == "__main__":
    asyncio.run(main())

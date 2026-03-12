# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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

import sys
import unittest

import bittensor as bt

from neurons.validator import Validator
from bittbridge.base.validator import BaseValidatorNeuron
from bittbridge.protocol import Challenge
from bittbridge.utils.uids import get_random_uids
from bittbridge.validator.reward import get_incentive_mechanism_rewards


class TemplateValidatorNeuronTestCase(unittest.TestCase):
    """
    This class contains unit tests for the RewardEvent classes.

    The tests cover different scenarios where completions may or may not be successful and the reward events are checked that they don't contain missing values.
    The `reward` attribute of all RewardEvents is expected to be a float, and the `is_filter_model` attribute is expected to be a boolean.
    """

    def setUp(self):
        sys.argv = sys.argv[0] + ["--config", "tests/configs/validator.json"]

        config = BaseValidatorNeuron.config()
        config.wallet._mock = True
        config.metagraph._mock = True
        config.subtensor._mock = True
        self.neuron = Validator(config)
        self.miner_uids = get_random_uids(self, k=10)

    def test_run_single_step(self):
        # TODO: Test a single step
        pass

    def test_sync_error_if_not_registered(self):
        # TODO: Test that the validator throws an error if it is not registered on metagraph
        pass

    def test_forward(self):
        # TODO: Test that the forward function returns the correct value
        pass

    def test_dummy_responses(self):
        # TODO: Test that the dummy responses are correctly constructed
        # Challenge now requires timestamp; use a sample ISO timestamp
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        responses = self.neuron.dendrite.query(
            axons=[
                self.neuron.metagraph.axons[uid] for uid in self.miner_uids
            ],
            synapse=Challenge(timestamp=ts),
            deserialize=True,
        )
        # Responses are predictions (float or None); basic sanity check
        for response in responses:
            self.assertIsInstance(response, (float, type(None)))

    def test_reward(self):
        # Test incentive mechanism with mocked LoadMw (actual_load_mw=12000)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        responses = [
            Challenge(timestamp=ts),
            Challenge(timestamp=ts),
        ]
        responses[0].prediction = 11900.0
        responses[0].interval = [11800.0, 12000.0]
        responses[1].prediction = 12100.0
        responses[1].interval = [12000.0, 12200.0]
        actual_load_mw = 12000.0  # Mock ground truth
        rewards, _ = get_incentive_mechanism_rewards(actual_load_mw, responses)
        self.assertEqual(len(rewards), 2)
        self.assertTrue(all(r >= 0 for r in rewards))

    def test_reward_with_nan(self):
        # Test that NaN rewards are correctly sanitized
        import numpy as np
        rewards = np.array([0.5, float("nan"), 0.3])
        with self.assertLogs(bt.logging, level="WARNING"):
            self.neuron.update_scores(rewards, self.miner_uids[:3])

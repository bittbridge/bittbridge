import os
import yaml
import typing
import bittensor as bt

import bittbridge
from bittbridge.base.miner import BaseMinerNeuron
from bittbridge.bittbridge import protocol

from .model_service import ModelService
from .features import features_from_timestamp


class ExampleMiner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        # Load example config
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yml")
        cfg_path = os.path.abspath(cfg_path)
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                self.example_cfg = yaml.safe_load(f)
        else:
            # Fallback to example config
            with open(os.path.join(os.path.dirname(__file__), "..", "config.example.yml"), "r") as f:
                self.example_cfg = yaml.safe_load(f)

        artifact_path = self.example_cfg["model"]["artifact_path"]
        backend = self.example_cfg["model"].get("backend", "sklearn")
        refresh_on_start = bool(self.example_cfg.get("update", {}).get("on_start", True))
        self.model = ModelService(artifact_path=artifact_path, backend=backend, refresh_on_start=refresh_on_start)
        self.model.maybe_update()

    async def forward(self, synapse: protocol.Challenge) -> protocol.Challenge:
        feats = features_from_timestamp(synapse.timestamp)
        try:
            y_hat, (lo, hi) = self.model.predict_with_interval(feats)
            synapse.prediction = float(y_hat)
            synapse.interval = [float(lo), float(hi)]
            bt.logging.success(f"Predicted: {y_hat}, Interval: {synapse.interval}")
        except Exception as e:
            bt.logging.warning(f"Prediction failed: {e}")
        return synapse

    async def blacklist(self, synapse: protocol.Challenge) -> typing.Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return True, "Missing dendrite or hotkey"
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            return True, "Unrecognized hotkey"
        if self.config.blacklist.force_validator_permit and not self.metagraph.validator_permit[uid]:
            return True, "Non-validator hotkey"
        return False, "Hotkey recognized!"

    async def priority(self, synapse: protocol.Challenge) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return float(self.metagraph.S[caller_uid])


if __name__ == "__main__":
    with ExampleMiner() as miner:
        miner.run()



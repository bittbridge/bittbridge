import argparse
import random
import time
from dataclasses import dataclass
from typing import Optional
import typing
import bittensor as bt

# Bittensor Miner Template:
import bittbridge

# import base miner class which takes care of most of the boilerplate
from bittbridge.base.miner import BaseMinerNeuron
from miner_model_energy.inference_runtime import (
    AdvancedModelPredictor,
    BaselineMovingAveragePredictor,
    PredictorRouter,
)
from miner_model_energy.ml_config import load_model_config
from miner_model_energy.pipeline import persist_training_result, train_model

# ---------------------------
# Miner Forward Logic for New England Energy Demand (LoadMw) Prediction
# ---------------------------
# This implementation is used inside the `forward()` method of the miner neuron.
# When a validator sends a Challenge synapse, this code:
#   1. Fetches latest LoadMw data from ISO-NE API (fiveminutesystemload/day/{day}).
#   2. Computes a simple moving average of the last N LoadMw values.
#   3. Uses the MA as the predicted next LoadMw (point forecast for the target timestamp).
#   4. Attaches the prediction to the synapse and returns it.
#
# Validators score the miner's point forecast against actual demand.

# Number of 5-minute steps for moving average (12 = 1 hour)
N_STEPS = 12
DEFAULT_PARAMS_PATH = "miner_model_energy/model_params.yaml"


@dataclass
class PreflightResult:
    mode: str
    training_result: object | None = None


def _ask_yes_no_preflight(prompt: str, default_yes: bool) -> bool:
    default_hint = "Y/n" if default_yes else "y/N"
    try:
        answer = input(f"{prompt} [{default_hint}] ").strip().lower()
    except EOFError:
        return default_yes
    if not answer:
        return default_yes
    return answer in {"y", "yes"}


def _ask_model_type_preflight() -> str:
    try:
        answer = input("Select advanced model to train (linear/cart/lstm) [linear]: ").strip().lower()
    except EOFError:
        return "linear"
    if not answer:
        return "linear"
    if answer not in {"linear", "cart", "lstm"}:
        print("Unknown model choice; defaulting to linear.")
        return "linear"
    return answer


def run_preflight(model_params_path: str, non_interactive: bool) -> PreflightResult:
    """
    Runs all interactive model-selection/training prompts before Miner() is constructed.
    This ensures no wallet/network/Bittensor objects are touched during setup decisions.
    """
    if non_interactive:
        print("Non-interactive mode enabled: using baseline moving-average model.")
        return PreflightResult(mode="baseline")

    if _ask_yes_no_preflight("Run baseline moving-average miner model?", default_yes=True):
        return PreflightResult(mode="baseline")

    selected_model = _ask_model_type_preflight()
    try:
        cfg = load_model_config(model_params_path)
        result = train_model(selected_model, cfg)
        print(
            f"[ML] Validation metrics ({selected_model}): "
            f"RMSE={result.metrics['rmse']:.3f}, "
            f"MAE={result.metrics['mae']:.3f}, "
            f"MAPE={result.metrics['mape']:.3f}%"
        )
        if _ask_yes_no_preflight("Deploy this trained model?", default_yes=False):
            if cfg.persistence.get("save_on_deploy", True):
                paths = persist_training_result(result, cfg, run_id="miner")
                print(f"[ML] Saved model artifacts to: {paths['artifact_dir']}")
            print(f"Deployed advanced model: {selected_model}")
            return PreflightResult(mode=f"advanced:{selected_model}", training_result=result)
        print("Deployment declined; continuing with baseline moving average.")
        return PreflightResult(mode="baseline")
    except Exception as exc:
        print(f"Advanced training flow failed; falling back to baseline: {exc}")
        return PreflightResult(mode="baseline")


class Miner(BaseMinerNeuron):
    """
    Miner neuron for New England energy demand (LoadMw) prediction.
    Uses ISO-NE API for latest 5-minute system load data.
    """

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super().add_args(parser)
        parser.add_argument(
            "--test",
            action="store_true",
            help="[Testing only] Add random noise to each prediction so multiple miners produce different values (e.g. for dashboard development).",
            default=False,
        )
        parser.add_argument(
            "--miner.model_params_path",
            type=str,
            default=DEFAULT_PARAMS_PATH,
            help="Path to model YAML config used for advanced training.",
        )
        parser.add_argument(
            "--miner.non_interactive",
            action="store_true",
            default=False,
            help="Disable terminal prompts and keep baseline MA model.",
        )

    def __init__(self, config=None, preflight_result: PreflightResult | None = None):
        super(Miner, self).__init__(config=config)
        self._add_test_noise = getattr(self.config, "test", False)
        self.predictor_router = PredictorRouter(BaselineMovingAveragePredictor(N_STEPS))
        if preflight_result and preflight_result.training_result is not None:
            self.predictor_router.set_predictor(
                AdvancedModelPredictor(result=preflight_result.training_result),
                mode=preflight_result.mode,
            )
            bt.logging.success(f"Using preflight-deployed model mode: {preflight_result.mode}")

    async def forward(self, synapse: bittbridge.protocol.Challenge) -> bittbridge.protocol.Challenge:
        """
        Responds to the Challenge synapse from the validator with a LoadMw point prediction
        (moving average of recent 5-min system load).
        """
        prediction = self.predictor_router.predict(synapse.timestamp)
        if prediction is None:
            return synapse

        # Step 3: [Testing only] Add noise scaled to load
        if self._add_test_noise:
            prediction += random.uniform(-50, 50)

        # Step 4: Assign point prediction
        synapse.prediction = prediction

        # Step 5: Log successful prediction
        if self._add_test_noise:
            bt.logging.success(
                f"Predicting LoadMw for timestamp={synapse.timestamp}: "
                f"{prediction:.1f} (with noise)"
            )
        else:
            bt.logging.success(
                f"[{self.predictor_router.mode}] Predicting LoadMw for timestamp={synapse.timestamp}: {prediction:.1f}"
            )
        return synapse

    async def blacklist(self, synapse: bittbridge.protocol.Challenge) -> typing.Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
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
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )
        priority = float(
            self.metagraph.S[caller_uid]
        )
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    preflight_arg_parser = argparse.ArgumentParser(add_help=False)
    preflight_arg_parser.add_argument(
        "--miner.model_params_path",
        dest="model_params_path",
        type=str,
        default=DEFAULT_PARAMS_PATH,
    )
    preflight_arg_parser.add_argument(
        "--miner.non_interactive",
        dest="non_interactive",
        action="store_true",
        default=False,
    )
    preflight_args, _ = preflight_arg_parser.parse_known_args()
    preflight_result = run_preflight(
        model_params_path=preflight_args.model_params_path,
        non_interactive=preflight_args.non_interactive,
    )

    with Miner(preflight_result=preflight_result) as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)

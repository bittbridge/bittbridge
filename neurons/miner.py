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
    SupabaseLiveAdvancedPredictor,
)
from miner_model_energy.ml_config import load_model_config
from miner_model_energy.pipeline import (
    persist_training_result,
    print_actual_vs_predicted_plotext,
    train_model,
)
from miner_model_energy.storage_train_io import (
    storage_cache_exists,
    storage_cache_last_updated_label,
)

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
DEFAULT_PARAMS_PATH = "model_params.yaml"
_SECTION_WIDTH = 72


@dataclass
class PreflightResult:
    mode: str
    training_result: object | None = None
    model_config: object | None = None


class PreflightExitRequested(Exception):
    """Raised when user requests to exit during preflight prompts."""


def _section(title: str) -> None:
    print()
    print("=" * _SECTION_WIDTH)
    print(f"  {title}")
    print("=" * _SECTION_WIDTH)


def _sub(text: str) -> None:
    print(f"  {text}")


def _format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.2f}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{int(m)}m {s:.1f}s"
    h, m2 = divmod(m, 60)
    return f"{int(h)}h {int(m2)}m {s:.0f}s"


def _print_training_timeline(result) -> None:
    d = getattr(result, "durations_sec", None) or {}
    if not d:
        return
    _sub("")
    _sub("Timing")
    _sub("-" * (_SECTION_WIDTH - 4))
    if "prepare_data_sec" in d:
        _sub(f"  Data prep (load + features):     {_format_seconds(d['prepare_data_sec'])}")
    if "split_arrays_sec" in d:
        _sub(f"  Arrays + temporal split:         {_format_seconds(d['split_arrays_sec'])}")
    if "fit_sec" in d:
        _sub(f"  Train (fit + predictions):       {_format_seconds(d['fit_sec'])}")
    if "metrics_sec" in d:
        _sub(f"  Metrics aggregation:             {_format_seconds(d['metrics_sec'])}")
    if "split_and_fit_sec" in d:
        _sub(f"  Split + train + metrics:         {_format_seconds(d['split_and_fit_sec'])}")
    if "total_sec" in d:
        _sub(f"  Total:                           {_format_seconds(d['total_sec'])}")


def _print_ml_report(selected_model: str, result) -> None:
    _section(f"Model: {selected_model}")
    _sub("")
    _sub("Tensor shapes")
    _sub("-" * (_SECTION_WIDTH - 4))
    _sub(f"  X_train : {result.shapes['X_train']}")
    _sub(f"  y_train : {result.shapes['y_train']}")
    _sub(f"  X_val   : {result.shapes['X_val']}")
    _sub(f"  y_val   : {result.shapes['y_val']}")
    _sub(f"  X_test  : {result.shapes['X_test']}")
    _print_training_timeline(result)
    tr = result.metrics["train"]
    va = result.metrics["validation"]
    _sub("")
    _sub("Train set")
    _sub("-" * (_SECTION_WIDTH - 4))
    _sub(
        f"  RMSE: {tr['rmse']:.3f}    "
        f"MAE: {tr['mae']:.3f}    "
        f"MAPE: {tr['mape']:.3f}%    "
        f"R²: {tr['r2']:.5f}"
    )
    _sub("")
    _sub("Validation set")
    _sub("-" * (_SECTION_WIDTH - 4))
    _sub(
        f"  RMSE: {va['rmse']:.3f}    "
        f"MAE: {va['mae']:.3f}    "
        f"MAPE: {va['mape']:.3f}%    "
        f"R²: {va['r2']:.5f}"
    )
    print_actual_vs_predicted_plotext(result, selected_model)
    print()


def _ask_yes_no_preflight(prompt: str, default_yes: bool) -> bool:
    default_hint = "Y/n" if default_yes else "y/N"
    try:
        answer = input(f"  {prompt} [{default_hint}] ").strip().lower()
    except EOFError:
        return default_yes
    if answer in {"3", "exit", "quit", "q"}:
        raise PreflightExitRequested()
    if not answer:
        return default_yes
    return answer in {"y", "yes"}


def _ask_model_type_preflight() -> str:
    try:
        answer = input("  Select model (linear / cart / rnn / lstm / [3] exit): ").strip().lower()
    except EOFError:
        return "linear"
    if not answer:
        return "linear"
    if answer in {"3", "exit", "quit", "q"}:
        raise PreflightExitRequested()
    if answer not in {"linear", "cart", "rnn", "lstm"}:
        print("  Unknown choice; defaulting to linear.")
        return "linear"
    return answer


def _ask_after_deploy_decline() -> str:
    """
    Returns:
      - 'baseline' to use moving-average miner
      - 'retrain' to pick another advanced model
      - 'exit' to stop before miner startup
    """
    _section("Deploy declined — what next?")
    _sub("  [1]  Continue with baseline moving-average model")
    _sub("  [2]  Train another advanced model (linear / cart / rnn / lstm)")
    _sub("  [3]  Exit miner")
    print()
    while True:
        try:
            answer = input("  Choose [1/2/3]: ").strip().lower()
        except EOFError:
            return "baseline"
        if answer in ("1", "baseline", "b", "ma"):
            return "baseline"
        if answer in ("2", "retrain", "r", "advanced", "train"):
            return "retrain"
        if answer in ("3", "exit", "quit", "q"):
            return "exit"
        if not answer:
            print("  Please enter 1, 2, or 3.")
            continue
        print("  Unrecognized choice. Enter 1 for baseline, 2 to retrain, or 3 to exit.")


def run_preflight(model_params_path: str, non_interactive: bool) -> PreflightResult:
    """
    Runs all interactive model-selection/training prompts before Miner() is constructed.
    This ensures no wallet/network/Bittensor objects are touched during setup decisions.
    """
    if non_interactive:
        _section("Miner preflight")
        _sub("Non-interactive mode: using baseline moving-average model.")
        print()
        return PreflightResult(mode="baseline")

    _section("Miner preflight — model selection")
    try:
        if _ask_yes_no_preflight("Run baseline moving-average miner model?", default_yes=True):
            _sub("")
            _sub("Starting miner with baseline moving-average predictions.")
            print()
            return PreflightResult(mode="baseline")

        try:
            cfg = load_model_config(model_params_path)
        except Exception as exc:
            print(f"  Failed to load model config: {exc}")
            return PreflightResult(mode="baseline")

        storage_force_refresh_decision = False
        force_refresh_used = False
        if cfg.data.get("source") == "supabase":
            _sub(
                "Data source: SUPABASE "
                f"(schema={cfg.data['supabase_schema']}, "
                f"train_table={cfg.data['supabase_train_table']}, "
                f"test_table={cfg.data['supabase_test_table']})"
            )
        elif cfg.data.get("source") == "supabase_storage":
            cache_ok = storage_cache_exists(cfg)
            _section("Supabase Storage training cache")
            if cache_ok:
                _sub(f"Last update: {storage_cache_last_updated_label(cfg)}")
                _sub("Refreshing training data usually takes 2-3 minutes.")
                storage_force_refresh_decision = _ask_yes_no_preflight(
                    "Update training data now?", default_yes=False
                )
            else:
                _sub("First-time training data fetch (~2-3m).")
                _sub("Fetching training data and building local cache usually takes 2-3 minutes.")
                storage_force_refresh_decision = False

        while True:
            selected_model = _ask_model_type_preflight()
            try:
                if cfg.data.get("source") == "supabase_storage":
                    # Only rebuild the cache once per miner run; subsequent models reuse the merged snapshot.
                    cfg.data["storage_force_refresh"] = storage_force_refresh_decision and not force_refresh_used
                result = train_model(selected_model, cfg)
            except Exception as exc:
                print(f"  Training failed: {exc}")
                if not _ask_yes_no_preflight("Try a different model?", default_yes=True):
                    print()
                    return PreflightResult(mode="baseline")
                continue

            if cfg.data.get("source") == "supabase_storage":
                # After the first successful training attempt, never force refresh again in this run.
                force_refresh_used = True
                cfg.data["storage_force_refresh"] = False

            _print_ml_report(selected_model, result)

            deploy_selected = _ask_yes_no_preflight("Deploy this trained model?", default_yes=False)
            dump_full_dataset = False
            if deploy_selected:
                dump_full_dataset = _ask_yes_no_preflight(
                    "Dump full training dataset with engineered features? (This may take a ~1-2 minutes)",
                    default_yes=False,
                )

            paths = persist_training_result(
                result,
                cfg,
                run_id="miner",
                dump_full_training_dataset=dump_full_dataset,
            )
            _sub(f"Saved artifacts: {paths['artifact_dir']}")

            if deploy_selected:
                _section("Ready")
                _sub(f"Deployed advanced model: {selected_model}")
                print()
                return PreflightResult(
                    mode=f"advanced:{selected_model}",
                    training_result=result,
                    model_config=cfg,
                )

            next_step = _ask_after_deploy_decline()
            if next_step == "baseline":
                _section("Ready")
                _sub("Using baseline moving-average model.")
                print()
                return PreflightResult(mode="baseline")
            if next_step == "exit":
                raise PreflightExitRequested()
            # retrain: loop again with new model choice
            _section("Train another model")
            _sub("")
    except PreflightExitRequested:
        _section("Exit")
        _sub("Exiting miner before startup.")
        print()
        return PreflightResult(mode="exit")


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
            predictor = AdvancedModelPredictor(result=preflight_result.training_result)
            if preflight_result.model_config and preflight_result.model_config.data.get("source") in {
                "supabase",
                "supabase_storage",
            }:
                predictor = SupabaseLiveAdvancedPredictor(
                    result=preflight_result.training_result,
                    config=preflight_result.model_config,
                )
            self.predictor_router.set_predictor(
                predictor,
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
    if preflight_result.mode == "exit":
        raise SystemExit(0)

    with Miner(preflight_result=preflight_result) as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)

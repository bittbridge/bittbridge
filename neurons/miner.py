import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import typing
import bittensor as bt
import numpy as np
import pandas as pd

# Bittensor Miner Template:
import bittbridge

# import base miner class which takes care of most of the boilerplate
from bittbridge.base.miner import BaseMinerNeuron
from bittbridge.utils.timestamp import get_now, round_minute_down, to_str
from miner_model_energy.custom_plugin_runtime import (
    CustomPluginDeployState,
    list_plugin_folders,
    read_feature_contract,
    resolve_plugin_dir,
    run_deploy_compatibility_probe,
    sanitize_plugin_folder_name,
    scan_model_candidates,
    update_plugin_metadata_after_deploy,
    write_plugin_export,
)
from miner_model_energy.inference_runtime import (
    AdvancedModelPredictor,
    BaselineMovingAveragePredictor,
    CustomModelPredictor,
    PredictorRouter,
    SupabaseLiveAdvancedPredictor,
)
from miner_model_energy.ml_config import load_model_config
from miner_model_energy.pipeline import (
    TrainingResult,
    load_training_bundle_from_manifest,
    prepare_training_data,
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
    custom_plugin: CustomPluginDeployState | None = None


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


def _ask_top_level_startup_mode() -> str:
    _section("Miner preflight — startup mode")
    _sub("  [1]  Baseline moving-average (ISO-NE)")
    _sub("  [2]  Train a built-in model (linear / cart / rnn / lstm)")
    _sub("  [3]  Custom model plugin (export training pack / deploy uploaded model)")
    _sub("  [4]  Deploy from saved artifact (manifest + model dump)")
    _sub("  [5]  Exit miner")
    print()
    while True:
        try:
            answer = input("  Choose [1/2/3/4/5]: ").strip().lower()
        except EOFError:
            return "baseline"
        if answer in ("1", "baseline", "b", "ma"):
            return "baseline"
        if answer in ("2", "prebuilt", "builtin", "train"):
            return "prebuilt"
        if answer in ("3", "custom", "plugin", "c"):
            return "custom"
        if answer in ("4", "artifact", "saved", "resume", "manifest"):
            return "artifact"
        if answer in ("5", "exit", "quit", "q"):
            raise PreflightExitRequested()
        print("  Enter 1, 2, 3, 4, or 5.")


def _iter_saved_artifact_manifests(artifact_root: str) -> list[Path]:
    root = Path(artifact_root)
    if not root.is_dir():
        return []
    manifests: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = d / "manifest.json"
        if not m.is_file():
            continue
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
            model_rel = data.get("model_path")
            if model_rel and (d / model_rel).exists():
                manifests.append(m)
        except Exception:
            continue
    manifests.sort(key=lambda p: p.parent.name, reverse=True)
    return manifests


def _fmt_manifest_line(manifest_path: Path) -> str:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = str(data.get("model_type", "unknown"))
        metrics = data.get("metrics", {}) or {}
        val = metrics.get("validation", {}) or {}
        rmse = val.get("rmse")
        mae = val.get("mae")
        r2 = val.get("r2")
        metric_txt = ""
        if rmse is not None and mae is not None and r2 is not None:
            metric_txt = f" | val RMSE={float(rmse):.2f}, MAE={float(mae):.2f}, R2={float(r2):.4f}"
        return f"{manifest_path.parent.name} ({model}){metric_txt}"
    except Exception:
        return manifest_path.parent.name


def _print_manifest_metrics(manifest_path: Path) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {}) or {}
    tr = metrics.get("train") or {}
    va = metrics.get("validation") or {}
    _section("Saved artifact metrics")
    _sub(f"  Folder: {manifest_path.parent.name}")
    _sub(f"  Model type: {data.get('model_type', 'unknown')}")
    if tr:
        _sub(
            f"  Train      RMSE={float(tr.get('rmse', float('nan'))):.3f}  "
            f"MAE={float(tr.get('mae', float('nan'))):.3f}  "
            f"MAPE={float(tr.get('mape', float('nan'))):.3f}%  "
            f"R2={float(tr.get('r2', float('nan'))):.5f}"
        )
    if va:
        _sub(
            f"  Validation RMSE={float(va.get('rmse', float('nan'))):.3f}  "
            f"MAE={float(va.get('mae', float('nan'))):.3f}  "
            f"MAPE={float(va.get('mape', float('nan'))):.3f}%  "
            f"R2={float(va.get('r2', float('nan'))):.5f}"
        )
    print()


def _ask_pick_saved_artifact(artifact_root: str) -> Optional[Path]:
    manifests = _iter_saved_artifact_manifests(artifact_root)
    if not manifests:
        return None
    _section("Saved artifacts with manifest.json")
    for i, m in enumerate(manifests, 1):
        _sub(f"  [{i}]  {_fmt_manifest_line(m)}")
    print()
    while True:
        try:
            answer = input(f"  Choose [1-{len(manifests)}] or [q] exit: ").strip().lower()
        except EOFError:
            return manifests[0]
        if answer in {"q", "quit", "exit"}:
            raise PreflightExitRequested()
        if answer.isdigit():
            idx = int(answer)
            if 1 <= idx <= len(manifests):
                return manifests[idx - 1]
        print("  Invalid choice.")


def _load_training_result_from_manifest_preflight(
    manifest_path: Path,
    cfg: object,
) -> TrainingResult:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_type = str(data.get("model_type", ""))
    features = list(data.get("features") or [])
    if not model_type or not features:
        raise ValueError(f"Manifest missing model_type/features: {manifest_path}")
    bundle = load_training_bundle_from_manifest(str(manifest_path))

    source = cfg.data.get("source", "csv")
    if source in {"supabase", "supabase_storage"}:
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()
        shapes = {
            "X_train": (0, len(features)),
            "X_val": (0, len(features)),
            "X_test": (1, len(features)),
            "y_train": (0,),
            "y_val": (0,),
        }
    else:
        train_frame, test_frame, features_live = prepare_training_data(cfg, show_progress=False)
        missing = [c for c in features if c not in test_frame.columns]
        if missing:
            raise ValueError(
                "Saved artifact features are missing from current CSV inference frame: "
                + ", ".join(missing[:20])
            )
        train_frame = train_frame[features].copy()
        test_frame = test_frame[features].copy()
        if set(features_live) != set(features):
            bt.logging.warning(
                "Saved artifact features differ from current CSV feature set; using manifest feature order."
            )
        shapes = {
            "X_train": tuple(train_frame.shape),
            "X_val": (0, len(features)),
            "X_test": tuple(test_frame.shape),
            "y_train": (0,),
            "y_val": (0,),
        }

    metrics = data.get("metrics", {}) or {}
    return TrainingResult(
        model_type=model_type,
        model_bundle=bundle,
        metrics=metrics,
        features=features,
        train_frame=train_frame,
        test_frame=test_frame,
        shapes=shapes,
        y_train=np.array([]),
        train_pred=np.array([]),
        y_val=np.array([]),
        val_pred=np.array([]),
        durations_sec=data.get("durations_sec", {}) or {},
    )


def _ask_custom_deploy_failure_next() -> str:
    """Returns exit | baseline | prebuilt."""
    _section("Custom model deploy failed — choose next step")
    _sub("  [1]  Exit miner")
    _sub("  [2]  Use baseline moving-average")
    _sub("  [3]  Train a built-in model locally instead")
    print()
    while True:
        try:
            answer = input("  Choose [1/2/3]: ").strip().lower()
        except EOFError:
            return "baseline"
        if answer in ("1", "exit", "quit", "q"):
            return "exit"
        if answer in ("2", "baseline", "b", "ma"):
            return "baseline"
        if answer in ("3", "prebuilt", "train", "builtin"):
            return "prebuilt"
        print("  Enter 1, 2, or 3.")


def _ask_pick_model_file(candidates: list[Path]) -> Optional[Path]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    _section("Multiple model files found — pick one")
    for i, p in enumerate(candidates, 1):
        _sub(f"  [{i}]  {p.name}")
    print()
    while True:
        try:
            answer = input(f"  Choose [1-{len(candidates)}]: ").strip()
        except EOFError:
            return candidates[0]
        if not answer.isdigit():
            print("  Enter a number.")
            continue
        idx = int(answer)
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]
        print("  Invalid choice.")


def _run_custom_plugin_preflight(
    cfg: object,
    model_params_path: str,
    storage_force_refresh_decision: bool,
    force_refresh_used_holder: list[bool],
) -> Tuple[Optional[PreflightResult], bool]:
    """
    Returns (terminal_preflight_result_or_none, continue_with_prebuilt).
    If the second value is True, run_preflight should enter the built-in training loop.
    """
    _section("Custom model plugin")
    create_new = _ask_yes_no_preflight(
        "Create a NEW plugin folder (training CSV + notebook)?", default_yes=True
    )

    artifact_root = cfg.persistence["artifact_dir"]
    if create_new:
        while True:
            try:
                raw_name = input(
                    "  Name for the new folder under artifacts (letters, numbers, -, _): "
                ).strip()
            except EOFError:
                return PreflightResult(mode="exit"), False
            try:
                safe = sanitize_plugin_folder_name(raw_name)
            except ValueError as exc:
                print(f"  {exc}")
                continue
            plugin_dir = resolve_plugin_dir(artifact_root, safe)
            if plugin_dir.exists() and any(plugin_dir.iterdir()):
                print(f"  Folder already exists with content: {plugin_dir}")
                if not _ask_yes_no_preflight("Pick a different name?", default_yes=True):
                    return PreflightResult(mode="exit"), False
                continue
            try:
                if cfg.data.get("source") == "supabase_storage":
                    cfg.data["storage_force_refresh"] = (
                        storage_force_refresh_decision and not force_refresh_used_holder[0]
                    )
                write_plugin_export(cfg, safe, model_params_path)
                if cfg.data.get("source") == "supabase_storage":
                    force_refresh_used_holder[0] = True
                    cfg.data["storage_force_refresh"] = False
            except Exception as exc:
                bt.logging.error(f"Plugin export failed: {exc}")
                print(f"  Export failed: {exc}")
                nxt = _ask_custom_deploy_failure_next()
                if nxt == "exit":
                    return PreflightResult(mode="exit"), False
                if nxt == "baseline":
                    return PreflightResult(mode="baseline"), False
                return None, True
            _section("Plugin folder ready")
            _sub(f"  Path: {plugin_dir}")
            _sub(
                "  Next: copy this folder to Colab/GCP, open custom_train_colab.ipynb, train, "
                "save model_custom.joblib or model_custom.keras"
            )
            _sub("  Do not change feature columns vs feature_contract.json. Restart miner, then Custom → deploy.")
            print()
            return PreflightResult(mode="exit"), False

    while True:
        try:
            raw_name = input("  Existing plugin folder name under artifacts: ").strip()
        except EOFError:
            return PreflightResult(mode="exit"), False
        try:
            safe = sanitize_plugin_folder_name(raw_name)
        except ValueError as exc:
            print(f"  {exc}")
            continue
        plugin_dir = resolve_plugin_dir(artifact_root, safe)
        if not plugin_dir.is_dir():
            print(f"  Not found: {plugin_dir}")
            known = list_plugin_folders(artifact_root)
            if known:
                _sub("  Known plugin folders (with plugin_metadata.json):")
                for k in known[:25]:
                    _sub(f"    - {k}")
            if not _ask_yes_no_preflight("Try another folder name?", default_yes=True):
                return PreflightResult(mode="exit"), False
            continue
        try:
            read_feature_contract(plugin_dir)
        except Exception as exc:
            bt.logging.error("Invalid plugin folder %s: %s", plugin_dir, exc)
            print(f"  {exc}")
            if not _ask_yes_no_preflight("Try another folder?", default_yes=True):
                return PreflightResult(mode="exit"), False
            continue
        break

    candidates = scan_model_candidates(plugin_dir)
    if not candidates:
        bt.logging.error("No model files (.joblib / .keras / SavedModel) in %s", plugin_dir)
        print("  No deployable model file found. Upload your trained artifact, then restart.")
        nxt = _ask_custom_deploy_failure_next()
        if nxt == "exit":
            return PreflightResult(mode="exit"), False
        if nxt == "baseline":
            return PreflightResult(mode="baseline"), False
        return None, True

    picked = _ask_pick_model_file(candidates)
    if picked is None:
        return PreflightResult(mode="exit"), False

    probe_ts = to_str(round_minute_down(get_now(), 5))
    try:
        wrapper, seq_steps, _x = run_deploy_compatibility_probe(
            cfg, plugin_dir, picked, probe_ts
        )
    except Exception as exc:
        bt.logging.error(f"Custom model compatibility check failed: {exc}")
        print(f"  Compatibility / load failed: {exc}")
        nxt = _ask_custom_deploy_failure_next()
        if nxt == "exit":
            return PreflightResult(mode="exit"), False
        if nxt == "baseline":
            return PreflightResult(mode="baseline"), False
        return None, True

    if not _ask_yes_no_preflight(f"Deploy this model? ({picked.name})", default_yes=True):
        next_step = _ask_after_deploy_decline()
        if next_step == "exit":
            return PreflightResult(mode="exit"), False
        if next_step == "baseline":
            return PreflightResult(mode="baseline"), False
        return None, True

    try:
        update_plugin_metadata_after_deploy(plugin_dir, picked, seq_steps)
    except Exception as exc:
        bt.logging.warning(f"Could not update plugin_metadata.json: {exc}")

    feats = read_feature_contract(plugin_dir)["features"]
    state = CustomPluginDeployState(
        plugin_dir=plugin_dir,
        model_path=picked,
        wrapper=wrapper,
        features=list(feats),
        sequence_n_steps=seq_steps,
    )
    _section("Ready")
    _sub(f"Deployed custom plugin model from {picked.name}")
    print()
    return (
        PreflightResult(
            mode="custom:deployed",
            model_config=cfg,
            custom_plugin=state,
        ),
        False,
    )


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

    try:
        startup = _ask_top_level_startup_mode()
    except PreflightExitRequested:
        _section("Exit")
        _sub("Exiting miner before startup.")
        print()
        return PreflightResult(mode="exit")

    if startup == "baseline":
        _section("Ready")
        _sub("Starting miner with baseline moving-average predictions.")
        print()
        return PreflightResult(mode="baseline")

    try:
        cfg = load_model_config(model_params_path)
    except Exception as exc:
        _section("Model config load failed")
        _sub(f"Could not load model config from: {model_params_path}")
        _sub(f"Reason: {exc}")
        _sub("Please fix model_params.yaml and restart.")
        print()
        return PreflightResult(mode="exit")

    storage_force_refresh_decision = False
    force_refresh_used_holder = [False]
    if cfg.data.get("source") == "supabase":
        _section("Data source")
        _sub(
            "SUPABASE "
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

    try:
        if startup == "custom":
            custom_res, need_prebuilt = _run_custom_plugin_preflight(
                cfg,
                model_params_path,
                storage_force_refresh_decision,
                force_refresh_used_holder,
            )
            if custom_res is not None:
                return custom_res
            if not need_prebuilt:
                return PreflightResult(mode="exit")
            startup = "prebuilt"

        if startup == "artifact":
            manifest_path = _ask_pick_saved_artifact(cfg.persistence["artifact_dir"])
            if manifest_path is None:
                _section("No saved artifacts")
                _sub("No manifest.json + model dumps found under artifacts. Choose another startup mode.")
                print()
                return PreflightResult(mode="baseline")
            if _ask_yes_no_preflight("Show training metrics from this artifact?", default_yes=True):
                _print_manifest_metrics(manifest_path)
            result = _load_training_result_from_manifest_preflight(manifest_path, cfg)
            if not _ask_yes_no_preflight(
                f"Deploy this saved artifact? ({manifest_path.parent.name})",
                default_yes=True,
            ):
                next_step = _ask_after_deploy_decline()
                if next_step == "exit":
                    raise PreflightExitRequested()
                if next_step == "baseline":
                    return PreflightResult(mode="baseline")
                startup = "prebuilt"
            else:
                _section("Ready")
                _sub(f"Deployed saved artifact model: {manifest_path.parent.name}")
                print()
                return PreflightResult(
                    mode=f"artifact:{result.model_type}:{manifest_path.parent.name}",
                    training_result=result,
                    model_config=cfg,
                )

        if startup == "prebuilt":
            while True:
                selected_model = _ask_model_type_preflight()
                try:
                    if cfg.data.get("source") == "supabase_storage":
                        cfg.data["storage_force_refresh"] = (
                            storage_force_refresh_decision and not force_refresh_used_holder[0]
                        )
                    result = train_model(selected_model, cfg)
                except Exception as exc:
                    print(f"  Training failed: {exc}")
                    if not _ask_yes_no_preflight("Try a different model?", default_yes=True):
                        print()
                        return PreflightResult(mode="baseline")
                    continue

                if cfg.data.get("source") == "supabase_storage":
                    force_refresh_used_holder[0] = True
                    cfg.data["storage_force_refresh"] = False

                _print_ml_report(selected_model, result)

                deploy_selected = _ask_yes_no_preflight(
                    "Deploy this trained model?", default_yes=False
                )
                dump_full_dataset = False
                if deploy_selected:
                    dump_full_dataset = _ask_yes_no_preflight(
                        "Dump full training dataset with engineered features? "
                        "(This may take ~1-2 minutes)",
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
                _section("Train another model")
                _sub("")
    except PreflightExitRequested:
        _section("Exit")
        _sub("Exiting miner before startup.")
        print()
        return PreflightResult(mode="exit")

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
        deployed_mode = "baseline"
        if preflight_result and preflight_result.custom_plugin is not None:
            cp = preflight_result.custom_plugin
            if not preflight_result.model_config:
                raise ValueError("custom_plugin requires model_config on PreflightResult")
            predictor = CustomModelPredictor(
                wrapper=cp.wrapper,
                config=preflight_result.model_config,
                features=cp.features,
                sequence_n_steps=cp.sequence_n_steps,
            )
            self.predictor_router.set_predictor(predictor, mode=preflight_result.mode)
            bt.logging.success(f"Using preflight-deployed model mode: {preflight_result.mode}")
            deployed_mode = preflight_result.mode
        elif preflight_result and preflight_result.training_result is not None:
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
            deployed_mode = preflight_result.mode
        elif preflight_result and preflight_result.mode:
            deployed_mode = preflight_result.mode

        bt.logging.success(
            f"Miner deployed and ready to answer validator requests. Active model mode: {deployed_mode}"
        )

    async def forward(self, synapse: bittbridge.protocol.Challenge) -> bittbridge.protocol.Challenge:
        """
        Responds to the Challenge synapse from the validator with a LoadMw point prediction
        (moving average of recent 5-min system load).
        """
        caller_hotkey = None
        if synapse.dendrite is not None:
            caller_hotkey = synapse.dendrite.hotkey
        bt.logging.info(
            f"Received validator prediction request: hotkey={caller_hotkey}, "
            f"timestamp={synapse.timestamp}, model_mode={self.predictor_router.mode}"
        )

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
        bt.logging.success(
            f"Prediction request input: hotkey={caller_hotkey}, timestamp={synapse.timestamp}, "
            f"model_mode={self.predictor_router.mode}, prediction={prediction:.1f}"
        )
        bt.logging.success(
            "Prediction model input context: "
            + json.dumps(self.predictor_router.last_prediction_context, default=str, ensure_ascii=True)
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

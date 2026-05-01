from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from .artifacts import feature_signature
from .data_io import TARGET_COLUMN, TARGET_COLUMN_HORIZON, TIMESTAMP_COLUMN
from .ml_config import ModelConfig
from .pipeline import live_probe_feature_matrix_for_custom, prepare_training_data

PLUGIN_SCHEMA_VERSION = 1
FEATURE_CONTRACT_NAME = "feature_contract.json"
PLUGIN_METADATA_NAME = "plugin_metadata.json"
TRAINING_DATASET_NAME = "training_dataset_full.csv"
NOTEBOOK_TEMPLATE_NAME = "custom_train_colab.ipynb"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_plugin_folder_name(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        raise ValueError("Plugin folder name cannot be empty.")
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "-", s).strip("-_.")
    if not s or s in {".", ".."}:
        raise ValueError("Invalid plugin folder name after sanitization.")
    return s[:120]


def resolve_plugin_dir(artifact_root: str | Path, folder_name: str) -> Path:
    root = Path(artifact_root).resolve()
    safe = sanitize_plugin_folder_name(folder_name)
    return root / safe


def list_plugin_folders(artifact_root: str | Path) -> List[str]:
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        return []
    names: List[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / PLUGIN_METADATA_NAME).is_file():
            names.append(p.name)
    return names


def template_notebook_path() -> Path:
    return Path(__file__).resolve().parent / "templates" / NOTEBOOK_TEMPLATE_NAME


def write_plugin_export(cfg: ModelConfig, folder_name: str, model_params_path: str) -> Path:
    """
    Create plugin folder with engineered training CSV, feature_contract.json,
    plugin_metadata.json, and Colab starter notebook copy.
    """
    plugin_dir = resolve_plugin_dir(cfg.persistence["artifact_dir"], folder_name)
    if plugin_dir.exists() and any(plugin_dir.iterdir()):
        raise FileExistsError(f"Plugin directory already exists and is not empty: {plugin_dir}")

    train_model, test, features = prepare_training_data(cfg, show_progress=False)
    y_col = TARGET_COLUMN_HORIZON if TARGET_COLUMN_HORIZON in train_model.columns else TARGET_COLUMN

    plugin_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plugin_dir / TRAINING_DATASET_NAME
    train_model.to_csv(csv_path, index=False)

    contract: Dict[str, Any] = {
        "schema_version": PLUGIN_SCHEMA_VERSION,
        "features": features,
        "n_features": len(features),
        "feature_signature": feature_signature(features),
        "target_y_column": y_col,
        "timestamp_column": TIMESTAMP_COLUMN,
        "raw_target_column": TARGET_COLUMN,
        "data_source": cfg.data.get("source", "csv"),
        "forecast_horizon_min": int(cfg.data.get("forecast_horizon_min", 0)),
        "train_feature_time_shift_min": int(cfg.data.get("train_feature_time_shift_min", 0)),
        "train_disable_horizon_label_shift_when_feature_shifted": bool(
            cfg.data.get("train_disable_horizon_label_shift_when_feature_shifted", False)
        ),
        "validation_split": float(cfg.training.get("validation_split", 0.2)),
        "random_state": int(cfg.training.get("random_state", 42)),
    }
    (plugin_dir / FEATURE_CONTRACT_NAME).write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )

    mp = Path(model_params_path)
    meta: Dict[str, Any] = {
        "schema_version": PLUGIN_SCHEMA_VERSION,
        "created_utc": _utc_iso(),
        "plugin_folder": plugin_dir.name,
        "model_params_path": str(mp.resolve()),
        "model_params_sha256_prefix": (
            hashlib.sha256(mp.read_bytes()).hexdigest()[:16] if mp.is_file() else ""
        ),
        "artifact_root": str(Path(cfg.persistence["artifact_dir"]).resolve()),
        "feature_contract_file": FEATURE_CONTRACT_NAME,
        "training_dataset_csv": TRAINING_DATASET_NAME,
        "notebook_file": NOTEBOOK_TEMPLATE_NAME,
        "feature_signature": contract["feature_signature"],
        "n_features": len(features),
        "selected_model_file": None,
        "keras_sequence_n_steps": None,
    }
    (plugin_dir / PLUGIN_METADATA_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    tpl = template_notebook_path()
    if tpl.is_file():
        shutil.copy2(tpl, plugin_dir / NOTEBOOK_TEMPLATE_NAME)
    else:
        (plugin_dir / NOTEBOOK_TEMPLATE_NAME).write_text(
            '{"nbformat":4,"nbformat_minor":5,"metadata":{},"cells":[]}', encoding="utf-8"
        )

    return plugin_dir


def read_feature_contract(plugin_dir: Path) -> Dict[str, Any]:
    path = plugin_dir / FEATURE_CONTRACT_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {FEATURE_CONTRACT_NAME} in {plugin_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_plugin_metadata(plugin_dir: Path) -> Dict[str, Any]:
    path = plugin_dir / PLUGIN_METADATA_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {PLUGIN_METADATA_NAME} in {plugin_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def scan_model_candidates(plugin_dir: Path) -> List[Path]:
    """Find user model files; ignore known non-model artifacts."""
    ignore_names = {
        FEATURE_CONTRACT_NAME,
        PLUGIN_METADATA_NAME,
        TRAINING_DATASET_NAME,
        NOTEBOOK_TEMPLATE_NAME,
        "config_snapshot.yaml",
        "metrics.json",
        "manifest.json",
        "actual_vs_predicted.csv",
    }
    ignore_suffixes = {".csv", ".json", ".yaml", ".yml", ".ipynb", ".txt", ".md"}
    candidates: List[Path] = []
    for p in sorted(plugin_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name in ignore_names:
            continue
        suf = p.suffix.lower()
        if suf in {".joblib", ".pkl", ".pickle", ".keras", ".h5"}:
            candidates.append(p)
        elif suf not in ignore_suffixes and p.name.startswith("model_"):
            candidates.append(p)
    # SavedModel directory: folder containing keras_metadata.pb or saved_model.pb
    for p in sorted(plugin_dir.iterdir()):
        if not p.is_dir():
            continue
        if (p / "keras_metadata.pb").is_file() or (p / "saved_model.pb").is_file():
            candidates.append(p)
    return candidates


def _load_keras_saved_model(path: str) -> Any:
    """Load .keras / SavedModel with version-tolerant options and clear errors."""
    try:
        from tensorflow.keras.models import load_model
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow/Keras is required to load Keras models. Install tensorflow."
        ) from exc
    last_err: Exception | None = None
    for compile_flag in (False, True):
        try:
            return load_model(path, compile=compile_flag)
        except Exception as exc:
            last_err = exc
            continue
    assert last_err is not None
    msg = str(last_err)
    if "quantization_config" in msg or "Unrecognized keyword" in msg:
        raise RuntimeError(
            "This Keras model was saved with a newer Keras (e.g. Colab) than your VM's TensorFlow "
            "can deserialize (often `quantization_config` on Dense). Fix: upgrade TensorFlow on the "
            "miner VM to match Colab (e.g. `pip install -U 'tensorflow>=2.16'` then retry), or re-save "
            "the model in Colab using the same TF/Keras major version as the VM."
        ) from last_err
    raise RuntimeError(f"Could not load Keras model from {path!r}: {last_err}") from last_err


def _infer_keras_sequence_steps(model: Any) -> Tuple[Optional[int], str]:
    """Returns (n_steps or None for dense 2D input, rank hint '2d'|'3d')."""
    shp = getattr(model, "input_shape", None)
    if shp is None and hasattr(model, "layers") and model.layers:
        try:
            shp = model.layers[0].input_shape
        except Exception:
            shp = None
    if not shp:
        return None, "2d"
    # Typical: (None, n_steps, n_features) or (None, n_features)
    if len(shp) == 3 and shp[1] is not None:
        return int(shp[1]), "3d"
    if len(shp) == 3 and shp[1] is None:
        # Unknown timesteps — treat as sequence, cannot probe without user metadata
        return None, "3d_unknown"
    return None, "2d"


@dataclass
class CustomModelWrapper:
    kind: str  # "sklearn" | "keras"
    model: Any
    keras_sequence_n_steps: Optional[int] = None

    def predict_values(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "sklearn":
            preds = self.model.predict(X)
            return np.asarray(preds, dtype=float).reshape(-1)
        if self.kind == "keras":
            preds = self.model.predict(X, verbose=0)
            return np.asarray(preds, dtype=float).reshape(-1)
        raise ValueError(f"Unknown custom model kind: {self.kind}")


def load_custom_model(path: Path) -> CustomModelWrapper:
    suf = path.suffix.lower()
    if path.is_dir():
        model = _load_keras_saved_model(str(path))
        n_steps, rank = _infer_keras_sequence_steps(model)
        if rank == "3d_unknown":
            raise ValueError(
                "Keras model has undefined sequence length in input_shape; "
                "use a fixed input_shape (timesteps, features) or train a dense-input model."
            )
        return CustomModelWrapper(kind="keras", model=model, keras_sequence_n_steps=n_steps)

    if suf in {".joblib", ".pkl", ".pickle"}:
        obj = joblib.load(path)
        if isinstance(obj, dict) and "model" in obj:
            inner = obj["model"]
            if hasattr(inner, "predict"):
                return CustomModelWrapper(kind="sklearn", model=inner)
        if hasattr(obj, "predict"):
            return CustomModelWrapper(kind="sklearn", model=obj)
        raise ValueError(
            f"joblib at {path} must load a scikit-learn estimator (or dict with 'model' key), "
            f"got {type(obj)}."
        )

    if suf == ".keras" or suf == ".h5":
        model = _load_keras_saved_model(str(path))
        n_steps, rank = _infer_keras_sequence_steps(model)
        if rank == "3d_unknown":
            raise ValueError(
                "Keras model has undefined sequence length; use explicit input_shape (n_steps, n_features)."
            )
        return CustomModelWrapper(kind="keras", model=model, keras_sequence_n_steps=n_steps)

    raise ValueError(f"Unsupported custom model path: {path}")


def validate_custom_model_probe(
    wrapper: CustomModelWrapper,
    X: np.ndarray,
    feature_list: List[str],
) -> None:
    if np.any(~np.isfinite(X)):
        raise ValueError("Probe feature matrix contains NaN or Inf; check live data and feature contract.")
    if wrapper.kind == "sklearn":
        if X.ndim != 2 or X.shape[1] != len(feature_list):
            raise ValueError(
                f"Sklearn probe expects X shape (1, {len(feature_list)}); got {X.shape}."
            )
    elif wrapper.kind == "keras":
        if wrapper.keras_sequence_n_steps and wrapper.keras_sequence_n_steps > 1:
            if X.ndim != 3 or X.shape[0] != 1:
                raise ValueError(
                    f"Keras sequence probe expects X shape (1, n_steps, n_features); got {X.shape}."
                )
            if X.shape[1] != int(wrapper.keras_sequence_n_steps):
                raise ValueError(
                    f"Keras model expects n_steps={wrapper.keras_sequence_n_steps}; "
                    f"probe X has timesteps={X.shape[1]}."
                )
            if X.shape[2] != len(feature_list):
                raise ValueError(
                    f"Keras probe n_features mismatch: model expects {X.shape[2]}, "
                    f"contract has {len(feature_list)}."
                )
        else:
            if X.ndim != 2 or X.shape[1] != len(feature_list):
                raise ValueError(
                    f"Keras dense probe expects X shape (1, {len(feature_list)}); got {X.shape}."
                )
    out = wrapper.predict_values(X)
    if out.size < 1 or not np.isfinite(float(out.ravel()[0])):
        raise ValueError("Model prediction is empty or non-finite on probe input.")


def run_deploy_compatibility_probe(
    cfg: ModelConfig,
    plugin_dir: Path,
    model_path: Path,
    timestamp_str: str,
) -> Tuple[CustomModelWrapper, Optional[int], np.ndarray]:
    """
    Load model + build live/CSV probe X; validate shapes and a forward pass.
    Returns (wrapper, sequence_n_steps_or_none, X_probe).
    """
    contract = read_feature_contract(plugin_dir)
    meta = read_plugin_metadata(plugin_dir)
    features: List[str] = list(contract.get("features") or [])
    if not features:
        raise ValueError("feature_contract.json has empty features list.")

    sig = contract.get("feature_signature")
    if sig and sig != feature_signature(features):
        raise ValueError(
            "feature_contract.json feature_signature does not match current features list "
            "(file may be corrupted or hand-edited)."
        )

    wrapper = load_custom_model(model_path)
    seq_steps: Optional[int] = None
    if wrapper.kind == "keras":
        if wrapper.keras_sequence_n_steps and int(wrapper.keras_sequence_n_steps) > 1:
            seq_steps = int(wrapper.keras_sequence_n_steps)
        meta_seq = meta.get("keras_sequence_n_steps")
        if meta_seq is not None and int(meta_seq) > 1:
            if seq_steps is None:
                seq_steps = int(meta_seq)
            elif int(meta_seq) != int(seq_steps):
                raise ValueError(
                    f"plugin_metadata keras_sequence_n_steps={meta_seq} disagrees with "
                    f"model input n_steps={seq_steps}."
                )

    X, _ctx = live_probe_feature_matrix_for_custom(
        cfg,
        timestamp_str,
        features,
        seq_steps,
        use_resilient_forecast_fetch=True,
    )
    validate_custom_model_probe(wrapper, X, features)
    return wrapper, seq_steps, X


@dataclass
class CustomPluginDeployState:
    """In-memory state for a deployed custom plugin model (not serialized)."""

    plugin_dir: Path
    model_path: Path
    wrapper: CustomModelWrapper
    features: List[str]
    sequence_n_steps: Optional[int]


def update_plugin_metadata_after_deploy(
    plugin_dir: Path,
    model_path: Path,
    sequence_n_steps: Optional[int],
) -> None:
    meta_path = plugin_dir / PLUGIN_METADATA_NAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["selected_model_file"] = model_path.name
    meta["keras_sequence_n_steps"] = sequence_n_steps
    meta["last_deploy_utc"] = _utc_iso()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def prepare_artifact_dir(root_dir: str, model_type: str, run_id: str | None = None) -> Path:
    suffix = run_id or "run"
    out = Path(root_dir) / f"{_utc_stamp()}_{model_type}_{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def feature_signature(features: List[str]) -> str:
    payload = ",".join(features).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_manifest(out_dir: Path, manifest: Dict[str, Any]) -> Path:
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def write_config_snapshot(out_dir: Path, config_dict: Dict[str, Any]) -> Path:
    path = out_dir / "config_snapshot.yaml"
    path.write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")
    return path


def load_manifest(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)

export PYTHONPATH="$ROOT_DIR"

python - <<'PY'
from bittbridge.miner_example.miner.miner import ExampleMiner

with ExampleMiner() as miner:
    miner.run()
PY



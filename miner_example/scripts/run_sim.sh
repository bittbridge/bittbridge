#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../../.." && pwd)

export PYTHONPATH="$ROOT_DIR"

python "$ROOT_DIR/bittbridge/miner_example/simulate_validator/simulate_validator.py"



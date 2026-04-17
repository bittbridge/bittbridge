"""Put the `bittbridge` package root on sys.path so `miner_model_energy` imports resolve."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

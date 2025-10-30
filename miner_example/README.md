## Miner Example (USDT/CNY)

Quickstart to run a plug-and-play miner that serves predictions from a local model artifact and can be tested with a local validator simulator.

### Prerequisites
- Python 3.12+
- Virtualenv (recommended)

### Setup
1) Create and activate a venv (optional):
```
python3 -m venv .venv && source .venv/bin/activate
```
2) Install project dependencies from repo root:
```
pip install -e .
```
3) Install baseline training deps:
```
pip install -r bittbridge/miner_example/baseline_model/requirements.txt
```

### Train the baseline and export artifacts
```
python bittbridge/miner_example/baseline_model/train.py \
  --csv bittbridge/miner_example/usdt_cny_1h.csv \
  --out_dir bittbridge/miner_example/miner/artifacts
```

This creates `model.pkl` and `schema.json` in `miner/artifacts/`.

### Configure
Copy and adapt the example config:
```
cp bittbridge/miner_example/config.example.yml bittbridge/miner_example/config.yml
```
Key fields in `config.yml`:
- `mode`: local | testnet | mainnet
- `model.artifact_path`: path to `model.pkl`
- `model.backend`: sklearn | custom (see below)
- `update.on_start`: true to refresh artifact on startup

### Run the example miner
```
bash bittbridge/miner_example/scripts/run_miner.sh
```

### Backend modes
- `sklearn` (default): loads a pickled scikit-learn model from `model.artifact_path` and expects a `.predict()` API. The baseline trainer produces a compatible artifact.
- `custom`: you provide your own artifact format and logic by editing `miner/model_service.py` (`load()` and `predict_with_interval()`), keeping the same method signatures. Point prediction and interval must be produced by your model (no fallback bands).

### Run the local validator simulator
Ensure the miner is running in another terminal, then:
```
bash bittbridge/miner_example/scripts/run_sim.sh
```

### Next steps
- See `GUIDE.md` to integrate your own model/backend and enable local auto-updates.



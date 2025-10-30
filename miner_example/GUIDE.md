## Guide: Integrate Your Own Predictive Model

This guide shows how to plug in a custom model (sklearn/torch/tf) into the example miner. The miner uses the `ModelService` to load a local artifact and serve predictions to `Challenge` requests.

### Key files
- `miner/model_service.py`: load, update, and predict from your model.
- `miner/miner.py`: Bittensor miner subclass that calls `ModelService`.
- `miner/features.py`: lightweight timestamp-based feature generation.
- `miner/artifacts/`: place your artifacts here (e.g., `model.pkl`).

### Configuration
`config.yml` fields:
- `mode`: `local|testnet|mainnet`
- `model.backend`: `sklearn|custom` — selects how `ModelService` loads/predicts
- `model.artifact_path`: path to your artifact file
- `update.on_start`: `true|false` – refresh artifact on startup

### Implement your backend
Update `ModelService` methods:
- `load()`: Load your artifact (e.g., pickle, torch `state_dict`, or SavedModel).
- `predict_with_interval(timestamp: str) -> Tuple[float, Tuple[float, float]]`:
  - Return both point prediction and interval; no fallback bands are added by the miner.
- `maybe_update()`: If `on_start` is enabled, refresh artifact from local path and hot-reload if changed.

#### Using `sklearn` backend
- Produce a pickle with a scikit-learn compatible estimator (has `.predict(X)`).
- The baseline trainer adds an `interval_width` attribute from residuals; `ModelService` uses it to return an interval.

#### Using `custom` backend
- Set `model.backend: custom` and update `miner/model_service.py`:
  - `load()`: load your artifact (Torch/TF/ONNX/etc.).
  - `predict_with_interval(features) -> (float, (float, float))`: run inference and return both point prediction and interval. The miner will not synthesize bands.
  - Keep method signatures stable so `miner/miner.py` remains unchanged.

### Auto-update (local path)
- Replace the artifact file on disk atomically (e.g., write to temp, then move).
- `ModelService` can periodically check for a new mtime and hot-reload (optional). The example enables refresh on start only by default.

### Testing locally
1) Start the miner with `mode: local`.
2) Run the simulator to send a `Challenge`.
3) Verify `prediction` and `interval` are non-empty and in expected ranges.

### Extending protocol (optional)
The example leaves `protocol.py` unchanged. If you need more inputs (e.g., horizon), extend `Challenge` in `bittbridge/bittbridge/protocol.py` and ensure validators and miners agree. Keep the example as-is for compatibility.



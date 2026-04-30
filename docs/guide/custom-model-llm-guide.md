# Add a custom model (LLM quick guide)

This is a high-level workflow for adding a new ML model to the miner (for example, RandomForest, XGBoost, or your own regressor).

Goal: give an LLM the right project context, let it implement/tune the model, then plug it back into miner preflight so users can select it at runtime.

---

## What to give the LLM

From repo root, provide these files first:

- `miner_model_energy/pipeline.py` (main train/predict/save/load routing)
- `miner_model_energy/ml_config.py` (YAML config normalization/validation)
- `neurons/miner.py` (interactive model selection before miner startup)
- `model_params.yaml` (model hyperparameters in YAML)
- `miner_model_energy/models_cart.py` (simple reference model structure)
- `miner_model_energy/models_linear.py` (another minimal reference pattern)

If your new model is sequence-based (like RNN/LSTM style), also provide:

- `miner_model_energy/models_lstm.py`
- `miner_model_energy/models_rnn.py`

---

## LLM prompt template (copy/paste)

Use this prompt with the files above attached:

```text
Add a new model type named "<your_model>" to this miner project.

Requirements:
1) Create miner_model_energy/models_<your_model>.py using the same bundle/train/predict/save/load pattern as existing models.
2) Wire it into miner_model_energy/pipeline.py everywhere model_type is routed:
   - train_model
   - predict_single_test_row
   - predict_for_timestamp (if applicable)
   - persist_training_result
   - load_training_bundle_from_manifest
3) Add a models.<your_model> block with defaults in model_params.yaml.
4) Update miner_model_energy/ml_config.py to normalize/validate the new model config.
5) Update neurons/miner.py interactive prompt so users can choose "<your_model>" in preflight.

Constraints:
- Keep behavior backward-compatible for existing model types.
- Follow existing naming and style conventions in this repo.
```

---

## Practical notes

- You can use `models_cart.py` as the easiest starting template for non-sequence regressors.
- Renaming classes alone is not enough; routing in `pipeline.py` and miner preflight selection must be updated.
- Keep model artifact naming explicit (for example `model_<your_model>.joblib`) to avoid collisions.
- Shift-based train/prod alignment knobs live in `model_params.yaml` under `data`:
  - `train_feature_time_shift_min`: shifts raw weather columns forward in training (currently applied for `supabase_storage` source).
  - `train_disable_horizon_label_shift_when_feature_shifted`: when enabled with shift mode, training uses same-row `Total Load` as label instead of creating `Total Load (horizon)`.
- This shift mode is a pragmatic approximation for matching production-style forecast inputs.


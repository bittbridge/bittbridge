# 07 — Custom Models (Export, Train, Deploy)

This guide explains the end-to-end flow for creating and deploying a custom model.
---
### Important rules

- You CAN experiment in Colab without stopping your miner
- Your current miner can run while you test models
- Only restart the miner in TMUX session when you are ready

---

### Feature selection

- ❌ You MUST choose features in config file to create training dataset
- ❌ You CANNOT create new features in Colab
- ❌ You CANNOT drop columns from dataset
- ❌ You CANNOT rename columns from dataset
- ❌ You CANNOT switch places columns from dataset
- ❌ DO NOT scale X_train manually in Colab
- ❌ DO NOT modify dataset outside model
- ✅ If needed, use scaling inside model (e.g., TensorFlow layer)

Reason:
External changes will break the deployment pipeline

---

## Supported model families

Use one of these model families:

- **scikit-learn regressors** saved as `.joblib` (for example: `HistGradientBoostingRegressor`, `RandomForestRegressor`, `Ridge`, `SVR`)
- **TensorFlow / Keras regressors** saved as `.keras` (dense input or fixed-length sequence input)

The model must be a regressor and must be compatible with the feature schema in `feature_contract.json`.

---

## IMPORTANT BEFORE YOU START

1) Follow update and restart flow: [5. Update and restart](05-update-and-restart.md)

2) Run this on the GCP VM first:

```bash
sudo apt update
sudo apt install -y zip unzip
```

3) Then you can play and update feature settings in `model_params.yaml` 

---

## 1) Export custom plugin folder from miner

Run miner from the VM:

```bash
cd ~/bittbridge
python -m neurons.miner \
  --netuid 183 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```

In preflight:

- choose `[3] Custom model plugin`
- choose `Create a NEW plugin folder`
- enter folder name (example: `custom_test`)

Miner creates:

- `artifacts/custom_test/training_dataset_full.csv`
- `artifacts/custom_test/feature_contract.json`
- `artifacts/custom_test/plugin_metadata.json`
- `artifacts/custom_test/custom_train_colab.ipynb`

---

## 2) Prepare archive and copy with GCP VM UI

On VM terminal:

```bash
cd ~/bittbridge/artifacts
zip -r custom_test.zip custom_test
realpath custom_test.zip
```

Copy the `realpath` output and use **GCP VM web terminal UI download**:

- Open the terminal menu (three dots / More)
- Choose **Download file**
- Paste the full `realpath` path
- Download to local machine

If UI download does not work, restart web terminal (`Ctrl + R`) and try again.

---

## 3) Train in Colab

In Colab:

1. Open Collab
2. Click Upload notebook and upload .ipynb file from zip archive
3. Create content folder
4. Upload to content folder:
  1. `training_dataset_full.csv`
  2. `feature_contract.json`
  3. `plugin_metadata.json`

1. Pin versions to match VM:
  ```python
   !pip install -q tensorflow==2.21.0 keras==3.12.1 scikit-learn==1.7.2 pandas==2.2.2 numpy
  ```
2. Train and save model in the same folder:
  - `model_custom.joblib` (sklearn), or
  - `model_custom.keras` (Keras)

### IMPORTANT

- Do **not** edit `feature_contract.json`.
- Do **not** reorder or rename feature columns.
- Change only model/training logic.

---

## 4) Download trained model from Colab

---

## 5) Upload trained model back to GCP VM UI

Use **GCP VM web terminal UI upload**:

- Open terminal menu (three dots / More)
- Choose **Upload file**
- Upload `model_custom.joblib` or `model_custom.keras`

Then move it into plugin folder on VM:

```bash
mv ~/model_custom.joblib ~/bittbridge/artifacts/custom_test/
# or
mv ~/model_custom.keras ~/bittbridge/artifacts/custom_test/
```

---

## 6) Restart miner and deploy custom model

Run miner again:

```bash
cd ~/bittbridge
python -m neurons.miner \
  --netuid 183 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```

In preflight:

- choose `[3] Custom model plugin`
- choose `Create a NEW plugin folder?` -> `n`
- enter existing folder name (`custom_test`)
- pick model file
- confirm deploy

If compatibility check fails, miner offers:

- `[1] Exit`
- `[2] Baseline`
- `[3] Train built-in locally`

---

## Compatibility rules

- Features must match `feature_contract.json` exactly (same names, same order).
- sklearn model must support `.predict(X)` with `X.shape == (n_samples, n_features)`.
- Keras model can be:
  - dense input `(batch, n_features)`, or
  - fixed sequence input `(batch, n_steps, n_features)`.
- Keep package versions aligned between Colab and VM.

---

## Troubleshooting

- `Supabase live probe found no forecast row`: probe could not find row for `forecast_horizon_min`.
- sklearn `InconsistentVersionWarning`: version mismatch between training and VM.
- Keras `quantization_config`/deserialize errors: TensorFlow/Keras version mismatch; align versions and re-save model.

---

## Security note

`joblib`/pickle can execute arbitrary code when loading. Only deploy artifacts you trust.

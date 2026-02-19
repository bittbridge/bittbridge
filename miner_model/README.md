# Miner Model Plugin – Quick Reference

For the full setup guide (fork, clone, wallets, registration, API keys, running miner/validator), see the **[Guide Index](../README.md#guide)** or start with [01 – Before You Start](../docs/guide/01-before-you-start.md).

This page covers only miner-model-specific details.

---

## Where to Put Your Model and Data

Place your `.h5` model file and `.csv` data file anywhere in the `miner_model/` directory (or a subdirectory). The `student_models/my_model.py` file automatically discovers both.

**Example layout:**

```
miner_model/
├── student_models/
│   ├── my_model.py         ← Auto-discovers .h5 and .csv (excludes LSTM_outside_example)
│   └── helpers.py
├── LSTM_outside_example/   ← Example only; excluded from auto-discovery
├── my_model.h5             ← Your model (or in a subdir like my_models/)
├── data/
│   └── my_data.csv         ← Your data (or at root level)
└── miner_plugin.py
```

**Valid locations (all work):**
- `miner_model/my_model.h5`, `miner_model/my_data.csv`
- `miner_model/my_models/my_model.h5`, `miner_model/data/my_data.csv`

---

## How It Works

- `my_model.py` searches for `.h5` files in `miner_model/` (excluding `LSTM_outside_example`)
- `my_model.py` searches for `.csv` files in `miner_model/` (excluding `LSTM_outside_example`)
- Uses the first file found of each type
- Includes a ready-to-use `predict()` function

No configuration needed if you have one `.h5` and one `.csv`. To select specific files when you have multiple, edit SECTION 1 and SECTION 2 in `my_model.py`.

---

## Running the Miner Plugin

From the repo root (`bittbridge/`) with venv activated:

```bash
python -m miner_model.miner_plugin \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_MINER_NAME \
  --wallet.hotkey YOUR_HOTKEY_NAME
```

Use the **hotkey name** (e.g., `default`), not the ss58 address.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No student model found" | Ensure `my_model.py` exists in `student_models/` |
| "Model file does not have a predict() function" | Don't remove the predict function from `my_model.py` |
| "Failed to load model" | Add a `.h5` file in `miner_model/` or customize `model_path` in SECTION 1 |
| "No .csv data files found" | Add a `.csv` file in `miner_model/` or customize `data_path` in SECTION 2 |
| "Prediction returned None" | Check data format and sufficient historical data |

For network connectivity, port forwarding, and `[NO_SUBMISSION]` errors, see **[09 – Troubleshooting](../docs/guide/09-troubleshooting.md)**.

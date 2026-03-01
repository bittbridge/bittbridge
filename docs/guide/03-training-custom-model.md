# 3. Training Custom Model (Advanced – Currently Not Working)

> **Note:** This is an advanced path. The basic scenario is to run `neurons/miner.py` (moving average miner) – see [06 – Run Miner](06-run-miner.md). Custom LSTM/RNN model loading via TensorFlow is **currently not working** due to a model-loading bug. Use the built-in moving average miner instead.

To use your own prediction model (when fixed), you would train it in a Jupyter notebook and plug it into the miner. Use `miner_model/LSTM_outside_example` as reference.

---

## Reference: LSTM_outside_example

The folder [miner_model/LSTM_outside_example/](../../miner_model/LSTM_outside_example/) contains:

- `USDT_CNY_RNN_LSTM.ipynb` – Jupyter notebook that trains an LSTM model
- `USDT-CNY_scraper.csv` – Data file used for training
- `lstm_model.h5` – Saved model

Open the notebook to understand the workflow: load data, train, save model as `.h5`.

---

## Workflow

1. **Use compatible TensorFlow** – Check `miner_model/requirements.txt` (e.g., `tensorflow>=2.13.0`). Train with a version that matches what the VM will use.

2. **Train in Jupyter** – Run the notebook -> You should train model on provided data (`USDT-CNY_scraper.csv`). Save the model:
   ```python
   model.save('my_model.h5')
   ```

3. **Place files in miner_model/** – Put your `.h5` model file and `.csv` data file in `miner_model/`

4. **Push to your fork** – Commit and push so the model files are in your repository (You can check if they have appeared on github page of your repo).

---

## File Layout

```
miner_model/
├── my_model.h5          ← Your model file
├── my_data.csv          ← Your data file
├── student_models/
│   └── my_model.py      ← Auto-discovers .h5 and .csv (excludes LSTM_outside_example)
└── miner_plugin.py      ← Main executable file 
```

The `student_models/my_model.py` file automatically searches for `.h5` and `.csv` files. No configuration needed if you have one of each.

See [miner_model/README.md](../../miner_model/README.md) for details.

---

**Prev:** [02 – Local Setup](02-local-setup.md) | **Next:** [04 – GCP VM Setup](04-gcp-vm-setup.md) | [Back to Guide Index](../../README.md#guide)

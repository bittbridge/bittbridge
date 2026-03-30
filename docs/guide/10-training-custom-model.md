# 10. Training Custom Model (Advanced – Currently Not Working)

> **Note:** This is an optional, advanced topic at the **end** of the guide. The path everyone uses in class is [04 – Run Miner](04-run-miner.md) (`neurons/miner.py`, moving average). Custom LSTM/RNN model loading via TensorFlow is **currently not working** due to a model-loading bug. Use the built-in moving average miner instead.

To use your own prediction model (when fixed), you would train it in a Jupyter notebook and plug it into the miner. Use `miner_model/LSTM_outside_example` as reference.

---

## Reference: LSTM_outside_example

The folder [miner_model/LSTM_outside_example/](../../miner_model/LSTM_outside_example/) contains:

- `USDT_CNY_RNN_LSTM.ipynb` – Jupyter notebook (legacy USDT/CNY; for LoadMw use energydata.csv or ISO-NE API data)
- Data: use energydata.csv (Total Load) or fetch from ISO-NE API for LoadMw training
- `lstm_model.h5` – Saved model

Open the notebook to understand the workflow: load data, train, save model as `.h5`.

---

## Workflow

1. **Use compatible TensorFlow** – Check `miner_model/requirements.txt` (e.g., `tensorflow>=2.13.0`). Train with a version that matches what the VM will use.

2. **Train in Jupyter** – Run the notebook -> Train on LoadMw data (energydata.csv or API-derived). Save the model:
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

**Prev:** [09 – Incentive Mechanism](09-incentive-mechanism.md) | **Next:** — | [Back to Guide Index](../../README.md#guide)

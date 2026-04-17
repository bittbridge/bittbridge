# Training custom model (advanced)

> TensorFlow custom path **currently not working** for loading. Use [4. Run Miner](../04-run-miner.md) (moving average). [← Advanced index](README.md)

When fixed, train in Jupyter and plug into the miner. See `miner_model/LSTM_outside_example/`.

---

## Reference

[miner_model/LSTM_outside_example/](../../../miner_model/LSTM_outside_example/) — notebook, sample `lstm_model.h5`. For LoadMw use energydata.csv or ISO-NE API data.

---

## Workflow

1. Match TensorFlow in `miner_model/requirements.txt` on the machine that runs the miner.
2. Train in Jupyter → `model.save('my_model.h5')`
3. Put `.h5` and `.csv` under `miner_model/`
4. Commit/push if you use a remote

---

## Layout

```
miner_model/
├── my_model.h5
├── my_data.csv
├── student_models/
│   └── my_model.py
└── miner_plugin.py
```

Details: [miner_model/README.md](../../../miner_model/README.md).

---

[Advanced index](README.md) · [Guide](../../../README.md#guide)

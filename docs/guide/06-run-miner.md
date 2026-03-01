# 6. Run Miner

Run the miner **on the GCP VM** from the repo root (`bittbridge/`) with venv activated. Use tmux session to run miner, and detach to leave it running 24/7.

---

## Using tmux

tmux keeps your miner (and validator) running even if you disconnect from SSH. Create a session, run your processes, then detach.

### Create a tmux session

```bash
tmux new -s miner
```

You are now inside a tmux session named `miner`

### Run the miner

Activate venv and run the miner:

```bash
cd ~/bittbridge
source venv/bin/activate
```

**Basic scenario – moving average miner **

```bash
python -m neurons.miner \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```

This uses the built-in moving average model with `neurons/data.csv`

Use the **hotkey name** (e.g., `default`), not the ss58 address.

**Optional: add noise for testing** (e.g., dashboard development with multiple miners)

```bash
python -m neurons.miner \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug \
  -- test
```

---

## Advanced: Custom model miner (currently not working)

Training and loading custom LSTM/RNN models via `miner_model` is advanced and **currently not working** due to a TensorFlow model-loading bug. Use the basic moving average miner above instead.

If you need the custom model path for future use:

```bash
python -m miner_model.miner_plugin \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```

### Detach – leave running 24/7

Press **`Ctrl+b`** then **`d`**.

Your session stays running in the background. You can close the SSH window; the miner keep running.

### Reattach to the session

When you SSH back into the VM:

```bash
tmux attach -t miner
```

---

## Quick Reference

| Action | Keys / Command |
|--------|----------------|
| Create session | `tmux new -s miner` |
| Detach (leave running) | `Ctrl+b` release then `d` |
| Reattach | `tmux attach -t miner` |

---

## Where to Run

- **Working directory:** `~/bittbridge` (repo root)
- **venv:** Always run `source venv/bin/activate` before miner

---

**Prev:** [05 – Wallets and Tokens](05-wallets-and-tokens.md) | **Next:** [07 – Run Validator](07-run-validator.md) | [Back to Guide Index](../../README.md#guide)

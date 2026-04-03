# Update and restart

Use this when the repo has new commits on `main` and you need to **pull** on your GCP VM and **restart** your miner.

**Assumes:** You run the miner in tmux as in [4. Run Miner](04-run-miner.md), session name `miner`.

If you also run a validator, see [Run validator](advanced/run-validator.md) and the **optional validator steps** below.

---

## 1. Stop the miner

```bash
tmux attach -t miner
```

Press **`Ctrl+C`** to stop the miner. Detach: **`Ctrl+b`** then **`d`**.

---

## 2. Pull the latest code

```bash
cd ~/bittbridge
git pull origin main
```

---

## 3. Start the miner again

```bash
tmux attach -t miner
cd ~/bittbridge
source venv/bin/activate
```

Run the same command as before (e.g. Arrow UP):

```bash
python -m neurons.miner \
  --netuid 183 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```

Detach: **`Ctrl+b`** then **`d`**.

---

## Optional: if you run a validator

See [Run validator](advanced/run-validator.md). After pulling, restart the validator in its tmux session (`validator`), same `python3 -m neurons.validator ...` command as in that doc.

---

| Step | Action |
|------|--------|
| 1 | `tmux attach -t miner` → `Ctrl+C` → detach |
| 2 | `cd ~/bittbridge` && `git pull origin main` |
| 3 | Restart miner in `miner` session → detach |

---

[← 4. Run Miner](04-run-miner.md) · [Guide](../../README.md#guide)

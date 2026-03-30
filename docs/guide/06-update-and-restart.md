# 6. Update the Repo and Restart Miner & Validator

Use this guide whenever the repository has new commits (for example on `main`) and you need to **pull** the latest code on your GCP VM and **restart** both the miner and the validator safely.

**Assumes:** You already run the miner and validator in tmux as in [04 – Run Miner](04-run-miner.md) and [05 – Run Validator](05-run-validator.md), with sessions named `miner` and `validator`.

---

## 1. Stop the miner

```bash
tmux attach -t miner
```

In the pane where the miner is running, press **`Ctrl+C`** to stop it.

Detach from tmux (leave the session alive): **`Ctrl+b`** then **`d`**.

---

## 2. Stop the validator

```bash
tmux attach -t validator
```

Press **`Ctrl+C`** to stop the validator.

Detach: **`Ctrl+b`** then **`d`**.

---

## 3. Pull the latest code

From the repo root (often `main`; use your usual branch if different):

```bash
cd ~/bittbridge
git pull
```

If you use a **fork** and `git pull` does not bring in changes from the upstream class repository, add that repo as a remote and merge from it

---

## 4. Start the miner again

```bash
tmux attach -t miner
```

```bash
cd ~/bittbridge
source venv/bin/activate
```

Run the same command you used before (Arrow UP):

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

## 5. Start the validator again

```bash
tmux attach -t validator
```

```bash
cd ~/bittbridge
source venv/bin/activate

export WANDB_API_KEY="PASTE_YOUR_WANDB_API_KEY"

python3 -m neurons.validator \
  --netuid 183 \
  --subtensor.network test \
  --wallet.name validator \
  --wallet.hotkey default \
  --logging.debug
```

Detach: **`Ctrl+b`** then **`d`**.

Use the same wallet and hotkey names as in [05 – Run Validator](05-run-validator.md).

---

## Quick checklist

| Step | Action |
|------|--------|
| 1 | `tmux attach -t miner` → `Ctrl+C` → detach |
| 2 | `tmux attach -t validator` → `Ctrl+C` → detach |
| 3 | `cd ~/bittbridge` && `git pull` |
| 4 | Restart miner in `miner` tmux session |
| 5 | Restart validator in `validator` tmux session |

---

**Prev:** [05 – Run Validator](05-run-validator.md) | **Next:** [07 – Local Run (Advanced)](07-local-run-advanced.md) | [Back to Guide Index](../../README.md#guide)

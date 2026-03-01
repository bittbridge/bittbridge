# 7. Run Validator

Run the validator **on the GCP VM** from the repo root (`bittbridge/`) with venv activated. Use tmux session to run validator, and detach to leave it running 24/7.

---

## Using tmux

tmux keeps your  validator running even if you disconnect from SSH. Create a session, run your processes, then detach.

### Create a tmux session

```bash
tmux new -s validator
```

You are now inside a tmux session named `validator` 

---

## Run Validator

```bash
cd ~/bittbridge
source venv/bin/activate

export COINGECKO_API_KEY=PASTE_YOUR_COINGECKO_API_KEY_HERE
export WANDB_API_KEY="PASTE_YOUR_WANDB_API_KEY"

python3 -m neurons.validator \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_VALIDATOR_NAME \
  --wallet.hotkey YOUR_VALIDATOR_HOTKEY_NAME \
  --logging.debug
```

Use the **hotkey name** (e.g., `default`), not the ss58 address.
---

### Detach – leave running 24/7

Press **`Ctrl+b`** release then **`d`** 

Validator keep running. Reconnect later with:

```bash
tmux attach -t validator
```

---

**Prev:** [06 – Run Miner](06-run-miner.md) | **Next:** [08 – Local Run (Advanced)](08-local-run-advanced.md) | [Back to Guide Index](../../README.md#guide)

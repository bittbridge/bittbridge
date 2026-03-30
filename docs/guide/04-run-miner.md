# 4. Run Miner

Run the miner **on the GCP VM** from the repo root (`bittbridge/`) with venv activated. Use a tmux session to run the miner, then detach to leave it running 24/7.

**Before running:** You need ISO-NE API access at [Sign up / Create account](https://www.iso-ne.com/isoexpress/login?p_p_id=com_liferay_login_web_portlet_LoginPortlet&p_p_lifecycle=0&p_p_state=maximized&p_p_mode=view&_com_liferay_login_web_portlet_LoginPortlet_mvcRenderCommandName=%2Flogin%2Fcreate_account&saveLastPath=false). Put your username and password in `.env` (see [03 – Wallets and Tokens](03-wallets-and-tokens.md)).
---

## Using tmux

tmux keeps your miner (and validator) running even if you disconnect from SSH. Create a session, run your processes, then detach.

### Create a tmux session

```bash
tmux new -s miner
```

You are now inside a tmux session named `miner`.

### Run the miner

Activate venv and go to the repo:

```bash
cd ~/bittbridge
source venv/bin/activate
```

### Moving average miner (default)

The default miner uses a simple moving average over recent LoadMw values from the ISO-NE API.

**Customize the window length (encouraged):** So that different students do not all use identical settings, change the lookback length in `neurons/miner.py`. At the top of the file you will find:

```python
# Number of 5-minute steps for moving average (12 = 1 hour)
N_STEPS = 12
```

- Each step is **5 minutes** of data, so `N_STEPS = 12` means a **1-hour** window.
- Pick a different integer (for example between **6** and **24**) and save the file before you start the miner. Use a value that makes sense to you; the goal is to avoid everyone using the same default.

<img width="682" height="445" alt="Screenshot 2026-03-01 at 4 50 09 PM" src="https://github.com/user-attachments/assets/da02295a-c3d0-4afe-b787-a0ea6790d1f0" />

Run the miner:

```bash
python -m neurons.miner \
  --netuid 183 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```

This uses the built-in moving average model with LoadMw data from the ISO-NE API.

Use the **hotkey name** (e.g., `default`), not the ss58 address.

**Optional: add noise for testing** (e.g., dashboard development with multiple miners)

```bash
python -m neurons.miner \
  --netuid 183 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug \
  --test
```

Use the flag **`--test`** (one word). Do not write `-- test` with a space: in the shell, `--` ends option parsing, so `test` is not treated as the `--test` flag and noise stays off.


---

### Detach – leave running 24/7

Press **`Ctrl+b`** then **`d`**.

Your session stays running in the background. You can close the SSH window; the miner **keeps** running.

### Reattach to the session

When you SSH back into the VM:

```bash
tmux attach -t miner
```

When the repo is updated and you need to pull and restart **both** the miner and the validator, use **[06 – Update the Repo and Restart Miner & Validator](06-update-and-restart.md)**.

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
- **venv:** Always run `source venv/bin/activate` before the miner

---

**Prev:** [03 – Wallets and Tokens](03-wallets-and-tokens.md) | **Next:** [05 – Run Validator](05-run-validator.md) | [Back to Guide Index](../../README.md#guide)

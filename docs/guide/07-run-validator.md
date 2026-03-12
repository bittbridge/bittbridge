# 7. Run Validator

Run the validator **on the GCP VM** from the repo root (`bittbridge/`) with venv activated. Use tmux session to run validator, and detach to leave it running 24/7.

**Before running:** You need to sign up for ISO-NE API access at [Sign up / Create account](https://www.iso-ne.com/isoexpress/login?p_p_id=com_liferay_login_web_portlet_LoginPortlet&p_p_lifecycle=0&p_p_state=maximized&p_p_mode=view&_com_liferay_login_web_portlet_LoginPortlet_mvcRenderCommandName=%2Flogin%2Fcreate_account&saveLastPath=false). Then add your username and password to `.env` (copy from `.env.example` and set `ISO_NE_USERNAME` and `ISO_NE_PASSWORD`). Run `python test.py` to verify API access.

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

# .env is loaded automatically (ISO_NE_USERNAME, ISO_NE_PASSWORD)
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

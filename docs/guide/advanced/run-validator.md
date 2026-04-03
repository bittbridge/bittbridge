# Run validator

> **Advanced — not required for class.** Default path: [4. Run Miner](../04-run-miner.md). [← Advanced index](README.md)

Run the validator **on the GCP VM** from the repo root with venv activated. Use tmux, then detach.

**Before running:**

- Register the validator hotkey on subnet 183 (if needed):

  ```bash
  btcli subnet register --netuid 183 --subtensor.network test --wallet.name validator --wallet.hotkey default
  ```

- ISO-NE: [Create account](https://www.iso-ne.com/isoexpress/login?p_p_id=com_liferay_login_web_portlet_LoginPortlet&p_p_lifecycle=0&p_p_state=maximized&p_p_mode=view&_com_liferay_login_web_portlet_LoginPortlet_mvcRenderCommandName=%2Flogin%2Fcreate_account&saveLastPath=false). Set `ISO_NE_USERNAME` and `ISO_NE_PASSWORD` in `.env`.

---

## tmux

```bash
tmux new -s validator
```

---

## Run

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

Detach: **`Ctrl+b`** then **`d`**. Reattach: `tmux attach -t validator`

---

[Update and restart](../update-and-restart.md) (validator section) · [Advanced index](README.md) · [Guide](../../../README.md#guide)

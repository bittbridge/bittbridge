# Local run (advanced)

> Not required for class. Default: [2. GCP VM Setup](../02-gcp-vm-setup.md). [← Advanced index](README.md)

Run the miner (and optionally a validator) on your own machine instead of GCP.

---

## When this applies

- Testing without a VM
- You can open the miner port (e.g. **8091**) to the internet

---

## Steps (local)

1. [1. Before You Start](../01-before-you-start.md) — clone URL
2. Install Python 3, git, tmux; clone `https://github.com/bittbridge/bittbridge.git`
3. venv, `pip install -r requirements.txt`, `.env`
4. Skip GCP VM guide
5. [3. Wallets and Tokens](../03-wallets-and-tokens.md)
6. [4. Run Miner](../04-run-miner.md)
7. (Optional) [Run validator](run-validator.md)
8. [Update and restart](../update-and-restart.md) when code changes

Port forwarding: [Troubleshooting](troubleshooting.md).

---

[← Advanced index](README.md) · [Troubleshooting](troubleshooting.md) · [Guide](../../../README.md#guide)

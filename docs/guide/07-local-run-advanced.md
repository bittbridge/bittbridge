# 7. Local Run (Advanced)

You can run miner and validator on your own machine instead of a GCP VM. This section is optional; the **default path** is [02 – GCP VM Setup](02-gcp-vm-setup.md).

---

## When to Use This

- You want to test without spinning up a VM
- You have a stable local connection and can configure port forwarding
- You understand that miner and validator must be reachable from the internet

---

## Same Steps, Different Location

Follow the same workflow as the main guide, but perform all steps on your local machine:

1. [01 – Before You Start](01-before-you-start.md) – GitHub fork, concepts
2. Install Python 3, git, and tmux locally; clone **your fork** into a directory of your choice
3. Create a venv, `pip install -r requirements.txt`, and `cp .env.example .env` (same idea as on the VM)
4. Skip [02 – GCP VM Setup](02-gcp-vm-setup.md) (no cloud VM)
5. [03 – Wallets and Tokens](03-wallets-and-tokens.md) – Create/import wallets, tTAO, register
6. [04 – Run Miner](04-run-miner.md) – Run locally
7. [05 – Run Validator](05-run-validator.md) – Run locally
8. [06 – Update the Repo and Restart Miner & Validator](06-update-and-restart.md) – Same `git pull` and restart flow when the code changes

You must still open the miner port (e.g., **8091**) to the internet so validators can reach you.

---

## Firewall and Ports

See [08 – Troubleshooting](08-troubleshooting.md) for port forwarding and firewall details.

---

**Prev:** [06 – Update the Repo and Restart Miner & Validator](06-update-and-restart.md) | **Next:** [08 – Troubleshooting](08-troubleshooting.md) | [Back to Guide Index](../../README.md#guide)

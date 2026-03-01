# 8. Local Run (Advanced)

You can do everything described for the GCP VM locally on your laptop. This section is for users who prefer to run miner and validator on their own machine instead of a cloud VM.

---

## When to Use This

- You want to test without spinning up a VM
- You have a stable local connection and can configure port forwarding
- You understand that miner and validator must be reachable from the internet

---

## Same Steps, Different Location

Follow the same workflow as the main guide, but perform all steps on your local machine:

1. [02 – Local Setup](02-local-setup.md) – Fork, clone, environment
2. [03 – Training Custom Model](03-training-custom-model.md) – Optional; advanced, currently not working
3. Skip GCP VM Setup
4. [05 – Wallets and Tokens](05-wallets-and-tokens.md) – Create/import wallets, tTAO, register (do this locally)
5. [06 – Run Miner](06-run-miner.md) – Run locally
6. [07 – Run Validator](07-run-validator.md) – Run locally

---

## Port Forwarding

**Critical:** Validators need to connect to miners over the internet. If you run locally behind a router, you must configure port forwarding for TCP port 8091 (or your chosen port).

See [09 – Troubleshooting](09-troubleshooting.md) for port forwarding and firewall details.

---

**Prev:** [07 – Run Validator](07-run-validator.md) | **Next:** [09 – Troubleshooting](09-troubleshooting.md) | [Back to Guide Index](../../README.md#guide)

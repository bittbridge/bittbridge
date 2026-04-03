# 1. Before You Start

---

## What is repo clone?

It is creating a local copy of a remote project (from GitHub) on your computer, including all its files, branches, and full version history

> **Never push to upstream main:** Do not push changes to `bittbridge/bittbridge` unless you are a maintainer.

You will use this remote repo clone on the GCP VM in [2. GCP VM setup](02-gcp-vm-setup.md).

---

## Key concepts

Understanding these will help you follow the setup:

| Concept | Description |
|---------|-------------|
| **Virtual Machine (VM)** | A remote computer on Google Cloud that can run 24/7. You SSH into it to run your miner and validator. |
| **Virtual environment (`venv`)** | An isolated Python environment for this project. Keeps Bittensor and dependencies separate from the system Python. Activate it before running miner or validator commands. |
| **Miner** | Serves predictions to the network. Validators send challenges; miners respond with predictions. |
| **Validator** | Scores miners' work and sets weights. Queries miners, evaluates predictions, and rewards accurate ones. |

---

## Prerequisites

- **Google Cloud Platform:** Free trial activated (you will create the VM in the next guide).

---

**Next:** [2. GCP VM setup](02-gcp-vm-setup.md) · [Guide](../../README.md#guide)

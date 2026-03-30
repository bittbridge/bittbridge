# 1. Before You Start

---

## GitHub: fork before you clone

**Make sure your GitHub workflow is clear before you start.** A common mistake is cloning the upstream repository and trying to push changes there. You work from **your own fork**.

- **Practice first:** Fork [bittbridge/bittbridge](https://github.com/bittbridge/bittbridge), make a small edit, commit, and push to **your** fork.
- **Never push to upstream main:** Do not push changes to `bittbridge/bittbridge` unless you are a maintainer.

You will clone **your fork** on the GCP VM in [02 – GCP VM Setup](02-gcp-vm-setup.md).

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

**Prev:** — | **Next:** [02 – GCP VM Setup](02-gcp-vm-setup.md) | [Back to Guide Index](../../README.md#guide)

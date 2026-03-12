# 1. Before You Start

---

## GitHub Knowledge

**Make sure your GitHub knowledge is solid before starting.** A common mistake is cloning the main repository and trying to commit changes there. You must work from your own fork.

- **Practice first:** Fork the repository, make an edit (~~remove this line~~), commit, and push to your fork before attempting this task.
- **Always fork:** Create your own fork of the Bittbridge repository, then clone your fork.
- **Never commit to main:** Do not clone `bittbridge/bittbridge` directly and push changes to the main branch.

---

## Key Concepts

Understanding these will help you follow the setup:

| Concept | Description |
|---------|-------------|
| **Virtual Machine (VM)** | A remote computer (e.g., on Google Cloud) that runs 24/7. You SSH into it (read connect to remote terminal) to run your miner/validator. |
| **Virtual Environment (.venv)** | An isolated Python environment for this project. Keeps Bittensor and TensorFlow dependencies separate from other projects. Always activate it before running commands. |
| **Miner** | Serves predictions to the network. Validators send challenges; miners respond with predictions. |
| **Validator** | Scores miners' work and sets weights. Queries miners, evaluates predictions, and rewards accurate ones. |

---

## Package Versioning (TensorFlow)

**Be conscious of TensorFlow and other package versions.** The version on your VM must match the version used when you trained your model.

- TensorFlow is specified in `miner_model/requirements.txt` (e.g., `tensorflow>=2.13.0`).
- If you train your model locally with a different TensorFlow version, the miner may fail to load the model on the VM.
- Ensure the VM environment and your training environment use compatible versions.

---

## Prerequisites

- **Google Cloud Platform:** Activated free trial
- **Windows:** [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/about) with Ubuntu (use Linux for this guide)
- **macOS/Linux:** Terminal with Python 3.9+

---

**Prev:** — | **Next:** [02 – Local Setup](02-local-setup.md) | [Back to Guide Index](../../README.md#guide)

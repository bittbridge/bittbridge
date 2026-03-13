<div align="center">

# **Bittbridge**
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bittbridge/bittbridge)

---

## The Incentivized AI

[Discord](https://discord.gg/) • [Network](https://taostats.io/)
</div>

---

## What is Bittbridge?

Bittbridge is a Bittensor subnet for New England energy demand (LoadMw) prediction. **Miners** serve predictions; **Validators** score them and set weights. The subnet rewards accurate predictions.

### High-Level Flow

```mermaid
flowchart LR
    subgraph validators [Validators]
        V1[Validator]
    end
    subgraph miners [Miners]
        M1[Miner]
    end
    subgraph chain [Bittensor Chain]
        S[Subnet 183]
    end
    V1 -->|"Challenge"| M1
    M1 -->|"Prediction"| V1
    V1 -->|"Set weights"| S
    S -->|"Rewards"| M1
```

Validators send challenges to miners. Miners respond with predictions. Validators evaluate accuracy, set weights on-chain, and miners earn rewards based on their scores.

---

## Quick Start

1. **You need:** GitHub account, GCP free trial, tTAO (testnet tokens)
2. **ISO-NE API:** Sign up at [ISO Express (Create account)](https://www.iso-ne.com/isoexpress/login?p_p_id=com_liferay_login_web_portlet_LoginPortlet&p_p_lifecycle=0&p_p_state=maximized&p_p_mode=view&_com_liferay_login_web_portlet_LoginPortlet_mvcRenderCommandName=%2Flogin%2Fcreate_account&saveLastPath=false), then add your username and password to `.env` (copy from `.env.example`).
3. **Verify API:** Run `python test.py` to confirm API access before starting miner or validator.
4. **Standard path:** Deploy on a GCP VM (recommended)
5. **Follow the guide:** Step-by-step instructions below

---

## Guide

| # | Topic | Description |
|---|-------|-------------|
| 1 | [Before You Start](docs/guide/01-before-you-start.md) | GitHub, concepts, versioning |
| 2 | [Local Setup](docs/guide/02-local-setup.md) | Fork, clone, environment |
| 3 | [Training Custom Model](docs/guide/03-training-custom-model.md) | Advanced – currently not working (TensorFlow bug) |
| 4 | [GCP VM Setup](docs/guide/04-gcp-vm-setup.md) | Ubuntu, tmux, venv, firewall |
| 5 | [Wallets and Tokens](docs/guide/05-wallets-and-tokens.md) | On VM: create/import, tTAO, register |
| 6 | [Run Miner](docs/guide/06-run-miner.md) | Basic: `neurons/miner.py` (moving average) |
| 7 | [Run Validator](docs/guide/07-run-validator.md) | On GCP VM |
| 8 | [Local Run (Advanced)](docs/guide/08-local-run-advanced.md) | Run everything locally instead of VM |
| 9 | [Troubleshooting](docs/guide/09-troubleshooting.md) | Port forwarding, connectivity |

---
## Final Checklist

| ✅ | Task |
|---|------|
| | Github repo forked |
| | VM created |
| | Repo cloned on VM, venv activated |
| | Two wallets created or imported (miner & validator); mnemonics stored |
| | tTAO balance is positive |
| | Miner and validator hotkeys registered to subnet 183 |
| | ISO-NE credentials in `.env` (ISO_NE_USERNAME, ISO_NE_PASSWORD); WandB API key (validator only) |
| | Miner running in one tmux session (`python -m neurons.miner`) |
| | Validator running in another tmux session |
| | Detached from tmux (`Ctrl+b` `d`) – both running 24/7 |
| | Logs show Metagraph sync and request/response traffic |


---

## License

This repository is licensed under the [MIT License](LICENSE).

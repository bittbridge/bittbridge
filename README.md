<div align="center">

# **Bittbridge** <!-- omit in toc -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bittbridge/bittbridge)

---

## The Incentivized Internet <!-- omit in toc -->

[Discord](https://discord.gg/) • [Network](https://taostats.io/)
</div>

---
- [Quickstarter guide](#Subnet_Deployment_Guide)
- [License](#license)

---
# Subnet Deployment Guide (Testnet)

> ⚠️ Always double-check which **network/subnet/wallet** you’re working on during each step

> ⚠️ Make sure that you're using most recent versions of Bittensor SDK & Btcli 

> ⚠️ For a good practice be sure that you're using Virtual env 

> ⚠️ To know your address / wallets -> ```*btcli wallet list*```

> ⚠️ You should have tTAO to register and stake on a testnet

---
## **Prerequisites**

 For Windows you should have [**WSL 2** (Windows Subsystem for Linux)](https://learn.microsoft.com/en-us/windows/wsl/about) 
 For Windows use Linux Ubuntu to run this guide ☝️

---
## Step 1 – Clone Template & Set Up Environment

> ⚠️ Create directory / folder on your machine where you will clone project. 

We’ll use our subnet code - main branch [link](https://github.com/bittbridge/bittbridge).  

```bash

# 2.1 Clone the repo

git clone https://github.com/bittbridge/bittbridge

cd bittbridge


# 2.2 Create and activate virtual environment

python3 -m venv venv

source venv/bin/activate

# 2.3 Install dependencies in editable mode

pip install -e .

# Dependencies that should be to run:
pip install bittensor
pip install bittensor-cli
pip install wandb
pip install pytz

```
  
Reference (step №1): 

🔗 https://github.com/opentensor/bittensor-subnet-template/blob/main/docs/running_on_testnet.md

---
## Step 2 – Create Wallets (Validator, Miner)

Follow the instructions to create wallets, you need to create 2 wallets:
a) One for - Miner
b) Second for - Validator

```bash
btcli w create
```

> ⚠️ Store mnemonic phrases for coldkey and hotkey, you will end up with **4** mnemonic

Reference (step №2): 
🔗 https://github.com/opentensor/bittensor-subnet-template/blob/main/docs/running_on_testnet.md

---
## Step 3 – (Optional) Get faucet tokens

If you don't have sufficient faucet (tTAO) tokens, ask the [Bittensor Discord](https://discord.com/channels/799672011265015819/830068283314929684) community for faucet tokens.
Workflow to get tokends:
1) Go to Bittensor discord channel
2) Go to help forum channel -> Requests for Testnet TAO
3) Leave request according to a special format mentioned in pinned message in this channel.
4) After some time you will receive on wallet your requested for tTAO, you can check balance with:
```bash
btcli w balance --network test
# Wallet name with balance
```

> ⚠️ If you have any issues or you're waiting tTAO more than 1 day -> ask Dmitrii to send it to you.

## Step 4 – Transfer Tokens to Wallets

Check wallet which should have tTAO balance.

```bash
btcli w balance --network test
# Wallet name with balance
```

Transfer tTAO from a wallet with a balance small amount to Miner & Validator wallets (to the wallets you just created), follow these commands:

```bash
# For miner:
btcli wallet transfer \
--amount 1 \
--wallet.name WALLET_NAME_WITH_tTAO \
--destination MINER_WALLET_COLDKEY_ADDRESS \
--network test

# For validator:
btcli wallet transfer \
--amount 1 \
--wallet.name WALLET_NAME_WITH_tTAO \
--destination VALIDATOR_WALLET_COLDKEY_ADDRESS \
--network test

```

---
## Step 5 – Check if subnet is running on a blockchain

Verify with:

```bash

btcli subnet show --network test --netuid 420

```

---
## Step 6 – Register Validator & Miner Hotkeys

```bash

btcli subnet register --netuid 420 --subtensor.network test --wallet.name YOUR_MINER_NAME --wallet.hotkey YOUR_MINER_HOTKEY_NAME

btcli subnet register --netuid 420 --subtensor.network test --wallet.name YOUR_VALIDATOR_NAME --wallet.hotkey YOUR_VALIDATOR_HOTKEY_NAME

```

Optional checks:

```bash

btcli wallet overview --wallet.name YOUR_VALIDATOR_HOTKEY_NAME --subtensor.network test

btcli wallet overview --wallet.name YOUR_MINER_HOTKEY_NAME --subtensor.network test

```

---
## Step 7 – Collect your API key at CoinGecko so Validator can evaluate Miner's work 

#### Obtain & Setup CoinGecko API Key (Validators Only)

Before starting the process, validators would be required to procure a CoinGecko API Key. Please follow the instructions mentioned below:  

- Log in to [CoinGecko](https://www.coingecko.com/en/developers/dashboard) and generate an API key in your account settings.
- Save it somewhere for next step

## Step 8 – Collect your API at WandB to store validator's work

#### Obtain & Setup WandB API Key (Validators Only)

Before starting the process, validators would be required to procure a WANDB API Key. Please follow the instructions mentioned below:  

- Log in to [Weights & Biases](https://wandb.ai/) and generate an API key in your account settings.
- Save it somewhere for next step

For help finding your wandb api key, look [here](https://docs.wandb.ai/support/find_api_key/)

---
## Step 9 – Run Miner & Validator

Run these commands **from the `bittbridge` directory with activated venv**:

> ⚠️ !!! **Run in 2 different terminals:** !!!
> ⚠️ !!! **Be sure that venv is activated in both terminals ** !!!

**Terminal A – Validator**
```bash

# Validator
# In the terminal where you will start validator paste these commands:

# Set the variable `COINGECKO_API_KEY` in your environment:
export COINGECKO_API_KEY="PASTE_YOUR_COINGECKO_API_KEY_HERE"

# Set the variable `WANDB_API_KEY` in your environment:
export WANDB_API_KEY="PASTE_YOUR_API_KEY"

# Run validator 
python3 -m neurons.validator \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_VALIDATOR_NAME \
  --wallet.hotkey YOUR_VALIDATOR_HOTKEY_NAME \
  --logging.debug
```

**Terminal B – Miner**
```bash
# Miner
# Set the variable `COINGECKO_API_KEY` in your environment for testing purposes:
export COINGECKO_API_KEY="PASTE_YOUR_COINGECKO_API_KEY_HERE"

python3 -m neurons.miner \
  --netuid 420 \
  --subtensor.network test \
  --wallet.name YOUR_MINER_NAME \
  --wallet.hotkey YOUR_MINER_HOTKEY_NAME \
  --logging.debug
```

---

## Final Checklist

| ✅   | Task                                                                                                         |
| --- | ------------------------------------------------------------------------------------------------------------ |
|     | new directory created                                                                                        |
|     | venv created & activated                                                                                     |
|     | Repo cloned (`bittbridge/`) and dependencies installed                                                       |
|     | Two wallets created: miner & validator (each has cold+hot); mnemonics stored; `btcli wallet list` shows them |
|     | Have tTAO on testnet (via Discord faucet or Dmitrii)                                                         |
|     | Sent small tTAO to both coldkeys (`btcli wallet transfer --dest ...`)                                        |
|     | Registered miner hotkey to subnet `netuid 420`                                                               |
|     | Registered validator hotkey to subnet `netuid 420`                                                           |
|     | Validator: CoinGecko API key + W&B API key set;                                                              |
|     | Two terminals open: validator terminal has env vars (CoinGecko & WandB) exported                             |
|     | Validator launched with                                                                                      |
|     | Miner launched with                                                                                          |
|     | Logs show metagraph sync and request/response traffic                                                        |

```

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```

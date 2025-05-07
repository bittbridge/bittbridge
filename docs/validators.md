# Validators

<div align="center">

| This repository is the official codebase<br>for Bittensor Subnet XX (SNXX) v1.0.0+,<br>which was released on MONTH DAY YEAR. | **Testnet UID:**  TBD <br> **Mainnet UID:**  TBD |
| - | - |

</div>

## Compute Requirements

| Validator |
| :-------: |
|  8gb RAM  |
|  2 vCPUs  |

## Installation

First, install PM2:
```
sudo apt update
sudo apt install nodejs npm
sudo npm install pm2@latest -g
```

Verify installation:
```
pm2 --version
```

Clone the repository:
```
git clone https://github.com/bittbridge/snporacle.git
cd snp_oracle
```

Create and source a python virtual environment:
```
python3 -m venv
source .venv/bin/activate
```

Install the requirements with poetry:
```
pip install poetry
poetry install
```

## Configuration

#### Environment Variables
First copy the `.env.validator.template` file to `.env`

```shell
cp .env.validator.template .env
```

Update the `.env` file with your validator's values for the following properties.

> TODO

#### Makefile
Edit the Makefile with your validator's values for the following properties.

> TODO

#### Obtain & Setup WandB API Key
Before starting the process, validators would be required to procure a WANDB API Key. Please follow the instructions mentioned below:<br>

- Log in to <a href="https://wandb.ai">Weights & Biases</a> and generate an API key in your account settings.
- Set the variable `WANDB_API_KEY` in the `.env` file.
- Finally, run `wandb login` and paste your API key. Now you're all set with weights & biases.

## Deploying a Validator
**IMPORTANT**
> Make sure your have activated your virtual environment before running your validator.
>
> Inspect the Makefile for port configurations.
1. Run the command:
    ```shell
    make validator ENV_FILE=.env.validator
    ```

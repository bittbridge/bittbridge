# Miners

# **DO NOT RUN THE BASE MINER ON MAINNET!**

> **The base miner provided in this repo is _not intended_ to be run on mainnet!**
>
> **If you run the base miner on mainnet, you are not guaranteed to earn anything!**
> It is provided as an example to help you build your own custom models!
>

<div align="center">

| This repository is the official codebase<br>for Bittensor Subnet XX (SNXX) v1.0.0+,<br>which was released on MONTH DAY YEAR. | **Testnet UID:**  TBD <br> **Mainnet UID:**  TBD |
| - | - |

</div>

## Compute Requirements

|   Miner   |
|-----------|
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
First copy the `.env.miner.template` file to `.env`

```shell
cp .env.miner.template .env
```

Update the `.env` file with your miner's values for the following properties.

> TODO

#### Makefile
Edit the Makefile with your miner's values for the following properties.

> TODO

## Deploying a Miner
We highly recommend that you run your miners on testnet before deploying on mainnet.

**IMPORTANT**
> Make sure you have activated your virtual environment before running your miner.


#### Base Miner

1. Run the command:
    ```
    make miner ENV_FILE=.env.miner
    ```

#### Custom Miner

1. Write a custom forward function stored in `snporacle/miners/your_file.py`
    - `miner.py` searches for a function called `forward` contained within your provided file `--forward_function your_file`
    - This function should handle how the miner responds to requests from the validator
    - Within the forward function, `synapse.predictions` and `synapse.interval` should be set.
    - See base_miner.py for an example
2. Add a command to Makefile.
    - copy the miner command and rename it (e.g. miner_custom) in Makefile
    - replace the `--forward_function base_miner` with `--forward_function your_file`
3. Run the Command:
    ```
    make miner_custom ENV_FILE=.env.custom
    ```

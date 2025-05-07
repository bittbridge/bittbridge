<div align="center">

# **Bittensor SN TBD - S&P 500 Oracle**

<br>

<div align="center">

*This subnet is supported by the University of Connecticut in partnership with Yuma, a core contributor to the Bittensor ecosystem, as part of the BittBridge program.*

| This repository is the official codebase<br>for Bittensor Subnet XX (SNXX) v1.0.0+,<br>which was released on MONTH DAY YEAR. | **Testnet UID:**  TBD <br> **Mainnet UID:**  TBD |
| - | - |

</div>

<br>

|     |     |
| :-: | :-: |
| **Status** | <img src="https://img.shields.io/github/v/release/bittbridge/snporacle?label=Release" height="25"/> <img src="https://img.shields.io/github/actions/workflow/status/bittbridge/snporacle/ci.yml?label=Build" height="25"/> <br> <a href="https://github.com/pre-commit/pre-commit" target="_blank"> <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&label=Pre-Commit" height="25"/> </a> <a href="https://github.com/psf/black" target="_blank"> <img src="https://img.shields.io/badge/code%20style-black-000000.svg?label=Code%20Style" height="25"/> </a> <br> <img src="https://img.shields.io/github/license/bittbridge/snporacle?label=License" height="25"/> |
| **Activity** | <img src="https://img.shields.io/github/commit-activity/m/bittbridge/snporacle?label=Commit%20Activity" height="25"/> <img src="https://img.shields.io/github/commits-since/bittbridge/snporacle/latest/dev?label=Commits%20Since%20Latest%20Release" height="25"/> <br> <img src="https://img.shields.io/github/release-date/bittbridge/snporacle?label=Latest%20Release%20Date" height="25"/> <img src="https://img.shields.io/github/last-commit/bittbridge/snporacle/dev?label=Last%20Commit" height="25"/> <br> <img src="https://img.shields.io/github/contributors/bittbridge/snporacle?label=Contributors" height="25"/> |
| **Compatibility** | <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbittbridge%2Fsnporacle%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.python&logo=python&label=Python&logoColor=yellow" height="25"/> <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbittbridge%2Fsnporacle%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.bittensor&prefix=v&label=Bittensor" height="25"/> <br> <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fbittbridge%2Fsnporacle%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.yfinance&label=yfinance" height="25"/> |
| **Social** | <a href="https://google.com/" target="_blank"> <img src="https://img.shields.io/website?url=https%3A%2F%2Fgoogle.com%2F&up_message=UConn%20Website&label=Website" height="25"/> </a> <br> <a href="https://google.com/" target="_blank"> <img src="https://img.shields.io/twitter/follow/bittbridge" height="25"/> </a> |

</div>

## Introduction

UConn BittBridge is launching the S&P 500 Oracle. This subnet incentivizes accurate short term price forecasts of the S&P 500 during market trading hours.

Miners perform short term price predictions on the S&P 500.

Validators store price forecasts for the S&P 500 and compare these predictions against the true price of the S&P 500 as the predictions mature.

## Usage

<div align="center">

| [Miner Docs](https://github.com/bittbridge/snporacle/blob/dev/docs/miners.md) | [Validator Docs](https://github.com/bittbridge/snporacle/blob/dev/docs/validators.md) |
| - | - |


</div>

## Incentive Mechanism
Please read the incentive mechanism white paper (location TBD) to understand exactly how miners are scored and ranked.

For transparency, there are two key metrics detailed in the white paper that will be calculated to score each miner:
1. **Directional Accuracy** - was the prediction in the same direction of the true price?
2. **Mean Absolute Error** - how far was the prediction from the true price?

## Design Decisions
Integration into financial markets will expose Bittensor to the largest system in the world; the global economy. The S&P 500 serves as a perfect starting place for financial predictions given its utility and name recognition. Financial market predictions were chosen for three main reasons:

#### Utility
Financial markets provide a massive userbase of professional traders, wealth managers, and individuals alike.

#### Objective Rewards Mechanism
By tying the rewards mechanism to an external source of truth, the defensibility of the subnet regarding gamification is quite strong.

#### Adversarial Environment
The adversarial environment, especially given the rewards mechanism, will allow for significant diversity of models. Miners will be driven to acquire different datasets, implement different training methods, and utilize different Neural Network architectures in order to develop the most performant models.

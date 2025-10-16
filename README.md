<p align="center">
  <img src="https://raw.githubusercontent.com/bittbridge/bittbridge/FS-Drafting-ReadMe/docs/logo_FXspresso.svg" width="150"/>
</p>

<h1 align="center">FXspresso Subnet</h1>

<div align="center">

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bittbridge/bittbridge)

</div>

<p align="center">
  <b>A decentralized Bittensor subnet for USD/CNY exchange rate forecasting</b><br>
  Developed by the 
  <a href="https://www.linkedin.com/posts/bittbridge-uconn_meet-the-bittbridge-team-behind-every-activity-7374484770837188608-UzTE">
    BittBridge Team
  </a> at the University of Connecticut (UConn)<br>
  In collaboration with Yuma, a DCG Company
</p>


<p align="center">
  <a href="https://bittensor.com">Bittensor</a> · 
  <a href="https://discord.gg">Discord</a> · 
  <a href="https://taostats.io/">Network</a>
</p>



---

## 💱 Introduction  

The FXspresso Subnet utilizes Bittensor’s decentralized intelligence framework to generate probabilistic, data-driven forecasts of the USD/CNY exchange rate through a network of competing and cooperating models.

Participants compete and collaborate to produce accurate and well-calibrated forecasts, while validators ensure transparent and fair evaluation.

This subnet is **experimental and research-oriented** — focused on testing how decentralized intelligence can improve financial forecasting.

---

## ⚙️ How It Works  

1. **Miners** provide one-hour-ahead predictions for the USD/CNY exchange rate.
   Each forecast includes a **point estimate** and an **interval** representing confidence.  
2. **Validators** collect predictions, compare them to the ground truth rate (from [CoinGecko](https://www.coingecko.com/en/api)), and compute accuracy scores.  
3. **Rewards** are distributed based on multi-step evaluation and reputation smoothing.

```
    ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
    │       Miners         │       │      Validators      │       │       Rewards        │
    │          ↓           │ ----→ │          ↓           │ ----→ │          ↓           │
    │     Predictions      │       │       Scoring        │       │ Distributed on-chain │
    └──────────────────────┘       └──────────────────────┘       └──────────────────────┘

```
---


## 💰 Incentive Mechanism  

Our scoring design ensures that miners are rewarded for **accuracy**, **confidence**, and **consistency**.

**Step 1 – Point Forecast Evaluation**  
- Miners are ranked by how close their prediction is to the actual rate.  
- Smaller error = higher score.  

**Step 2 – Interval Forecast Evaluation**  
- Forecast intervals are checked for whether the actual price falls inside.  
- Narrower intervals (that still capture the truth) score higher.  

**Step 3 – Combined Share**  
- Point and interval scores are averaged for a balanced reward.  

**Step 4 – Exponential Moving Average (EMA)**  
- Each miner’s final weight blends new performance with historical performance:  
  **wₜ = 0.58 × sₜ + 0.42 × wₜ₋₁**
- This stabilizes reputation and reduces the impact of lucky guesses.  

🔗 For full scoring examples and formulas, see the [Incentive Mechanism Methodology](docs/incentive_methodology.md).

---

## 🧠 Roles in the Subnet  

- **Miners** → Submit hourly predictions (expected value + range).  
- **Validators** → Evaluate forecasts, compute scores, and submit weights.  
- **Observers/Users** → Access results for analytics, visualization, financial modeling, applications, or any other purpose that benefits from decentralized predictive insights.

---

## 🚀 Getting Started  

### Run a Miner  
Follow the step-by-step guide here: [Miner Setup](docs/running_miner_guide.md)

### Run a Validator  
Follow the setup guide here: [Validator Setup](docs/running_validator_guide.md)

---

## 🔒 Integrity  

- **Hash Verification:** All model files and predictions are hashed (SHA-256).  
- **Cutoff Enforcement:** Predictions must be submitted before each round’s deadline.  
- **Reputation Decay:** Inactive miners gradually lose weight (EMA decay).  

---

## 📅 Roadmap  

| Phase | Milestone | Status |
|-------|------------|--------|
| Q4 2025 | Launch testnet for USD/CNY pair | ✅ |
| Q1 2026 | Public leaderboard and analytics dashboard | 🔜 |
| Q2 2026 | Expand to multiple currency pairs | 🔜 |
| Q3 2026 | Introduce backtesting-based scoring | 🔜 |

---

## ⚖️ License  

This repository is licensed under the MIT License:

```
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

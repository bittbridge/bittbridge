
# **BittBridge Subnet Incentive Mechanism**

## **Energy Demand Forecasting**
### **Overview**

The BittBridge subnet rewards participants (miners) for producing accurate forecasts of short-term electricity demand. Each miner submits a prediction of future energy demand for a specified timestamp. Validators compare the predictions to the observed ground truth and compute a score for each miner.

The goal of the incentive mechanism is to:

- reward **forecast accuracy**
- provide **continuous incentives for improvement**
- remain **transparent and easy to understand**
- allow fair comparison between different forecasting models

All miners who submit a valid prediction receive a positive score, but more accurate predictions receive higher rewards.

```mermaid
flowchart TD
    A[Validator requests forecast for target timestamp]
    B[Miners submit point forecasts]
    C[Actual energy demand becomes available]
    D[Compute percentage error for each miner]
    E[Convert error to score using exponential decay]
    F[Normalize scores into weight vector]
    G[Validator submits weights to blockchain]
    H[Yuma Consensus aggregates validator weights]
    I[Emissions distributed to miners]
    J[Leaderboard updates based on scores]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    E --> J
```
---
# **Prediction Task**
Each round:

1. The validator requests a prediction for a future timestamp.
2. Miners submit a **point forecast** of energy demand.
3. Once the actual demand value becomes available, the validator evaluates all submitted predictions.

Example prediction round:

|**Miner**|**Prediction**|**Actual**|
|---|---|---|
|Good Miner|10340|10353.585|
|Medium Miner|10150|10353.585|
|Bad Miner|9500|10353.585|
# **Error Calculation**
Forecast accuracy is measured using **absolute percentage error**:
$error_i = \frac{|prediction_i - actual|}{|actual|}$

Where:
- prediction_i = miner prediction
- actual = observed energy demand
- error_i = relative prediction error

This metric is scale-independent, meaning it works correctly regardless of whether demand values are in hundreds or tens of thousands.

Example calculation:

| **Miner**    | **Prediction** | **Actual** | **Absolute Error** | **Percentage Error** |
| ------------ | -------------- | ---------- | ------------------ | -------------------- |
| Good Miner   | 10340          | 10353.585  | 13.585             | 0.001312 (0.131%)    |
| Medium Miner | 10150          | 10353.585  | 203.585            | 0.019663 (1.966%)    |
| Bad Miner    | 9500           | 10353.585  | 853.585            | 0.082443 (8.244%)    |
Smaller errors indicate more accurate forecasts.

# **Converting Error into Score**

  

Errors are converted into scores using an exponential decay function:

$score_i = e^{-error_i / T}$

This produces scores between **0 and 1**:
- perfect prediction → score close to **1**
- moderate error → score decreases smoothly
- very large error → score becomes small but remains positive

Example scoring using T = 0.097634:

|**Miner**|**Prediction**|**Actual**|**Percentage Error**|**Score**|
|---|---|---|---|---|
|Good Miner|10340|10353.585|0.001312|0.986651|
|Medium Miner|10150|10353.585|0.019663|0.817587|
|Bad Miner|9500|10353.585|0.082443|0.429810|

Because the scoring function is continuous, **every improvement in forecast accuracy increases the score**.

## **Parameter** T

The parameter T determines how quickly scores decrease as forecast error increases. Smaller values of T create stricter scoring, while larger values make scoring more forgiving.

To calibrate this parameter objectively, we analyzed **one year (March 2026-2026) of historical energy demand data** using a simple baseline (5 hours Moving Average) forecasting method.
### **Baseline Model**

A **5-hour moving average** forecast was used as a naive baseline model. Since the dataset is sampled every 5 minutes, the moving average window consisted of **60 observations**.

The baseline prediction error was computed for the entire dataset using the same absolute percentage error metric used in the scoring mechanism.
### **Baseline Error Distribution**

The distribution of baseline errors was:

| **Percentile** | **Percentage Error** |
| -------------- | -------------------- |
| 10%            | 2.09%                |
| 25%            | 5.08%                |
| 50%            | **9.76%**            |
| 75%            | 15.78%               |
| 90%            | 21.41%               |
The **median error** of the baseline model was:

T = 0.097634 or approximately **9.76%**.

### **Why the Median Error Was Chosen**

Using the median baseline error as T provides a natural calibration:
- the naive baseline model receives a score near **0.37**
- better models receive significantly higher scores
- weaker models receive lower scores
- the scoring system preserves meaningful differentiation between forecasting approaches
This ensures the incentive mechanism rewards **model improvements beyond simple baselines**.
# **Reward Distribution**
Once scores are computed for all miners in a round, they are converted into a normalized weight vector.
$weight_i = \frac{score_i}{\sum_j score_j}$

These weights represent each miner’s relative performance and are submitted by validators to the blockchain.

Example reward distribution:

| **Miner**    | **Prediction** | **Actual** | **Error** | **Score** | **Weight** |
| ------------ | -------------- | ---------- | --------- | --------- | ---------- |
| Good Miner   | 10340          | 10353.585  | 0.001312  | 0.986651  | 0.441643   |
| Medium Miner | 10150          | 10353.585  | 0.019663  | 0.817587  | 0.365967   |
| Bad Miner    | 9500           | 10353.585  | 0.082443  | 0.429810  | 0.192390   |
More accurate predictions therefore receive proportionally larger rewards.

# **Key Properties of the Mechanism**

The scoring mechanism has several important characteristics:
### **Continuous Incentives**
Even small improvements in prediction accuracy increase a miner’s score and rewards.
### **Fair Comparison**
Relative error ensures forecasts are evaluated consistently regardless of demand magnitude.
### **Robustness**
All valid predictions receive positive scores, preventing hard cutoffs or unstable rankings.
### **Transparency**
The scoring process is simple and reproducible using publicly available formulas.

---
# **Summary**

The BittBridge incentive mechanism evaluates forecasting performance using absolute percentage error and converts this error into rewards through an exponential scoring function.

The parameter T was calibrated using **one year of historical demand data** and reflects the median error of a simple baseline forecasting model.

This design ensures:

- clear incentives for improved forecasting models
- fair competition among miners
- transparent and reproducible evaluation

Participants are encouraged to develop models that outperform the baseline in order to maximize their rewards.

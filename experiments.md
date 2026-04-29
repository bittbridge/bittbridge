# BittBridge Final Project Experiment Tracker

Goal: find the model with the lowest validation MAPE and stable live deployment.

| Exp | Model | Features | Val MAE | Val RMSE | Val MAPE | R2 | Decision |
|---|---|---|---:|---:|---:|---:|---|
| 001 | linear | 4 features | 1996.657 | 2483.328 | 14.212% | -0.1285 | no |
| 001 | cart | 4 features | 1298.927 | 1634.362 | 9.124% | 0.5112 | current checkpoint |
| 002 | cart | weather + lags 12/24/72/144/288 | 1118.909 | 1417.825 | 8.200% | 0.6316 | better than Exp 001 |
| 003 | linear | same as Exp 002 | | | | | compare |
| 003 | cart | same as Exp 002 | | | | | compare |
| 004 | cart | + rolling 12/24/72 | | | | | run after lags |
| 005 | rf/hgb | best features | | | | | likely champion candidate |
| 006 | rf/hgb | nonlinear weather | | | | | deploy if best |
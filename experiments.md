# BittBridge Final Project Experiment Tracker

Goal: find the model with the lowest validation MAPE and stable live deployment.

| Exp | Model | Features | Val MAE | Val RMSE | Val MAPE | R2 | Decision |
|---|---|---|---:|---:|---:|---:|---|
| 001 | linear | 4 features | 1996.657 | 2483.328 | 14.212% | -0.1285 | no |
| 001 | cart | 4 features | 1298.927 | 1634.362 | 9.124% | 0.5112 | current checkpoint |
| 002 | cart | weather + lags 12/24/72/144/288 | 1118.909 | 1417.825 | 8.200% | 0.6316 | better than Exp 001 |
| 003 | linear | same as Exp 002 | 1106.155 | 1417.301 | 8.363% | 0.6319 | close, but worse MAPE than CART |
| 003 | cart | same as Exp 002 | 1118.909 | 1417.825 | 8.200% | 0.6316 | current best by MAPE |
| 004 | cart | lags 12/24/72/144/288 + rolling 12/24/72 | 973.875 | 1246.259 | 7.275% | 0.7154 | new best |
| 005 | rf | lags 12/24/72/144/288 + rolling 12/24/72 | 712.334 | 951.109 | 5.383% | 0.8342 | current champion candidate |
| 006B | hgb | same 83 features as Exp 005 | 505.320 | 657.598 | 3.784% | 0.9235 | DEPLOY - beats RF |
| 007A/007B | hgb tuned | 800 iter / lr .025 or equivalent tuned HGB | 500.931 | 649.911 | 3.751% | 0.9252 | best candidate |
| 007C | hgb | tuned HGB + lags 1/6/12/24/72/144/288 | 492.686 | 646.335 | 3.698% | 0.9261 | new best candidate |
| 007D | hgb | 007C + weather aggregate nonlinear features | 500.186 | 653.611 | 3.744% | 0.9244 | worse than 007C, do not deploy |
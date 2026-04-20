# 6. Advanced miner models

Hey !!!
if you made it through the **moving average** baseline in [4. Run Miner](04-run-miner.md), here is the next level: **advanced machine learning models** built into the repo (linear regression, decision tree, RNN, LSTM). You can experiment on your VM and **compete with other miners** to find the **strongest** setup for New England load forecasting.

This page is about **how to run** that path and **what to tune**. You do **not** need to change data pipelines, Supabase settings, or persistence—the project keeps that for you. Focus on **`features`** and **`models hyperparameters`**, at the repo root.

---

## Performance vs. compute

Your GCP VM has **limited** RAM, CPU, and time. Bigger models and longer training runs can score better in theory, but they also use more memory and minutes. Become real ML engineer with limited resources, as it is in real world! Be ready for a **tradeoff** between prediction quality and what your machine can finish comfortably.

**Practical tip:** start with **smaller** hyperparameters—fewer `epochs`, smaller `units` / `batch_size` for neural nets, modest tree depth for CART—so you learn **how long** a run takes on **your** VM. Then increase complexity once you have a feel for runtime and stability.

---

## Document your experiments

As part of your project work, keep a simple log of what you tried: **which model** (linear, cart, rnn, lstm), **which parameters** you changed, **what you changed in `features`**, and **what you observed** (metrics on screen, training time, errors). That habit makes comparisons easy and helps you explain your final choice later.

---

## Edit the config (nano)

From the repo root (`~/bittbridge`):

```bash
cd ~/bittbridge
nano model_params.yaml
```

Work only in the sections below:

- **`features`** — Turn on or off and edit if needed optional inputs (time features, cyclical encodings, weather groups, load lags, rolling stats, load deltas, etc.). The file comments describe each toggle.
- **`models`** — Per-model settings: `linear`, `cart`, `lstm`, `rnn` (learning rate, epochs, batch size, units, and similar).

**Do not** edit `data`, storage URLs, Supabase tables, or `persistence` for the class workflow—the defaults keep the app flow consistent.

Save in nano: **Ctrl+O**, Enter; exit: **Ctrl+X**.

---

## Run the miner (same command as step 4)

Activate your venv, then run **exactly**:

```bash
python -m neurons.miner \
  --netuid 183 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```

When the miner starts, it runs a short **interactive** setup:

1. It asks whether to run the **baseline moving-average** miner. Answer **No** if you want to train and use an **advanced** model with your YAML settings.
2. Follow the prompts to pick a model type (for example **linear**, **cart**, **rnn**, or **lstm**).
3. Training runs once; you will see metrics on screen.
4. When asked whether to **deploy** the trained model, confirm if you want the miner to serve that model to validators.

---
## How to download files on CSV actual vs predicted example
> CSV actual vs predicted is saved during each model training session run in artifacts folder under bittbridge directory

1. Navigating to folder with needed file
```bash
cd artifacts
```
2. Navigate to model run 
```bash
# EXAMPPLE ONLY, YOU WILL HAVE DIFFERENT NAME
cd 20260418T192815Z_linear_miner
```   
3. Get real path of a file
```bash
realpath actual_vs_predicted.csv
```

<img width="954" height="547" alt="Screenshot 2026-04-20 at 10 24 11 AM" src="https://github.com/user-attachments/assets/869984eb-56ef-407b-809b-5f75a74fd3cd" />

4. Select Download file
   
<img width="953" height="399" alt="Screenshot 2026-04-20 at 10 23 45 AM" src="https://github.com/user-attachments/assets/4e3e3684-e7fa-428d-8a3a-acfab105e540" />

6. Enter the full path and click Download

<img width="365" height="241" alt="Screenshot 2026-04-20 at 10 25 22 AM" src="https://github.com/user-attachments/assets/0a1db6fc-7f4c-4865-b6b4-8b669f753314" />

### !! Sometimes it's not working, just refresh your terminal webpage with Ctrl+R !!

---

## !! If something gets stuck !!

Bugs and long steps happen. If the process **hangs**, spams errors, or you want to abort: press **`Ctrl+C`** (your best friend in dev stage) in the terminal. That **stops** the running command; you can fix config or retry afterward.
You can also share bugs or issues you continue to face with dmitrii.tuzov@uconn.edu

---

## Optional: background on data and Supabase

Curious how live rows and tables and data fit together? Read [App workflow and Supabase](app-workflow-supabase.md)—for **context**; 

---

[← 5. Update and restart](05-update-and-restart.md) · [Guide](../../README.md#guide)

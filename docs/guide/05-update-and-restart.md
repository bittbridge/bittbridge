# 5. Update and restart

Use this when the repo has new commits on main, or when you want to train a better model and deploy it.

**Assumption:** You run the miner in tmux as in [4. Run Miner](04-run-miner.md), session name `miner`.

## ⚠️ Important mindset (READ FIRST)

* Your miner can keep running while you experiment
* You do NOT need to stop tmux to test models
* restart the miner after you are ready to deploy a new model

👉 Think of tmux as a running container
👉 Your experiments happen outside of it

---

## Linux commands you should be familiar with

Quick reference for the shell on your VM (run these inside SSH):

| Command | What it does |
|--------|----------------|
| `pwd` | **P**rint **w**orking **d**irectory (shows your current folder path). |
| `ls` | **L**i**s**ts files and folders in the current directory. |
| `cd folder` | **C**hange **d**irectory into `folder` (relative name or full path). |
| `cd ..` | Go **up** one level (parent folder). |
| `nano file` | Open `file` in a simple text editor; save with **Ctrl+O**, exit with **Ctrl+X**. |
| `cat file` | Print `file` contents to the terminal (read-only). |
|Arrow UP / DOWN | Find previously used commands |
| CTRL+R | Search for previously used commands |
|Press Ctrl + b, then [ |Scroll in Tmux session|

**`To scroll in Tmux: Press Ctrl + b, then [`**

**`Navigate: Use the Arrow Keys, Page Up, or Page Down`**

**`Exit: Press q`**

---

## 1. Pull the latest code

```bash
cd ~/bittbridge
git pull
```

---

## 2. Update Python dependencies

After every pull, **sync your venv** so new or updated libraries from the repo are installed:

```bash
cd ~/bittbridge
source venv/bin/activate
pip install -r requirements.txt
```

---

## If Git refuses to pull (any local changes)

Sometimes `git pull` shows a message like: *please commit your changes or stash them before you merge.*

If you **do not need** any uncommitted edits on the VM (you just want a clean copy of `main`), you can reset the working tree and pull:

```bash
cd ~/bittbridge
git reset --hard
git pull
```

Run it one more time just in case!
```bash
cd ~/bittbridge
source venv/bin/activate
pip install -r requirements.txt
```
## 3. Experiment with your models 

* Train models
* Try different feature combinations
* Evaluate performance (MAE, R², scatter plot)

## 5. Restart miner (DEPLOY)

(if first time -> tmux new -s miner)

```bash
tmux attach -t miner
```

Press **`Ctrl+C`** to stop the miner. 

Run miner with updated specs:

```bash
cd ~/bittbridge
python -m neurons.miner \
  --netuid 183 \
  --subtensor.network test \
  --wallet.name miner \
  --wallet.hotkey default \
  --logging.debug
```
Detach from tmux session: **`Ctrl+b`** then **`d`**

---

#❗ Important notes

* Your old model will continue running until you restart the miner
* Your new model is only used after restart
* Leaderboard updates only after predictions + ground truth (~6 hours delay)

---

[← 4. Run Miner](04-run-miner.md) · [6. Advanced miner models](06-advanced-miner-models.md) · [Guide](../../README.md#guide)

# 5. Update and restart

Use this when the repo has new commits on `main` and you need to **pull** on your GCP VM, **refresh dependencies**, and **restart** your miner.

**Assumes:** You run the miner in tmux as in [4. Run Miner](04-run-miner.md), session name `miner`.

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

## 1. Stop the miner

```bash
tmux attach -t miner
```

Press **`Ctrl+C`** to stop the miner. Detach: **`Ctrl+b`** then **`d`**.

---

## 2. Pull the latest code

```bash
cd ~/bittbridge
git pull
```

---

## 3. Update Python dependencies

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

---

| Step | Action |
|------|--------|
| 1 | `tmux attach -t miner` → `Ctrl+C` → detach |
| 2 | `cd ~/bittbridge` && `git pull origin main` |
| 3 | `source venv/bin/activate` && `pip install -r requirements.txt` |
| 4 | Restart miner in `miner` session → detach |

---

[← 4. Run Miner](04-run-miner.md) · [6. Advanced miner models](06-advanced-miner-models.md) · [Guide](../../README.md#guide)

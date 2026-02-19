# 4. GCP VM Setup

This guide walks you through creating a Google Cloud VM, configuring the firewall, and setting up the environment. All steps after VM creation happen in the SSH terminal.

---

## Part 1: GCP Console (Browser)

### Step 1 – Go to GCP

1. Open [Google Cloud Console](https://console.cloud.google.com/welcome)
2. Sign in with your Google account

### Step 2 – Activate Free Trial

1. If prompted, activate the free trial (includes $300 credit)
2. You may need to add a payment method; you won't be charged unless you exceed the free tier

### Step 3 – Create Firewall Rule

Create a firewall rule **before** the VM so you can apply it during VM creation.

1. Go to **VPC network** → **Firewall** (or search "Firewall" in the top search bar)

<img width="1512" height="854" alt="Screenshot 2026-02-18 at 12 18 51 PM" src="https://github.com/user-attachments/assets/cbcb6989-1329-44b9-a0b2-4beb11d4ac04" />

3. Click **Create firewall rule**
4. Set:
   - **Name:** `allow-tcp-8091`
   - **Direction:** Ingress
   - **Targets:** All instances in the network (or select "Specified target tags" and add a tag like `bittensor-miner`)
   - **Source IPv4 ranges:** `0.0.0.0/0`
   - **Protocols and ports:** Check "TCP" and enter `8091`
5. Click **Create**



<img width="590" height="721" alt="Screenshot 2026-02-18 at 12 21 09 PM" src="https://github.com/user-attachments/assets/e16045fd-837b-4861-9fa5-f56eafad332f" />

### Step 4 – Go to VM Instances

1. Go to [VM instances](https://console.cloud.google.com/compute/instances)
2. Or: **Navigation menu** → **Compute Engine** → **VM instances**
3. Or: Search "VM instances" in the top search bar

### Step 5 – Create Instance

1. Click **Create instance**
2. **Name:** e.g. `bittbridge-vm` (or leave default)
3. **Region:** Choose one close to you
4. **Machine type:** e2-medium (2 vCPU, 4 GB) is sufficient for testing and low cost;
<img width="1512" height="850" alt="Screenshot 2026-02-18 at 12 15 53 PM" src="https://github.com/user-attachments/assets/80cdd183-a810-4f09-9b0b-dd523e0aa72e" />

### Step 6 – Change Boot Disk (OS)

1. Under **Boot disk**, click **Change**

<img width="1512" height="767" alt="Screenshot 2026-02-18 at 12 28 05 PM" src="https://github.com/user-attachments/assets/26cd8692-98a3-4db5-8dbd-b57e6380e286" />


3. Select **Ubuntu**
4. **Version:** Ubuntu 22.04 LTS x86/64
5. **Disk size:** 25 GB (or more if you have large models)
6. Click **Select**



<img width="704" height="758" alt="Screenshot 2026-02-18 at 12 28 50 PM" src="https://github.com/user-attachments/assets/e2f6e48f-63c2-4749-b019-f1e0aa8eaf03" />

### Step 7 – Create VM

1. Click **Create**
2. Wait for the VM to start (green checkmark)



<img width="1512" height="772" alt="Screenshot 2026-02-18 at 12 31 08 PM" src="https://github.com/user-attachments/assets/2e5b2501-3542-4068-bdaf-d2da2c4b0c05" />

### Step 8 – Connect via SSH

1. Click **SSH** (or **Connect** → **SSH**) next to your VM
2. Authorize if prompted
3. You should see a terminal prompt, e.g. `your_username@bittbridge-vm:~$`


<img width="1512" height="776" alt="Screenshot 2026-02-18 at 12 31 55 PM" src="https://github.com/user-attachments/assets/16c70140-e6ee-41cc-94de-151776090b3b" />


<img width="908" height="718" alt="Screenshot 2026-02-18 at 12 32 58 PM" src="https://github.com/user-attachments/assets/97b251c8-066f-4300-a75f-22bbec1a67f2" />


---

## Part 2: SSH Terminal – Environment Setup

All commands below run **inside the SSH terminal** after you connect.

### Step 9 – Update System and Install Git

```bash
sudo apt update
sudo apt install -y git
```

### Step 10 – Install Python and venv

For **Ubuntu 22.04 LTS** (Python is installed be default):

```bash
sudo apt install -y python3-venv python3-pip
```

### Step 11 – Install tmux

tmux lets you run miner and validator in separate panes; if the SSH connection drops, processes keep running.

```bash
sudo apt install -y tmux
```

### Step 12 – Clone Your Fork

Replace `YOUR_USERNAME` with your GitHub username:

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/bittbridge.git
cd bittbridge
```
### Step 13 – Create Virtual Environment and Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You should see `(venv)` in your prompt. Always run `source venv/bin/activate` in new terminals/tmux panes before running miner or validator.

### Step 14 – Verify Setup

```bash
btcli --version
python3 -c "import bittensor; print('Bittensor OK')"
```

If both succeed, you're ready for [05 – Wallets and Tokens](05-wallets-and-tokens.md).

---

## Quick Reference: tmux Usage

| Action | Command |
|--------|---------|
| Create new session | `tmux new -s bittbridge` |
| Split pane horizontally | `Ctrl+b` then `"` |
| Split pane vertically | `Ctrl+b` then `%` |
| Switch between panes | `Ctrl+b` then arrow keys |
| Detach (leave running) | `Ctrl+b` then `d` |
| Reattach | `tmux attach -t bittbridge` |
| Kill session | `tmux kill-session -t bittbridge` |

---

**Prev:** [03 – Training Custom Model](03-training-custom-model.md) | **Next:** [05 – Wallets and Tokens](05-wallets-and-tokens.md) | [Back to Guide Index](../../README.md#guide)

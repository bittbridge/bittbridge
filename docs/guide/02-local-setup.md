# 2. Local Setup

Minimal setup on your machine to get the repo and environment ready. The main deployment happens on a GCP VM (see [04 – GCP VM Setup](04-gcp-vm-setup.md)).

---

## Step 1 – Fork and Clone

1. Go to [https://github.com/bittbridge/bittbridge](https://github.com/bittbridge/bittbridge)
2. Click **Fork** to create your own copy.
3. Create a directory on your machine for the project, then clone **your fork**:

```bash
git clone https://github.com/YOUR_USERNAME/bittbridge.git
cd bittbridge
```

---

## Step 2 – Set Up Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

This gives you a working local environment. You will repeat a similar setup on the GCP VM.

---

**Prev:** [01 – Before You Start](01-before-you-start.md) | **Next:** [03 – Training Custom Model](03-training-custom-model.md) | [Back to Guide Index](../../README.md#guide)

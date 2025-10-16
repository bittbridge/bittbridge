# System Requirements for Bittensor Nodes (Validator and Miner)

## I. Introduction and Operational Context

### 1.1 Why System Specs Matter: Speed and Stability

To successfully run a node on the Bittensor network—whether as a **validator** (who scores work) or a **miner** (who provides the work)—you need specialized computer hardware and software. Simply getting a node to run is the minimum bar; earning tokens (TAO) requires a **min-performance setup**.

The key to earning on Bittensor is **Uptime and Speed**. The network rewards nodes that are fast, stable, and available. If your node is too slow or crashes often, it will lose rank and risk being removed from the network.

The requirements below are split into two categories:
* **Minimum Specs:** Enough power for local testing, development, or participation on the testnet.
* **Recommended Specs:** The high-end setup necessary for stable, competitive, operation and maximum profitability on the mainnet.

### 1.2 Universal Node Environment

#### 1.2.1 Operating System and Environment

The foundation of a competitive node is the Operating System (OS).

| OS/Environment | Status | Use Case & Key Requirements |
| :--- | :--- | :--- |
| **Linux (Recommended)** | Best for 24/7 operation. | **Ubuntu 22.04 LTS** is the industry standard for stability, and optimized support for high-performance GPU drivers (like NVIDIA CUDA). |
| **macOS (Apple Silicon)** | Supported for Dev/Testing. | Supported on newer Apple Silicon M-series (ARM64) and older x86_64 Macs.  |
| **PC/Windows** | Limited, Non-Native Support. | Bittensor does not run natively on Windows. You **must** install the **Windows Subsystem for Linux 2 (WSL 2)** and an Ubuntu distribution. |

#### 1.2.2 Python Version and Core Dependencies

Bittensor is built on Python and uses a standard Python package manager (`pip`).

* **Python Version:** The Bittensor SDK supports Python versions **3.9 through 3.13**.
* **Virtual Environments:** It is **mandatory** to use a **Python Virtual Environment (venv)** for installation to keep Bittensor's required libraries separate from your system's other software.
* **Core Dependencies:**
    * Bittensor SDK: `pip install bittensor`.
    * Bittensor CLI: `pip install bittensor-cli`.
    * GPU Support: If you use a GPU (almost all miners do), you must install the PyTorch extras: `pip install "bittensor[torch]"`.
    * Rust: Linux deployments require the **Rust compiler toolchain** for compiling certain core infrastructure components.
    * Competitive Registration (Optional): Miners can install `cubit` for GPU-accelerated Proof-of-Work (PoW) registration, which can be faster than CPU-only PoW.

---

## II. Security and Operations (SecOps) Baseline

### 2.1 Key Management: Hotkey vs. Coldkey

Bittensor uses a dual-key system for security.

| Key Type | Security Mandate | Operational Use |
| :--- | :--- | :--- |
| **Coldkey** | The "bank account" key; controls all your TAO funds and stakes. It **must never be stored or used on the operational node server**. Keep it completely offline (air-gapped device or hardware wallet). Losing or leaking your Coldkey means immediate, irreversible loss of all your TAO. | None. It is purely for securing your assets and stake.|
| **Hotkey** | The "day-to-day" operational key used for mining, validating. It is considered expendable.| Used for signing all network operations. If a leak is suspected, you can immediately swap it for a new one using the `btcli wallet swap-hotkey` command. |

> ⚠️ Do not mine with coldkeys
Miners will need coldkeys to manage their TAO and alpha currency, as well as hotkeys to serve requests. Miners must ensure that there is a clear boundary—the coldkey should never be on an environment with untrusted ML code from containers, frameworks, or libraries that might exfiltrate secrets.

### 2.2 Configuration and Secrets Management

> ⚠️ Never hardcode passwords or private keys into your code.

* **.env File Setup:** Use a local `.env` file to store sensitive information like wallet names, API keys, or custom network endpoints (e.g., `BT_SUBTENSOR_CHAIN_ENDPOINT`).
* **Security:** Ensure the `.env` file is excluded from version control systems (e.g., in your `.gitignore`).
* For professional, high-stakes deployments, dedicated secret managers (like HashiCorp Vault) are recommended over simple `.env` files for enhanced security.

### 2.3 Logging and Monitoring Baseline

To achieve the required **99.9% uptime**:

* **Process Management:** Use process managers or Linux Systemd services**. These tools ensure the Bittensor application automatically restarts if it crashes and manages log files.
* **Validator Tools:** Validators should use external platform - **Weights & Biases (WandB)** to track performance scores and key metrics.

---

## III. Validator Requirements

Validators are the nodes responsible for scoring and setting weights. They need highly stable, low-latency machines with massive memory capacity to handle complex reward calculations quickly.

### 3.1 Validator Compute Profile (Stability Focus)

| Resource | Minimum Specification (Testnet/Local) | Recommended Specification (24/7 Mainnet) |
| :--- | :--- | :--- |
| **CPU** | 2 Cores (Modern x86\_64) | 4+ Cores @ 3.5 GHz+|
| **System RAM** | 8 GB | **16+ GB**|
| **Storage** | 32 GB SSD | **125+ GB SSD** |
| **GPU/VRAM** | CPU-Only | **8+ GB VRAM** |
| **Network** | 10+ Mbps Stable | **500+ Mbps**|

### 3.2 Validator-Specific Needs

* **API Connectivity:** Validators should connect to external data sources (like **CoinGecko** or **WandB**) to check and log their work.
* **Using paid API keys could be necessary to secure higher request quotas**.

---

## IV. Miner Requirements

Miners are the resource providers, primarily serving machine learning models. Their requirements are entirely dependent on the size and speed of the models they must serve, making **GPU VRAM** critical resource.

### 4.1 Miner Compute Profile (Inference and Model Focus)

| Resource | Minimum Specification (Testnet/Quantized) | Recommended Specification (Competitive Mainnet) |
| :--- | :--- | :--- |
| **CPU** | 2 Cores (Modern x86\_64) | 4+ Cores @ 3.5 GHz+|
| **System RAM** | 8 GB | **16+ GB**|
| **Storage** | 32 GB SSD | **125+ GB SSD** |
| **GPU VRAM** | **8 GB VRAM** | **24+ GB VRAM** |
| **GPU Type** | CUDA support | **NVIDIA A100/H100** |
| **Network** | 10+ Mbps Stable | **500+ Mbps**|

**VRAM is Key:** While you can run a basic model with 8 GB VRAM, more capacity allows the miner to run larger, higher-quality models faster, which increases the score assigned by validators.

### 4.2 Miner-Specific Needs

* **Configuration Management (Axon):** Miners must start an **Axon server** to receive requests and send responses to validators. This server needs to be configured to listen on a specific IP address and port (e.g., `axon.port 8901`) and be publicly reachable.

---

## V. Network and Firewall Configuration Checklist

All nodes need to open specific ports for the network to function.

| Component | Port (Default) | Protocol | Direction | Firewall Rule | Usage / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Subtensor P2P Sync** | 30333 | TCP | Inbound/Outbound | **OPEN** to Public Internet | Allows your node to connect to other nodes and stay synchronized with the blockchain. |
| **Node Axon Server** | 8091 (Default) | TCP | Inbound | **OPEN** to Public Internet | This is your public address (IP:Port) used for Validator-Miner communication. Must be in range 8091 to 8999. |

The network connection itself is crucial; aiming for **10 Gbps unmetered bandwidth** is highly recommended.

---

## VI. Summary Reference Table

The table below summarizes the minimum and recommended requirements for a Bittensor node.

### System Requirements Comparison: Minimum vs. Recommended

| Category | Resource | Minimum Specification (Entry/Testing) | Recommended Specification (24/7 Competitive Mainnet) |
| :--- | :--- | :--- | :--- |
| **Operating Environment** | OS | Linux (Ubuntu 20.04+ x86\_64) | Linux (Ubuntu 22.04 LTS x86\_64) |
| | Python | 3.10 (In venv) | 3.10 or 3.11 (In venv) |
| | SecOps | .env file for secrets | PM2/Systemd, Coldkey/Hotkey separation, WandB monitoring enabled |
| **Universal Hardware** | CPU Cores | 2 Cores | 4+ Cores |
| | System RAM | 8 GB | **16+ GB** |
| | Storage I/O | 32 GB SSD | **126 GB+** |
| | Network B/W | 10+ Mbps (Stable connection) | **500 Mbps+**  |
| **Miner Specifics (Computation)** | GPU VRAM | 8 GB VRAM | **24+ GB VRAM** |
| | GPU Type | CUDA support | NVIDIA A100/H100 |
| **Validator Specifics (Verification)** | GPU VRAM | CPU scoring possible | **8+ GB VRAM** |
| | API Tools | WandB + CoinGecko API | Dedicated API Keys, Rate Limit Handling implementation |

# System Requirements for Bittensor Nodes (Validator and Miner)

## I. Introduction and Operational Context

### 1.1 Why System Specs Matter: Speed and Stability

[cite_start]To successfully run a node on the Bittensor network—whether as a **validator** (who scores work) or a **miner** (who provides the work)—you need specialized computer hardware and software[cite: 4]. [cite_start]Simply getting a node to run is the minimum bar; earning tokens (TAO) requires a **high-performance setup**[cite: 5].

[cite_start]The key to earning on Bittensor is **Uptime and Speed**[cite: 6]. [cite_start]The network rewards nodes that are fast, stable, and available[cite: 7]. [cite_start]If your node is too slow or crashes often, it will lose rank and risk being removed from the network (pruned)[cite: 8].

[cite_start]The requirements below are split into two categories[cite: 9]:
* [cite_start]**Minimum Specs:** Enough power for local testing, development, or participation on the testnet[cite: 10].
* [cite_start]**Recommended Specs:** The high-end setup necessary for stable, competitive, operation and maximum profitability on the mainnet[cite: 11].

[cite_start]**Important Note on Subnets:** Every Bittensor subnet (a specialized market) has different needs[cite: 12]. [cite_start]For example, a video generation subnet needs a huge GPU, while a data scraping subnet might only need a powerful CPU[cite: 13]. [cite_start]**Always check the specific subnet's documentation** (often in a `min_compute.yml` file or `README`) for exact GPU requirements before investing in hardware[cite: 14].

### 1.2 Universal Node Environment

#### 1.2.1 Operating System and Environment

[cite_start]The foundation of a competitive node is the Operating System (OS)[cite: 17].

| OS/Environment | Status | Use Case & Key Requirements |
| :--- | :--- | :--- |
| **Linux (Recommended)** | Best for 24/7 operation. | [cite_start]**Ubuntu 22.04 LTS** is the industry standard for stability, superior resource management, and optimized support for high-performance GPU drivers (like NVIDIA CUDA). [cite: 18] |
| **macOS (Apple Silicon)** | Supported for Dev/Testing. | Supported on newer Apple Silicon M-series (ARM64) and older x86_64 Macs. [cite_start]**Not recommended for competitive mainnet operation** due to potential overhead and stability issues. [cite: 18] |
| **PC/Windows** | Limited, Non-Native Support. | Bittensor does not run natively on Windows. You **must** install the **Windows Subsystem for Linux 2 (WSL 2)** and an Ubuntu distribution. [cite_start]This setup is generally only reliable for wallet transactions and limited development, **not** for competitive mining or validation. [cite: 18] |

#### 1.2.2 Python Version and Core Dependencies

[cite_start]Bittensor is built on Python and uses a standard Python package manager (`pip`)[cite: 20].

* [cite_start]**Python Version:** The Bittensor SDK supports Python versions **3.9 through 3.13**[cite: 21].
* [cite_start]**Virtual Environments:** It is **mandatory** to use a **Python Virtual Environment (venv)** for installation to keep Bittensor's required libraries separate from your system's other software[cite: 22].
* **Core Dependencies:**
    * [cite_start]Bittensor SDK: `pip install bittensor`[cite: 24].
    * [cite_start]GPU Support: If you use a GPU (almost all miners do), you must install the PyTorch extras: `pip install "bittensor[torch]"`[cite: 25].
    * [cite_start]Rust: Linux deployments require the **Rust compiler toolchain** for compiling certain core infrastructure components[cite: 26].
    * [cite_start]Competitive Registration (Optional): Miners can install `cubit` for GPU-accelerated Proof-of-Work (PoW) registration, which can be faster than CPU-only PoW[cite: 27].

---

## II. Security and Operations (SecOps) Baseline

### 2.1 Key Management: Hotkey vs. Coldkey

[cite_start]Bittensor uses a dual-key system for security, which is non-negotiable[cite: 30].

| Key Type | Security Mandate | Operational Use |
| :--- | :--- | :--- |
| **Coldkey** | The "bank account" key; controls all your TAO funds and stakes. It **must never be stored or used on the operational node server**. Keep it completely offline (air-gapped device or hardware wallet). [cite_start]Losing or leaking your Coldkey means immediate, irreversible loss of all your TAO. [cite: 31] | None. [cite_start]It is purely for securing your assets and stake. [cite: 31] |
| **Hotkey** | The "day-to-day" operational key used for mining, validating, and performing network tasks. [cite_start]It is considered expendable. [cite: 31] | Used for signing all network operations. [cite_start]If a leak is suspected, you can immediately swap it for a new one using the `btcli wallet swap-hotkey` command. [cite: 31] |

### 2.2 Configuration and Secrets Management

[cite_start]Never hardcode passwords or private keys into your code[cite: 33].

* [cite_start]**.env File Setup:** Use a local `.env` file to store sensitive information like wallet names, API keys, or custom network endpoints (e.g., `BT_SUBTENSOR_CHAIN_ENDPOINT`)[cite: 34].
* [cite_start]**Security:** Ensure the `.env` file is excluded from version control systems (e.g., in your `.gitignore`)[cite: 35].
* [cite_start]For professional, high-stakes deployments, dedicated secret managers (like HashiCorp Vault) are recommended over simple `.env` files for enhanced security[cite: 36].

### 2.3 Logging and Monitoring Baseline

[cite_start]To achieve the required **99.9% uptime**[cite: 38]:

* [cite_start]**Process Management:** Use process managers like **PM2 (Process Manager 2)** or **Linux Systemd services**[cite: 39]. [cite_start]These tools ensure the Bittensor application automatically restarts if it crashes and manages log files[cite: 40].
* [cite_start]**Node Health Monitoring:** Continuously track key metrics[cite: 41]:
    * [cite_start]Synchronization Status: Ensure your node is fully synced (`isSyncing` should be false)[cite: 42].
    * [cite_start]Peer Count: Maintain healthy network connectivity (ideally **over 150 peers**)[cite: 43].
    * [cite_start]Hardware Health: Monitor CPU/GPU usage, temperature, and memory consumption in real time[cite: 44].
* [cite_start]**Validator Tools:** Validators often use external platforms like **Weights & Biases (WandB)** to track performance scores and key metrics[cite: 45].

---

## III. Validator Requirements

[cite_start]Validators are the nodes responsible for scoring and setting weights[cite: 47]. [cite_start]They need highly stable, low-latency machines with massive memory capacity to handle complex reward calculations quickly[cite: 48].

### 3.1 Validator Compute Profile (Stability Focus)

| Resource | Minimum Specification (Testnet/Local) | Recommended Specification (24/7 Mainnet) |
| :--- | :--- | :--- |
| **CPU** | 4 Cores (Modern x86\_64) | [cite_start]8+ Cores @ 3.5 GHz+ [cite: 50] |
| **System RAM** | 16 GB | [cite_start]**256 GB (Minimum) - 512 GB+ (Optimal)** [cite: 50] |
| **Storage** | 128 GB SSD | [cite_start]**500 GB+ NVMe SSD** (Fastest disk available) [cite: 50] |
| **GPU/VRAM** | CPU-Only (Sufficient for simple subnets) | [cite_start]**12 GB VRAM** (e.g., RTX 3060/A4000 equivalent) [cite: 50] |
| **Network** | 100 Mbps Stable | [cite_start]**1 Gbps Minimum, 10 Gbps Recommended** (Unmetered) [cite: 50] |

[cite_start]**Why so much RAM?** The high memory requirement (**256 GB+**) is critical for speed[cite: 51]. [cite_start]Validators must quickly process and store information from every miner (the metagraph) and run complex scoring models simultaneously[cite: 52]. [cite_start]Having ample RAM prevents the system from slowing down by using the hard drive (disk swap), which would ruin your latency and competitive standing[cite: 53].

### 3.2 Validator-Specific Needs

* [cite_start]**API Connectivity:** Validators often connect to external data sources (like **CoinGecko** or **WandB**) to check their work[cite: 55].
* [cite_start]**Rate Limit Handling:** When querying external APIs, validators must implement robust retry logic and **Exponential Backoff** to pause and try again, ensuring they don't get blocked during critical scoring periods[cite: 56]. [cite_start]Using paid API keys is often necessary to secure higher request quotas[cite: 57].

---

## IV. Miner Requirements

[cite_start]Miners are the resource providers, primarily serving machine learning models[cite: 59]. [cite_start]Their requirements are entirely dependent on the size and speed of the models they must serve, making **GPU VRAM** the most critical resource[cite: 60].

### 4.1 Miner Compute Profile (Inference and Model Focus)

| Resource | Minimum Specification (Testnet/Quantized) | Recommended Specification (Competitive Mainnet) |
| :--- | :--- | :--- |
| **CPU** | 4 Cores | [cite_start]8+ Cores @ 3.5 GHz+ [cite: 62] |
| **System RAM** | 16 GB | [cite_start]256 GB - 512 GB+ [cite: 62] |
| **Storage** | 128 GB SSD | [cite_start]**500 GB+ NVMe SSD** (Fastest disk available) [cite: 62] |
| **GPU VRAM** | **8 GB VRAM** (for small, quantized models) | [cite_start]**24 GB VRAM** (RTX 3090/4090 or better) [cite: 62] |
| **GPU Type** | NVIDIA with CUDA support | [cite_start]**NVIDIA A100/H100** or high-end consumer cards [cite: 62] |

[cite_start]**VRAM is Key:** For subnets that run Large Language Models (LLMs) or complex AI, VRAM is the competitive barrier[cite: 63]. [cite_start]While you can run a basic model with 8 GB VRAM (using 4-bit quantization), the widely accepted floor for competitive earning in high-demand subnets is **24 GB VRAM**[cite: 64]. [cite_start]This capacity allows the miner to run larger, higher-quality models faster, which increases the score assigned by validators[cite: 65].

### 4.2 Miner-Specific Needs

* [cite_start]**Ability to Run Models Locally:** The primary function of a miner is to quickly run and serve models on its GPU[cite: 67]. [cite_start]The **GPU VRAM** dictates the maximum size and quality of the model you can run[cite: 68]. [cite_start]High-end GPUs prioritize low inference latency (quick response time) because speed directly increases earnings[cite: 69].
* [cite_start]**Configuration Management (Axon):** Miners must start an **Axon server** to receive requests from validators[cite: 70]. [cite_start]This server needs to be configured to listen on a specific IP address and port (e.g., `axon.port 8901`) and be publicly reachable[cite: 71].
* [cite_start]**Subnet-Specific Configurations:** Miners must comply with specific rules set by the subnet's creators (hyperparameters)[cite: 72].

---

## V. Network and Firewall Configuration Checklist

[cite_start]All nodes need to open specific ports for the network to function[cite: 74].

| Component | Port (Default) | Protocol | Direction | Firewall Rule | Usage / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Subtensor P2P Sync** | 30333 | TCP | Inbound/Outbound | **OPEN** to Public Internet | [cite_start]Allows your node to connect to other nodes and stay synchronized with the blockchain. [cite: 75] |
| **Subtensor Websocket** | 9944 | TCP | Localhost Only | **BLOCKED** from Public Internet | Used by your local Bittensor SDK to communicate with the chain. [cite_start]Must be restricted to your local machine for security. [cite: 75] |
| **Node Axon Server** | 8091 (Default) | TCP | Inbound | **OPEN** to Public Internet | This is your public address (IP:Port) used for Validator-Miner communication. [cite_start]Must be in range 8091 to 8999. [cite: 75] |
| **Subnet-Specific Ports** | Varies (e.g., 8091) | TCP | Inbound | **OPEN** to Public Internet | Some subnets (like Compute SN27) require additional open ports. [cite_start]Check the subnet's specific documentation. [cite: 75] |

[cite_start]The network connection itself is crucial; aiming for **10 Gbps unmetered bandwidth** is highly recommended[cite: 76].

---

## VI. Summary Reference Table

[cite_start]The table below summarizes the minimum and recommended requirements for a Bittensor node[cite: 78].

### System Requirements Comparison: Minimum vs. Recommended

| Category | Resource | Minimum Specification (Entry/Testing) | Recommended Specification (24/7 Competitive Mainnet) |
| :--- | :--- | :--- | :--- |
| **Operating Environment** | OS | [cite_start]Linux (Ubuntu 20.04+ x86\_64) [cite: 80] | [cite_start]Linux (Ubuntu 22.04 LTS x86\_64) [cite: 80] |
| | Python | [cite_start]3.10 (In venv) [cite: 80] | [cite_start]3.10 or 3.11 (In venv) [cite: 80] |
| | SecOps | [cite_start].env file for secrets [cite: 80] | [cite_start]PM2/Systemd, Coldkey/Hotkey separation, WandB monitoring enabled [cite: 80] |
| **Universal Hardware** | CPU Cores | [cite_start]4 Cores [cite: 80] | [cite_start]8+ Cores (High Clock Speed) [cite: 80] |
| | System RAM | [cite_start]16 GB [cite: 80] | [cite_start]**256 GB - 512 GB+** [cite: 80] |
| | Storage I/O | [cite_start]128 GB SSD [cite: 80] | [cite_start]**500 GB+ NVMe SSD** [cite: 80] |
| | Network B/W | [cite_start]100 Mbps (Stable connection) [cite: 80] | [cite_start]**1 Gbps Minimum, 10 Gbps Recommended** [cite: 80] |
| **Miner Specifics (Computation)** | GPU VRAM | [cite_start]8 GB VRAM (for Q4/test models) [cite: 80] | [cite_start]**24 GB VRAM** (RTX 3090/4090 or better) [cite: 80] |
| | GPU Type | [cite_start]NVIDIA with CUDA support [cite: 80] | [cite_start]NVIDIA A100/H100 or competitive consumer cards [cite: 80] |
| **Validator Specifics (Verification)** | GPU VRAM | [cite_start]N/A (CPU scoring possible) [cite: 80] | [cite_start]**12 GB VRAM** [cite: 80] |
| | API Tools | [cite_start]Bittensor SDK/CLI [cite: 80] | [cite_start]Dedicated API Keys, Rate Limit Handling implementation [cite: 80] |

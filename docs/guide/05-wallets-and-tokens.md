# 5. Wallets and Tokens

All steps in this section are done **on the GCP VM** (or locally if using [08 – Local Run (Advanced)](08-local-run-advanced.md)).

---

## Create or Import Wallets

You need two wallets: one for the Miner, one for the Validator. Each wallet has a **coldkey** (holds funds) and a **hotkey** (used for mining/validating).

### Creating New Wallets

To create both coldkey and hotkey for a wallet:

```bash
btcli wallet create
```

Example for miner and validator:

```bash
# Miner wallet
btcli wallet create --wallet.name miner --wallet.hotkey default

# Validator wallet
btcli wallet create --wallet.name validator --wallet.hotkey default
```

You will be prompted to set a password for the coldkey and choose mnemonic length. A unique mnemonic is generated for each key (coldkey and hotkey) and shown in the terminal.

### Storing Your Mnemonic Phrases – Critical

> **Store your mnemonics securely.** You will need them in the future.

**Why mnemonics matter:**

- **New VM or machine:** When you set up a new VM, reinstall, or switch computers, you must regenerate your keys from the mnemonics. There is no other way to recover access.
- **Recovery:** If you lose the wallet files (e.g., VM deleted, disk failure), the mnemonic is the only way to restore your wallet and funds.
- **No recovery without mnemonic:** Without the mnemonic, lost keys mean permanent loss of access to your TAO.

Write down each mnemonic and keep it in a safe place. You will have **4 mnemonics** total (2 coldkeys + 2 hotkeys for miner and validator).

### Importing Existing Wallets

If you already have wallets from another machine, regenerate them on this VM using the mnemonics:

1. **Regenerate coldkey first** (one per wallet):
   ```bash
   btcli wallet regen_coldkey --wallet.name miner --mnemonic "word1 word2 word3 ... word12"
   btcli wallet regen_coldkey --wallet.name validator --mnemonic "word1 word2 word3 ... word12"
   ```

2. **Regenerate hotkey** (use the coldkey name and hotkey mnemonic):
   ```bash
   btcli wallet regen_hotkey --wallet.name miner --mnemonic "word1 word2 word3 ... word12"
   btcli wallet regen_hotkey --wallet.name validator --mnemonic "word1 word2 word3 ... word12"
   ```

You must regenerate **both** coldkey and hotkey for each wallet. Use the same wallet names you used originally so commands in this guide work correctly.

### Important: Wallet Names vs. Addresses

- **`--wallet.name`** = The coldkey/wallet name (e.g., `miner`, `validator`).
- **`--wallet.hotkey`** = The **hotkey name** (e.g., `default`), **not** the ss58 address.

---

## Get Faucet Tokens

Ask Dmitrii to send tTAO to your miner and validator wallets. You need to send him SS58 coldkey address (starts with "5"), see screenshot. 


<img width="690" height="84" alt="Screenshot 2026-02-19 at 12 30 29 PM" src="https://github.com/user-attachments/assets/a01ed835-f9fb-43d4-9061-790c073f3366" />


Check your balance:

```bash
btcli w balance --network test
```

---

## Register Validator & Miner Hotkeys

```bash
btcli subnet register --netuid 420 --subtensor.network test --wallet.name miner --wallet.hotkey default

btcli subnet register --netuid 420 --subtensor.network test --wallet.name validator --wallet.hotkey default
```

Optional checks (use wallet/coldkey name, not hotkey name):

---

## Collect API Keys

### CoinGecko (Validators)

- Log in to [CoinGecko](https://www.coingecko.com/en/developers/dashboard) and generate an API key.
- Save it for the next step.

### WandB (Validators)

- Log in to [Weights & Biases](https://wandb.ai/) and generate an API key.
- Save it for the next step.
- Help: [Find your WandB API key](https://docs.wandb.ai/support/find_api_key/)

---

**Prev:** [04 – GCP VM Setup](04-gcp-vm-setup.md) | **Next:** [06 – Run Miner](06-run-miner.md) | [Back to Guide Index](../../README.md#guide)

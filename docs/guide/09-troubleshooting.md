# 9. Troubleshooting

**If you see `[NO_SUBMISSION]` errors or validators can't connect to miners, this section is for you!**

---

## The Problem

Validators need to connect to miners over the internet. If miners are behind firewalls/routers, validators can't reach them.

---

## Quick Diagnosis

**Test if your miner is accessible from the internet:**
```bash
# From any external machine (not your local network)
telnet YOUR_EXTERNAL_IP YOUR_MINER_PORT

# Example:
telnet 69.115.169.144 8091
```

**Expected results:**
- **Connection successful** = Your miner is accessible
- **Connection refused/hangs** = Your miner is not accessible (needs port forwarding)

---

## Solution: Port Forwarding

**Step 1: Find your internal IP**
```bash
ifconfig | grep inet
# Look for something like: inet 192.168.1.63
```

**Step 2: Access your router admin panel**
- Usually: http://192.168.1.1 or http://192.168.0.1
- Login with router credentials

**Step 3: Add port forwarding rule**
- **Service Name**: Bittensor Miner
- **External Port**: 8091 (or your chosen port)
- **Internal IP**: 192.168.1.63 (your machine's internal IP)
- **Internal Port**: 8091 (same as external)
- **Protocol**: TCP

**Step 4: Restart your miner with explicit port**

Use lowercase flags (`--axon.port`, not `--Axon.port`):

```bash
python3 -m neurons.miner --axon.port 8091 --axon.external_port 8091 --netuid 420 --subtensor.network test --wallet.name miner --wallet.hotkey default
```

For custom model miner (advanced, currently not working):
```bash
python -m miner_model.miner_plugin --axon.port 8091 --axon.external_port 8091 --netuid 420 --subtensor.network test --wallet.name miner --wallet.hotkey default
```

---

## Alternative: Use Different Ports

If your ISP blocks certain ports, try:
```bash
python3 -m neurons.miner --axon.port 80 --axon.external_port 80 --netuid 420 --subtensor.network test --wallet.name miner --wallet.hotkey default
python3 -m neurons.miner --axon.port 8080 --axon.external_port 8080 --netuid 420 --subtensor.network test --wallet.name miner --wallet.hotkey default
python3 -m neurons.miner --axon.port 443 --axon.external_port 443 --netuid 420 --subtensor.network test --wallet.name miner --wallet.hotkey default
```

---

## Testing Connectivity

**From your machine:**
```bash
telnet YOUR_EXTERNAL_IP YOUR_PORT
```

**From external machine (VPS/friend's machine):**
```bash
telnet YOUR_EXTERNAL_IP YOUR_PORT
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `Connection refused` | Port forwarding not configured |
| `Connection hangs/timeout` | Firewall blocking or ISP restrictions |
| `Cannot connect to host` | Wrong IP address or port |
| `100% packet loss on ping` | ICMP blocked (normal), test with telnet instead |

---

## Firewall Configuration

**Linux/macOS:**
```bash
sudo ufw allow 8091
# or
sudo iptables -A INPUT -p tcp --dport 8091 -j ACCEPT
```

**Windows:**
- Windows Defender Firewall → Inbound Rules → New Rule → Port → TCP 8091

---

## VPS Validators

If running validators on VPS:
- VPS usually has open outbound connections
- Main issue is miners not being accessible from VPS
- Follow port forwarding steps above

---

## Success Indicators

**Working correctly:**
```
[COLLECT] UID=3, Prediction=7.2456, Interval=[7.1200, 7.3700]
```

**Not working:**
```
[NO_SUBMISSION] UID=3 provided no prediction - will receive zero reward
```

---

**Prev:** [08 – Local Run (Advanced)](08-local-run-advanced.md) | **Next:** — | [Back to Guide Index](../../README.md#guide)

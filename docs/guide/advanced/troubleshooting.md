# Troubleshooting

> [Advanced](README.md). For `[NO_SUBMISSION]` or miners not reachable.

Validators must reach your miner over the internet. Firewalls and home routers often block inbound connections.

---

## Quick check

```bash
telnet YOUR_EXTERNAL_IP YOUR_MINER_PORT
# e.g. telnet 69.115.169.144 8091
```

- Works → miner reachable  
- Refused / hangs → configure port forwarding

---

## Port forwarding (summary)

1. Find internal IP: `ifconfig | grep inet`
2. Router admin (often `http://192.168.1.1`)
3. Forward external port → your machine IP, same port (e.g. TCP 8091)

Restart miner with explicit ports (lowercase flags):

```bash
python3 -m neurons.miner --axon.port 8091 --axon.external_port 8091 --netuid 183 --subtensor.network test --wallet.name miner --wallet.hotkey default
```

Custom model miner:

```bash
python -m miner_model.miner_plugin --axon.port 8091 --axon.external_port 8091 --netuid 183 --subtensor.network test --wallet.name miner --wallet.hotkey default
```

---

## Logs

**OK:**
```
[COLLECT] UID=3, Prediction=7.2456, Interval=[7.1200, 7.3700]
```

**Problem:**
```
[NO_SUBMISSION] UID=3 provided no prediction - will receive zero reward
```

---

[← Local run](local-run.md) · [Advanced index](README.md) · [Guide](../../../README.md#guide)

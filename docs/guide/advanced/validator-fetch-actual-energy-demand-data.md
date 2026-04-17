# Validator Example: Fetch Actual Energy Demand Data

> **Advanced.** Companion page for [Run validator](run-validator.md).

This page shows how the validator fetches the **actual ground-truth demand** (`LoadMw`) for scoring miner predictions.

---

## Why validator needs actual demand

The validator first asks miners to predict demand for a target 5-minute slot, then later fetches the real value for that same slot and computes rewards.

At a high level:

1. Validator sends challenge timestamp (future 5-minute slot).
2. Miners return predictions.
3. Validator waits for data publication delay.
4. Validator fetches actual `LoadMw` from ISO-NE API.
5. Validator scores predictions vs. actual value.

---

## Code path in this repo

The ground-truth fetch path is:

- `neurons/validator.py` -> `get_actual_load_mw(timestamp)`
- `bittbridge/validator/reward.py` -> `get_actual_load_mw()`
- `bittbridge/utils/iso_ne_api.py` -> `get_load_mw_for_timestamp()`

The request is backed by ISO-NE endpoint:

- `https://webservices.iso-ne.com/api/v1.1/fiveminutesystemload/day/{YYYYMMDD}`

Authentication uses `.env` variables:

- `ISO_NE_USERNAME`
- `ISO_NE_PASSWORD`

---

## Validator timing behavior (important)

The validator stores predictions first and evaluates them later:

- `evaluation_delay=600` seconds (10 minutes) in `neurons/validator.py`.
- This delay gives ISO-NE time to publish the actual value for the requested slot.
- If actual data is not available yet, validator logs retry message and keeps those predictions in queue until data appears.

---

## Minimal Python example (same logic as validator)

Run from repo root (`~/bittbridge`) with venv activated:

```bash
python - <<'PY'
from dotenv import load_dotenv
from bittbridge.utils.timestamp import get_before, round_to_interval, to_str
from bittbridge.utils.iso_ne_api import get_load_mw_for_timestamp

load_dotenv()

# Use a timestamp in the recent past so actual data is likely published
target = round_to_interval(get_before(minutes=15), interval_minutes=5)
ts = to_str(target)
actual = get_load_mw_for_timestamp(ts)

print(f"Timestamp: {ts}")
print(f"Actual LoadMw: {actual}")
PY
```

If the result is `None`, wait a few minutes and retry (API publish lag can vary).

---

## Quick built-in check script

You can run the included script that tests both miner-style and validator-style API access:

```bash
python scripts/check_iso_ne_api.py
```

Expected validator-style output looks like:

```text
Validator-style: actual LoadMw for slot 2026-04-17T13:45:00.000000-04:00 = 14320.4 MW
SUCCESS: API is working. You can run the miner and validator.
```

---

## Troubleshooting

- `actual` is `None`: target slot may not be published yet; retry with an older slot (15-30 minutes ago).
- Empty API result: verify `.env` credentials (`ISO_NE_USERNAME`, `ISO_NE_PASSWORD`).
- Connection/auth errors: run `python scripts/check_iso_ne_api.py` to isolate API issues before running validator.

---

[Run validator](run-validator.md) · [Advanced index](README.md) · [Guide](../../../README.md#guide)

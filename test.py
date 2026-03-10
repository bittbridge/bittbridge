#!/usr/bin/env python3
"""
Test script to verify ISO-NE API access and credentials.

Uses the same logic as miner and validator:
- Miner-style: fetch latest N LoadMw values, compute MA
- Validator-style: get actual LoadMw for a specific 5-min timestamp

Run after setting .env (copy from .env.example, set ISO_NE_USERNAME and ISO_NE_PASSWORD).
Usage: python test.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

# Load .env before importing bittbridge (which may use env vars)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add bittbridge to path if running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bittbridge.utils.iso_ne_api import fetch_fiveminute_system_load, get_load_mw_for_timestamp
from bittbridge.utils.timestamp import get_now, round_to_interval, to_str, get_before

N_STEPS = 12


def main():
    username = os.getenv("ISO_NE_USERNAME")
    password = os.getenv("ISO_NE_PASSWORD")

    if not username or not password:
        print("ERROR: ISO_NE_USERNAME and ISO_NE_PASSWORD must be set in .env")
        print("  1. Copy .env.example to .env")
        print("  2. Sign up at https://www.iso-ne.com/isoexpress/login (Create account)")
        print("  3. Add your username and password to .env")
        sys.exit(1)

    print("Testing ISO-NE API access...")
    print()

    # Use yesterday for testing (today might not have full data yet)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
    day_yyyymmdd = yesterday

    # Miner-style check: fetch day, take last N, compute MA
    data = fetch_fiveminute_system_load(day_yyyymmdd, use_cache=False)
    if not data:
        print("FAIL: Could not fetch data from API (check credentials and network)")
        sys.exit(1)

    load_values = [load_mw for _, load_mw in data]
    if len(load_values) < N_STEPS:
        print(f"FAIL: Insufficient data ({len(load_values)} rows, need {N_STEPS})")
        sys.exit(1)

    recent = load_values[-N_STEPS:]
    ma = sum(recent) / len(recent)
    print(f"Miner-style: latest {N_STEPS} LoadMw -> MA = {ma:.1f} MW")
    print()

    # Validator-style check: get actual LoadMw for a 5-min slot.
    # Use a slot 15 minutes ago so the API has published it (real-time data has ~5–15 min delay).
    past = get_before(minutes=15)
    slot_rounded = round_to_interval(past, interval_minutes=5)
    ts_str = to_str(slot_rounded)
    actual = get_load_mw_for_timestamp(ts_str)
    if actual is None:
        print(f"FAIL: Could not get actual LoadMw for timestamp {ts_str}")
        print("      (API may not have this slot yet, or check credentials/network.)")
        sys.exit(1)

    print(f"Validator-style: actual LoadMw for slot {ts_str} = {actual:.1f} MW")
    print()
    print("SUCCESS: API is working. You can run the miner and validator.")
    sys.exit(0)


if __name__ == "__main__":
    main()

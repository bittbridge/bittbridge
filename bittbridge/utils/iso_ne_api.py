"""
ISO-NE Web Services API helper for Five Minute System Load.

Fetches LoadMw data from https://webservices.iso-ne.com/api/v1.1/fiveminutesystemload/day/{YYYYMMDD}
using HTTP Basic Auth. Credentials from .env: ISO_NE_USERNAME, ISO_NE_PASSWORD.
"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Tuple

import requests
from pytz import timezone

ISO_NE_BASE_URL = "https://webservices.iso-ne.com/api/v1.1"
UTC = timezone("UTC")
# ISO-NE uses Eastern time for "day" in the API (day/YYYYMMDD)
EASTERN = timezone("America/New_York")

# Day-level cache for validator: {day_yyyymmdd: [(datetime_utc, load_mw), ...]}
_day_cache: dict = {}


def _get_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Get ISO-NE credentials from environment."""
    username = os.getenv("ISO_NE_USERNAME")
    password = os.getenv("ISO_NE_PASSWORD")
    return username, password


def _parse_xml_response(text: str) -> List[Tuple[datetime, float]]:
    """
    Parse XML response from fiveminutesystemload endpoint.
    Returns list of (datetime_utc, load_mw) sorted by datetime.
    """
    root = ET.fromstring(text)
    # Handle namespace: xmlns="http://WEBSERV.iso-ne.com"
    ns = {"iso": "http://WEBSERV.iso-ne.com"}
    # Try with namespace first
    elements = root.findall(".//iso:FiveMinSystemLoad", ns)
    if not elements:
        elements = root.findall(".//FiveMinSystemLoad")
    if not elements:
        # Fallback: search without namespace
        for elem in root.iter():
            if "FiveMinSystemLoad" in elem.tag:
                elements = root.findall(f".//{{*}}{elem.tag.split('}')[-1]}")
                break
        if not elements:
            elements = list(root.iter("FiveMinSystemLoad"))

    results = []
    for elem in elements:
        begin_date_elem = elem.find("iso:BeginDate", ns) or elem.find("BeginDate")
        load_mw_elem = elem.find("iso:LoadMw", ns) or elem.find("LoadMw")
        if begin_date_elem is None:
            for child in elem:
                if "BeginDate" in child.tag:
                    begin_date_elem = child
                    break
        if load_mw_elem is None:
            for child in elem:
                if "LoadMw" in child.tag:
                    load_mw_elem = child
                    break

        if begin_date_elem is None or load_mw_elem is None:
            continue

        begin_str = begin_date_elem.text
        load_str = load_mw_elem.text
        if not begin_str or not load_str:
            continue

        try:
            # Parse BeginDate e.g. "2026-03-09T00:00:00.000-04:00"
            dt = datetime.fromisoformat(begin_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            dt_utc = dt.astimezone(UTC)
            load_mw = float(load_str)
            results.append((dt_utc, load_mw))
        except (ValueError, TypeError):
            continue

    results.sort(key=lambda x: x[0])
    return results


def fetch_fiveminute_system_load(
    day_yyyymmdd: str,
    use_cache: bool = True,
) -> List[Tuple[datetime, float]]:
    """
    Fetch Five Minute System Load for a given day.

    Args:
        day_yyyymmdd: Day in YYYYMMDD format (e.g. "20260309")
        use_cache: If True, return cached data for this day when available

    Returns:
        List of (datetime_utc, load_mw) sorted by datetime. Empty list on failure.
    """
    if use_cache and day_yyyymmdd in _day_cache:
        return _day_cache[day_yyyymmdd]

    username, password = _get_credentials()
    if not username or not password:
        return []

    url = f"{ISO_NE_BASE_URL}/fiveminutesystemload/day/{day_yyyymmdd}"
    try:
        response = requests.get(
            url,
            auth=(username, password),
            headers={"Accept": "application/xml"},
            timeout=30,
        )
        response.raise_for_status()
        results = _parse_xml_response(response.text)
        if use_cache:
            _day_cache[day_yyyymmdd] = results
        return results
    except Exception:
        return []


def _parse_timestamp(timestamp: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime in global timezone (Eastern). Handles various ISO formats."""
    try:
        from bittbridge.utils.timestamp import to_datetime
        return to_datetime(timestamp)
    except (ValueError, TypeError):
        pass
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except (ValueError, TypeError):
        return None


def get_load_mw_for_timestamp(timestamp: str) -> Optional[float]:
    """
    Get actual LoadMw for a 5-minute slot matching the given timestamp.
    Used by validator for ground truth.

    Args:
        timestamp: ISO format string (e.g. "2024-01-15T10:30:00.000Z" or "2024-01-15T10:30:00+00:00")

    Returns:
        LoadMw for that 5-min slot, or None if not found
    """
    from bittbridge.utils.timestamp import round_to_interval

    try:
        dt = _parse_timestamp(timestamp)
        if dt is None:
            return None
        dt_rounded = round_to_interval(dt, interval_minutes=5)
        # dt_rounded is already in Eastern (timestamp module uses Eastern); API day is Eastern
        day_yyyymmdd = dt_rounded.strftime("%Y%m%d")
        data = fetch_fiveminute_system_load(day_yyyymmdd, use_cache=True)
        dt_rounded = dt_rounded.replace(second=0, microsecond=0)
        for slot_dt, load_mw in data:
            slot_normalized = slot_dt.replace(second=0, microsecond=0)
            if slot_normalized == dt_rounded:
                return load_mw
        return None
    except Exception:
        return None


def clear_cache() -> None:
    """Clear the day cache (e.g. for testing)."""
    global _day_cache
    _day_cache = {}

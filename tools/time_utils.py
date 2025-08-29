
from __future__ import annotations
from datetime import datetime, timezone
from typing import Union

_SEC_PER_YEAR = 365*24*60*60

def to_unix_ms(ts: Union[str, int, float, datetime, None]) -> int | None:
    if ts is None:
        return None
    if isinstance(ts, int):
        return ts if ts > 10_000_000_000 else int(ts * 1000)
    if isinstance(ts, float):
        return int(ts * 1000)
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int(ts.timestamp() * 1000)
    if isinstance(ts, str):
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    raise TypeError(f"Unsupported ts type: {type(ts)}")

def infer_period_seconds(interval: str) -> int:
    """Infer bar length in seconds from strings like '1m','15m','1h','1d'."""
    s = interval.strip().lower()
    try:
        if s.endswith('ms'):
            return max(1, int(s[:-2]) // 1000)
        if s.endswith('s'):
            return int(s[:-1])
        if s.endswith('m'):
            return int(s[:-1]) * 60
        if s.endswith('h'):
            return int(s[:-1]) * 3600
        if s.endswith('d'):
            return int(s[:-1]) * 86400
        # default assume minutes
        return int(s) * 60
    except Exception:
        return 60

def annualization_factor(interval: str) -> float:
    sec = infer_period_seconds(interval)
    return (_SEC_PER_YEAR / max(1, sec)) ** 0.5

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock

_LOCK = Lock()
_LAST: datetime | None = None


def utcnow_iso() -> str:
    """UTC timestamp as ISO string, guaranteed strictly monotonic across calls.

    Windows clocks only tick at ~15ms, so two back-to-back calls can otherwise
    produce identical values. When that happens, the returned timestamp is
    bumped by 1 microsecond past the previous result.
    """
    global _LAST
    with _LOCK:
        now = datetime.now(timezone.utc)
        if _LAST is not None and now <= _LAST:
            now = _LAST + timedelta(microseconds=1)
        _LAST = now
        return now.isoformat()

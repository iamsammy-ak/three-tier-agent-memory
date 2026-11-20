"""Injectable clocks: real time for production, a manual clock for simulations, tests and the demo."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Current time as UNIX seconds."""
        ...


class SystemClock:
    """Wall-clock time."""

    def now(self) -> float:
        return time.time()


class ManualClock:
    """A clock that only moves when told to, so decay and TTLs can be simulated over months in milliseconds."""

    def __init__(self, start: float | datetime = 0.0) -> None:
        self._t = start.timestamp() if isinstance(start, datetime) else float(start)

    def now(self) -> float:
        return self._t

    def set(self, ts: float) -> None:
        self._t = float(ts)

    def advance(self, *, days: float = 0.0, hours: float = 0.0, minutes: float = 0.0, seconds: float = 0.0) -> float:
        self._t += days * 86400 + hours * 3600 + minutes * 60 + seconds
        return self._t


def parse_when(value: str) -> float:
    """Parse an ISO date or date-time (UTC assumed when no offset is given) into UNIX seconds."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()

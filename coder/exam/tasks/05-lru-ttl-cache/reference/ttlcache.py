"""A least-recently-used cache whose entries expire after a fixed time."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any

_MISSING = object()


class TTLCache:
    """LRU cache with per-entry expiry. Each instance has its own storage; thread-safe."""

    def __init__(self, maxsize: int, ttl: float, clock: Callable[[], float] = time.monotonic) -> None:
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError(f"maxsize must be a positive integer, got {maxsize!r}")
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError(f"ttl must be a positive number of seconds, got {ttl!r}")
        self.maxsize = maxsize
        self.ttl = ttl
        self._clock = clock
        # Recency order: least recently used first.
        self._values: OrderedDict[Hashable, Any] = OrderedDict()
        # Expiry order: the ttl is the same for every entry, so the entry set
        # longest ago always expires first.
        self._expires: OrderedDict[Hashable, float] = OrderedDict()
        self._lock = threading.RLock()

    def _remove(self, key: Hashable) -> None:
        del self._values[key]
        del self._expires[key]

    def _purge_expired(self, now: float) -> None:
        while self._expires:
            key, expires_at = next(iter(self._expires.items()))
            if expires_at > now:
                break
            self._remove(key)

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            expires_at = self._expires.get(key, _MISSING)
            if expires_at is _MISSING:
                return default
            if expires_at <= self._clock():
                self._remove(key)
                return default
            self._values.move_to_end(key)
            return self._values[key]

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            if key in self._values:
                self._remove(key)
            while len(self._values) >= self.maxsize:
                oldest = next(iter(self._values))
                self._remove(oldest)
            self._values[key] = value
            self._expires[key] = now + self.ttl

    def delete(self, key: Hashable) -> None:
        with self._lock:
            if key in self._values:
                self._remove(key)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            expires_at = self._expires.get(key)
            return expires_at is not None and expires_at > self._clock()

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired(self._clock())
            return len(self._values)

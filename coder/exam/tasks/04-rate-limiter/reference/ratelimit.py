"""Token-bucket rate limiting with an injectable clock."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Hashable

Clock = Callable[[], float]


def _positive(name: str, value: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
        return float(value)
    raise ValueError(f"{name} must be a positive finite number, got {value!r}")


class TokenBucket:
    """A bucket that refills at `rate` tokens per second up to `capacity` tokens."""

    def __init__(self, rate: float, capacity: float, clock: Clock = time.monotonic) -> None:
        self.rate = _positive("rate", rate)
        self.capacity = _positive("capacity", capacity)
        self._clock = clock
        self._tokens = self.capacity
        self._last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        if elapsed > 0:  # a clock that goes backwards must not drain the bucket
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last = max(self._last, now)

    def allow(self, cost: float = 1) -> bool:
        cost = _positive("cost", cost)
        with self._lock:
            self._refill()
            if cost <= self._tokens:
                self._tokens -= cost
                return True
            return False


class KeyedLimiter:
    """One token bucket per key, created on first use."""

    def __init__(self, rate: float, capacity: float, clock: Clock = time.monotonic) -> None:
        self.rate = _positive("rate", rate)
        self.capacity = _positive("capacity", capacity)
        self._clock = clock
        self._buckets: dict[Hashable, TokenBucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: Hashable, cost: float = 1) -> bool:
        _positive("cost", cost)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = self._buckets[key] = TokenBucket(self.rate, self.capacity, self._clock)
        return bucket.allow(cost)

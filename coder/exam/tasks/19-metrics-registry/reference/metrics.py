"""Thread-safe counters and gauges with a plain-text exporter."""

from __future__ import annotations

import re
import threading
from numbers import Real

_NAME = re.compile(r"[a-z][a-z0-9_]*")


def _check_amount(amount: object) -> None:
    if isinstance(amount, bool) or not isinstance(amount, Real):
        raise TypeError(f"amount must be a number, got {amount!r}")


class Counter:
    """A value that only goes up."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._value: float = 0

    def inc(self, amount: float = 1) -> None:
        _check_amount(amount)
        if amount < 0:
            raise ValueError("a counter can only go up")
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class Gauge:
    """A value that can go up and down."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._lock = threading.Lock()
        self._value: float = 0

    def set(self, value: float) -> None:
        _check_amount(value)
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1) -> None:
        _check_amount(amount)
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1) -> None:
        _check_amount(amount)
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        with self._lock:
            return self._value


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, Counter | Gauge] = {}

    def _get_or_create(self, name: str, kind: type) -> Counter | Gauge:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"invalid metric name: {name!r}")
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None:
                metric = kind(name)
                self._metrics[name] = metric
            elif not isinstance(metric, kind):
                raise ValueError(f"{name!r} is already a {type(metric).__name__.lower()}")
            return metric

    def counter(self, name: str) -> Counter:
        return self._get_or_create(name, Counter)

    def gauge(self, name: str) -> Gauge:
        return self._get_or_create(name, Gauge)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            metrics = list(self._metrics.items())
        return {name: metric.value for name, metric in sorted(metrics)}

    def render(self) -> str:
        return "".join(f"{name} {value}\n" for name, value in self.snapshot().items())

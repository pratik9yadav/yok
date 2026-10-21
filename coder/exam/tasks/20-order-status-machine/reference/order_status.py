"""Order status rules: placed -> packed -> shipped -> delivered, cancel only before shipping."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

PLACED = "placed"
PACKED = "packed"
SHIPPED = "shipped"
DELIVERED = "delivered"
CANCELLED = "cancelled"

_ALLOWED: dict[str, frozenset[str]] = {
    PLACED: frozenset({PACKED, CANCELLED}),
    PACKED: frozenset({SHIPPED, CANCELLED}),
    SHIPPED: frozenset({DELIVERED}),
    DELIVERED: frozenset(),
    CANCELLED: frozenset(),
}


class InvalidTransition(Exception):
    """The requested status change is not allowed from the current status."""


class Order:
    def __init__(self, order_id: str, clock: Callable[[], float] = time.time) -> None:
        if not isinstance(order_id, str) or not order_id.strip():
            raise ValueError("order_id must be a non-empty string")
        self._order_id = order_id
        self._clock = clock
        self._lock = threading.Lock()
        self._status = PLACED
        self._history: list[tuple[str, float]] = [(PLACED, clock())]
        self._tracking_id: str | None = None
        self._cancel_reason: str | None = None

    @property
    def order_id(self) -> str:
        return self._order_id

    @property
    def status(self) -> str:
        return self._status

    @property
    def history(self) -> list[tuple[str, float]]:
        with self._lock:
            return list(self._history)

    @property
    def tracking_id(self) -> str | None:
        return self._tracking_id

    @property
    def cancel_reason(self) -> str | None:
        return self._cancel_reason

    def _move(self, new_status: str) -> None:
        # caller holds self._lock
        if new_status not in _ALLOWED[self._status]:
            raise InvalidTransition(f"order {self._order_id}: cannot go from {self._status} to {new_status}")
        self._status = new_status
        self._history.append((new_status, self._clock()))

    def pack(self) -> None:
        with self._lock:
            self._move(PACKED)

    def ship(self, tracking_id: str) -> None:
        if not isinstance(tracking_id, str) or not tracking_id.strip():
            raise ValueError("a non-empty tracking id is required to ship")
        with self._lock:
            self._move(SHIPPED)
            self._tracking_id = tracking_id.strip()

    def deliver(self) -> None:
        with self._lock:
            self._move(DELIVERED)

    def cancel(self, reason: str = "") -> None:
        with self._lock:
            self._move(CANCELLED)
            self._cancel_reason = reason

    def __repr__(self) -> str:
        return f"Order({self._order_id!r}, status={self._status!r})"

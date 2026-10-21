"""Stock ledger for a small shop: receive and issue stock, idempotent by move id."""

from __future__ import annotations

from collections import defaultdict
from numbers import Integral


class InsufficientStock(Exception):
    """Raised when an issue asks for more units than are in stock."""


def _check_qty(qty: object) -> int:
    """The quantity as an int; ValueError unless it is a whole number above zero."""
    if isinstance(qty, Integral) and not isinstance(qty, bool) and qty > 0:
        return int(qty)
    raise ValueError(f"quantity must be a whole number above zero, got {qty!r}")


class Ledger:
    """In-memory stock ledger. Each instance keeps its own state."""

    def __init__(self) -> None:
        self._stock: dict[str, int] = defaultdict(int)
        self._history: dict[str, list[dict]] = defaultdict(list)
        self._seen_moves: set = set()

    def receive(self, sku: str, qty: int, move_id: object = None) -> int:
        """Add qty units of sku. Returns the new stock level."""
        return self._apply(sku, qty, move_id, "receive")

    def issue(self, sku: str, qty: int, move_id: object = None) -> int:
        """Take qty units of sku out. Returns the new stock level."""
        return self._apply(sku, qty, move_id, "issue")

    def stock(self, sku: str) -> int:
        return self._stock.get(sku, 0)

    def history(self, sku: str) -> list[dict]:
        return [dict(move) for move in self._history.get(sku, ())]

    def _apply(self, sku: str, qty: int, move_id: object, kind: str) -> int:
        qty = _check_qty(qty)
        if move_id is not None and move_id in self._seen_moves:
            return self.stock(sku)
        current = self._stock.get(sku, 0)
        if kind == "issue":
            if qty > current:
                raise InsufficientStock(f"{sku}: asked for {qty}, only {current} in stock")
            new = current - qty
        else:
            new = current + qty
        self._stock[sku] = new
        self._history[sku].append({"move_id": move_id, "kind": kind, "qty": qty})
        if move_id is not None:
            self._seen_moves.add(move_id)
        return new

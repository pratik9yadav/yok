"""Shared stock counts for all the tills in the shop.

Every till talks to one StockBook object inside the shop server (one thread per request).
Each change is also written to the ledger (a slow append to a file on disk).

Check-and-update of a count happens under a lock, so two tills can never both take the
last unit and concurrent restocks are never lost. The ledger is written while the lock is
held so that a failed ledger write leaves the stock unchanged, exactly as before.
"""

import threading


class OutOfStock(Exception):
    pass


class NullLedger:
    def record(self, action, sku, qty):
        pass


class StockBook:
    def __init__(self, stock=None, ledger=None):
        self._stock = dict(stock or {})
        self._ledger = ledger if ledger is not None else NullLedger()
        self._lock = threading.Lock()

    def available(self, sku):
        with self._lock:
            return self._stock.get(sku, 0)

    @staticmethod
    def _check_qty(qty):
        if qty <= 0:
            raise ValueError("qty must be positive")

    def reserve(self, sku, qty):
        """Take `qty` units for a customer's bill. Raises OutOfStock if there aren't enough."""
        self._check_qty(qty)
        with self._lock:
            available = self._stock.get(sku, 0)
            if qty > available:
                raise OutOfStock(f"{sku}: wanted {qty}, have {available}")
            self._ledger.record("reserve", sku, qty)
            self._stock[sku] = available - qty

    def _add(self, action, sku, qty):
        self._check_qty(qty)
        with self._lock:
            current = self._stock.get(sku, 0)
            self._ledger.record(action, sku, qty)
            self._stock[sku] = current + qty

    def release(self, sku, qty):
        """Put units back when a bill is cancelled."""
        self._add("release", sku, qty)

    def restock(self, sku, qty):
        """Add units from a supplier delivery."""
        self._add("restock", sku, qty)

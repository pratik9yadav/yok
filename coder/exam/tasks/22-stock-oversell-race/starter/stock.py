"""Shared stock counts for all the tills in the shop.

Every till talks to one StockBook object inside the shop server (one thread per request).
Each change is also written to the ledger (a slow append to a file on disk).
"""


class OutOfStock(Exception):
    pass


class NullLedger:
    def record(self, action, sku, qty):
        pass


class StockBook:
    def __init__(self, stock=None, ledger=None):
        self._stock = dict(stock or {})
        self._ledger = ledger if ledger is not None else NullLedger()

    def available(self, sku):
        return self._stock.get(sku, 0)

    def reserve(self, sku, qty):
        """Take `qty` units for a customer's bill. Raises OutOfStock if there aren't enough."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        available = self._stock.get(sku, 0)
        if qty > available:
            raise OutOfStock(f"{sku}: wanted {qty}, have {available}")
        self._ledger.record("reserve", sku, qty)
        self._stock[sku] = available - qty

    def release(self, sku, qty):
        """Put units back when a bill is cancelled."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        current = self._stock.get(sku, 0)
        self._ledger.record("release", sku, qty)
        self._stock[sku] = current + qty

    def restock(self, sku, qty):
        """Add units from a supplier delivery."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        current = self._stock.get(sku, 0)
        self._ledger.record("restock", sku, qty)
        self._stock[sku] = current + qty

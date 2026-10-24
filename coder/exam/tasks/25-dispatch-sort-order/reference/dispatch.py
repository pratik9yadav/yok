"""Which order the packers should work on next."""

from dataclasses import dataclass
from datetime import date


@dataclass
class Order:
    order_id: str
    express: bool
    promised_by: date | None  # None = no promised date (e.g. pickup at the counter)


def _dispatch_key(order):
    # express first; then dated orders by date; undated last. sorted() is stable, so
    # orders with the same key keep the order they were placed in (first come, first served).
    return (not order.express, order.promised_by is None, order.promised_by or date.min)


def dispatch_queue(orders):
    """Return the orders in the sequence the packers should work on them.

    `orders` is the list of open orders in the order they were placed. It is not modified.
    """
    return sorted(orders, key=_dispatch_key)

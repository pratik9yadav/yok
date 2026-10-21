"""Which order the packers should work on next."""

from dataclasses import dataclass
from datetime import date
from functools import cmp_to_key


@dataclass
class Order:
    order_id: str
    express: bool
    promised_by: date | None  # None = no promised date (e.g. pickup at the counter)


def _compare(a, b):
    # express orders first
    if a.express != b.express:
        return -1 if a.express else 1
    # then the earliest promised date first
    return -1 if a.promised_by <= b.promised_by else 1


def dispatch_queue(orders):
    """Return the orders in the sequence the packers should work on them.

    `orders` is the list of open orders in the order they were placed.
    """
    orders.sort(key=cmp_to_key(_compare))
    return orders

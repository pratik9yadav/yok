"""Orders: a list of dicts like {"item": "MCB", "qty": 2, "price": 220.0, "day": datetime.date(...)}."""

import datetime


def calc_total(order: dict) -> float:
    return order["qty"] * order["price"]


def in_stock_count(stock: dict[str, int]) -> int:
    """How many items have at least one piece in stock."""
    return sum(1 for qty in stock.values() if qty > 0)


def weekly_totals(orders: list[dict]) -> dict[str, float]:
    """Total sales per weekday name, Monday to Sunday."""
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    totals = {name: 0.0 for name in names}
    for order in orders:
        day: datetime.date = order["day"]
        totals[names[day.isoweekday() % 7]] += calc_total(order)
    return totals

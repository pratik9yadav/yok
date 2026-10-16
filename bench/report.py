"""A text report of orders."""

from bench.orders import order_total


def order_lines(orders: list[dict]) -> list[str]:
    return [f"{o['item']}: {o['qty']} x {o['price']} = {order_total(o)}" for o in orders]

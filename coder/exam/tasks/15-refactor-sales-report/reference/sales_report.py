"""Daily sales report in several output formats.

To add a format, write a function that takes a Summary and returns text,
and register it in FORMATTERS.
"""

import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

PAISA = Decimal("0.01")
TOP_PRODUCTS = 3


def _money(amount):
    return str(amount.quantize(PAISA))


@dataclass
class Totals:
    orders: set = field(default_factory=set)
    items: int = 0
    amount: Decimal = Decimal(0)

    def add(self, sale):
        self.orders.add(sale["order_id"])
        self.items += sale["qty"]
        self.amount += sale["amount"]


@dataclass
class Summary:
    days: list          # [(date, Totals)] sorted by date
    total: Totals
    top_products: list  # [(sku, items, amount)] most items first, then by sku


def summarise(sales):
    by_day = defaultdict(Totals)
    by_sku = defaultdict(Totals)
    total = Totals()
    for sale in sales:
        by_day[sale["date"]].add(sale)
        by_sku[sale["sku"]].add(sale)
        total.add(sale)
    products = sorted(
        ((sku, t.items, t.amount) for sku, t in by_sku.items()),
        key=lambda p: (-p[1], p[0]),
    )
    return Summary(days=sorted(by_day.items()), total=total, top_products=products[:TOP_PRODUCTS])


def _totals_text(t):
    return f"orders: {len(t.orders)}  items: {t.items}  amount: {_money(t.amount)}"


def format_text(summary):
    lines = ["Sales report", "============"]
    if not summary.days:
        lines.append("No sales.")
        return "\n".join(lines) + "\n"
    lines.extend(f"{day}  {_totals_text(t)}" for day, t in summary.days)
    lines.append(f"Total  {_totals_text(summary.total)}")
    lines.append("")
    lines.append("Top products:")
    lines.extend(
        f"  {rank}. {sku}  {items} items  {_money(amount)}"
        for rank, (sku, items, amount) in enumerate(summary.top_products, start=1)
    )
    return "\n".join(lines) + "\n"


def format_csv(summary):
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(["date", "orders", "items", "amount"])
    for day, t in summary.days:
        writer.writerow([day, len(t.orders), t.items, _money(t.amount)])
    return out.getvalue()


FORMATTERS = {"text": format_text, "csv": format_csv}


def build_report(sales, fmt="text"):
    """Sales report for sales (dicts with order_id, date, sku, qty, amount) in the given format."""
    try:
        formatter = FORMATTERS[fmt]
    except (KeyError, TypeError):
        raise ValueError(f"unknown report format {fmt!r}; choose from {', '.join(FORMATTERS)}") from None
    return formatter(summarise(sales))

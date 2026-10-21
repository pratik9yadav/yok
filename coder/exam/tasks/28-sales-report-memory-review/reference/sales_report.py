"""Nightly sales report from a branch's sales log.

The log is a CSV file with the header `date,sku,qty,price_paise` and one line per item sold.
The till appends to it all year, so the file is read as a stream, one line at a time: memory
use depends on the number of distinct days and SKUs, never on the size of the file.
"""

import heapq


def iter_rows(path):
    """Yield (date, sku, qty, price_paise) for each sale, streaming the file."""
    with open(path, encoding="utf-8", newline=None) as f:
        f.readline()  # header
        for n, line in enumerate(f, start=2):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split(",")
            if len(parts) != 4:
                raise ValueError(f"{path} line {n}: expected 4 fields, got {len(parts)}")
            date, sku, qty, price = parts
            try:
                yield date, sku, int(qty), int(price)
            except ValueError:
                raise ValueError(f"{path} line {n}: bad number") from None


def read_rows(path):
    """All rows as dicts. Loads the whole file: only for small files; reports use iter_rows."""
    return [{"date": d, "sku": s, "qty": q, "price_paise": p} for d, s, q, p in iter_rows(path)]


def _totals(path):
    revenue = units = 0
    by_day: dict[str, int] = {}
    by_sku_units: dict[str, int] = {}
    by_sku_revenue: dict[str, int] = {}
    for date, sku, qty, price in iter_rows(path):
        amount = qty * price
        revenue += amount
        units += qty
        by_day[date] = by_day.get(date, 0) + amount
        by_sku_units[sku] = by_sku_units.get(sku, 0) + qty
        by_sku_revenue[sku] = by_sku_revenue.get(sku, 0) + amount
    return revenue, units, by_day, by_sku_units, by_sku_revenue


def summarize(path):
    """Totals for the report.

    Returns {"revenue_paise": int, "units": int, "by_day": {date: revenue_paise}, "by_sku": {sku: units}},
    with by_day and by_sku sorted by key.
    """
    revenue, units, by_day, by_sku_units, _ = _totals(path)
    return {
        "revenue_paise": revenue,
        "units": units,
        "by_day": dict(sorted(by_day.items())),
        "by_sku": dict(sorted(by_sku_units.items())),
    }


def top_skus(path, n=10):
    """The `n` SKUs with the most revenue as [(sku, revenue_paise)], highest first, ties by sku."""
    if n <= 0:
        return []
    totals = _totals(path)[4]
    return heapq.nsmallest(n, totals.items(), key=lambda kv: (-kv[1], kv[0]))

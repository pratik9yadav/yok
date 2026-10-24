"""Nightly sales report from a branch's sales log.

The log is a CSV file with the header `date,sku,qty,price_paise` and one line per item sold.
The till appends to it all year.
"""


def read_rows(path):
    f = open(path)
    lines = f.read().splitlines()
    rows = []
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 4:
            raise ValueError(f"{path} line {n}: expected 4 fields, got {len(parts)}")
        date, sku, qty, price = parts
        try:
            rows.append({"date": date, "sku": sku, "qty": int(qty), "price_paise": int(price)})
        except ValueError:
            raise ValueError(f"{path} line {n}: bad number")
    return rows


def summarize(path):
    """Totals for the report.

    Returns {"revenue_paise": int, "units": int, "by_day": {date: revenue_paise}, "by_sku": {sku: units}},
    with by_day and by_sku sorted by key.
    """
    rows = read_rows(path)
    revenue = sum(r["qty"] * r["price_paise"] for r in rows)
    units = sum(r["qty"] for r in rows)
    by_day = {}
    for r in rows:
        by_day[r["date"]] = by_day.get(r["date"], 0) + r["qty"] * r["price_paise"]
    by_sku = {}
    for r in rows:
        by_sku[r["sku"]] = by_sku.get(r["sku"], 0) + r["qty"]
    return {
        "revenue_paise": revenue,
        "units": units,
        "by_day": dict(sorted(by_day.items())),
        "by_sku": dict(sorted(by_sku.items())),
    }


def top_skus(path, n=10):
    """The `n` SKUs with the most revenue as [(sku, revenue_paise)], highest first, ties by sku."""
    rows = read_rows(path)
    rows.sort(key=lambda r: r["sku"])
    totals = {}
    for r in rows:
        totals[r["sku"]] = totals.get(r["sku"], 0) + r["qty"] * r["price_paise"]
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]

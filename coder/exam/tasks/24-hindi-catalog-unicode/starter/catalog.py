"""Product catalog helpers for the billing counter.

The catalog is a CSV file with the columns sku,name,price_paise. Names can be in English or Hindi.
"""

import csv

FIELDS = ["sku", "name", "price_paise"]


def load_catalog(path):
    """Read the catalog CSV and return a list of dicts (price_paise as int)."""
    items = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            items.append({"sku": row["sku"], "name": row["name"], "price_paise": int(row["price_paise"])})
    return items


def save_catalog(path, items):
    """Write the catalog CSV (same columns as load_catalog reads)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow({k: item[k] for k in FIELDS})


def search(items, query):
    """Products whose name contains `query`, ignoring case."""
    q = query.strip().lower()
    return [item for item in items if q in item["name"].lower()]


def receipt_label(name, max_bytes=24):
    """The product name as printed on the thermal printer, which takes at most `max_bytes` UTF-8 bytes."""
    data = name.encode("utf-8")
    if len(data) <= max_bytes:
        return name
    return data[:max_bytes].decode("utf-8")

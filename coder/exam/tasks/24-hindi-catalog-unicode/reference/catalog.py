"""Product catalog helpers for the billing counter.

The catalog is a CSV file with the columns sku,name,price_paise. Names can be in English or Hindi.
Files are always UTF-8 (a BOM written by Excel is accepted), whatever the computer's locale.
"""

import csv
import unicodedata

FIELDS = ["sku", "name", "price_paise"]


def load_catalog(path):
    """Read the catalog CSV and return a list of dicts (price_paise as int)."""
    items = []
    # utf-8-sig also strips the byte-order mark that Excel puts at the start of "CSV UTF-8" files
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            items.append({"sku": row["sku"], "name": row["name"], "price_paise": int(row["price_paise"])})
    return items


def save_catalog(path, items):
    """Write the catalog CSV (same columns as load_catalog reads)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow({k: item[k] for k in FIELDS})


def _fold(text):
    # The same letter can be typed as different code points (e.g. ज़ as U+095B or as ज + nukta U+093C);
    # normalizing both sides to one form makes them compare equal.
    return unicodedata.normalize("NFD", text).casefold()


def search(items, query):
    """Products whose name contains `query`, ignoring case and Unicode spelling differences."""
    q = _fold(query.strip())
    return [item for item in items if q in _fold(item["name"])]


def receipt_label(name, max_bytes=24):
    """The product name as printed on the thermal printer, which takes at most `max_bytes` UTF-8 bytes.

    Cuts on character boundaries and never separates a vowel sign (matra) or other combining
    mark from the letter it belongs to.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must not be negative")
    if len(name.encode("utf-8")) <= max_bytes:
        return name
    end = 0
    used = 0
    for i, ch in enumerate(name):
        used += len(ch.encode("utf-8"))
        if used > max_bytes:
            break
        end = i + 1
    while 0 < end < len(name) and unicodedata.category(name[end]).startswith("M"):
        end -= 1
    return name[:end]

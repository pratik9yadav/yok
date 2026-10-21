"""Import orders from a spreadsheet CSV export, collecting bad rows instead of failing."""

from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from os import PathLike

REQUIRED = ("order_id", "customer", "sku", "qty", "price")


def _parse_row(row: dict[str, str], columns: dict[str, str]) -> dict:
    values = {}
    for name in REQUIRED:
        raw = row.get(columns[name])
        if raw is None:
            raise ValueError(f"missing value for {name}")
        raw = raw.strip()
        if not raw:
            raise ValueError(f"empty {name}")
        values[name] = raw
    try:
        qty = int(values["qty"])
    except ValueError:
        raise ValueError(f"qty is not a whole number: {values['qty']!r}") from None
    if qty <= 0:
        raise ValueError(f"qty must be positive: {qty}")
    try:
        price = Decimal(values["price"])
    except InvalidOperation:
        raise ValueError(f"price is not a number: {values['price']!r}") from None
    if not price.is_finite() or price < 0:
        raise ValueError(f"bad price: {values['price']!r}")
    values["qty"] = qty
    values["price"] = price
    return values


def import_orders(path: str | PathLike) -> tuple[list[dict], list[tuple[int, str]]]:
    """Read orders from the CSV at path. Returns (orders, errors)."""
    orders: list[dict] = []
    errors: list[tuple[int, str]] = []
    seen: set[str] = set()
    # utf-8-sig strips the byte-order mark that Excel puts at the start of CSV exports.
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError("file is empty: no header row")
        columns: dict[str, str] = {}
        normalised = [h.strip().lower() for h in header]
        for name in REQUIRED:
            if name not in normalised:
                raise ValueError(f"missing required column {name!r}")
            columns[name] = name
        for row_values in reader:
            line = reader.line_num
            if not any(v.strip() for v in row_values):
                continue
            if len(row_values) != len(normalised):
                errors.append((line, f"expected {len(normalised)} fields, got {len(row_values)}"))
                continue
            row = {}
            for key, value in zip(normalised, row_values, strict=True):
                row.setdefault(key, value)
            try:
                order = _parse_row(row, columns)
            except ValueError as exc:
                errors.append((line, str(exc)))
                continue
            if order["order_id"] in seen:
                errors.append((line, f"duplicate order_id {order['order_id']!r}"))
                continue
            seen.add(order["order_id"])
            orders.append(order)
    return orders, errors

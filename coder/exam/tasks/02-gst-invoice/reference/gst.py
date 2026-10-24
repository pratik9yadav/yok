"""Invoice totals with Indian GST, using exact decimal arithmetic."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from numbers import Integral

GST_RATES = frozenset({0, 5, 12, 18, 28})
PAISA = Decimal("0.01")
RUPEE = Decimal(1)


def _money(value: Decimal) -> Decimal:
    return value.quantize(PAISA, rounding=ROUND_HALF_UP)


def _price(raw: object) -> Decimal:
    """The price as a Decimal. Floats are refused: they cannot hold paise exactly."""
    price = None
    if isinstance(raw, (str, int, Decimal)) and not isinstance(raw, bool):
        try:
            price = Decimal(raw.strip() if isinstance(raw, str) else raw)
        except InvalidOperation:
            price = None
    if price is None or not price.is_finite() or price < 0:
        raise ValueError(f"bad price {raw!r} (give a string, int or Decimal of at least 0)")
    return price


def _qty(raw: object) -> int:
    if isinstance(raw, bool) or not isinstance(raw, Integral) or raw <= 0:
        raise ValueError(f"bad quantity {raw!r}")
    return int(raw)


def _rate(raw: object) -> int:
    if isinstance(raw, bool) or raw not in GST_RATES:
        raise ValueError(f"unknown GST rate {raw!r}")
    return int(raw)


def invoice_totals(lines: list[dict]) -> dict[str, Decimal]:
    """Return subtotal, gst, total and payable (nearest rupee) for the invoice lines."""
    subtotal = Decimal(0)
    gst = Decimal(0)
    for line in lines:
        price = _price(line["price"])
        qty = _qty(line["qty"])
        rate = _rate(line["gst_rate"])
        amount = _money(price * qty)
        subtotal += amount
        gst += _money(amount * rate / 100)
    subtotal = _money(subtotal)
    gst = _money(gst)
    total = subtotal + gst
    payable = total.quantize(RUPEE, rounding=ROUND_HALF_UP)
    return {"subtotal": subtotal, "gst": gst, "total": total, "payable": payable}

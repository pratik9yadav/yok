"""Counter invoices: the numbers (`invoice_totals`) and the printed bill (`make_invoice`).

order = {"invoice_no": str, "date": "2024-11-05", "customer": str (optional),
         "items": [{"name": str, "qty": int, "price_paise": int}, ...], "discount_pct": number (optional)}

Both functions use the same calculation, so the app and the printed bill always agree.
The printed layout is byte-for-byte the historical one.
"""

SHOP_NAME = "SHARMA KIRANA STORE"
FOOTER = "Thank you! Visit again."
WIDTH = 45
NAME_WIDTH = 20
LABEL_WIDTH = 35
MONEY_WIDTH = 10


def _rupees(paise):
    # historical formatting (float division), kept so the printed bill does not change
    return "%.2f" % (paise / 100)


def _line_amount(item):
    return item["qty"] * item["price_paise"]


def invoice_totals(order):
    """Subtotal, discount, round-off and total in paise, exactly as printed on the bill."""
    items = order["items"]
    if not items:
        raise ValueError("no items")
    subtotal = sum(_line_amount(item) for item in items)
    pct = order.get("discount_pct") or 0
    # truncation and float maths kept from the original bill
    discount = int(subtotal * pct / 100) if pct else 0
    net = subtotal - discount
    round_off = round(net / 100) * 100 - net  # nearest rupee (Python's round: halves to even)
    return {
        "subtotal_paise": subtotal,
        "discount_paise": discount,
        "round_off_paise": round_off,
        "total_paise": net + round_off,
    }


def _item_line(item):
    name = item["name"]
    if len(name) > NAME_WIDTH:
        name = name[: NAME_WIDTH - 1] + "~"
    return (
        name.ljust(NAME_WIDTH)
        + str(item["qty"]).rjust(4)
        + " x "
        + _rupees(item["price_paise"]).rjust(8)
        + _rupees(_line_amount(item)).rjust(MONEY_WIDTH)
    )


def _total_line(label, amount_text):
    return label.ljust(LABEL_WIDTH) + amount_text.rjust(MONEY_WIDTH)


def make_invoice(order):
    """The printed bill as text."""
    totals = invoice_totals(order)
    rule = "-" * WIDTH
    lines = [
        SHOP_NAME,
        "Invoice: " + order["invoice_no"] + "   Date: " + order["date"],
        "Customer: " + (order.get("customer") or "Cash"),
        rule,
    ]
    lines += [_item_line(item) for item in order["items"]]
    lines.append(rule)
    lines.append(_total_line("Subtotal", _rupees(totals["subtotal_paise"])))
    if order.get("discount_pct"):
        label = "Discount (" + str(order["discount_pct"]) + "%)"
        lines.append(_total_line(label, "-" + _rupees(totals["discount_paise"])))
    if totals["round_off_paise"] != 0:
        lines.append(_total_line("Round off", "%+.2f" % (totals["round_off_paise"] / 100)))
    lines.append(_total_line("TOTAL", _rupees(totals["total_paise"])))
    lines.append(rule)
    lines.append(FOOTER)
    return "\n".join(lines) + "\n"

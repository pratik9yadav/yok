import copy
import csv
import io
import random
from decimal import Decimal as D

import pytest

from sales_report import build_report

SALES = [
    {"order_id": "A1", "date": "2026-10-02", "sku": "ATTA-5KG", "qty": 2, "amount": D("490.00")},
    {"order_id": "A1", "date": "2026-10-02", "sku": "SALT-1KG", "qty": 1, "amount": D("28.00")},
    {"order_id": "A2", "date": "2026-10-01", "sku": "OIL-1L", "qty": 3, "amount": D("540.50")},
    {"order_id": "A3", "date": "2026-10-02", "sku": "SALT-1KG", "qty": 4, "amount": D("112")},
    {"order_id": "A4", "date": "2026-10-03", "sku": "DAL-1KG", "qty": 2, "amount": D("240.25")},
    {"order_id": "A4", "date": "2026-10-04", "sku": "ATTA-5KG", "qty": 1, "amount": D("245.00")},
    {"order_id": "A5", "date": "2026-10-03", "sku": "OIL-1L", "qty": 1, "amount": D("180.5")},
]

TEXT = (
    "Sales report\n"
    "============\n"
    "2026-10-01  orders: 1  items: 3  amount: 540.50\n"
    "2026-10-02  orders: 2  items: 7  amount: 630.00\n"
    "2026-10-03  orders: 2  items: 3  amount: 420.75\n"
    "2026-10-04  orders: 1  items: 1  amount: 245.00\n"
    "Total  orders: 5  items: 14  amount: 1836.25\n"
    "\n"
    "Top products:\n"
    "  1. SALT-1KG  5 items  140.00\n"
    "  2. OIL-1L  4 items  721.00\n"
    "  3. ATTA-5KG  3 items  735.00\n"
)

TIES = [
    {"order_id": "B1", "date": "2026-11-01", "sku": "TEA-250G", "qty": 2, "amount": D("150")},
    {"order_id": "B2", "date": "2026-11-01", "sku": "COFFEE-100G", "qty": 2, "amount": D("199.99")},
    {"order_id": "B3", "date": "2026-11-01", "sku": "SUGAR-1KG", "qty": 2, "amount": D("45")},
    {"order_id": "B3", "date": "2026-11-01", "sku": "BISCUIT", "qty": 2, "amount": D("20")},
    {"order_id": "B4", "date": "2026-11-01", "sku": "ATTA-5KG", "qty": 1, "amount": D("245")},
]

TIES_TEXT = (
    "Sales report\n"
    "============\n"
    "2026-11-01  orders: 4  items: 9  amount: 659.99\n"
    "Total  orders: 4  items: 9  amount: 659.99\n"
    "\n"
    "Top products:\n"
    "  1. BISCUIT  2 items  20.00\n"
    "  2. COFFEE-100G  2 items  199.99\n"
    "  3. SUGAR-1KG  2 items  45.00\n"
)


def parse_csv(text):
    return list(csv.reader(io.StringIO(text)))


def test_text_report_unchanged():
    assert build_report(SALES) == TEXT
    assert build_report(SALES, fmt="text") == TEXT
    assert build_report(SALES, "text") == TEXT


def test_text_report_ties_unchanged():
    assert build_report(TIES) == TIES_TEXT


def test_text_report_empty_unchanged():
    assert build_report([]) == "Sales report\n============\nNo sales.\n"


def test_text_report_single_sale():
    one = [{"order_id": "Z", "date": "2026-10-09", "sku": "PEN", "qty": 1, "amount": D("10")}]
    assert build_report(one) == (
        "Sales report\n============\n"
        "2026-10-09  orders: 1  items: 1  amount: 10.00\n"
        "Total  orders: 1  items: 1  amount: 10.00\n"
        "\nTop products:\n  1. PEN  1 items  10.00\n"
    )


def test_order_of_input_does_not_matter():
    shuffled = list(SALES)
    random.Random(7).shuffle(shuffled)
    assert build_report(shuffled) == TEXT
    assert build_report(shuffled, fmt="csv") == build_report(SALES, fmt="csv")


def test_csv_report():
    rows = parse_csv(build_report(SALES, fmt="csv"))
    assert rows == [
        ["date", "orders", "items", "amount"],
        ["2026-10-01", "1", "3", "540.50"],
        ["2026-10-02", "2", "7", "630.00"],
        ["2026-10-03", "2", "3", "420.75"],
        ["2026-10-04", "1", "1", "245.00"],
    ]


def test_csv_report_is_a_string():
    out = build_report(SALES, fmt="csv")
    assert isinstance(out, str)
    assert out.splitlines()[0] == "date,orders,items,amount"


def test_csv_report_empty():
    rows = parse_csv(build_report([], fmt="csv"))
    assert rows in ([], [["date", "orders", "items", "amount"]])


@pytest.mark.parametrize("fmt", ["pdf", "CSV ", "", None, "json"])
def test_unknown_format_raises(fmt):
    with pytest.raises(ValueError):
        build_report(SALES, fmt=fmt)


def test_input_not_modified_and_repeatable():
    sales = copy.deepcopy(SALES)
    first = build_report(sales)
    second = build_report(sales)
    build_report(sales, fmt="csv")
    assert first == second == TEXT
    assert sales == SALES

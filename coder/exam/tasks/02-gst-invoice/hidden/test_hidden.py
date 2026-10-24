import copy
import time
from decimal import Decimal

import pytest

from gst import invoice_totals


def D(s):
    return Decimal(s)


def test_single_line():
    r = invoice_totals([{"price": "100", "qty": 2, "gst_rate": 18}])
    assert r == {"subtotal": D("200"), "gst": D("36"), "total": D("236"), "payable": D("236")}
    assert all(isinstance(v, Decimal) for v in r.values())


def test_mixed_input_types():
    r = invoice_totals([
        {"price": "199.50", "qty": 1, "gst_rate": 5},
        {"price": 40, "qty": 3, "gst_rate": 0},
        {"price": D("12.25"), "qty": 4, "gst_rate": 12},
    ])
    # 199.50 + 120 + 49.00 ; gst 9.975->9.98 + 0 + 5.88
    assert r["subtotal"] == D("368.50")
    assert r["gst"] == D("15.86")
    assert r["total"] == D("384.36")
    assert r["payable"] == D("384")


def test_gst_rounds_half_up_to_paisa():
    # 0.25 * 18% = 0.045 -> 0.05 (half up); binary floats give 0.04
    r = invoice_totals([{"price": "0.25", "qty": 1, "gst_rate": 18}])
    assert r["gst"] == D("0.05")
    r = invoice_totals([{"price": "1.15", "qty": 1, "gst_rate": 5}])
    # 0.0575 -> 0.06
    assert r["gst"] == D("0.06")


def test_gst_is_rounded_per_line():
    line = {"price": "0.25", "qty": 1, "gst_rate": 18}
    r = invoice_totals([line, dict(line)])
    assert r["gst"] == D("0.10")  # 0.05 + 0.05, not round(0.09)
    assert r["total"] == D("0.60")


def test_no_float_drift():
    r = invoice_totals([{"price": "0.1", "qty": 3, "gst_rate": 0}, {"price": "0.2", "qty": 1, "gst_rate": 0}])
    assert r["subtotal"] == D("0.50")
    assert str(r["total"]) == "0.50"


@pytest.mark.parametrize("price, payable", [("100.50", "101"), ("100.49", "100"), ("102.50", "103"), ("99.99", "100")])
def test_payable_rounds_half_up_to_rupee(price, payable):
    r = invoice_totals([{"price": price, "qty": 1, "gst_rate": 0}])
    assert r["payable"] == D(payable)


def test_money_values_have_paise():
    r = invoice_totals([{"price": "10", "qty": 1, "gst_rate": 5}])
    assert r["total"] == D("10.50")
    assert r["total"].as_tuple().exponent == -2
    assert r["gst"].as_tuple().exponent == -2


def test_empty_invoice_is_zero():
    r = invoice_totals([])
    assert r == {"subtotal": 0, "gst": 0, "total": 0, "payable": 0}


@pytest.mark.parametrize("rate", [15, -5, 100, "eighteen", None])
def test_unknown_rate_raises(rate):
    with pytest.raises(ValueError):
        invoice_totals([{"price": "10", "qty": 1, "gst_rate": rate}])


@pytest.mark.parametrize("qty", [0, -1, 1.5, "abc"])
def test_bad_quantity_raises(qty):
    with pytest.raises(ValueError):
        invoice_totals([{"price": "10", "qty": qty, "gst_rate": 5}])


@pytest.mark.parametrize("price", ["abc", "", "-5", "NaN", None])
def test_bad_price_raises(price):
    with pytest.raises(ValueError):
        invoice_totals([{"price": price, "qty": 1, "gst_rate": 5}])


def test_input_is_not_modified():
    lines = [{"price": "99.99", "qty": 2, "gst_rate": 12}, {"price": 5, "qty": 1, "gst_rate": 0}]
    before = copy.deepcopy(lines)
    invoice_totals(lines)
    assert lines == before


def test_same_input_same_output():
    lines = [{"price": "33.33", "qty": 3, "gst_rate": 28}]
    assert invoice_totals(lines) == invoice_totals(lines)


def test_large_invoice_is_fast_and_exact():
    lines = [{"price": "0.10", "qty": 1, "gst_rate": 0}] * 100_000
    start = time.perf_counter()
    r = invoice_totals(lines)
    assert time.perf_counter() - start < 3.0
    assert r["subtotal"] == D("10000.00")

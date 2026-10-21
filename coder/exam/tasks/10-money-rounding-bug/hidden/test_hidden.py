import time
from decimal import Decimal

import pytest

from billing import apply_discount, cart_total, split_bill


def D(s):
    return Decimal(s)


def test_cart_total_is_exact():
    assert cart_total([D("0.10"), D("0.20")]) == D("0.30")
    assert str(cart_total([D("0.10"), D("0.20")])) in ("0.30", "0.3")


def test_cart_total_empty():
    assert cart_total([]) == 0
    assert isinstance(cart_total([]), Decimal)


def test_cart_total_accepts_generator():
    assert cart_total(D("1.10") for _ in range(3)) == D("3.30")


def test_cart_total_many_items_exact_and_fast():
    prices = [D("0.10")] * 100_000
    start = time.perf_counter()
    assert cart_total(prices) == D("10000.00")
    assert time.perf_counter() - start < 2.0


def test_cart_total_large_amounts_keep_paise():
    assert cart_total([D("9999999999999.99"), D("0.01")]) == D("10000000000000.00")


@pytest.mark.parametrize("price, percent, expected", [
    ("2.25", 10, "2.03"),     # 2.025 -> half up
    ("100.00", 0, "100.00"),
    ("100.00", 100, "0.00"),
    ("199.99", 15, "169.99"),  # 169.9915
    ("1.05", 50, "0.53"),      # 0.525 -> half up
    ("80.00", D("12.5"), "70.00"),
])
def test_discount_rounds_half_up(price, percent, expected):
    result = apply_discount(D(price), percent)
    assert isinstance(result, Decimal)
    assert result == D(expected)


@pytest.mark.parametrize("percent", [-1, 101, 150])
def test_discount_out_of_range(percent):
    with pytest.raises(ValueError):
        apply_discount(D("10"), percent)


def test_split_reported_case():
    assert split_bill(D("100"), 3) == [D("33.34"), D("33.33"), D("33.33")]


def test_split_always_adds_up():
    for total in ["0.01", "0.05", "1.00", "10.01", "99.99", "100.00", "12345.67"]:
        for people in range(1, 12):
            shares = split_bill(D(total), people)
            assert len(shares) == people
            assert sum(shares) == D(total), (total, people)
            assert max(shares) - min(shares) <= D("0.01")
            assert shares == sorted(shares, reverse=True)  # extra paise go to the first people
            assert all(isinstance(s, Decimal) for s in shares)


def test_split_less_than_one_paisa_each():
    assert split_bill(D("0.02"), 3) == [D("0.01"), D("0.01"), D("0.00")]


def test_split_one_person():
    assert split_bill(D("57.50"), 1) == [D("57.50")]


@pytest.mark.parametrize("people", [0, -2])
def test_split_bad_people(people):
    with pytest.raises(ValueError):
        split_bill(D("100"), people)


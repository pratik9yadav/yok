from bench.prices import apply_discount
import datetime

from bench.orders import calc_total
from bench.prices import parse_price
from bench.report import order_lines


def test_parse_plain_price() -> None:
    assert parse_price("1250") == 1250.0


def test_total() -> None:
    assert calc_total({"item": "MCB", "qty": 2, "price": 220.0, "day": datetime.date(2026, 10, 12)}) == 440.0


def test_lines() -> None:
    order = {"item": "MCB", "qty": 2, "price": 220.0, "day": datetime.date(2026, 10, 12)}
    assert order_lines([order]) == ["MCB: 2 x 220.0 = 440.0"]

def test_apply_discount() -> None:
    assert apply_discount(1000, 10) == 900.0
    assert apply_discount(500, 20) == 400.0
    assert apply_discount(100, 0) == 100.0
    assert apply_discount(100, 100) == 0.0

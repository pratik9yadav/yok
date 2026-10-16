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

from bench.orders import in_stock_count
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

def test_in_stock_count_with_zero() -> None:
    # Test that items with 0 quantity are not counted as in stock
    stock = {'item1': 3, 'item2': 0, 'item3': 5}
    assert in_stock_count(stock) == 2
    # Test edge case: all zero quantities
    stock_all_zero = {'item1': 0, 'item2': 0}
    assert in_stock_count(stock_all_zero) == 0
    # Test edge case: no items
    assert in_stock_count({}) == 0



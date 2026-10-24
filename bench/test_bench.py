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

def test_weekly_totals_bug_fix() -> None:
    """Test that weekly totals correctly assign sales to the right days."""
    import datetime
    from bench.orders import weekly_totals
    
    # Create orders for Monday and Sunday to test the bug fix
    orders = [
        {'item': 'A', 'qty': 1, 'price': 100.0, 'day': datetime.date(2023, 10, 2)},  # Monday
        {'item': 'B', 'qty': 1, 'price': 200.0, 'day': datetime.date(2023, 10, 1)},  # Sunday
    ]
    
    result = weekly_totals(orders)
    
    # Before the fix, Monday sales would be under Tuesday and Sunday under Monday
    # After the fix, they should be under their correct days
    assert result['Monday'] == 100.0
    assert result['Tuesday'] == 0.0  # No Tuesday sales
    assert result['Sunday'] == 200.0
    assert result['Monday'] != result['Tuesday']  # Monday sales should not be under Tuesday
    assert result['Sunday'] != result['Monday']  # Sunday sales should not be under Monday

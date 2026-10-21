import random
import time
from datetime import date, timedelta

from dispatch import Order, dispatch_queue

D1 = date(2024, 11, 1)
D2 = date(2024, 11, 2)
D3 = date(2024, 11, 3)


def ids(orders):
    return [o.order_id for o in orders]


def test_express_first_then_by_date():
    orders = [
        Order("n-d3", False, D3),
        Order("e-d2", True, D2),
        Order("n-d1", False, D1),
        Order("e-d1", True, D1),
    ]
    assert ids(dispatch_queue(orders)) == ["e-d1", "e-d2", "n-d1", "n-d3"]


def test_same_day_is_first_come_first_served():
    orders = [Order(f"o{i:02d}", i % 3 == 0, D1) for i in range(20)]
    result = ids(dispatch_queue(orders))
    express = [f"o{i:02d}" for i in range(20) if i % 3 == 0]
    normal = [f"o{i:02d}" for i in range(20) if i % 3 != 0]
    assert result == express + normal


def test_no_promised_date_goes_last_in_its_group_in_placed_order():
    orders = [
        Order("pickup-1", False, None),
        Order("n-d2", False, D2),
        Order("e-pickup", True, None),
        Order("pickup-2", False, None),
        Order("e-d3", True, D3),
        Order("n-d1", False, D1),
    ]
    assert ids(dispatch_queue(orders)) == ["e-d3", "e-pickup", "n-d1", "n-d2", "pickup-1", "pickup-2"]


def test_does_not_reorder_the_callers_list():
    orders = [Order("a", False, D3), Order("b", True, D1), Order("c", False, D1)]
    original = list(orders)
    result = dispatch_queue(orders)
    assert orders == original
    assert [o.order_id for o in orders] == ["a", "b", "c"]
    assert ids(result) == ["b", "c", "a"]
    result.append(Order("x", False, None))
    assert len(orders) == 3


def test_empty_and_single():
    assert list(dispatch_queue([])) == []
    one = [Order("only", False, None)]
    assert ids(dispatch_queue(one)) == ["only"]


def test_matches_stable_reference_on_random_data():
    rng = random.Random(42)
    days = [D1 + timedelta(days=k) for k in range(4)] + [None]
    for _ in range(50):
        orders = [Order(f"o{i}", rng.random() < 0.3, rng.choice(days)) for i in range(60)]
        expected = sorted(
            orders, key=lambda o: (not o.express, o.promised_by is None, o.promised_by or date.min)
        )
        assert ids(dispatch_queue(orders)) == ids(expected)


def test_big_queue_is_fast():
    rng = random.Random(1)
    days = [D1 + timedelta(days=k) for k in range(30)] + [None]
    orders = [Order(f"o{i}", rng.random() < 0.2, rng.choice(days)) for i in range(100_000)]
    start = time.perf_counter()
    result = dispatch_queue(orders)
    assert time.perf_counter() - start < 2
    assert len(result) == 100_000


def test_running_twice_gives_the_same_queue():
    orders = [Order(f"o{i}", i % 4 == 0, [D2, D1, None][i % 3]) for i in range(30)]
    first = dispatch_queue(orders)
    assert ids(dispatch_queue(orders)) == ids(first)
    assert ids(dispatch_queue(list(first))) == ids(first)


def test_express_pickup_still_before_normal_dated_orders():
    orders = [Order("n-d1", False, D1), Order("e-none", True, None)]
    assert ids(dispatch_queue(orders)) == ["e-none", "n-d1"]

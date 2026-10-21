import random
import time
from fractions import Fraction

import pytest

from splitpay import settle, split_bill


def expected_split(total, weights):
    s = sum(weights)
    base = [total * w // s for w in weights]
    rems = sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % s), i))
    for i in rems[: total - sum(base)]:
        base[i] += 1
    return base


@pytest.mark.parametrize(
    "total,weights,expected",
    [
        (100, [1, 1, 1], [34, 33, 33]),
        (1001, [1, 1, 2], [250, 250, 501]),
        (10, [1] * 7, [2, 2, 2, 1, 1, 1, 1]),
        (7, [3, 0, 1], [5, 0, 2]),
        (0, [1, 2, 3], [0, 0, 0]),
        (99999, [5], [99999]),
        (5, [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0, 0]),
    ],
)
def test_split_known_cases(total, weights, expected):
    assert split_bill(total, weights) == expected


def test_split_huge_amount_is_exact():
    total = 10**18 + 1
    parts = split_bill(total, [1, 2])
    assert parts == [333333333333333334, 666666666666666667]
    assert sum(parts) == total


def test_split_random_properties():
    rng = random.Random(2024)
    for _ in range(300):
        n = rng.randint(1, 12)
        weights = [rng.randint(0, 9) for _ in range(n)]
        if sum(weights) == 0:
            weights[0] = 1
        total = rng.randint(0, 5_000_000)
        parts = split_bill(total, weights)
        assert all(isinstance(p, int) for p in parts)
        assert sum(parts) == total
        for p, w in zip(parts, weights):
            assert abs(p - Fraction(total * w, sum(weights))) < 1
        assert parts == expected_split(total, weights)


def test_split_does_not_modify_weights():
    weights = [1, 2, 3]
    split_bill(100, weights)
    assert weights == [1, 2, 3]


@pytest.mark.parametrize("weights", [[], [0, 0], [1, -1, 2]])
def test_split_bad_weights(weights):
    with pytest.raises(ValueError):
        split_bill(100, weights)


@pytest.mark.parametrize("total", [100.5, 10.0, "100"])
def test_split_rejects_non_integer_total(total):
    with pytest.raises((TypeError, ValueError)):
        split_bill(total, [1, 1])


def test_split_is_fast_for_many_people():
    start = time.perf_counter()
    parts = split_bill(123_456_789, [1 + i % 5 for i in range(200_000)])
    assert sum(parts) == 123_456_789
    assert time.perf_counter() - start < 2


def apply(balances, payments):
    left = dict(balances)
    for payer, payee, amount in payments:
        left[payer] += amount
        left[payee] -= amount
    return left


def test_settle_simple():
    balances = {"Amit": 500, "Bunty": -300, "Chintu": -200}
    payments = settle(balances)
    assert sorted(payments) == [("Bunty", "Amit", 300), ("Chintu", "Amit", 200)]


def test_settle_all_even():
    assert list(settle({"a": 0, "b": 0})) == []
    assert list(settle({})) == []


def test_settle_unbalanced_raises():
    with pytest.raises(ValueError):
        settle({"a": 100, "b": -99})


def test_settle_rejects_float_money():
    with pytest.raises((TypeError, ValueError)):
        settle({"a": 10.5, "b": -10.5})


def test_settle_random_properties():
    rng = random.Random(7)
    for _ in range(200):
        names = [f"p{i}" for i in range(rng.randint(2, 10))]
        balances = {n: rng.randint(-50_000, 50_000) for n in names[:-1]}
        balances[names[-1]] = -sum(balances.values())
        before = dict(balances)
        payments = [tuple(p) for p in settle(balances)]
        assert balances == before  # input untouched
        assert all(v == 0 for v in apply(balances, payments).values())
        for payer, payee, amount in payments:
            assert isinstance(amount, int) and amount > 0
            assert payer != payee
            assert balances[payer] < 0 < balances[payee]
        nonzero = sum(1 for v in balances.values() if v)
        assert len(payments) <= max(nonzero - 1, 0)


def test_settle_same_balances_same_sheet_regardless_of_order():
    items = [("Amit", 700), ("Bunty", -300), ("Chintu", -300), ("Dolly", 200), ("Eshan", -300)]
    first = [tuple(p) for p in settle(dict(items))]
    for seed in range(10):
        shuffled = items[:]
        random.Random(seed).shuffle(shuffled)
        assert [tuple(p) for p in settle(dict(shuffled))] == first

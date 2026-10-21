import copy
import math
from decimal import Decimal

import pytest

from shipping import shipping_fee


def legacy_fee(order):
    """The fee rules exactly as they were before the clean-up (test oracle)."""
    rates = {"local": (30, 10, 8, 499), "state": (50, 15, 12, 999), "national": (80, 25, 20, 1499)}
    zone = order["zone"].lower()
    if zone not in rates:
        raise ValueError(zone)
    w = order["weight_kg"]
    if w <= 0 or w > 30:
        raise ValueError(w)
    first, to5, above5, free_from = rates[zone]
    if order["subtotal"] >= free_from:
        fee = 0
    elif w <= 1:
        fee = first
    elif w <= 5:
        fee = first + math.ceil(w - 1) * to5
    else:
        fee = first + 4 * to5 + math.ceil(w - 5) * above5
    if order.get("cod"):
        fee += max(25, math.ceil(order["subtotal"] * 2 / 100))
    return fee


WEIGHTS = [0.2, 1, 1.01, 2.5, 4.999, 5, 5.0001, 5.5, 12, 29.9, 30]
SUBTOTALS = [0, 100, 498.99, 499, 998, 999, 1250, 1498.5, 1499, 1998, 1999, 2500, 10000]


@pytest.mark.parametrize("zone", ["local", "state", "national"])
def test_existing_zones_unchanged(zone):
    for w in WEIGHTS:
        for s in SUBTOTALS:
            for cod in (False, True):
                order = {"zone": zone, "weight_kg": w, "subtotal": s, "cod": cod}
                assert shipping_fee(order) == legacy_fee(order), order


@pytest.mark.parametrize("order, fee", [
    ({"zone": "local", "weight_kg": 0.5, "subtotal": 200}, 30),
    ({"zone": "local", "weight_kg": 3, "subtotal": 200}, 50),
    ({"zone": "state", "weight_kg": 7.2, "subtotal": 500}, 146),
    ({"zone": "national", "weight_kg": 5, "subtotal": 1000, "cod": True}, 205),
    ({"zone": "national", "weight_kg": 5, "subtotal": 1499, "cod": True}, 30),
])
def test_known_fees(order, fee):
    assert shipping_fee(order) == fee


def test_cod_defaults_to_false_and_zone_case():
    assert shipping_fee({"zone": "Local", "weight_kg": 1, "subtotal": 100}) == 30
    assert shipping_fee({"zone": "STATE", "weight_kg": 1, "subtotal": 100, "cod": False}) == 50


def test_decimal_subtotal():
    assert shipping_fee({"zone": "local", "weight_kg": 2, "subtotal": Decimal("498.99"), "cod": True}) == 40 + 25
    assert shipping_fee({"zone": "local", "weight_kg": 2, "subtotal": Decimal("499.00")}) == 0
    assert shipping_fee({"zone": "state", "weight_kg": 2, "subtotal": Decimal("1300.50"), "cod": True}) == 27


def test_fee_is_an_int():
    for zone in ("local", "state", "national", "northeast"):
        fee = shipping_fee({"zone": zone, "weight_kg": 2.5, "subtotal": 300.0, "cod": True})
        assert isinstance(fee, int) and not isinstance(fee, bool)


@pytest.mark.parametrize("zone", ["local", "state", "national", "northeast"])
@pytest.mark.parametrize("weight", [0, -1, 30.01, 45])
def test_bad_weight_raises(zone, weight):
    with pytest.raises(ValueError):
        shipping_fee({"zone": zone, "weight_kg": weight, "subtotal": 100})


@pytest.mark.parametrize("zone", ["mars", "", "north east"])
def test_unknown_zone_raises(zone):
    with pytest.raises(ValueError):
        shipping_fee({"zone": zone, "weight_kg": 1, "subtotal": 100})


@pytest.mark.parametrize(
    "weight, fee", [(0.5, 120), (1, 120), (1.2, 155), (3, 190), (5, 260), (5.1, 290), (7.2, 350), (30, 1010)]
)
def test_northeast_rates(weight, fee):
    assert shipping_fee({"zone": "northeast", "weight_kg": weight, "subtotal": 500}) == fee


def test_northeast_free_shipping_and_cod():
    assert shipping_fee({"zone": "northeast", "weight_kg": 3, "subtotal": 1998.99}) == 190
    assert shipping_fee({"zone": "northeast", "weight_kg": 3, "subtotal": 1999}) == 0
    assert shipping_fee({"zone": "northeast", "weight_kg": 3, "subtotal": 1999, "cod": True}) == 40
    assert shipping_fee({"zone": "Northeast", "weight_kg": 3, "subtotal": 100, "cod": True}) == 215


def test_order_is_not_modified():
    order = {"zone": "northeast", "weight_kg": 3, "subtotal": 100, "cod": True}
    before = copy.deepcopy(order)
    shipping_fee(order)
    shipping_fee(order)
    assert order == before

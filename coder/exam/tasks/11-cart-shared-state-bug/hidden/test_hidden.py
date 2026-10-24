import copy
from decimal import Decimal

import pytest

from cart import Cart


def D(s):
    return Decimal(s)


def test_new_carts_start_empty():
    a = Cart("ravi")
    a.add("ATTA-5KG", 1, D("245"))
    b = Cart("meena")
    assert b.lines() == []
    assert b.total() == 0


def test_carts_are_independent():
    a = Cart("ravi")
    b = Cart("meena")
    a.add("SALT-1KG", 2, D("28"))
    b.add("OIL-1L", 1, D("180"))
    assert [line["sku"] for line in a.lines()] == ["SALT-1KG"]
    assert [line["sku"] for line in b.lines()] == ["OIL-1L"]


def test_starting_items_are_not_shared_with_caller():
    start = [{"sku": "DAL-1KG", "qty": 1, "price": D("120")}]
    before = copy.deepcopy(start)
    a = Cart("ravi", start)
    b = Cart("asha", start)
    a.add("SALT-1KG", 1, D("28"))
    a.add("DAL-1KG", 1, D("120"))
    assert start == before
    assert b.total() == D("120")
    assert a.total() == D("268")


def test_same_product_merges_into_one_line():
    c = Cart("ravi")
    c.add("ATTA-5KG", 1, D("245"))
    c.add("SALT-1KG", 1, D("28"))
    c.add("ATTA-5KG", 2, D("245"))
    lines = c.lines()
    assert [(line["sku"], line["qty"]) for line in lines] == [("ATTA-5KG", 3), ("SALT-1KG", 1)]
    assert c.total() == D("763")


def test_lines_cannot_change_the_cart():
    c = Cart("ravi")
    c.add("A", 1, D("10"))
    lines = c.lines()
    lines.append({"sku": "B", "qty": 1, "price": D("99")})
    assert c.total() == D("10")
    assert len(c.lines()) == 1


def test_remove_and_missing_remove():
    c = Cart("ravi")
    c.add("A", 2, D("10"))
    c.add("B", 1, D("5"))
    c.remove("A")
    assert [line["sku"] for line in c.lines()] == ["B"]
    with pytest.raises(KeyError):
        c.remove("A")


def test_add_after_remove_starts_fresh():
    c = Cart("ravi")
    c.add("A", 2, D("10"))
    c.remove("A")
    c.add("A", 1, D("10"))
    assert [(line["sku"], line["qty"]) for line in c.lines()] == [("A", 1)]


@pytest.mark.parametrize("qty", [0, -1])
def test_bad_qty_still_rejected(qty):
    c = Cart("ravi")
    c.add("A", 1, D("10"))
    with pytest.raises(ValueError):
        c.add("A", qty, D("10"))
    assert c.lines()[0]["qty"] == 1


def test_price_accepts_strings():
    c = Cart("ravi")
    c.add("A", 3, "19.99")
    assert c.total() == D("59.97")

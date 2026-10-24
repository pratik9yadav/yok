from decimal import Decimal

from cart import Cart


def test_total():
    c = Cart("ravi")
    c.add("ATTA-5KG", 2, Decimal("245"))
    assert c.total() == Decimal("490")

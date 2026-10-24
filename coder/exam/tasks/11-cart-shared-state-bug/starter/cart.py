"""Shopping cart for the shop app."""

from decimal import Decimal


class Cart:
    def __init__(self, customer, items=[]):
        """items: optional starting lines, each {"sku", "qty", "price"}."""
        self.customer = customer
        self.items = items

    def add(self, sku, qty=1, price=Decimal("0")):
        """Add qty units of sku at the given unit price. qty must be at least 1."""
        if qty < 1:
            raise ValueError("qty must be at least 1")
        self.items.append({"sku": sku, "qty": qty, "price": Decimal(price)})

    def remove(self, sku):
        """Remove the line for sku. Raises KeyError if it is not in the cart."""
        for line in self.items:
            if line["sku"] == sku:
                self.items.remove(line)
                return
        raise KeyError(sku)

    def lines(self):
        """The cart lines, in the order they were first added."""
        return self.items

    def total(self):
        return sum((line["price"] * line["qty"] for line in self.items), Decimal("0"))

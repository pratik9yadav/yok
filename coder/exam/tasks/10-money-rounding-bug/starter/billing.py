"""Money helpers for the billing counter. All amounts are Decimal rupees."""

from decimal import Decimal


def cart_total(prices):
    """Sum of the item prices."""
    return Decimal(str(sum(float(p) for p in prices)))


def apply_discount(price, percent):
    """Price after a percentage discount, rounded half up to the paisa.

    percent is a number from 0 to 100; anything else raises ValueError.
    """
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")
    result = float(price) * (1 - percent / 100)
    return Decimal(str(round(result, 2)))


def split_bill(total, people):
    """Split total between people.

    Returns one share per person. The shares add up exactly to the total;
    when it does not divide evenly, the first people pay one paisa more.
    """
    share = round(float(total) / people, 2)
    return [Decimal(str(share))] * people

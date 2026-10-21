"""Money helpers for the billing counter. All amounts are Decimal rupees."""

from decimal import ROUND_HALF_UP, Decimal

PAISA = Decimal("0.01")


def _to_decimal(value):
    if isinstance(value, float):
        # str() gives the shortest repr, e.g. 0.1 -> "0.1", not 0.1000000000000000055...
        return Decimal(str(value))
    return Decimal(value)


def cart_total(prices):
    """Sum of the item prices."""
    return sum((_to_decimal(p) for p in prices), Decimal(0))


def apply_discount(price, percent):
    """Price after a percentage discount, rounded half up to the paisa.

    percent is a number from 0 to 100; anything else raises ValueError.
    """
    percent = _to_decimal(percent)
    if not percent.is_finite() or percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")
    result = _to_decimal(price) * (100 - percent) / 100
    return result.quantize(PAISA, rounding=ROUND_HALF_UP)


def split_bill(total, people):
    """Split total between people.

    Returns one share per person. The shares add up exactly to the total;
    when it does not divide evenly, the first people pay one paisa more.
    """
    if isinstance(people, bool) or not isinstance(people, int) or people < 1:
        raise ValueError("people must be a whole number of at least 1")
    total_paise = int((_to_decimal(total) / PAISA).to_integral_value(rounding=ROUND_HALF_UP))
    base, extra = divmod(total_paise, people)
    return [Decimal(base + (1 if i < extra else 0)) * PAISA for i in range(people)]

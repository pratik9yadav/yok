"""Prices as shopkeepers and apps write them."""


def parse_price(text: str) -> float:
    """'₹1,250.50' / 'Rs. 1250' / '1250' -> 1250.5 / 1250.0 / 1250.0"""
    cleaned = text.replace("Rs.", "").replace("₹", "").replace(",", "").strip()
    return float(cleaned)


def apply_discount(price: float, percent: float) -> float:
    """The price after a percent discount, rounded to 2 decimals. percent must be 0-100."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * percent / 100, 2)

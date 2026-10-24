"""Prices as shopkeepers and apps write them."""


def parse_price(text: str) -> float:
    """'₹1,250.50' / 'Rs. 1250' / '1250' -> 1250.5 / 1250.0 / 1250.0"""
    cleaned = text.replace("Rs.", "").replace(",", "").strip()
    return float(cleaned)


def apply_discount(price: float, percent: float) -> float:
    """The price after a percent discount, rounded to 2 decimals. percent must be 0-100."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * percent / 100, 2)


def format_inr(amount: float) -> str:
    """Format amount as Indian Rupees with Indian thousands separator."""
    # Convert to string with 2 decimal places
    str_amount = f"{amount:.2f}"
    
    # Split into integer and fractional parts
    if '.' in str_amount:
        integer_part, decimal_part = str_amount.split('.')
    else:
        integer_part, decimal_part = str_amount, '00'
    
    # If it's less than 1000, no special formatting needed for integer part
    if len(integer_part) < 4:
        if decimal_part == '00':
            return f"₹{integer_part}"
        else:
            return f"₹{integer_part}.{decimal_part}"
    
    # For amounts >= 1000, apply Indian thousands separator
    # Pattern: 12,50,000 (first group of 3 digits from right, then every 2 digits)
    # So we split the integer part into groups from right to left
    # First 3 digits from right, then every 2 digits from right
    result = []
    i = len(integer_part)
    
    # Add last 3 digits first
    if i > 3:
        result.append(integer_part[max(0, i-3):i])
        i -= 3
    else:
        result.append(integer_part)
        i = 0
    
    # Add remaining groups of 2 digits from right to left
    while i > 0:
        result.append(integer_part[max(0, i-2):i])
        i -= 2
    
    # Reverse the order and join with commas
    formatted_integer = ','.join(reversed(result))
    
    # Combine with decimal part
    if decimal_part == '00':
        return f"₹{formatted_integer}"
    else:
        return f"₹{formatted_integer}.{decimal_part}"

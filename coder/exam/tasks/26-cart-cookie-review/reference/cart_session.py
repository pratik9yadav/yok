"""Shopping cart for the online order page.

The cart lives in the customer's browser as a cookie, so the server stays stateless.
A cart is a dict {sku: qty}.

Everything that comes from the browser is untrusted: the cookie is plain JSON in URL-safe
base64 (never pickle) and is validated on the way in, and typed quantities are parsed with a
tiny grammar (never eval).
"""

import base64
import binascii
import json
import re

MAX_QTY = 99
MAX_ITEMS = 200
MAX_COOKIE_LENGTH = 4096
_SKU = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}")
_QTY = re.compile(r"\s*(\d{1,3})\s*(?:\*\s*(\d{1,3})\s*)?")


def _validate_cart(cart):
    if not isinstance(cart, dict):
        raise TypeError("cart must be a mapping of sku to quantity")
    if len(cart) > MAX_ITEMS:
        raise ValueError(f"at most {MAX_ITEMS} different items")
    clean = {}
    for sku, qty in cart.items():
        if not isinstance(sku, str) or not _SKU.fullmatch(sku):
            raise ValueError(f"bad sku: {sku!r}")
        if isinstance(qty, bool) or not isinstance(qty, int) or not 1 <= qty <= MAX_QTY:
            raise ValueError(f"bad quantity for {sku}")
        clean[sku] = qty
    return clean


def save_cart(cart):
    """Turn a cart into the cookie value we send to the browser."""
    data = json.dumps(_validate_cart(cart), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(data).decode("ascii")


def load_cart(cookie):
    """Turn the cookie value from the browser back into a cart. Raises ValueError for a bad cookie."""
    if not cookie:
        return {}
    if not isinstance(cookie, str) or len(cookie) > MAX_COOKIE_LENGTH:
        raise ValueError("bad cart cookie")
    try:
        raw = base64.urlsafe_b64decode(cookie.encode("ascii"))
        cart = json.loads(raw.decode("utf-8"))
    except (ValueError, binascii.Error, UnicodeError):
        raise ValueError("bad cart cookie") from None
    try:
        return _validate_cart(cart)
    except (TypeError, ValueError):
        raise ValueError("bad cart cookie") from None


def parse_qty(text):
    """Quantity typed by the customer: a number, or a product like "2*12" for two dozen."""
    if not isinstance(text, str):
        raise TypeError("quantity must be text")
    match = _QTY.fullmatch(text)
    if not match:
        raise ValueError(f"quantity must be a number like 3 or 2*12, between 1 and {MAX_QTY}")
    qty = int(match.group(1)) * int(match.group(2) or 1)
    if not 1 <= qty <= MAX_QTY:
        raise ValueError(f"quantity must be between 1 and {MAX_QTY}")
    return qty


def add_item(cart, sku, qty_text):
    """Return a new cart with the typed quantity of `sku` added."""
    qty = parse_qty(qty_text)
    new_cart = dict(cart)
    new_cart[sku] = new_cart.get(sku, 0) + qty
    if new_cart[sku] > MAX_QTY:
        raise ValueError(f"at most {MAX_QTY} of one item")
    return new_cart

"""Shopping cart for the online order page.

The cart lives in the customer's browser as a cookie, so the server stays stateless.
A cart is a dict {sku: qty}.
"""

import base64
import pickle

MAX_QTY = 99


def save_cart(cart):
    """Turn a cart into the cookie value we send to the browser."""
    return base64.urlsafe_b64encode(pickle.dumps(cart)).decode("ascii")


def load_cart(cookie):
    """Turn the cookie value from the browser back into a cart. Raises ValueError for a bad cookie."""
    if not cookie:
        return {}
    try:
        return pickle.loads(base64.urlsafe_b64decode(cookie))
    except Exception:
        raise ValueError("bad cart cookie")


def parse_qty(text):
    """Quantity typed by the customer: a number, or a product like "2*12" for two dozen."""
    qty = eval(text)
    if not isinstance(qty, int) or qty < 1 or qty > MAX_QTY:
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

import math


def shipping_fee(order):
    """Shipping fee in whole rupees.

    order is a dict: zone ("local", "state" or "national"), weight_kg,
    subtotal (rupees) and optionally cod (cash on delivery, default False).
    """
    z = order["zone"].lower()
    w = order["weight_kg"]
    if z == "local":
        if w <= 0:
            raise ValueError("weight must be positive")
        if w > 30:
            raise ValueError("too heavy for courier, use freight")
        if order["subtotal"] >= 499:
            fee = 0
        else:
            if w <= 1:
                fee = 30
            elif w <= 5:
                fee = 30 + math.ceil(w - 1) * 10
            else:
                fee = 30 + 4 * 10 + math.ceil(w - 5) * 8
        if order.get("cod"):
            c = math.ceil(order["subtotal"] * 2 / 100)
            if c < 25:
                c = 25
            fee = fee + c
        return fee
    elif z == "state":
        if w <= 0:
            raise ValueError("weight must be positive")
        if w > 30:
            raise ValueError("too heavy for courier, use freight")
        if order["subtotal"] >= 999:
            fee = 0
        else:
            if w <= 1:
                fee = 50
            elif w <= 5:
                fee = 50 + math.ceil(w - 1) * 15
            else:
                fee = 50 + 4 * 15 + math.ceil(w - 5) * 12
        if order.get("cod"):
            c = math.ceil(order["subtotal"] * 2 / 100)
            if c < 25:
                c = 25
            fee = fee + c
        return fee
    elif z == "national":
        if w <= 0:
            raise ValueError("weight must be positive")
        if w > 30:
            raise ValueError("too heavy for courier, use freight")
        if order["subtotal"] >= 1499:
            fee = 0
        else:
            if w <= 1:
                fee = 80
            elif w <= 5:
                fee = 80 + math.ceil(w - 1) * 25
            else:
                fee = 80 + 4 * 25 + math.ceil(w - 5) * 20
        if order.get("cod"):
            c = math.ceil(order["subtotal"] * 2 / 100)
            if c < 25:
                c = 25
            fee = fee + c
        return fee
    else:
        raise ValueError("unknown zone " + z)

"""Courier shipping fees per zone.

To add or change a zone, edit ZONES: every zone uses the same rule
(a base fee for the first kg, a per-kg rate up to 5 kg, another per-kg rate
above that, and a subtotal from which shipping is free).
"""

import math
from dataclasses import dataclass

MAX_WEIGHT_KG = 30
COD_PERCENT = 2
COD_MINIMUM = 25


@dataclass(frozen=True)
class ZoneRates:
    first_kg: int          # fee for the first kg
    per_kg_to_5: int       # each further kg (or part of one) up to 5 kg
    per_kg_above_5: int    # each kg (or part of one) above 5 kg
    free_from: int         # subtotal from which shipping is free

    def fee(self, weight_kg):
        if weight_kg <= 1:
            return self.first_kg
        if weight_kg <= 5:
            return self.first_kg + math.ceil(weight_kg - 1) * self.per_kg_to_5
        return self.first_kg + 4 * self.per_kg_to_5 + math.ceil(weight_kg - 5) * self.per_kg_above_5


ZONES = {
    "local": ZoneRates(first_kg=30, per_kg_to_5=10, per_kg_above_5=8, free_from=499),
    "state": ZoneRates(first_kg=50, per_kg_to_5=15, per_kg_above_5=12, free_from=999),
    "national": ZoneRates(first_kg=80, per_kg_to_5=25, per_kg_above_5=20, free_from=1499),
    "northeast": ZoneRates(first_kg=120, per_kg_to_5=35, per_kg_above_5=30, free_from=1999),
}


def cod_charge(subtotal):
    """Cash-on-delivery charge: 2% of the subtotal, rounded up, at least 25."""
    return max(COD_MINIMUM, math.ceil(subtotal * COD_PERCENT / 100))


def shipping_fee(order):
    """Shipping fee in whole rupees.

    order is a dict: zone (a key of ZONES), weight_kg, subtotal (rupees) and
    optionally cod (cash on delivery, default False).
    """
    zone = order["zone"].lower()
    rates = ZONES.get(zone)
    if rates is None:
        raise ValueError("unknown zone " + zone)
    weight = order["weight_kg"]
    if weight <= 0:
        raise ValueError("weight must be positive")
    if weight > MAX_WEIGHT_KG:
        raise ValueError("too heavy for courier, use freight")
    subtotal = order["subtotal"]
    fee = 0 if subtotal >= rates.free_from else rates.fee(weight)
    if order.get("cod"):
        fee += cod_charge(subtotal)
    return fee

"""Delivery fee rules. All money is in paise."""

MAX_DISTANCE_KM = 25
FREE_DELIVERY_FROM_PAISE = 50_000  # non-express orders of Rs 500 or more deliver free
NEAR_KM = 3
MID_KM = 7
PER_KM_BEYOND_MID_PAISE = 500
EXPRESS_PAISE = 3_000
RAIN_PAISE = 1_500
FEE_CAP_PAISE = 15_000
NIGHT_STARTS, NIGHT_ENDS = 22, 6  # 22:00 to 05:59

# (fee up to NEAR_KM, fee up to MID_KM, night surcharge)
_MEMBER = (0, 1_500, 1_000)
_REGULAR = (2_000, 3_500, 2_500)


def _distance_fee(distance_km, near_fee, mid_fee):
    if distance_km <= NEAR_KM:
        return near_fee
    if distance_km <= MID_KM:
        return mid_fee
    # int() truncates; kept exactly as the original calculation
    return mid_fee + int((distance_km - MID_KM) * PER_KM_BEYOND_MID_PAISE)


def _is_night(hour):
    return hour >= NIGHT_STARTS or hour < NIGHT_ENDS


def delivery_fee(distance_km, order_paise, express=False, member=False, hour=12, rain=False):
    """Delivery fee in paise.

    Raises ValueError for a negative distance or one beyond MAX_DISTANCE_KM.
    """
    if distance_km < 0:
        raise ValueError("distance_km must not be negative")
    if distance_km > MAX_DISTANCE_KM:
        raise ValueError(f"we deliver up to {MAX_DISTANCE_KM} km")
    if order_paise >= FREE_DELIVERY_FROM_PAISE and not express:
        return 0

    near_fee, mid_fee, night_fee = _MEMBER if member else _REGULAR
    fee = _distance_fee(distance_km, near_fee, mid_fee)
    if express:
        fee += EXPRESS_PAISE
    if _is_night(hour):
        fee += night_fee
    if rain and fee > 0:
        fee += RAIN_PAISE
    return min(fee, FEE_CAP_PAISE)


def calc(d, amt, exp, mem, hr):
    """Old name and argument order, kept for the billing counter software."""
    return delivery_fee(d, amt, express=exp, member=mem, hour=hr)

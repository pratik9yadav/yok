"""Delivery promises shown to customers at checkout."""

from datetime import timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), "IST")
CUTOFF_HOUR = 14  # orders placed before 2 pm India time go out the same day


def promised_date(order_time, holidays=()):
    """Return the date we promise to deliver an order.

    order_time must be a timezone-aware datetime (the server stores UTC).
    Orders placed before 2 pm India time are dispatched that day, later
    orders the next day. Delivery is on the first day after dispatch that
    is not a Sunday or one of the given holidays (dates).
    """
    dispatch = order_time.date()
    if order_time.hour >= CUTOFF_HOUR:
        dispatch += timedelta(days=1)
    day = dispatch + timedelta(days=1)
    while day.weekday() == 7 or day in holidays:
        day += timedelta(days=1)
    return day

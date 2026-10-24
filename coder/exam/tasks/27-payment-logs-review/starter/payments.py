"""Card payments for the online order page, through our payment gateway."""

import logging

log = logging.getLogger("shop.payments")

_recent = []  # shown on the support dashboard


class PaymentError(Exception):
    pass


def charge_card(gateway, order_id, amount_paise, card_number, expiry, cvv, phone):
    """Charge a card through `gateway` and return the gateway's transaction id.

    `gateway` has `.url`, `.api_key` and `.charge(request: dict) -> {"txn_id": ...}`, and raises on failure.
    Raises PaymentError if the charge fails.
    """
    if amount_paise <= 0:
        raise ValueError("amount must be positive")
    request = {
        "order_id": order_id,
        "amount_paise": amount_paise,
        "card_number": card_number,
        "expiry": expiry,
        "cvv": cvv,
        "phone": phone,
    }
    log.info("charging order %s: %s", order_id, request)
    log.debug("POST %s Authorization: Bearer %s", gateway.url, gateway.api_key)
    _recent.append(request)
    try:
        response = gateway.charge(request)
    except Exception as exc:
        log.error("charge failed for %s: %s", request, exc)
        raise PaymentError(f"payment failed for {request}: {exc}")
    log.info("charged order %s (customer phone %s): txn %s", order_id, phone, response["txn_id"])
    return response["txn_id"]


def recent_charges():
    """The latest charges, for the support dashboard."""
    return list(_recent)

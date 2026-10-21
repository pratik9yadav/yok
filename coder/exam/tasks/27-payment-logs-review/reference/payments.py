"""Card payments for the online order page, through our payment gateway.

Card numbers, CVVs, phone numbers and the gateway key never reach logs, exception messages
or the support dashboard: only masked values do. The dashboard history is bounded.
"""

import logging
import threading
from collections import deque

log = logging.getLogger("shop.payments")

RECENT_LIMIT = 200
_recent = deque(maxlen=RECENT_LIMIT)  # shown on the support dashboard
_recent_lock = threading.Lock()


class PaymentError(Exception):
    pass


def mask_card(card_number):
    digits = "".join(ch for ch in str(card_number) if ch.isdigit())
    return f"**** **** **** {digits[-4:]}" if len(digits) >= 4 else "****"


def mask_phone(phone):
    digits = "".join(ch for ch in str(phone) if ch.isdigit())
    return f"******{digits[-4:]}" if len(digits) >= 4 else "****"


def charge_card(gateway, order_id, amount_paise, card_number, expiry, cvv, phone):
    """Charge a card through `gateway` and return the gateway's transaction id.

    `gateway` has `.url`, `.api_key` and `.charge(request: dict) -> {"txn_id": ...}`, and raises on failure.
    Raises PaymentError if the charge fails.
    """
    if isinstance(amount_paise, bool) or not isinstance(amount_paise, int):
        raise TypeError("amount_paise must be an integer number of paise")
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
    safe = {
        "order_id": order_id,
        "amount_paise": amount_paise,
        "card": mask_card(card_number),
        "phone": mask_phone(phone),
    }
    log.info("charging order %s: amount_paise=%s card=%s", order_id, amount_paise, safe["card"])
    log.debug("POST %s", gateway.url)
    try:
        response = gateway.charge(request)
    except Exception as exc:  # noqa: BLE001 - any gateway failure becomes a PaymentError
        # the gateway's own message may echo request fields, so only log the error type
        log.error("charge failed for order %s (%s)", order_id, type(exc).__name__)
        with _recent_lock:
            _recent.append({**safe, "status": "failed"})
        raise PaymentError(f"payment failed for order {order_id}") from None
    txn_id = response["txn_id"]
    with _recent_lock:
        _recent.append({**safe, "status": "charged", "txn_id": txn_id})
    log.info("charged order %s: txn %s", order_id, txn_id)
    return txn_id


def recent_charges():
    """The latest charges, for the support dashboard (masked)."""
    with _recent_lock:
        return [dict(item) for item in _recent]

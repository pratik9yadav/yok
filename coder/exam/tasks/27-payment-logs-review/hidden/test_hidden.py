import logging
import re
import threading

import pytest

import payments
from payments import PaymentError, charge_card, recent_charges

CARD = "4111111111111111"
CVV = "846"
EXPIRY = "12/27"
PHONE = "9876543210"
API_KEY = "sk_live_51HxQaZ0secretKEY"
ORDER = "ORD-2024-0001"


class FakeGateway:
    url = "https://pay.example-gateway.in/v1/charges"
    api_key = API_KEY

    def __init__(self, fail=False):
        self.fail = fail
        self.requests = []
        self.lock = threading.Lock()

    def charge(self, request):
        with self.lock:
            self.requests.append(dict(request))
            n = len(self.requests)
        if self.fail:
            raise RuntimeError("card declined: insufficient funds")
        return {"txn_id": f"TXN-{n:05d}"}


def leaked(text):
    found = []
    if CARD in re.sub(r"[ -]", "", text):
        found.append("card number")
    if re.search(rf"(?<!\d){CVV}(?!\d)", text):
        found.append("cvv")
    if PHONE in text:
        found.append("phone")
    if API_KEY in text:
        found.append("api key")
    return found


def all_log_text(caplog):
    parts = []
    for record in caplog.records:
        parts.append(record.getMessage())
        if record.exc_info:
            parts.append(logging.Formatter().formatException(record.exc_info))
        if record.exc_text:
            parts.append(record.exc_text)
        if record.stack_info:
            parts.append(record.stack_info)
    return "\n".join(parts) + "\n" + caplog.text


def charge(gateway, order_id=ORDER, amount=129900):
    return charge_card(gateway, order_id, amount, CARD, EXPIRY, CVV, PHONE)


def test_successful_charge_returns_txn_and_gateway_gets_full_details():
    gw = FakeGateway()
    txn = charge(gw)
    assert txn == "TXN-00001"
    sent = gw.requests[0]
    assert CARD in sent.values()
    assert CVV in sent.values()
    assert 129900 in sent.values()


def test_success_logs_contain_no_secrets(caplog):
    caplog.set_level(logging.DEBUG)
    charge(FakeGateway())
    text = all_log_text(caplog)
    assert leaked(text) == []
    assert ORDER in text, "support must still find the charge by order id in the logs"


def test_failure_raises_payment_error_without_secrets(caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(PaymentError) as excinfo:
        charge(FakeGateway(fail=True), order_id="ORD-2024-0002")
    message = str(excinfo.value) + repr(excinfo.value) + repr(excinfo.value.args)
    assert leaked(message) == []
    text = all_log_text(caplog)
    assert leaked(text) == []
    assert "ORD-2024-0002" in text


def test_card_with_spaces_is_not_logged_either(caplog):
    caplog.set_level(logging.DEBUG)
    charge_card(FakeGateway(), "ORD-2024-0003", 5000, "4111 1111 1111 1111", EXPIRY, CVV, "+91 98765 43210")
    text = all_log_text(caplog)
    assert "4111 1111 1111 1111" not in text
    assert CARD not in text.replace(" ", "")
    assert re.search(rf"(?<!\d){CVV}(?!\d)", text) is None
    assert "98765 43210" not in text and PHONE not in text.replace(" ", "")


def test_dashboard_has_no_card_or_cvv_but_has_order_id():
    charge(FakeGateway(), order_id="ORD-2024-0004")
    with pytest.raises(PaymentError):
        charge(FakeGateway(fail=True), order_id="ORD-2024-0005")
    shown = repr(recent_charges())
    assert CARD not in shown
    assert re.search(rf"(?<!\d){CVV}(?!\d)", shown) is None
    assert "ORD-2024-0004" in shown


def test_dashboard_history_does_not_grow_forever():
    gw = FakeGateway()
    for i in range(5000):
        charge(gw, order_id=f"BULK-{i}", amount=100)
    recent = recent_charges()
    assert len(recent) < 5000
    assert "BULK-4999" in repr(recent)  # the newest are kept


def test_dashboard_list_is_a_copy():
    charge(FakeGateway(), order_id="ORD-2024-0006")
    before = len(recent_charges())
    recent_charges().clear()
    assert len(recent_charges()) == before


@pytest.mark.parametrize("amount", [0, -100])
def test_non_positive_amount_rejected_before_calling_gateway(amount):
    gw = FakeGateway()
    with pytest.raises(ValueError):
        charge(gw, amount=amount)
    assert gw.requests == []


@pytest.mark.parametrize("amount", [1299.5, "129900"])
def test_non_integer_amount_rejected(amount):
    gw = FakeGateway()
    with pytest.raises((TypeError, ValueError)):
        charge(gw, amount=amount)
    assert gw.requests == []


def test_concurrent_charges_all_succeed():
    gw = FakeGateway()
    barrier = threading.Barrier(8)
    txns, lock = [], threading.Lock()

    def worker(n):
        barrier.wait(timeout=5)
        for i in range(50):
            txn = charge(gw, order_id=f"T{n}-{i}", amount=100)
            with lock:
                txns.append(txn)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(txns) == 400 and len(set(txns)) == 400
    recent_charges()


def test_module_has_payment_error():
    assert issubclass(payments.PaymentError, Exception)

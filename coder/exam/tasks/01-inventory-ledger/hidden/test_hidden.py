import time

import pytest

import inventory
from inventory import InsufficientStock, Ledger


def test_receive_and_stock():
    led = Ledger()
    led.receive("ATTA-5KG", 10)
    led.receive("ATTA-5KG", 5)
    assert led.stock("ATTA-5KG") == 15


def test_unknown_sku_is_zero():
    led = Ledger()
    assert led.stock("NEVER-SEEN") == 0
    assert led.history("NEVER-SEEN") == []


def test_issue_reduces_stock():
    led = Ledger()
    led.receive("SALT-1KG", 10)
    led.issue("SALT-1KG", 4)
    assert led.stock("SALT-1KG") == 6
    led.issue("SALT-1KG", 6)
    assert led.stock("SALT-1KG") == 0


def test_insufficient_stock_leaves_state_unchanged():
    led = Ledger()
    led.receive("OIL-1L", 3)
    with pytest.raises(InsufficientStock):
        led.issue("OIL-1L", 4)
    assert led.stock("OIL-1L") == 3
    assert len(led.history("OIL-1L")) == 1
    with pytest.raises(InsufficientStock):
        led.issue("NEW-SKU", 1)
    assert led.stock("NEW-SKU") == 0


def test_insufficient_stock_is_an_exception_class():
    assert issubclass(inventory.InsufficientStock, Exception)


@pytest.mark.parametrize("bad", [0, -1, -100, 1.5])
@pytest.mark.parametrize("method", ["receive", "issue"])
def test_invalid_quantity_raises_value_error(method, bad):
    led = Ledger()
    led.receive("X", 10)
    with pytest.raises(ValueError):
        getattr(led, method)("X", bad)
    assert led.stock("X") == 10
    assert len(led.history("X")) == 1


@pytest.mark.parametrize("bad", ["5", None])
def test_non_number_quantity_rejected(bad):
    led = Ledger()
    with pytest.raises((ValueError, TypeError)):
        led.receive("X", bad)
    assert led.stock("X") == 0


def test_duplicate_move_id_counts_once():
    led = Ledger()
    led.receive("SUGAR-1KG", 10, move_id="m-1")
    led.receive("SUGAR-1KG", 10, move_id="m-1")
    assert led.stock("SUGAR-1KG") == 10
    led.issue("SUGAR-1KG", 3, move_id="m-2")
    led.issue("SUGAR-1KG", 3, move_id="m-2")
    led.issue("SUGAR-1KG", 3, move_id="m-2")
    assert led.stock("SUGAR-1KG") == 7
    assert len(led.history("SUGAR-1KG")) == 2


def test_moves_without_id_always_count():
    led = Ledger()
    led.receive("A", 1)
    led.receive("A", 1)
    assert led.stock("A") == 2


def test_failed_move_does_not_burn_its_id():
    led = Ledger()
    with pytest.raises(ValueError):
        led.receive("A", 0, move_id="r-1")
    led.receive("A", 5, move_id="r-1")
    assert led.stock("A") == 5
    with pytest.raises(InsufficientStock):
        led.issue("A", 8, move_id="i-1")
    led.receive("A", 5)
    led.issue("A", 8, move_id="i-1")  # retried after restock: must apply now
    assert led.stock("A") == 2


def test_history_order_and_content():
    led = Ledger()
    led.receive("DAL-1KG", 7, move_id="a")
    led.issue("DAL-1KG", 2, move_id="b")
    led.receive("DAL-1KG", 1)
    hist = led.history("DAL-1KG")
    assert [(h["move_id"], h["kind"], h["qty"]) for h in hist] == [
        ("a", "receive", 7),
        ("b", "issue", 2),
        (None, "receive", 1),
    ]


def test_history_cannot_be_changed_from_outside():
    led = Ledger()
    led.receive("A", 5, move_id="a")
    hist = led.history("A")
    hist.clear()
    assert len(led.history("A")) == 1
    assert led.stock("A") == 5


def test_ledgers_do_not_share_state():
    a = Ledger()
    b = Ledger()
    a.receive("A", 5, move_id="m")
    b.receive("A", 2, move_id="m")
    assert a.stock("A") == 5
    assert b.stock("A") == 2
    assert len(b.history("A")) == 1


def test_many_moves_are_fast():
    led = Ledger()
    start = time.perf_counter()
    for i in range(50_000):
        led.receive(f"SKU-{i % 100}", 2, move_id=f"r{i}")
    for i in range(50_000):
        led.issue(f"SKU-{i % 100}", 1, move_id=f"i{i}")
    elapsed = time.perf_counter() - start
    assert led.stock("SKU-7") == 500
    assert len(led.history("SKU-7")) == 1000
    assert elapsed < 3.0

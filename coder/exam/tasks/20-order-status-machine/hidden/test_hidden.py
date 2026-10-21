import threading

import pytest

from order_status import InvalidTransition, Order


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now


def make(order_id="ORD-1001"):
    return Order(order_id, clock=FakeClock(1001.0))


def test_happy_path_with_history():
    clock = FakeClock(1001.0)
    o = Order("ORD-1001", clock=clock)
    assert o.status == "placed"
    clock.now = 1002.0
    o.pack()
    clock.now = 1003.0
    o.ship("DTDC123456")
    clock.now = 1004.0
    o.deliver()
    assert o.status == "delivered"
    assert list(o.history) == [("placed", 1001.0), ("packed", 1002.0), ("shipped", 1003.0), ("delivered", 1004.0)]


@pytest.mark.parametrize("steps", [[], ["pack"]])
def test_cancel_before_shipping(steps):
    o = make()
    for step in steps:
        getattr(o, step)()
    o.cancel("customer changed mind")
    assert o.status == "cancelled"
    assert o.history[-1][0] == "cancelled"


def advance(o, upto):
    if upto in ("packed", "shipped", "delivered"):
        o.pack()
    if upto in ("shipped", "delivered"):
        o.ship("TRK-1")
    if upto == "delivered":
        o.deliver()


@pytest.mark.parametrize(
    "state,action",
    [
        ("placed", "deliver"),
        ("placed", "ship"),
        ("packed", "pack"),
        ("packed", "deliver"),
        ("shipped", "cancel"),
        ("shipped", "pack"),
        ("shipped", "ship"),
        ("delivered", "cancel"),
        ("delivered", "deliver"),
        ("delivered", "pack"),
    ],
)
def test_illegal_moves_raise_and_change_nothing(state, action):
    o = make()
    advance(o, state)
    before_history = list(o.history)
    args = ("TRK-2",) if action == "ship" else ()
    with pytest.raises(InvalidTransition):
        getattr(o, action)(*args)
    assert o.status == state
    assert list(o.history) == before_history


@pytest.mark.parametrize("action", ["pack", "deliver", "cancel"])
def test_cancelled_is_final(action):
    o = make()
    o.cancel()
    with pytest.raises(InvalidTransition):
        getattr(o, action)()
    with pytest.raises((InvalidTransition, ValueError)):
        o.ship("TRK-3")
    assert o.status == "cancelled"


@pytest.mark.parametrize("tracking", ["", "   ", None])
def test_ship_needs_tracking_id(tracking):
    o = make()
    o.pack()
    with pytest.raises(ValueError):
        o.ship(tracking)
    assert o.status == "packed"
    assert [s for s, _ in o.history] == ["placed", "packed"]


def test_invalid_transition_is_an_exception():
    assert issubclass(InvalidTransition, Exception)
    o = make()
    with pytest.raises(InvalidTransition):
        o.deliver()


def test_status_cannot_be_set_directly():
    o = make()
    try:
        o.status = "delivered"
    except (AttributeError, TypeError):
        pass
    assert o.status == "placed"


def test_history_cannot_be_changed_from_outside():
    o = make()
    h = o.history
    try:
        h.append(("delivered", 0))
    except (AttributeError, TypeError):
        pass
    assert list(o.history) == [("placed", 1001.0)]


def test_orders_are_independent():
    a, b = make("A"), make("B")
    a.pack()
    assert b.status == "placed"
    assert len(b.history) == 1


def test_only_one_of_many_concurrent_packs_wins():
    o = make()
    n = 8
    barrier = threading.Barrier(n)
    results = []
    lock = threading.Lock()

    def worker():
        barrier.wait(timeout=5)
        try:
            o.pack()
            outcome = "ok"
        except InvalidTransition:
            outcome = "rejected"
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert sorted(results) == ["ok"] + ["rejected"] * (n - 1)
    assert [s for s, _ in o.history] == ["placed", "packed"]

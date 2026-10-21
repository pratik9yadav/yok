import threading

import pytest

from metrics import MetricsRegistry


def parse(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        name, value = line.split()
        rows.append((name, float(value)))
    return rows


def run_threads(target, n, *args):
    threads = [threading.Thread(target=target, args=args) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not any(t.is_alive() for t in threads)


def test_counter_starts_at_zero_and_counts():
    reg = MetricsRegistry()
    c = reg.counter("orders_placed_total")
    assert c.value == 0
    c.inc()
    c.inc(4)
    assert c.value == 5


def test_same_name_same_counter():
    reg = MetricsRegistry()
    reg.counter("sms_sent_total").inc()
    reg.counter("sms_sent_total").inc(2)
    assert reg.counter("sms_sent_total") is reg.counter("sms_sent_total")
    assert reg.counter("sms_sent_total").value == 3


def test_counter_cannot_go_down():
    c = MetricsRegistry().counter("refunds_total")
    c.inc(2)
    with pytest.raises(ValueError):
        c.inc(-1)
    assert c.value == 2


def test_counter_rejects_non_numbers():
    c = MetricsRegistry().counter("x_total")
    with pytest.raises((TypeError, ValueError)):
        c.inc("5")
    assert c.value == 0


def test_gauge_up_down_and_set():
    g = MetricsRegistry().gauge("open_carts")
    g.inc()
    g.inc(3)
    g.dec(5)
    assert g.value == -1
    g.set(2.5)
    assert g.value == 2.5


@pytest.mark.parametrize("name", ["", "Orders", "1orders", "_orders", "orders-total", "orders total", "ऑर्डर",
                                  "orders.total"])
def test_bad_names_rejected(name):
    reg = MetricsRegistry()
    with pytest.raises(ValueError):
        reg.counter(name)
    with pytest.raises(ValueError):
        reg.gauge(name)
    assert reg.snapshot() == {}


def test_kind_conflict_rejected():
    reg = MetricsRegistry()
    reg.counter("orders_total")
    reg.gauge("queue_depth")
    with pytest.raises(ValueError):
        reg.gauge("orders_total")
    with pytest.raises(ValueError):
        reg.counter("queue_depth")


def test_snapshot_is_a_copy():
    reg = MetricsRegistry()
    reg.counter("a_total").inc(2)
    snap = reg.snapshot()
    assert snap == {"a_total": 2}
    snap["a_total"] = 100
    snap["fake"] = 1
    assert reg.snapshot() == {"a_total": 2}


def test_render_sorted_name_value_lines():
    reg = MetricsRegistry()
    reg.gauge("temperature_c").set(2.5)
    reg.counter("b_total").inc(3)
    reg.counter("a_total").inc()
    assert parse(reg.render()) == [("a_total", 1.0), ("b_total", 3.0), ("temperature_c", 2.5)]


def test_render_empty_registry():
    assert parse(MetricsRegistry().render()) == []


def test_concurrent_increments_are_not_lost():
    reg = MetricsRegistry()
    n_threads, per_thread = 8, 5000
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait(timeout=5)
        for _ in range(per_thread):
            reg.counter("hits_total").inc()
            reg.gauge("in_flight").inc()
            reg.gauge("in_flight").dec()

    run_threads(worker, n_threads)
    assert reg.counter("hits_total").value == n_threads * per_thread
    assert reg.gauge("in_flight").value == 0


def test_render_while_new_metrics_are_created():
    reg = MetricsRegistry()
    stop = threading.Event()
    errors = []

    def writer():
        for i in range(20000):
            reg.counter(f"sku_{i}_sold_total").inc()
        stop.set()

    def reader():
        try:
            while not stop.is_set():
                reg.render()
                reg.snapshot()
        except Exception as exc:
            errors.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(2)]
    w = threading.Thread(target=writer)
    for t in readers:
        t.start()
    w.start()
    w.join(timeout=20)
    stop.set()
    for t in readers:
        t.join(timeout=20)
    assert errors == []
    assert len(reg.snapshot()) == 20000

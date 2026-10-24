import threading

import pytest

from stock import OutOfStock, StockBook


class ListLedger:
    def __init__(self):
        self.lock = threading.Lock()
        self.entries = []

    def record(self, action, sku, qty):
        with self.lock:
            self.entries.append((action, sku, qty))


class SlowLedger(ListLedger):
    """A ledger whose disk write is slow: each write waits (briefly) for the other till.

    With correct locking the second till is not inside the critical section, the barrier
    times out and the call simply continues, so the test is deterministic either way.
    """

    def __init__(self, parties=2):
        super().__init__()
        self.barrier = threading.Barrier(parties)

    def record(self, action, sku, qty):
        try:
            self.barrier.wait(timeout=0.3)
        except threading.BrokenBarrierError:
            pass
        super().record(action, sku, qty)


def run_together(n, target):
    results, lock = [], threading.Lock()

    def wrapped():
        try:
            target()
            outcome = "ok"
        except OutOfStock:
            outcome = "out"
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=wrapped) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "threads deadlocked"
    return sorted(results)


def test_reserve_and_available():
    book = StockBook({"COOKER-5L": 3})
    book.reserve("COOKER-5L", 2)
    assert book.available("COOKER-5L") == 1
    assert book.available("UNKNOWN") == 0


def test_out_of_stock_leaves_stock_alone():
    ledger = ListLedger()
    book = StockBook({"COOKER-5L": 1}, ledger=ledger)
    with pytest.raises(OutOfStock):
        book.reserve("COOKER-5L", 2)
    assert book.available("COOKER-5L") == 1
    assert ledger.entries == []


@pytest.mark.parametrize("method", ["reserve", "release", "restock"])
@pytest.mark.parametrize("qty", [0, -1])
def test_bad_qty_still_rejected(method, qty):
    book = StockBook({"X": 5})
    with pytest.raises(ValueError):
        getattr(book, method)("X", qty)
    assert book.available("X") == 5


def test_release_and_restock_add_stock():
    ledger = ListLedger()
    book = StockBook({"DIYA-BOX": 2}, ledger=ledger)
    book.restock("DIYA-BOX", 5)
    book.release("DIYA-BOX", 1)
    book.restock("NEW-SKU", 4)
    assert book.available("DIYA-BOX") == 8
    assert book.available("NEW-SKU") == 4
    assert ledger.entries == [("restock", "DIYA-BOX", 5), ("release", "DIYA-BOX", 1), ("restock", "NEW-SKU", 4)]


def test_two_tills_cannot_both_sell_the_last_cooker():
    ledger = SlowLedger()
    book = StockBook({"COOKER-5L": 1}, ledger=ledger)
    results = run_together(2, lambda: book.reserve("COOKER-5L", 1))
    assert results == ["ok", "out"]
    assert book.available("COOKER-5L") == 0
    assert ledger.entries == [("reserve", "COOKER-5L", 1)]


def test_concurrent_restocks_are_not_lost():
    book = StockBook({"DIYA-BOX": 10}, ledger=SlowLedger())
    run_together(2, lambda: book.restock("DIYA-BOX", 5))
    assert book.available("DIYA-BOX") == 20


def test_concurrent_releases_are_not_lost():
    book = StockBook({"DIYA-BOX": 0}, ledger=SlowLedger())
    run_together(2, lambda: book.release("DIYA-BOX", 3))
    assert book.available("DIYA-BOX") == 6


def test_reserve_racing_restock():
    book = StockBook({"OIL-1L": 5}, ledger=SlowLedger())
    barrier = threading.Barrier(2)

    def sell():
        barrier.wait(timeout=5)
        book.reserve("OIL-1L", 5)

    def deliver():
        barrier.wait(timeout=5)
        book.restock("OIL-1L", 10)

    threads = [threading.Thread(target=sell), threading.Thread(target=deliver)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert book.available("OIL-1L") == 10


def test_many_tills_sell_exactly_the_stock():
    ledger = ListLedger()
    book = StockBook({"SUGAR-1KG": 100}, ledger=ledger)
    n_threads, attempts = 8, 25
    barrier = threading.Barrier(n_threads)
    sold, lock = [0], threading.Lock()

    def till():
        barrier.wait(timeout=5)
        for _ in range(attempts):
            try:
                book.reserve("SUGAR-1KG", 1)
            except OutOfStock:
                continue
            with lock:
                sold[0] += 1

    threads = [threading.Thread(target=till) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert sold[0] == 100
    assert book.available("SUGAR-1KG") == 0
    assert len(ledger.entries) == 100


def test_ledger_failure_changes_nothing_and_does_not_lock_up():
    class BrokenLedger:
        def __init__(self):
            self.fail = True

        def record(self, action, sku, qty):
            if self.fail:
                raise OSError("disk full")

    ledger = BrokenLedger()
    book = StockBook({"RICE-5KG": 3}, ledger=ledger)
    with pytest.raises(OSError):
        book.reserve("RICE-5KG", 1)
    with pytest.raises(OSError):
        book.restock("RICE-5KG", 1)
    assert book.available("RICE-5KG") == 3
    ledger.fail = False
    done = []
    t = threading.Thread(target=lambda: (book.reserve("RICE-5KG", 1), done.append(True)))
    t.start()
    t.join(timeout=5)
    assert done, "StockBook stayed locked after a ledger error"
    assert book.available("RICE-5KG") == 2

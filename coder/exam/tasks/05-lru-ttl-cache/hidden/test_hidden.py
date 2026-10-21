import time

import pytest

from ttlcache import TTLCache


class FakeClock:
    def __init__(self, now=500.0):
        self.now = now

    def __call__(self):
        return self.now


def make(maxsize=3, ttl=10):
    clock = FakeClock()
    return TTLCache(maxsize, ttl, clock=clock), clock


def test_set_and_get():
    c, _ = make()
    c.set("atta", 245)
    assert c.get("atta") == 245
    assert "atta" in c
    assert len(c) == 1


def test_missing_key_returns_default():
    c, _ = make()
    assert c.get("nope") is None
    assert c.get("nope", "dflt") == "dflt"
    assert "nope" not in c


def test_falsy_and_none_values_are_cached():
    c, _ = make()
    c.set("a", None)
    c.set("b", 0)
    assert c.get("a", "dflt") is None
    assert c.get("b", "dflt") == 0
    assert "a" in c and len(c) == 2


def test_expiry_boundary():
    c, clock = make(ttl=10)
    c.set("k", 1)
    clock.now += 9.999
    assert c.get("k") == 1
    clock.now = 510.0
    assert c.get("k") is None
    assert "k" not in c
    assert len(c) == 0


def test_get_does_not_extend_ttl_but_set_does():
    c, clock = make(ttl=10)
    c.set("k", 1)
    clock.now += 6
    assert c.get("k") == 1
    clock.now += 6
    assert c.get("k") is None
    c.set("k", 2)
    clock.now += 6
    c.set("k", 3)
    clock.now += 6
    assert c.get("k") == 3


def test_lru_eviction_order():
    c, _ = make(maxsize=2)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")
    c.set("c", 3)
    assert c.get("b") is None
    assert c.get("a") == 1 and c.get("c") == 3
    assert len(c) == 2


def test_updating_key_does_not_evict_others():
    c, _ = make(maxsize=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("a", 10)
    assert len(c) == 2
    c.set("c", 3)  # a was set most recently, so b goes
    assert c.get("b") is None
    assert c.get("a") == 10 and c.get("c") == 3


def test_expired_entries_never_push_out_live_ones():
    c, clock = make(maxsize=2, ttl=10)
    c.set("b", "old")      # expires at 510
    clock.now += 5
    c.set("a", "live")     # expires at 515
    clock.now += 1
    c.get("b")             # b is now most recently used, a least
    clock.now = 512.0      # b expired, a still live: only 1 live entry
    c.set("c", "new")
    assert c.get("a") == "live"
    assert c.get("c") == "new"


def test_len_counts_only_live_entries():
    c, clock = make(maxsize=5, ttl=10)
    c.set("a", 1)
    clock.now += 5
    c.set("b", 2)
    clock.now += 6
    assert len(c) == 1


def test_delete():
    c, _ = make()
    c.set("a", 1)
    c.delete("a")
    assert c.get("a") is None
    c.delete("a")
    c.delete("never")
    assert len(c) == 0


@pytest.mark.parametrize("maxsize, ttl", [(0, 10), (-1, 10), (3, 0), (3, -1), (3, float("nan"))])
def test_bad_settings_raise(maxsize, ttl):
    with pytest.raises(ValueError):
        TTLCache(maxsize, ttl, clock=FakeClock())


def test_caches_do_not_share_state():
    c1, _ = make()
    c2, _ = make()
    c1.set("a", 1)
    assert c2.get("a") is None
    assert len(c2) == 0


def test_default_clock_works():
    c = TTLCache(2, 60)
    c.set("a", 1)
    assert c.get("a") == 1


def test_many_operations_are_fast():
    clock = FakeClock()
    c = TTLCache(1000, 30, clock=clock)
    start = time.perf_counter()
    for i in range(100_000):
        clock.now += 0.001
        c.set(i, i)
        c.get(i - 500)
    assert time.perf_counter() - start < 3.0
    assert len(c) == 1000
    assert c.get(99_999) == 99_999
    assert c.get(0) is None

import time

import pytest

from ratelimit import KeyedLimiter, TokenBucket


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_starts_full():
    clock = FakeClock()
    b = TokenBucket(rate=1, capacity=3, clock=clock)
    assert [b.allow() for _ in range(4)] == [True, True, True, False]


def test_refills_over_time():
    clock = FakeClock()
    b = TokenBucket(rate=2, capacity=4, clock=clock)
    for _ in range(4):
        assert b.allow()
    assert not b.allow()
    clock.advance(1)
    assert b.allow()
    assert b.allow()
    assert not b.allow()


def test_fractional_rate():
    clock = FakeClock()
    b = TokenBucket(rate=0.5, capacity=1, clock=clock)
    assert b.allow()
    clock.advance(1)
    assert not b.allow()
    clock.advance(1)
    assert b.allow()


def test_never_exceeds_capacity():
    clock = FakeClock()
    b = TokenBucket(rate=10, capacity=5, clock=clock)
    clock.advance(10_000)
    assert sum(b.allow() for _ in range(20)) == 5


def test_denied_request_takes_nothing():
    clock = FakeClock()
    b = TokenBucket(rate=1, capacity=5, clock=clock)
    assert b.allow(cost=3)
    assert not b.allow(cost=3)
    assert b.allow(cost=2)
    assert not b.allow()


def test_clock_going_backwards_does_not_drain():
    clock = FakeClock()
    b = TokenBucket(rate=1, capacity=2, clock=clock)
    clock.advance(-5)
    assert b.allow()
    assert b.allow()
    assert not b.allow()
    clock.advance(6)
    assert b.allow()


def test_uses_injected_clock_only():
    clock = FakeClock()
    b = TokenBucket(rate=1, capacity=1, clock=clock)
    assert b.allow()
    time.sleep(0.05)
    assert not b.allow()  # real time passed, fake time did not


@pytest.mark.parametrize("rate, capacity", [(0, 1), (-1, 1), (1, 0), (1, -5), (float("nan"), 1), (1, float("inf"))])
def test_bad_settings_raise(rate, capacity):
    with pytest.raises(ValueError):
        TokenBucket(rate=rate, capacity=capacity, clock=FakeClock())
    with pytest.raises(ValueError):
        KeyedLimiter(rate=rate, capacity=capacity, clock=FakeClock())


@pytest.mark.parametrize("cost", [0, -1])
def test_bad_cost_raises(cost):
    clock = FakeClock()
    b = TokenBucket(rate=1, capacity=5, clock=clock)
    with pytest.raises(ValueError):
        b.allow(cost=cost)
    assert b.allow(cost=5)  # the bad call took nothing
    k = KeyedLimiter(rate=1, capacity=5, clock=clock)
    with pytest.raises(ValueError):
        k.allow("98450", cost=cost)


def test_buckets_are_independent():
    clock = FakeClock()
    a = TokenBucket(rate=1, capacity=1, clock=clock)
    b = TokenBucket(rate=1, capacity=1, clock=clock)
    assert a.allow()
    assert b.allow()


def test_keyed_limiter_separate_per_key():
    clock = FakeClock()
    lim = KeyedLimiter(rate=1, capacity=2, clock=clock)
    assert lim.allow("9876543210")
    assert lim.allow("9876543210")
    assert not lim.allow("9876543210")
    assert lim.allow("9123456780")
    clock.advance(1)
    assert lim.allow("9876543210")
    assert not lim.allow("9876543210")


def test_keyed_limiters_do_not_share_state():
    clock = FakeClock()
    a = KeyedLimiter(rate=1, capacity=1, clock=clock)
    b = KeyedLimiter(rate=1, capacity=1, clock=clock)
    assert a.allow("k")
    assert b.allow("k")


def test_new_key_bucket_starts_full_even_later():
    clock = FakeClock()
    lim = KeyedLimiter(rate=1, capacity=3, clock=clock)
    clock.advance(100)
    assert sum(lim.allow("late") for _ in range(5)) == 3


def test_many_calls_are_fast():
    clock = FakeClock()
    lim = KeyedLimiter(rate=5, capacity=10, clock=clock)
    start = time.perf_counter()
    allowed = 0
    for i in range(100_000):
        clock.advance(0.001)
        allowed += lim.allow(i % 1000)
    assert time.perf_counter() - start < 3.0
    assert allowed > 0

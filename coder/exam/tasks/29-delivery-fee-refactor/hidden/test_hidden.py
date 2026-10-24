import itertools

import pytest

from delivery import calc, delivery_fee


def legacy_calc(d, amt, exp, mem, hr):
    """The original implementation, copied verbatim as the oracle."""
    if d < 0:
        raise ValueError("bad distance")
    if d > 25:
        raise ValueError("too far")
    if amt >= 50000 and exp == False:  # noqa: E712
        return 0
    if mem == True:  # noqa: E712
        if d <= 3:
            f = 0
        else:
            if d <= 7:
                f = 1500
            else:
                f = 1500 + int((d - 7) * 500)
        if exp == True:  # noqa: E712
            f = f + 3000
        if hr >= 22 or hr < 6:
            f = f + 1000
    else:
        if d <= 3:
            f = 2000
        else:
            if d <= 7:
                f = 3500
            else:
                f = 3500 + int((d - 7) * 500)
        if exp == True:  # noqa: E712
            f = f + 3000
        if hr >= 22 or hr < 6:
            f = f + 2500
    if f > 15000:
        f = 15000
    return f


DISTANCES = [0, 0.5, 1, 2.9, 3, 3.1, 4.5, 6.9, 7, 7.1, 7.3, 7.7, 8, 9.9, 12.4, 15, 19.7, 23.3, 25]
AMOUNTS = [0, 100, 49_999, 50_000, 120_000]
HOURS = list(range(24))
GRID = list(itertools.product(DISTANCES, AMOUNTS, [False, True], [False, True], HOURS))


def test_calc_positional_matches_old_behaviour_everywhere():
    mismatches = [args for args in GRID if calc(*args) != legacy_calc(*args)]
    assert mismatches == [], f"{len(mismatches)} fees changed, e.g. calc{mismatches[0]}"


def test_delivery_fee_without_rain_matches_old_behaviour():
    for d, amt, exp, mem, hr in GRID:
        got = delivery_fee(distance_km=d, order_paise=amt, express=exp, member=mem, hour=hr, rain=False)
        assert got == legacy_calc(d, amt, exp, mem, hr), (d, amt, exp, mem, hr)


def test_delivery_fee_defaults():
    assert delivery_fee(5, 10_000) == legacy_calc(5, 10_000, False, False, 12)
    assert delivery_fee(2, 10_000, member=True) == 0
    assert delivery_fee(10, 10_000, express=True, hour=23) == legacy_calc(10, 10_000, True, False, 23)


def test_rain_adds_15_rupees_to_non_zero_fees_with_cap():
    checked_nonzero = checked_zero = checked_cap = 0
    for d, amt, exp, mem, hr in GRID:
        old = legacy_calc(d, amt, exp, mem, hr)
        wet = delivery_fee(d, amt, express=exp, member=mem, hour=hr, rain=True)
        if old == 0:
            assert wet == 0, (d, amt, exp, mem, hr)
            checked_zero += 1
        else:
            uncapped = delivery_fee(d, amt, express=exp, member=mem, hour=hr)
            assert wet == min(uncapped + 1500, 15000), (d, amt, exp, mem, hr)
            checked_nonzero += 1
            checked_cap += wet == 15000
    assert checked_nonzero and checked_zero and checked_cap


def test_rain_examples():
    assert delivery_fee(5, 10_000, rain=True) == 3500 + 1500
    assert delivery_fee(5, 60_000, rain=True) == 0  # free delivery stays free
    assert delivery_fee(1, 10_000, member=True, rain=True) == 0
    assert delivery_fee(25, 10_000, express=True, hour=23, rain=True) == 15000


def test_calc_keeps_ignoring_rain():
    assert calc(5, 10_000, False, False, 12) == 3500


@pytest.mark.parametrize("d", [-0.1, -5, 25.1, 40])
def test_out_of_range_distance_still_rejected(d):
    with pytest.raises(ValueError):
        calc(d, 10_000, False, False, 12)
    with pytest.raises(ValueError):
        delivery_fee(d, 10_000)
    with pytest.raises(ValueError):
        delivery_fee(d, 10_000, rain=True)


def test_results_are_whole_paise():
    for d, amt, exp, mem, hr in GRID[::7]:
        fee = delivery_fee(d, amt, express=exp, member=mem, hour=hr, rain=True)
        assert isinstance(fee, int) and not isinstance(fee, bool)

from datetime import date, datetime, timedelta, timezone

import pytest

from delivery import promised_date

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))


def ist(*args):
    return datetime(*args, tzinfo=IST).astimezone(UTC)


def test_reported_case_saturday_afternoon():
    # Sat 3 Oct 2026, 15:10 IST -> dispatch Sun 4 Oct -> deliver Mon 5 Oct
    assert promised_date(ist(2026, 10, 3, 15, 10)) == date(2026, 10, 5)


def test_saturday_morning_skips_sunday():
    # dispatch Sat 3 Oct, next day is Sunday -> Monday
    assert promised_date(ist(2026, 10, 3, 9, 0)) == date(2026, 10, 5)


def test_weekday_before_cutoff():
    assert promised_date(ist(2026, 10, 6, 13, 59)) == date(2026, 10, 7)


def test_cutoff_is_two_pm_india_time():
    # 14:00 IST is 08:30 UTC: the UTC hour is below 14 but the order is after the cutoff
    assert promised_date(ist(2026, 10, 6, 14, 0)) == date(2026, 10, 8)
    assert promised_date(ist(2026, 10, 6, 19, 0)) == date(2026, 10, 8)


def test_late_night_order_uses_india_date():
    # 00:30 IST on Wed 7 Oct is still Tue 6 Oct in UTC
    t = ist(2026, 10, 7, 0, 30)
    assert t.date() == date(2026, 10, 6)
    assert promised_date(t) == date(2026, 10, 8)
    # 23:45 IST Tue 6 Oct -> after cutoff -> dispatch Wed -> deliver Thu
    assert promised_date(ist(2026, 10, 6, 23, 45)) == date(2026, 10, 8)


def test_any_timezone_is_accepted():
    t = datetime(2026, 10, 6, 12, 0, tzinfo=timezone(timedelta(hours=-5)))  # 22:30 IST
    assert promised_date(t) == date(2026, 10, 8)


def test_holidays_are_skipped():
    holidays = {date(2026, 10, 7)}
    assert promised_date(ist(2026, 10, 6, 10, 0), holidays) == date(2026, 10, 8)


def test_holiday_next_to_sunday():
    # dispatch Fri 2 Oct; Sat 3 holiday, Sun 4 Sunday, Mon 5 holiday -> Tue 6
    holidays = [date(2026, 10, 3), date(2026, 10, 5)]
    assert promised_date(ist(2026, 10, 2, 11, 0), holidays) == date(2026, 10, 6)


def test_holidays_as_generator_or_tuple():
    gen = (d for d in [date(2026, 10, 8), date(2026, 10, 7)])
    assert promised_date(ist(2026, 10, 6, 10, 0), gen) == date(2026, 10, 9)
    assert promised_date(ist(2026, 10, 6, 10, 0), (date(2026, 10, 7),)) == date(2026, 10, 8)


def test_never_promises_a_sunday():
    start = ist(2026, 1, 1, 0, 0)
    for hours in range(0, 24 * 60, 5):
        d = promised_date(start + timedelta(hours=hours))
        assert d.weekday() != 6


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        promised_date(datetime(2026, 10, 6, 10, 0))


def test_returns_date_and_does_not_depend_on_today():
    d = promised_date(ist(2020, 2, 28, 10, 0))  # leap year: Fri 28 Feb -> Sat 29 Feb
    assert d == date(2020, 2, 29)
    assert type(d) is date


def test_year_end():
    # Thu 31 Dec 2026 15:00 IST -> dispatch Fri 1 Jan -> deliver Sat 2 Jan 2027
    assert promised_date(ist(2026, 12, 31, 15, 0)) == date(2027, 1, 2)

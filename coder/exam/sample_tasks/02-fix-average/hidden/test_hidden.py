import pytest

from stats import average


def test_two_numbers():
    assert average([2, 4]) == 3


def test_one_number():
    assert average([5]) == 5


def test_floats():
    assert average([1.5, 2.5]) == pytest.approx(2.0)


def test_no_numbers():
    with pytest.raises(ValueError):
        average([])

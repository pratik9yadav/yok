import pytest

from words import count_words


def test_example():
    assert count_words("Hi, hi there!") == {"hi": 2, "there": 1}


def test_empty_and_only_punctuation():
    assert count_words("") == {}
    assert count_words("... !!! ,") == {}


def test_case_spaces_and_lines():
    assert count_words("Hello\n\thello   WORLD") == {"hello": 2, "world": 1}


def test_not_text():
    with pytest.raises(TypeError):
        count_words(None)

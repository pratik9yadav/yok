import pytest

from words import count_words


def test_counts_ignoring_case_and_punctuation():
    assert count_words("Hi, hi there!") == {"hi": 2, "there": 1}


def test_nothing_to_count():
    assert count_words("") == {}


def test_rejects_non_text():
    with pytest.raises(TypeError):
        count_words(42)

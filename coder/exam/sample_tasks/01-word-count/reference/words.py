"""Count the words in a text."""

import re
from collections import Counter

WORD = re.compile(r"[^\W_]+")


def count_words(text: str) -> dict[str, int]:
    """How many times each word appears in text, ignoring case and punctuation."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, not {type(text).__name__}")
    return dict(Counter(WORD.findall(text.lower())))

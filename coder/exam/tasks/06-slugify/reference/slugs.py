"""URL-safe slugs for product names, with collision handling."""

from __future__ import annotations

import re
import unicodedata

FALLBACK = "item"
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _check_max_length(max_length: int) -> int:
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < len(FALLBACK):
        raise ValueError(f"max_length must be an integer >= {len(FALLBACK)}, got {max_length!r}")
    return max_length


def _cut(slug: str, max_length: int) -> str:
    return slug[:max_length].strip("-")


def slugify(text: str, max_length: int = 60) -> str:
    """Turn text into a lowercase ASCII slug of at most max_length characters."""
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    _check_max_length(max_length)
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _NON_ALNUM.sub("-", ascii_text.lower()).strip("-")
    return _cut(slug, max_length) or FALLBACK


class SlugRegistry:
    """Hands out unique slugs. Each registry keeps its own set of claimed slugs."""

    def __init__(self, max_length: int = 60) -> None:
        self.max_length = _check_max_length(max_length)
        self._taken: set[str] = set()
        self._next_suffix: dict[str, int] = {}

    def claim(self, text: str) -> str:
        base = slugify(text, self.max_length)
        if base not in self._taken:
            self._taken.add(base)
            return base
        n = self._next_suffix.get(base, 2)
        while True:
            suffix = f"-{n}"
            room = self.max_length - len(suffix)
            candidate = (_cut(base, room) or FALLBACK[:room]) + suffix
            n += 1
            if candidate not in self._taken:
                break
        self._next_suffix[base] = n
        self._taken.add(candidate)
        return candidate

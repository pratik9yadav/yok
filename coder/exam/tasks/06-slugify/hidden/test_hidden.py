import re
import time

import pytest

from slugs import SlugRegistry, slugify

SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


@pytest.mark.parametrize("text, expected", [
    ("Tata Salt 1kg", "tata-salt-1kg"),
    ("  Amul -- Butter!!  ", "amul-butter"),
    ("Café Crème", "cafe-creme"),
    ("Parle-G (Pack of 10)", "parle-g-pack-of-10"),
    ("MAGGI_Masala/Noodles", "maggi-masala-noodles"),
])
def test_basic_slugs(text, expected):
    assert slugify(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "!!!", "नमक", "---"])
def test_empty_result_falls_back_to_item(text):
    assert slugify(text) == "item"


def test_mixed_script_keeps_latin_part():
    assert slugify("Namak नमक 1kg") == "namak-1kg"


def test_max_length_default_and_custom():
    assert len(slugify("a" * 100)) == 60
    assert slugify("abcd efgh", max_length=5) == "abcd"  # no dangling dash after cutting
    assert slugify("abcd efgh", max_length=6) == "abcd-e"


def test_slugify_is_idempotent_and_deterministic():
    for text in ["Tata Salt 1kg", "Café Crème", "a" * 80, "x - y - z"]:
        once = slugify(text)
        assert slugify(once) == once
        assert slugify(text) == once


def test_registry_suffixes():
    reg = SlugRegistry()
    assert [reg.claim("Tata Salt") for _ in range(3)] == ["tata-salt", "tata-salt-2", "tata-salt-3"]
    assert reg.claim("Amul Butter") == "amul-butter"


def test_registry_avoids_slugs_that_look_like_suffixes():
    reg = SlugRegistry()
    got = [reg.claim("Salt 2"), reg.claim("Salt"), reg.claim("Salt"), reg.claim("Salt"), reg.claim("salt-2")]
    assert len(set(got)) == len(got)
    assert got[0] == "salt-2" and got[1] == "salt"
    assert all(SLUG.match(s) for s in got)


def test_registry_respects_max_length():
    reg = SlugRegistry(max_length=12)
    got = [reg.claim("Super Long Product Name") for _ in range(15)]
    assert len(set(got)) == 15
    assert all(len(s) <= 12 for s in got)
    assert all(SLUG.match(s) for s in got)


def test_registry_fallback_names_are_unique():
    reg = SlugRegistry()
    got = [reg.claim("नमक") for _ in range(3)] + [reg.claim("") for _ in range(2)]
    assert got[0] == "item"
    assert len(set(got)) == 5
    assert all(SLUG.match(s) for s in got)


def test_registries_do_not_share_state():
    a = SlugRegistry()
    b = SlugRegistry()
    assert a.claim("Tata Salt") == "tata-salt"
    assert b.claim("Tata Salt") == "tata-salt"


def test_many_same_names_are_fast():
    reg = SlugRegistry()
    start = time.perf_counter()
    got = [reg.claim("Parle-G") for _ in range(10_000)]
    assert time.perf_counter() - start < 1.5
    assert len(set(got)) == 10_000
    assert got[-1] == "parle-g-10000"

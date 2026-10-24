import base64
import json
import os
import pickle
import time

import pytest

from cart_session import MAX_QTY, add_item, load_cart, parse_qty, save_cart

COOKIE_UNSAFE = set(' \t\r\n;,"\\')


class Evil:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (os.mkdir, (self.path,))


def encodings_of(obj):
    """The same payload in the ways a cookie could plausibly be encoded."""
    raw = json.dumps(obj).encode("utf-8")
    return [
        base64.urlsafe_b64encode(raw).decode("ascii"),
        base64.b64encode(raw).decode("ascii"),
        raw.decode("utf-8"),
    ]


def test_round_trip():
    cart = {"ATTA-5KG": 2, "TOOR_DAL.1KG": 1, "MAGGI-12PK": MAX_QTY}
    cookie = save_cart(cart)
    assert isinstance(cookie, str)
    assert load_cart(cookie) == cart


def test_cookie_value_is_cookie_safe():
    cookie = save_cart({"ATTA-5KG": 2, "SUGAR-1KG": 3})
    assert not (set(cookie) & COOKIE_UNSAFE)
    assert cookie.isascii()


def test_empty_cookie_is_empty_cart():
    assert load_cart("") == {}
    assert load_cart(save_cart({})) == {}


def test_old_pickle_cookie_is_rejected_without_running_code(tmp_path):
    marker = tmp_path / "pwned"
    cookie = base64.urlsafe_b64encode(pickle.dumps(Evil(str(marker)))).decode("ascii")
    with pytest.raises(ValueError):
        load_cart(cookie)
    assert not marker.exists()


def test_plain_pickled_cart_is_rejected():
    cookie = base64.urlsafe_b64encode(pickle.dumps({"ATTA-5KG": 2})).decode("ascii")
    with pytest.raises(ValueError):
        load_cart(cookie)


@pytest.mark.parametrize("cookie", ["!!!", "abc", "e30", "%7B%7D", "a" * 10, "=" * 8, "अटा"])
def test_garbage_cookie_raises_value_error(cookie):
    with pytest.raises(ValueError):
        load_cart(cookie)


@pytest.mark.parametrize(
    "bad",
    [
        [1, 2, 3],
        "ATTA-5KG",
        42,
        None,
        {"ATTA-5KG": -5},
        {"ATTA-5KG": 0},
        {"ATTA-5KG": MAX_QTY + 1},
        {"ATTA-5KG": 1_000_000_000},
        {"ATTA-5KG": "2"},
        {"ATTA-5KG": 1.5},
        {"ATTA-5KG": True},
        {"ATTA-5KG": None},
        {"ATTA-5KG": {"nested": 1}},
    ],
)
def test_tampered_cart_never_loads(bad):
    for cookie in encodings_of(bad):
        with pytest.raises(ValueError):
            load_cart(cookie)
    try:
        cookie = save_cart(bad)
    except (ValueError, TypeError):
        return
    with pytest.raises(ValueError):
        load_cart(cookie)


@pytest.mark.parametrize("text,expected", [("3", 3), ("2*12", 24), (" 2 * 6 ", 12), ("99", 99), ("1", 1), ("12*2", 24)])
def test_parse_qty_normal_input(text, expected):
    assert parse_qty(text) == expected


@pytest.mark.parametrize("text", ["0", "100", "-1", "abc", "", "  ", "2*", "*3", "2**3", "50*2", "1.5", "2*0",
                                  "True", "[5]", "0x10"])
def test_parse_qty_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_qty(text)


def test_parse_qty_does_not_run_code(tmp_path):
    marker = tmp_path / "pwned"
    for attack in (
        f"__import__('os').mkdir({str(marker)!r})",
        f"[__import__('os').mkdir({str(marker)!r}), 1][1]",
        f"(lambda: __import__('os').mkdir({str(marker)!r}) or 1)()",
    ):
        with pytest.raises(ValueError):
            parse_qty(attack)
    assert not marker.exists()


def test_parse_qty_huge_expressions_are_rejected_quickly():
    start = time.perf_counter()
    for attack in ("9" * 5000, "99999999*99999999", "9**999999"):
        with pytest.raises(ValueError):
            parse_qty(attack)
    assert time.perf_counter() - start < 1


def test_add_item_still_works_and_caps():
    cart = add_item({}, "ATTA-5KG", "2")
    cart = add_item(cart, "ATTA-5KG", "2*3")
    assert cart == {"ATTA-5KG": 8}
    with pytest.raises(ValueError):
        add_item({"ATTA-5KG": 98}, "ATTA-5KG", "2")
    assert load_cart(save_cart(cart)) == cart

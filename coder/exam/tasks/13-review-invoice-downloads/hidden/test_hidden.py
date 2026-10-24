import contextlib
import gzip
import hashlib
import logging
import random
import re
from pathlib import Path

import pytest

import invoices

PASSWORD = "Gup!t-Pa55word"
USERS = {"ravi": hashlib.sha256(PASSWORD.encode()).hexdigest()}


@pytest.fixture
def shop(tmp_path):
    base = tmp_path / "invoices"
    (base / "2026-10").mkdir(parents=True)
    (base / "2026-10" / "INV-0001.txt").write_bytes(b"Invoice 1: Atta 5kg x2 = 490.00\n")
    (base / "INV 0002.txt").write_bytes(b"Invoice 2\n")
    (tmp_path / "secret.txt").write_bytes(b"DB_PASSWORD=hunter2\n")
    return base


def test_read_invoice_in_month_folder(shop):
    assert invoices.read_invoice(str(shop), "2026-10/INV-0001.txt") == b"Invoice 1: Atta 5kg x2 = 490.00\n"
    assert invoices.read_invoice(shop, "INV 0002.txt") == b"Invoice 2\n"


def test_missing_invoice_raises(shop):
    with pytest.raises(FileNotFoundError):
        invoices.read_invoice(str(shop), "2026-10/INV-9999.txt")


@pytest.mark.parametrize("name", ["../secret.txt", "2026-10/../../secret.txt", "..\\secret.txt"])
def test_read_cannot_escape_folder(shop, name):
    try:
        data = invoices.read_invoice(str(shop), name)
    except Exception:
        return
    assert b"hunter2" not in data


def test_read_rejects_absolute_path(shop):
    secret = shop.parent / "secret.txt"
    try:
        data = invoices.read_invoice(str(shop), str(secret))
    except Exception:
        return
    assert b"hunter2" not in data


def test_compress_works_and_keeps_original(shop):
    out = Path(invoices.compress_invoice(str(shop), "2026-10/INV-0001.txt"))
    assert out == shop / "2026-10" / "INV-0001.txt.gz"
    assert gzip.decompress(out.read_bytes()) == b"Invoice 1: Atta 5kg x2 = 490.00\n"
    assert (shop / "2026-10" / "INV-0001.txt").exists()


def test_compress_name_with_space(shop):
    out = Path(invoices.compress_invoice(str(shop), "INV 0002.txt"))
    assert gzip.decompress(out.read_bytes()) == b"Invoice 2\n"


@pytest.mark.parametrize("name", [
    "INV 0002.txt; touch pwned",
    "INV-0001.txt && touch pwned",
    "$(touch pwned)",
    "`touch pwned`",
])
def test_compress_cannot_run_commands(shop, monkeypatch, name):
    monkeypatch.chdir(shop)
    with contextlib.suppress(Exception):
        invoices.compress_invoice(str(shop), name)
    assert not list(shop.parent.rglob("pwned*"))


def test_compress_cannot_write_outside(shop):
    with contextlib.suppress(Exception):
        invoices.compress_invoice(str(shop), "../secret.txt")
    assert not (shop.parent / "secret.txt.gz").exists()


def test_login_ok_and_failed():
    token = invoices.login("ravi", PASSWORD, USERS)
    assert isinstance(token, str) and token
    assert invoices.login("ravi", "wrong", USERS) is None
    assert invoices.login("nobody", PASSWORD, USERS) is None


def test_password_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    invoices.login("ravi", PASSWORD, USERS)
    invoices.login("ravi", PASSWORD + "x", USERS)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert PASSWORD not in text
    assert "ravi" in text  # still useful logs


def test_tokens_are_url_safe_and_long():
    for _ in range(50):
        t = invoices.new_download_token()
        assert re.fullmatch(r"[A-Za-z0-9_\-]+", t)
        assert len(t) >= 22


def test_tokens_are_unique():
    assert len({invoices.new_download_token() for _ in range(2000)}) == 2000


def test_tokens_cannot_be_predicted_from_random_seed():
    random.seed(1234)
    a = invoices.new_download_token()
    random.seed(1234)
    b = invoices.new_download_token()
    assert a != b

"""Invoice downloads for the shop's customer portal."""

import gzip
import hashlib
import hmac
import logging
import secrets
import shutil
from pathlib import Path

log = logging.getLogger("invoices")


def _safe_path(base_dir, name):
    """Resolve name inside base_dir; refuse anything that points outside it."""
    base = Path(base_dir).resolve()
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base) or candidate == base:
        raise ValueError(f"invalid invoice name: {name!r}")
    return candidate


def read_invoice(base_dir, name):
    """Return the bytes of invoice `name` stored under base_dir.

    name comes from the download link and may include the month folder,
    e.g. "2026-10/INV-0001.txt".
    """
    return _safe_path(base_dir, name).read_bytes()


def compress_invoice(base_dir, name):
    """Write a gzip copy of the invoice next to it (name + ".gz").

    The original file is kept. Returns the path of the .gz file.
    """
    path = _safe_path(base_dir, name)
    target = path.with_name(path.name + ".gz")
    with open(path, "rb") as src, gzip.open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return str(target)


def new_download_token():
    """A random token for a one-time download link (URL-safe)."""
    return secrets.token_urlsafe(32)


def login(username, password, users):
    """Check a password; users maps username -> sha256 hex digest of the password.

    Returns a new download token on success, None otherwise.
    """
    log.info("login attempt user=%s", username)
    expected = users.get(username)
    actual = hashlib.sha256(password.encode()).hexdigest()
    if expected and hmac.compare_digest(actual, expected):
        log.info("login ok for %s", username)
        return new_download_token()
    log.warning("failed login for %s", username)
    return None

"""Invoice downloads for the shop's customer portal."""

import hashlib
import logging
import os
import random
import string
import subprocess

log = logging.getLogger("invoices")


def read_invoice(base_dir, name):
    """Return the bytes of invoice `name` stored under base_dir.

    name comes from the download link and may include the month folder,
    e.g. "2026-10/INV-0001.txt".
    """
    with open(os.path.join(base_dir, name), "rb") as f:
        return f.read()


def compress_invoice(base_dir, name):
    """Write a gzip copy of the invoice next to it (name + ".gz").

    The original file is kept. Returns the path of the .gz file.
    """
    path = os.path.join(base_dir, name)
    subprocess.run(f"gzip -kf {path}", shell=True, check=True)
    return path + ".gz"


def new_download_token():
    """A random token for a one-time download link (URL-safe)."""
    return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(32))


def login(username, password, users):
    """Check a password; users maps username -> sha256 hex digest of the password.

    Returns a new download token on success, None otherwise.
    """
    log.info("login attempt user=%s password=%s", username, password)
    expected = users.get(username)
    if expected and hashlib.sha256(password.encode()).hexdigest() == expected:
        log.info("login ok for %s", username)
        return new_download_token()
    log.warning("failed login for %s", username)
    return None

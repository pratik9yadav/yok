"""Customer records for the shop app, stored in SQLite."""

import os
import sqlite3
import urllib.parse
import urllib.request

SMS_URL = "https://sms.example.in/send"
SMS_KEY_ENV = "SMS_API_KEY"  # provided by ops in the server environment; never commit the key


class DuplicateCustomer(Exception):
    """A customer with this phone number is already registered."""


def _row_to_dict(row):
    return {"id": row[0], "name": row[1], "phone": row[2], "city": row[3]}


class CustomerStore:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS customers ("
                "id INTEGER PRIMARY KEY, name TEXT NOT NULL, phone TEXT NOT NULL UNIQUE, city TEXT)"
            )

    def close(self):
        self.conn.close()

    def _insert(self, name, phone, city):
        try:
            cur = self.conn.execute(
                "INSERT INTO customers (name, phone, city) VALUES (?, ?, ?)", (name, phone, city)
            )
        except sqlite3.IntegrityError as exc:
            if phone is not None and self.get_by_phone(phone) is not None:
                raise DuplicateCustomer(f"phone {phone} is already registered") from exc
            raise
        return cur.lastrowid

    def add(self, name, phone, city=None):
        """Add a customer and return the new id.

        Raises DuplicateCustomer if the phone number is already registered.
        """
        with self.conn:  # commits on success, rolls back on error
            return self._insert(name, phone, city)

    def get_by_phone(self, phone):
        """The customer with this phone number as a dict, or None."""
        row = self.conn.execute(
            "SELECT id, name, phone, city FROM customers WHERE phone = ?", (phone,)
        ).fetchone()
        return None if row is None else _row_to_dict(row)

    def search(self, text):
        """Customers whose name contains text (ignoring case), sorted by name."""
        pattern = "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        rows = self.conn.execute(
            "SELECT id, name, phone, city FROM customers WHERE name LIKE ? ESCAPE '\\' ORDER BY name",
            (pattern,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def import_many(self, customers):
        """Add many customers (dicts with name, phone and optional city).

        All or nothing: if any customer cannot be added, none are added and
        the error is raised. Returns the number of customers added.
        """
        added = 0
        with self.conn:  # one transaction: rolled back if any insert fails
            for c in customers:
                self._insert(c["name"], c["phone"], c.get("city"))
                added += 1
        return added


def send_welcome_sms(phone, name, opener=urllib.request.urlopen):
    """Send the welcome SMS through the gateway."""
    key = os.environ.get(SMS_KEY_ENV)
    if not key:
        raise RuntimeError(f"{SMS_KEY_ENV} is not set; cannot send SMS")
    query = urllib.parse.urlencode(
        {"key": key, "to": phone, "msg": f"Welcome to Sharma Kirana, {name}!"}
    )
    return opener(f"{SMS_URL}?{query}")

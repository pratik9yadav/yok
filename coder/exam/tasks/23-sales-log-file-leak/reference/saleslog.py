"""Day-wise sales log: one CSV file per day (`<day>.csv`, lines `order_id,amount_paise`).

Files are opened only for the duration of one operation (`with` blocks), so nothing leaks
file descriptors and every recorded sale is on disk as soon as `record` returns.
"""

import os
import threading


class SalesLog:
    def __init__(self, folder):
        self.folder = folder
        self._lock = threading.Lock()

    def _path(self, day):
        return os.path.join(self.folder, f"{day}.csv")

    def record(self, day, order_id, amount_paise):
        """Append one sale. `day` is like "2024-11-01"."""
        line = f"{order_id},{amount_paise}\n"
        with self._lock, open(self._path(day), "a", encoding="utf-8") as f:
            f.write(line)

    def total(self, day):
        """Total paise sold on `day` (0 if nothing was sold). Raises ValueError on a corrupt line."""
        path = self._path(day)
        if not os.path.exists(path):
            return 0
        total = 0
        with open(path, encoding="utf-8") as f:
            for n, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 2:
                    raise ValueError(f"{path} line {n}: expected order_id,amount_paise")
                try:
                    total += int(parts[1])
                except ValueError:
                    raise ValueError(f"{path} line {n}: bad amount {parts[1]!r}") from None
        return total

    def close(self):
        """Nothing stays open between calls; kept so existing callers keep working."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

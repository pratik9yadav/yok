"""Day-wise sales log: one CSV file per day (`<day>.csv`, lines `order_id,amount_paise`)."""

import os


class SalesLog:
    def __init__(self, folder):
        self.folder = folder
        self._files = {}

    def _path(self, day):
        return os.path.join(self.folder, f"{day}.csv")

    def _file_for(self, day):
        # keep the file open so we don't reopen it for every sale
        if day not in self._files:
            self._files[day] = open(self._path(day), "a", encoding="utf-8")
        return self._files[day]

    def record(self, day, order_id, amount_paise):
        """Append one sale. `day` is like "2024-11-01"."""
        f = self._file_for(day)
        f.write(f"{order_id},{amount_paise}\n")

    def total(self, day):
        """Total paise sold on `day` (0 if nothing was sold). Raises ValueError on a corrupt line."""
        path = self._path(day)
        if not os.path.exists(path):
            return 0
        f = open(path, encoding="utf-8")
        total = 0
        for n, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            order_id, amount = line.split(",")
            try:
                total += int(amount)
            except ValueError:
                raise ValueError(f"{path} line {n}: bad amount {amount!r}")
        return total

    def close(self):
        for f in self._files.values():
            f.close()

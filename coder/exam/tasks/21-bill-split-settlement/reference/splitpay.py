"""Split bills and settle balances in integer paise (no floating point anywhere)."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence


def _require_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an integer number of paise, got {value!r}")
    return value


def split_bill(total_paise: int, weights: Sequence[int]) -> list[int]:
    """Split `total_paise` by `weights` with the largest-remainder method.

    Every part is the exact share rounded down or up, the parts sum to the total, and the
    leftover paise go to the largest remainders (earlier people win ties).
    """
    _require_int(total_paise, "total_paise")
    weights = list(weights)
    if not weights:
        raise ValueError("need at least one weight")
    for w in weights:
        if isinstance(w, bool) or not isinstance(w, int):
            raise TypeError(f"weights must be whole numbers, got {w!r}")
        if w < 0:
            raise ValueError("weights must not be negative")
    weight_sum = sum(weights)
    if weight_sum == 0:
        raise ValueError("at least one weight must be positive")

    sign = -1 if total_paise < 0 else 1
    total = abs(total_paise)
    parts = []
    remainders = []
    for i, w in enumerate(weights):
        q, r = divmod(total * w, weight_sum)
        parts.append(q)
        remainders.append((-r, i))
    leftover = total - sum(parts)
    for _, i in sorted(remainders)[:leftover]:
        parts[i] += 1
    return [sign * p for p in parts]


def settle(balances: Mapping[str, int]) -> list[tuple[str, str, int]]:
    """Return payments (payer, payee, amount) that bring every balance to zero.

    Greedy: the biggest debtor pays the biggest creditor. Uses at most n-1 payments and is
    deterministic (ties are broken by name, never by dict order).
    """
    for name, amount in balances.items():
        if not isinstance(name, str):
            raise TypeError(f"names must be strings, got {name!r}")
        _require_int(amount, f"balance of {name}")
    if sum(balances.values()) != 0:
        raise ValueError(f"balances must add up to zero, they add up to {sum(balances.values())}")

    creditors = [(-amount, name) for name, amount in balances.items() if amount > 0]
    debtors = [(amount, name) for name, amount in balances.items() if amount < 0]
    heapq.heapify(creditors)
    heapq.heapify(debtors)
    payments: list[tuple[str, str, int]] = []
    while creditors and debtors:
        neg_credit, payee = heapq.heappop(creditors)
        debt, payer = heapq.heappop(debtors)
        amount = min(-neg_credit, -debt)
        payments.append((payer, payee, amount))
        if -neg_credit > amount:
            heapq.heappush(creditors, (neg_credit + amount, payee))
        if -debt > amount:
            heapq.heappush(debtors, (debt + amount, payer))
    return payments

"""Shared by the notebook and the tests: the task's prompt, and how an answer is scored.

The task: read what a shopkeeper said on a phone call and return JSON with four fields.
"""

import json
import re

SYSTEM = (
    "You read what a shopkeeper said on a phone call about a product. "
    "Reply with only a JSON object with these keys: "
    "in_stock (true, false or null), price_inr (a number or null), "
    "delivery (true, false or null), alternative (text or null). "
    "price_inr is the price of the product that was asked for. "
    "Use null when the shopkeeper did not say."
)

FIELDS = ("in_stock", "price_inr", "delivery", "alternative")


def user_message(product: str, said: str) -> str:
    return f"Product: {product}\nShopkeeper: {said}"


def answer_text(answer: dict) -> str:
    """The exact text the model is trained to reply with."""
    return json.dumps({k: answer[k] for k in FIELDS}, ensure_ascii=False)


def parse(text: str) -> dict | None:
    """The first JSON object in the model's reply, or None if there is none."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def same(field: str, got, want) -> bool:
    if want is None or got is None:
        return got is None and want is None
    if field == "price_inr":
        try:
            return float(got) == float(want)
        except (TypeError, ValueError):
            return False
    if field == "alternative":
        # Wording may differ ("40 mm" vs "40 mm lock"); one containing the other counts.
        got, want = _norm(got), _norm(want)
        return want in got or got in want
    return got == want


def score(replies: list[str], examples: list[dict]) -> dict:
    """Per-field accuracy, all-four-right accuracy, and the mistakes."""
    right = {f: 0 for f in FIELDS}
    all_right = json_ok = 0
    mistakes = []
    for reply, ex in zip(replies, examples, strict=True):
        got = parse(reply)
        if got is not None:
            json_ok += 1
        got = got or {}
        wrong = [f for f in FIELDS if not same(f, got.get(f), ex["answer"][f])]
        for f in FIELDS:
            right[f] += f not in wrong
        if wrong:
            mistakes.append({"said": ex["said"], "want": ex["answer"], "reply": reply, "wrong": wrong})
        else:
            all_right += 1
    n = len(examples)
    return {
        "examples": n,
        "json_ok": json_ok / n,
        "all_correct": all_right / n,
        **{f: right[f] / n for f in FIELDS},
        "mistakes": mistakes,
    }

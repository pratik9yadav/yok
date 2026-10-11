"""Shared by the notebook and the tests: the task's prompt, and how an answer is scored.

The task: read what a shopkeeper said on a phone call and return JSON with four fields.
"""

import json
import re
import time

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
    """The last JSON object in the reply that has exactly the four keys, or None.

    Every "{" is tried, so braces inside a value ("M-seal {2 pack}") or other text around the
    answer do not hide it.
    """
    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and set(value) == set(FIELDS):
            found = value
    return found


def _tokens(text: str) -> set[str]:
    """Lower-case words, with "40mm" read as "40 mm" and punctuation dropped."""
    text = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", text.lower())
    return set(re.sub(r"[-_.,;:!?()\[\]{}\"'/]", " ", text).split())


def same(field: str, got, want) -> bool:
    if want is None or got is None:
        return got is None and want is None
    if field in ("in_stock", "delivery"):
        return isinstance(got, bool) and got == want  # 1 and 0 are not true and false
    if field == "price_inr":
        if isinstance(got, bool):
            return False
        try:
            return float(got) == float(want)
        except (TypeError, ValueError):
            return False
    # alternative: the wanted words must all be there, with at most two extra ("40 mm padlock").
    if not isinstance(got, str):
        return False
    have, need = _tokens(got), _tokens(want)
    return bool(have) and need <= have and len(have - need) <= 2


def score(replies: list[str], examples: list[dict]) -> dict:
    """Per-field accuracy, all-four-right accuracy, and the mistakes.

    A reply with no valid answer object (or a missing or extra key) gets every field wrong.
    """
    right = {f: 0 for f in FIELDS}
    all_right = json_ok = 0
    mistakes = []
    for reply, ex in zip(replies, examples, strict=True):
        got = parse(reply)
        if got is not None:
            json_ok += 1
        wrong = list(FIELDS) if got is None else [f for f in FIELDS if not same(f, got[f], ex["answer"][f])]
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


KEYS = ("all_correct", "json_ok", *FIELDS)


def evaluate(ask, sets: dict[str, list[dict]], deadline: float | None = None) -> dict:
    """Run ask(product, said) over every test set and score it. Every reply is kept.

    After `deadline` (a time.time() value) no more questions are asked; what was answered is
    scored, and "of" says how many there were.
    """
    out = {}
    for name, rows in sets.items():
        replies = []
        for ex in rows:
            if deadline is not None and time.time() > deadline:
                break
            replies.append(ask(ex["product"], ex["said"]))
        done = rows[: len(replies)]
        scored = score(replies, done) if done else {"examples": 0, "mistakes": []}
        out[name] = {**scored, "replies": replies, "of": len(rows)}
    return out


def show(label: str, results: dict, max_mistakes: int = 8) -> None:
    for name, res in results.items():
        n = res["examples"]
        if not n:
            print(f"== {label} [{name}]: no calls answered")
            continue
        partial = f" (only {n} of {res['of']} answered)" if n < res["of"] else ""
        print(f"== {label} [{name}]: all 4 fields right in {res['all_correct']:.0%} of {n} calls{partial}")
        print("   " + "   ".join(f"{k} {res[k]:.0%}" for k in KEYS[1:]))
        for m in res["mistakes"][:max_mistakes]:
            print("\n   said: ", m["said"])
            print("   want: ", answer_text(m["want"]))
            print("   reply:", m["reply"][:200])
            print("   wrong:", ", ".join(m["wrong"]))


def results_payload(model: str, epochs: int, before: dict, after: dict) -> dict:
    """What results.json holds: scores, mistakes and every raw reply, so scores can be recomputed."""
    sets = {}
    for name, a in after.items():
        b = before[name]
        sets[name] = {
            "examples": a["of"],
            "before": {k: round(b[k], 3) for k in KEYS},
            "after": {k: round(a[k], 3) for k in KEYS},
            "mistakes_before": b["mistakes"],
            "mistakes_after": a["mistakes"],
            "replies_before": b["replies"],
            "replies_after": a["replies"],
        }
    return {"model": model, "epochs": epochs, "sets": sets}


def make_modelfile(gguf_name: str) -> str:
    """An Ollama Modelfile whose template is exactly the text the model was trained on.

    Unsloth's own Qwen3 template differs a little (a blank line after the system message, and
    thinking-mode syntax that needs a newer Ollama), so it is not used.
    """
    return (
        f"FROM ./{gguf_name}\n"
        'TEMPLATE """{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}'
        '{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}'
        '<|im_start|>assistant\n{{ .Response }}<|im_end|>\n"""\n'
        'PARAMETER stop "<|im_end|>"\n'
        "PARAMETER temperature 0\n"
        f'SYSTEM """{SYSTEM}"""\n'
    )

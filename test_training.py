"""Checks the data files and the scoring.  Run:  python -m pytest"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from make_data import WORDS  # noqa: E402
from score import FIELDS, SYSTEM, answer_text, parse, score  # noqa: E402


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / "data" / name).read_text(encoding="utf-8").splitlines()]


def test_data_files_are_well_formed() -> None:
    for name in ("train.jsonl", "test.jsonl"):
        for ex in load(name):
            assert set(ex) == {"product", "said", "answer"}
            assert set(ex["answer"]) == set(FIELDS)
            a = ex["answer"]
            assert a["in_stock"] in (True, False, None)
            assert a["delivery"] in (True, False, None)
            assert a["price_inr"] is None or isinstance(a["price_inr"], int)
            if a["in_stock"] is not True:
                assert a["price_inr"] is None


def test_train_price_is_actually_said() -> None:
    for ex in load("train.jsonl"):
        p = ex["answer"]["price_inr"]
        if p is not None:
            said_as = {str(p)} | {words[p] for words in WORDS.values() if p in words}
            assert any(s in ex["said"] for s in said_as), ex


def test_test_set_is_not_in_training_data() -> None:
    train = {ex["said"] for ex in load("train.jsonl")}
    assert not train & {ex["said"] for ex in load("test.jsonl")}


def test_perfect_replies_score_full_marks() -> None:
    test = load("test.jsonl")
    result = score([answer_text(ex["answer"]) for ex in test], test)
    assert result["all_correct"] == 1.0 and result["json_ok"] == 1.0 and not result["mistakes"]


def test_scoring_catches_mistakes_and_reads_fenced_json() -> None:
    ex = {"said": "x", "answer": {"in_stock": True, "price_inr": 60, "delivery": None, "alternative": None}}
    fenced = '```json\n{"in_stock": true, "price_inr": "60", "delivery": null, "alternative": null}\n```'
    wrong = '{"in_stock": true, "price_inr": 6, "delivery": false, "alternative": null}'
    result = score([fenced, wrong, "no json here"], [ex, ex, ex])
    assert result["json_ok"] == 2 / 3
    assert result["all_correct"] == 1 / 3
    assert result["mistakes"][0]["wrong"] == ["price_inr", "delivery"]
    assert parse("no json here") is None


def test_alternative_wording_can_differ() -> None:
    ex = {"said": "x", "answer": {"in_stock": False, "price_inr": None, "delivery": None, "alternative": "40 mm"}}
    reply = '{"in_stock": false, "price_inr": null, "delivery": null, "alternative": "40 mm padlock"}'
    assert score([reply], [ex])["all_correct"] == 1.0


def test_notebook_is_built_from_current_files() -> None:
    nb = json.loads((HERE / "train_colab.ipynb").read_text(encoding="utf-8"))
    data_cell = "".join(next(c for c in nb["cells"] if "TRAIN_DATA = " in "".join(c["source"]))["source"])
    scope: dict = {}
    exec(data_cell, scope)  # noqa: S102 - our own generated cell
    assert scope["TRAIN_DATA"] == load("train.jsonl")
    assert scope["TEST"] == load("test.jsonl")
    assert scope["SYSTEM"] == SYSTEM

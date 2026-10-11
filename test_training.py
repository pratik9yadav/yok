"""Checks the data files and the scoring.  Run:  python -m pytest"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from make_data import WORDS
from score import FIELDS, KEYS, SYSTEM, answer_text, evaluate, parse, results_payload, score, show


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / "data" / name).read_text(encoding="utf-8").splitlines()]


def test_data_files_are_well_formed() -> None:
    for name in ("train.jsonl", "test.jsonl", "hard_test.jsonl"):
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


def test_test_sets_are_not_in_training_data_or_each_other() -> None:
    train, test, hard = (load(n) for n in ("train.jsonl", "test.jsonl", "hard_test.jsonl"))
    train_said, test_said, hard_said = ({ex["said"] for ex in rows} for rows in (train, test, hard))
    assert not train_said & test_said and not train_said & hard_said and not test_said & hard_said
    assert len(hard_said) == len(hard) == 56


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
    assert scope["HARD"] == load("hard_test.jsonl")
    assert scope["SETS"] == {"test": scope["TEST"], "hard": scope["HARD"]}
    assert scope["SYSTEM"] == SYSTEM


class FakeModel:
    """Stands in for the trained model: writes the files Unsloth would write."""

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def save_pretrained_gguf(self, path: str, tokenizer, quantization_method: str) -> dict:
        out = Path(path + "_gguf")
        out.mkdir(parents=True)
        (out / "model.BF16.gguf").write_text("x" * 100)
        (out / "model.Q4_K_M.gguf").write_text("x" * 30)
        (out / "Modelfile").write_text('FROM model.BF16.gguf\nTEMPLATE """{{ .Prompt }}"""\nPARAMETER temperature 0.6\n')
        return {"gguf_files": [str(out / "model.BF16.gguf"), str(out / "model.Q4_K_M.gguf")],
                "modelfile_location": str(out / "Modelfile")}


def test_save_cell_keeps_the_small_model_and_a_working_modelfile(tmp_path: Path) -> None:
    import shutil

    nb = json.loads((HERE / "train_colab.ipynb").read_text(encoding="utf-8"))
    save = "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith("#@title 10."))["source"])
    scope = {"model": FakeModel(), "tokenizer": FakeModel(), "shutil": shutil, "os": __import__("os"),
             "SYSTEM": SYSTEM, "OUTPUT_NAME": "m", "MAKE_OLLAMA_FILE": True,
             "WORK": str(tmp_path / "work"), "SAVE_DIR": str(tmp_path / "out")}
    exec(save, scope)  # noqa: S102 - our own generated cell
    saved = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert saved == ["Modelfile", "lora", "m.Q4_K_M.gguf"]
    modelfile = (tmp_path / "out" / "Modelfile").read_text()
    assert modelfile.startswith("FROM ./m.Q4_K_M.gguf\n") and modelfile.count("FROM ") == 1
    assert "PARAMETER temperature 0\n" in modelfile and "0.6" not in modelfile
    assert f'SYSTEM """{SYSTEM}"""' in modelfile


def test_kaggle_runner_pushes_waits_and_fetches(tmp_path: Path, monkeypatch, capsys) -> None:
    import enum
    from types import SimpleNamespace

    import run_on_kaggle

    Status = enum.Enum("Status", "RUNNING COMPLETE")
    calls: dict = {}

    class FakeApi:
        def kernels_push(self, folder, acc=None):
            calls["meta"] = json.loads((Path(folder) / "kernel-metadata.json").read_text())
            calls["acc"] = acc
            calls["has_notebook"] = (Path(folder) / "train_colab.ipynb").exists()
            return SimpleNamespace(error=None, url="https://kaggle/x")

        def kernels_status(self, kernel):
            calls["polls"] = calls.get("polls", 0) + 1
            return SimpleNamespace(status=Status.COMPLETE if calls["polls"] > 1 else Status.RUNNING,
                                   failure_message="")

        def kernels_output(self, kernel, path, file_pattern=None, force=False):
            row = {f: 1.0 for f in KEYS}
            (Path(path) / "results.json").write_text(json.dumps(
                {"sets": {"test": {"examples": 30, "before": row, "after": row, "mistakes_after": []}}}))
            (Path(path) / "results_gguf.json").write_text(json.dumps(
                {"gguf": "m.gguf", "seconds": 5, "sets": {"test": {"examples": 30, "of": 30, "all_correct": 1.0}}}))

        def kernels_list_files(self, kernel, page_size=20):
            return SimpleNamespace(files=[SimpleNamespace(name="wimb-shop-model/m.Q4_K_M.gguf", size=2.5e9)])

    monkeypatch.setattr(run_on_kaggle, "OUT", tmp_path / "out")
    monkeypatch.setattr(run_on_kaggle.time, "sleep", lambda s: None)
    api = FakeApi()
    run_on_kaggle.push(api, "me/wimb-shop-model-training")
    assert run_on_kaggle.wait(api, "me/wimb-shop-model-training") is True
    run_on_kaggle.fetch(api, "me/wimb-shop-model-training")
    out = capsys.readouterr().out
    assert "test set: 30 calls" in out and "m.Q4_K_M.gguf  2,500 MB" in out and "GGUF file m.gguf" in out
    assert calls["acc"] == "NvidiaTeslaT4" and calls["has_notebook"]
    assert calls["meta"]["id"] == "me/wimb-shop-model-training" and calls["meta"]["enable_internet"]
    assert (tmp_path / "out" / "results.json").exists()


NULL = {"in_stock": None, "price_inr": None, "delivery": None, "alternative": None}


def one(reply: str, **want) -> dict:
    return score([reply], [{"said": "x", "answer": {**NULL, **want}}])


def reply(**kw) -> str:
    return json.dumps({**NULL, **kw})


def test_garbage_is_never_correct_even_on_all_null_rows() -> None:
    for garbage in ("", "{}", "no json", '{"in_stock": true}', reply(note="x")[:-1] + ', "note": "x"}'):
        res = one(garbage)
        assert res["all_correct"] == 0 and res["json_ok"] == 0 and res["mistakes"][0]["wrong"] == list(FIELDS)


def test_booleans_are_not_numbers() -> None:
    res = one('{"in_stock": 1, "price_inr": true, "delivery": 0, "alternative": null}',
              in_stock=True, price_inr=1, delivery=False)
    assert res["mistakes"][0]["wrong"] == ["in_stock", "price_inr", "delivery"]


def test_alternative_must_really_match() -> None:
    for bad in ('""', '"m"', '"4"', "40", '["40 mm"]', '"padlock"'):
        assert one(reply(in_stock=False, alternative=json.loads(bad)), in_stock=False, alternative="40 mm")[
            "all_correct"] == 0, bad
    for good in ("40mm", "40 mm padlock", "40 MM", "Padlock 40 mm"):
        assert one(reply(in_stock=False, alternative=good), in_stock=False, alternative="40 mm")["all_correct"] == 1


def test_parse_finds_the_answer_among_other_text_and_braces() -> None:
    assert parse(reply(alternative="M-seal {2 pack}"))["alternative"] == "M-seal {2 pack}"
    assert parse("<think>shape {a,b}</think>" + reply(price_inr=5))["price_inr"] == 5
    assert parse(reply(in_stock=False) + " " + reply(in_stock=True))["in_stock"] is True  # the last one counts
    assert parse(reply() [:-1] + ', "note": "x"}') is None  # extra key


def test_evaluate_show_and_results_payload(capsys) -> None:
    rows = load("test.jsonl")
    sets = {"test": rows, "hard": load("hard_test.jsonl")}
    said_to_answer = {ex["said"]: answer_text(ex["answer"]) for rows_ in sets.values() for ex in rows_}
    before = evaluate(lambda product, said: "{}", sets)
    after = evaluate(lambda product, said: said_to_answer[said], sets)
    assert before["test"]["all_correct"] == 0 and after["hard"]["all_correct"] == 1.0
    assert len(after["hard"]["replies"]) == 56 and after["test"]["of"] == 30
    payload = results_payload("m", 2, before, after)
    json.dumps(payload)  # must be JSON-safe
    assert payload["sets"]["hard"]["examples"] == 56 and payload["sets"]["hard"]["after"]["all_correct"] == 1.0
    assert len(payload["sets"]["test"]["replies_after"]) == 30
    show("BEFORE", before)
    assert "BEFORE [test]: all 4 fields right in 0% of 30 calls" in capsys.readouterr().out
    # a deadline in the past answers nothing, and that is reported rather than crashing
    late = evaluate(lambda product, said: "{}", sets, deadline=0)
    assert late["test"]["examples"] == 0 and late["test"]["of"] == 30
    show("LATE", late)
    assert "no calls answered" in capsys.readouterr().out


def notebook_cell(title_start: str) -> str:
    nb = json.loads((HERE / "train_colab.ipynb").read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith(title_start))["source"])


def data_scope() -> dict:
    scope: dict = {}
    exec(notebook_cell("#@title 4."), scope)  # noqa: S102 - our own generated cell
    return scope


def test_before_and_after_cells_run_with_a_stand_in_model(tmp_path: Path) -> None:
    import contextlib
    from types import SimpleNamespace

    scope = data_scope()
    answers = {ex["said"]: answer_text(ex["answer"]) for rows in scope["SETS"].values() for ex in rows}
    seen = {"said": None, "modes": []}

    class Encoded(dict):
        def to(self, device):
            return self

    class Tokenizer:
        reply = staticmethod(lambda said: "{}")

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["role"] == "system" and messages[1]["content"].startswith("Product: ")
            seen["said"] = messages[1]["content"].split("Shopkeeper: ", 1)[1]
            return Encoded(input_ids=SimpleNamespace(shape=(1, 7)))

        def decode(self, ids, skip_special_tokens=True):
            return "  " + self.reply(seen["said"]) + "\n"

    class Model:
        def generate(self, **inputs):
            assert inputs["max_new_tokens"] == 96 and inputs["do_sample"] is False
            return [[0] * 7 + [1, 2]]

    class FastLanguageModel:
        for_inference = staticmethod(lambda m: seen["modes"].append("inference"))
        for_training = staticmethod(lambda m: seen["modes"].append("training"))

    tokenizer = Tokenizer()
    scope.update(tokenizer=tokenizer, model=Model(), FastLanguageModel=FastLanguageModel,
                 torch=SimpleNamespace(no_grad=contextlib.nullcontext), ON_KAGGLE=True,
                 MODEL_NAME="m", EPOCHS=2)
    exec(notebook_cell("#@title 6."), scope)  # noqa: S102
    assert scope["before"]["test"]["all_correct"] == 0 and scope["before"]["hard"]["of"] == 56
    assert seen["modes"] == ["inference", "training"]

    tokenizer.reply = staticmethod(lambda said: answers[said])
    after_cell = notebook_cell("#@title 8.").replace("/kaggle/working/", str(tmp_path) + "/")
    exec(after_cell, scope)  # noqa: S102
    written = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert written["sets"]["test"]["before"]["all_correct"] == 0
    assert written["sets"]["test"]["after"]["all_correct"] == 1.0 and written["sets"]["hard"]["examples"] == 56
    assert len(written["sets"]["hard"]["replies_after"]) == 56


FAKE_LLAMA_SERVER = """#!/usr/bin/env python3
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])
answers = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "answers.json"), encoding="utf-8"))

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.respond({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["temperature"] == 0 and body["messages"][0]["role"] == "system"
        said = body["messages"][1]["content"].split("Shopkeeper: ", 1)[1]
        self.respond({"choices": [{"message": {"content": " " + answers[said] + " "}}]})

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def gguf_cell_scope(tmp_path: Path, binary: str) -> tuple[dict, str]:
    import os
    import subprocess
    import time
    import urllib.request

    scope = data_scope()
    answers = {ex["said"]: answer_text(ex["answer"]) for rows in scope["SETS"].values() for ex in rows}
    (tmp_path / "answers.json").write_text(json.dumps(answers), encoding="utf-8")
    server = tmp_path / "llama-server"
    server.write_text(FAKE_LLAMA_SERVER)
    server.chmod(0o755)
    cell = notebook_cell("#@title 11.").replace("/kaggle/working/", str(tmp_path) + "/")
    find = "find / -xdev -name llama-server -type f -perm -u+x 2>/dev/null | head -1"
    assert find in cell
    cell = cell.replace(find, f"echo {binary or server}" if binary != "none" else "true")
    scope.update(os=os, subprocess=subprocess, time=time, urllib=urllib, ON_KAGGLE=True,
                 MAKE_OLLAMA_FILE=True, gguf=str(tmp_path / "m.Q4_K_M.gguf"))
    return scope, cell


def test_gguf_cell_asks_the_server_and_scores_the_answers(tmp_path: Path) -> None:
    scope, cell = gguf_cell_scope(tmp_path, "")
    exec(cell, scope)  # noqa: S102
    report = json.loads((tmp_path / "results_gguf.json").read_text(encoding="utf-8"))
    assert "error" not in report and report["gguf"] == "m.Q4_K_M.gguf"
    assert report["sets"]["test"]["all_correct"] == 1.0 and report["sets"]["hard"]["examples"] == 56


def test_gguf_cell_never_fails_the_run_when_there_is_no_server(tmp_path: Path) -> None:
    scope, cell = gguf_cell_scope(tmp_path, "none")
    exec(cell, scope)  # noqa: S102 - must not raise
    assert "llama-server was not found" in json.loads((tmp_path / "results_gguf.json").read_text())["error"]

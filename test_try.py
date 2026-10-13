"""Checks try_model.ipynb (Ollama on Kaggle) and try_on_kaggle.py, with a fake Ollama.  Run:  python -m pytest"""

import enum
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import build_notebook
import try_on_kaggle
from score import SYSTEM, answer_text, user_message
from test_training import assert_well_formed, load, nfc_lower, runs

# Made-up rows for the fake runs below (not from any data file).
ROWS = [
    {"product": "Test bulb 9W", "said": "hai bhai, 95 rupaye, ghar bhej denge",
     "answer": {"in_stock": True, "price_inr": 95, "delivery": True, "alternative": None}},
    {"product": "Test tape", "said": "abhi khatam, kal aayega",
     "answer": {"in_stock": False, "price_inr": None, "delivery": None, "alternative": None}},
    {"product": "Test wire", "said": "woh nahi hai, dusri company ka le lo",
     "answer": {"in_stock": False, "price_inr": None, "delivery": None, "alternative": "dusri company"}},
]
WRONG = '{"in_stock": false, "price_inr": 1, "delivery": true, "alternative": "dusri company"}'  # row 3: 2 wrong
OTHER = '{"in_stock": true, "price_inr": 450, "delivery": false, "alternative": null}'  # any other question
LOAD, CPU_SLEEP = 0.25, 0.3  # the fake Ollama's load_duration, and how long it takes per answer on the CPU


def try_cell(title: str, nb: dict | None = None) -> str:
    nb = nb or json.loads((HERE / "try_model.ipynb").read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith(title))["source"])


def notebook_scope(tmp_path: Path, nb: dict | None = None) -> dict:
    """The settings cell and the hidden data cell, run as the notebook would, saving into tmp_path."""
    scope: dict = {}
    exec(try_cell("#@title 1.", nb), scope)  # noqa: S102 - our own generated cell
    scope["WORKING_DIR"] = str(tmp_path)
    exec(try_cell("#@title 2.", nb), scope)  # noqa: S102
    scope["OLLAMA_LOG"] = str(tmp_path / "ollama.log")
    return scope


@pytest.fixture
def demo_rows(tmp_path: Path) -> list[dict]:
    path = HERE / "data" / "demo_questions.jsonl"
    if not path.exists():  # not written yet: check a stand-in, so the checks themselves still run
        path = tmp_path / "demo_questions.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in ROWS[:2]) + "\n", encoding="utf-8")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def fake_ollama():
    """A tiny stand-in for Ollama's API: /api/version, /api/ps and /api/generate (answers chosen from the prompt).

    Like Ollama, num_gpu 0 loads the model on the CPU, and later requests without num_gpu keep that CPU copy;
    only an unload (no prompt, keep_alive 0) lets the next request load it on the GPU again.
    Set state["no_gpu"] to load it on the CPU always (as when Ollama cannot use the GPU).
    """
    answers = {ex["said"]: answer_text(ex["answer"]) for ex in ROWS[:2]} | {ROWS[2]["said"]: WRONG}
    state: dict = {"requests": [], "loaded": None, "ran_on": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def respond(self, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/api/ps":
                size = 3_500_000_000
                models = [{"name": "wimb-shop:latest", "size": size,
                           "size_vram": size if state["loaded"] == "GPU" else 0}] if state["loaded"] else []
                self.respond({"models": models})
            else:
                self.respond({"version": "0.0.0-fake"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(body)
            gpu = body.get("options", {}).get("num_gpu", -1) != 0 and not state.get("no_gpu")
            if "prompt" not in body and body.get("keep_alive") == 0:
                state["loaded"] = None
                self.respond({"done": True, "done_reason": "unload"})
                return
            if state["loaded"] is None or (state["loaded"] == "GPU" and not gpu):
                state["loaded"] = "GPU" if gpu else "CPU"
            if "prompt" not in body:
                self.respond({"done": True, "done_reason": "load"})
                return
            state["ran_on"].append(state["loaded"])
            if state["loaded"] == "CPU":
                time.sleep(CPU_SLEEP)
            said = body["prompt"].split("Shopkeeper: ", 1)[1]
            self.respond({"response": " " + answers.get(said, OTHER) + "\n", "done": True,
                          "eval_count": 20, "eval_duration": 1e9, "load_duration": LOAD * 1e9})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


def test_all_notebooks_are_built_from_current_files() -> None:
    built = build_notebook.notebooks()
    assert set(built) == {"train_colab.ipynb", "try_model.ipynb", "ask_model.ipynb"}
    for name, nb in built.items():
        current = (HERE / name).read_text(encoding="utf-8") == build_notebook.text(nb)
        assert current, f"{name} is out of date: run python build_notebook.py"


def test_try_notebook_holds_the_demo_scoring_and_the_repo_modelfile(tmp_path: Path) -> None:
    scope = notebook_scope(tmp_path)
    demo = HERE / "data" / "demo_questions.jsonl"
    assert scope["DEMO"] == (load("demo_questions.jsonl") if demo.exists() else [])
    assert scope["MODELFILE_REPO"] == (HERE / "Modelfile").read_text(encoding="utf-8")
    assert scope["make_modelfile"]("wimb-shop-model.Q4_K_M.gguf") == scope["MODELFILE_REPO"]
    assert scope["SYSTEM"] == SYSTEM and scope["QUESTIONS"] == [] and scope["CPU_CHECK"] == 3
    nb = json.loads((HERE / "try_model.ipynb").read_text(encoding="utf-8"))
    assert nb["metadata"]["accelerator"] == "GPU"


def test_demo_questions_are_well_formed_and_not_from_other_data(demo_rows: list[dict]) -> None:
    assert demo_rows
    for ex in demo_rows:
        assert_well_formed(ex)
    said = [nfc_lower(ex["said"]).strip() for ex in demo_rows]
    assert len(set(said)) == len(said)
    others = set()
    for name in ("train.jsonl", "train_v1.jsonl", "test.jsonl", "hard_test.jsonl", "fresh_test.jsonl",
                 "real/xpqa_hi_examples.jsonl"):
        others |= {nfc_lower(ex["said"]).strip() for ex in load(name)}
    assert not set(said) & others
    long_runs = set().union(*(runs(ex["said"]) for name in ("train.jsonl", "test.jsonl", "hard_test.jsonl",
                                                             "fresh_test.jsonl") for ex in load(name)))
    shared = [ex["said"] for ex in demo_rows if runs(ex["said"]) & long_runs]
    assert not shared, shared[:3]
    # Not even close to a held-out row: no run of 6 words, no same product name.
    held_out = [ex for name in ("test.jsonl", "hard_test.jsonl", "fresh_test.jsonl") for ex in load(name)]
    six = set().union(*(runs(ex["said"], 6) for ex in held_out))
    close = [ex["said"] for ex in demo_rows if runs(ex["said"], 6) & six]
    assert not close, close[:3]
    products = {nfc_lower(ex["product"]).strip() for ex in held_out}
    same_product = [ex["product"] for ex in demo_rows if nfc_lower(ex["product"]).strip() in products]
    assert not same_product, same_product


TRICKY = [
    {"product": "it's a \"quoted\" ''' item", "said": "a ''' b \"\"\" c \\ d \\n e\nnew line, \\'"},
    {"product": "", "said": "हाँ है, २२० का पड़ेगा\tटैब   x \x0b y"},
]


def test_questions_line_survives_any_characters() -> None:
    nb = try_on_kaggle.notebook_with(TRICKY)
    cell = try_cell("#@title 1.", nb)
    compile(cell, "settings", "exec")
    scope: dict = {}
    exec(cell, scope)  # noqa: S102 - our own generated cell
    assert scope["QUESTIONS"] == TRICKY
    lines = cell.splitlines()
    assert lines.count('PRODUCT = ""') == 1 and lines.count('SAID = ""') == 1
    assert sum(line.startswith("QUESTIONS = ") for line in lines) == 1
    # the untouched notebook asks nothing
    assert "QUESTIONS = json.loads('[]')" in try_cell("#@title 1.").splitlines()


def test_questions_are_checked() -> None:
    ok = try_on_kaggle.questions_from([["16A MCB", " haan hai "]], '[{"product": "x", "said": "nahi"}, {"said": "y"}]')
    assert ok == [{"product": "16A MCB", "said": "haan hai"}, {"product": "x", "said": "nahi"},
                  {"product": "", "said": "y"}]
    assert try_on_kaggle.questions_from([], None) == []
    for bad in ("not json", '{"said": "x"}', '"x"', "[1]", '[{"product": "x"}]', '[{"said": ""}]', '[{"said": "  "}]',
                '[{"said": 5}]', '[{"product": 5, "said": "x"}]', json.dumps([{"said": "x" * 501}]),
                json.dumps([{"said": "x"}] * 31), '[{"said": "haan \\ud800 hai"}]'):
        with pytest.raises((TypeError, ValueError)):
            try_on_kaggle.questions_from([], bad)
    with pytest.raises(ValueError, match="non-empty"):
        try_on_kaggle.questions_from([["x", ""]], None)
    with pytest.raises(ValueError, match="normal text"):  # what a broken byte in --ask becomes
        try_on_kaggle.questions_from([["x", "haan \udcff hai"]], None)


def test_bad_questions_stop_before_kaggle(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["try_on_kaggle.py", "--questions-json", '{"said": "x"}'])
    monkeypatch.setattr(try_on_kaggle, "connect", lambda slug: pytest.fail("must not reach Kaggle"))
    with pytest.raises(SystemExit) as stop:
        try_on_kaggle.main()
    assert stop.value.code == 2 and "JSON list" in capsys.readouterr().err


Status = enum.Enum("Status", "RUNNING COMPLETE ERROR")

RESULT = {
    "ollama_version": "0.12.0", "gpu": "Tesla T4", "model_file": "/kaggle/input/t/wimb-shop-model/wimb-shop-model.Q4_K_M.gguf",
    "model_mb": 2497, "create_seconds": 31.2, "load_seconds": 3.4,
    "demo": [{"product": "Test bulb 9W", "said": ROWS[0]["said"], "want": ROWS[0]["answer"],
              "reply": answer_text(ROWS[0]["answer"]), "wrong": [], "seconds": 0.8, "tokens_per_s": 61.0},
             {"product": "Test wire", "said": ROWS[2]["said"], "want": ROWS[2]["answer"], "reply": WRONG,
              "wrong": ["price_inr", "delivery"], "seconds": 0.9, "tokens_per_s": 59.0}],
    "demo_asked": 2, "demo_right": 1, "demo_total": 2, "processor": "100% GPU", "template": "gguf_chat_template",
    "gpu_avg_seconds": 0.85, "gpu_tokens_per_s": 60.0,
    "gpu_memory": "3500 MiB, 15360 MiB", "cpu": [{"said": ROWS[0]["said"], "seconds": 12.0, "load_seconds": 2.5}],
    "cpu_avg_seconds": 9.5, "ollama_ps": "NAME  ID  SIZE  PROCESSOR\nwimb-shop:latest  x  3.5 GB  100% GPU",
    "questions_asked": [{"product": "Fevicol 1 kg", "said": "haan hai, 450 ka"}, {"product": "Test pipe", "said": "pata nahi"}],
    "questions": [{"product": "Fevicol 1 kg", "said": "haan hai, 450 ka", "reply": OTHER, "parsed": json.loads(OTHER)},
                  {"product": "Test pipe", "said": "pata nahi", "reply": "sorry", "parsed": None}],
    "questions_processor": "100% GPU",
    "errors": ["CPU check: TimeoutError()"],
}


class FakeApi:
    """Stands in for Kaggle: records the push, finishes the run, and hands out a results file."""

    def __init__(self, results: dict | None = RESULT, final: Status = Status.COMPLETE,
                 invalid_sources: list[str] | None = None) -> None:
        self.results, self.final, self.invalid_sources, self.calls = results, final, invalid_sources or [], {"polls": 0}

    def kernels_status(self, kernel):
        if kernel.endswith("/wimb-shop-model-training"):
            return SimpleNamespace(status=Status.COMPLETE, failure_message="")
        self.calls["polls"] += 1
        return SimpleNamespace(status=self.final if self.calls["polls"] > 1 else Status.RUNNING, failure_message="")

    def kernels_push(self, folder, acc=None):
        meta = json.loads((Path(folder) / "kernel-metadata.json").read_text())
        self.calls.update(meta=meta, acc=acc, files=sorted(p.name for p in Path(folder).iterdir()),
                          notebook=json.loads((Path(folder) / meta["code_file"]).read_text(encoding="utf-8")))
        return SimpleNamespace(error=None, url="https://kaggle/x", invalid_kernel_sources=self.invalid_sources)

    def kernels_output(self, kernel, path, file_pattern=None, force=False):
        self.calls["pattern"] = file_pattern
        log = [{"stream_name": "stdout", "time": 1, "data": "RuntimeError: No model file\n"}]
        (Path(path) / "wimb-shop-model-try.log").write_text(json.dumps(log))
        if self.results is not None:
            (Path(path) / "try_results.json").write_text(json.dumps(self.results, ensure_ascii=False),
                                                         encoding="utf-8")


def test_runner_attaches_the_training_output_and_sends_the_questions() -> None:
    api = FakeApi()
    try_on_kaggle.push(api, "me/wimb-shop-model-try", TRICKY)
    meta = api.calls["meta"]
    assert meta["kernel_sources"] == ["me/wimb-shop-model-training"]
    assert meta["id"] == "me/wimb-shop-model-try" and meta["code_file"] == "try_model.ipynb"
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert api.calls["acc"] == "NvidiaTeslaT4"
    scope: dict = {}
    exec(try_cell("#@title 1.", api.calls["notebook"]), scope)  # noqa: S102 - our own generated cell
    assert scope["QUESTIONS"] == TRICKY


def test_runner_stops_when_kaggle_cannot_attach_the_training_output() -> None:
    api = FakeApi(invalid_sources=["me/wimb-shop-model-training"])
    with pytest.raises(SystemExit, match="could not attach"):
        try_on_kaggle.push(api, "me/wimb-shop-model-try", [])


def test_runner_prints_the_report(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(try_on_kaggle, "OUT", tmp_path / "out")
    assert try_on_kaggle.fetch(FakeApi(), "me/wimb-shop-model-try") == RESULT
    out = capsys.readouterr().out
    for text in ("Ollama 0.12.0; GPU: Tesla T4", "wimb-shop-model.Q4_K_M.gguf (2,497 MB)", "ollama create took 31.2 s",
                 "loading the model took 3.4 s", "Prompt template: the chat template inside the .gguf",
                 "Example calls (2, on the GPU)", "✓", "wrong: price_inr, delivery", "1 of 2 fully right",
                 "Speed on the GPU: 0.85 s per answer on average, 60.0 tokens/s", "CPU only: 9.5 s per answer",
                 "100% GPU", "Your questions (2, on the GPU)", "Product: Fevicol 1 kg",
                 "Answer: in stock: yes; price: ₹450; delivery: no; other product offered: none",
                 "Answer: no answer", "CPU check: TimeoutError()"):
        assert text in out, text
    assert "Kaggle's log" not in out


def test_report_says_where_ollama_really_ran_it(capsys) -> None:
    try_on_kaggle.report({**RESULT, "processor": "100% CPU", "questions_processor": None, "template": "go_template"})
    out = capsys.readouterr().out
    assert "Example calls (2, on the CPU only (Ollama did not use the GPU))" in out
    assert "Speed on the CPU only" in out and "on the GPU" not in out
    assert "Your questions (2, on GPU or CPU (not known))" in out
    assert "Prompt template: the Modelfile's TEMPLATE" in out
    assert try_on_kaggle.ran_on("30%/70% CPU/GPU") == "GPU and CPU (30%/70% CPU/GPU)"


def test_runner_without_results_shows_the_log_and_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(try_on_kaggle, "OUT", tmp_path / "out")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "try_results.json").write_text("{}")  # an older run's file must not count
    api = FakeApi(results=None)
    assert try_on_kaggle.fetch(api, "me/wimb-shop-model-try") is None
    out = capsys.readouterr().out
    assert "No try_results.json" in out and "RuntimeError: No model file" in out
    assert api.calls["pattern"] == r"(try_results\.json|.*\.log|.*\.ipynb)$"


def run_main(api: FakeApi, tmp_path: Path, monkeypatch, *argv: str) -> int:
    """try_on_kaggle.main() against the fake Kaggle: its exit code."""
    import run_on_kaggle

    monkeypatch.setattr(try_on_kaggle, "OUT", tmp_path / "out")
    monkeypatch.setattr(try_on_kaggle, "connect", lambda slug: (api, f"me/{slug}"))
    monkeypatch.setattr(try_on_kaggle, "check_gpu", lambda api, need_hours: None)
    monkeypatch.setattr(run_on_kaggle.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["try_on_kaggle.py", *argv])
    with pytest.raises(SystemExit) as stop:
        try_on_kaggle.main()
    return stop.value.code


ASKED = ("--ask", "Fevicol 1 kg", "haan hai, 450 ka", "--ask", "Test pipe", "pata nahi")  # RESULT's questions


def test_runner_main_pushes_waits_and_reports(tmp_path: Path, monkeypatch, capsys) -> None:
    api = FakeApi()
    assert run_main(api, tmp_path, monkeypatch, *ASKED) == 0
    scope: dict = {}
    exec(try_cell("#@title 1.", api.calls["notebook"]), scope)  # noqa: S102
    assert scope["QUESTIONS"] == RESULT["questions_asked"]
    out = capsys.readouterr().out
    assert "1 of 2 fully right" in out and "Not answered in full" not in out


@pytest.mark.parametrize(("api", "argv", "says"), [
    (FakeApi(results=None), ASKED, "No try_results.json"),
    (FakeApi(final=Status.ERROR), ASKED, "The run did not finish. Kaggle's log:"),  # e.g. a time-out in cell 4
    (FakeApi({**RESULT, "demo": [], "demo_total": 0, "demo_right": 0, "questions": []}), ASKED,
     "the model answered nothing"),
    (FakeApi({**RESULT, "demo": RESULT["demo"][:1], "demo_total": 1}), ASKED, "1 of 2 example calls got no answer"),
    (FakeApi({**RESULT, "questions": RESULT["questions"][:1]}), ASKED, "1 of your 2 questions got no answer"),
    (FakeApi(), ASKED[:3], "these answers are for other questions"),
])
def test_runner_fails_when_not_everything_was_answered(api, argv, says, tmp_path, monkeypatch, capsys) -> None:
    assert run_main(api, tmp_path, monkeypatch, *argv) == 1
    out = capsys.readouterr().out
    assert says in out
    if "log" in says:
        assert "RuntimeError: No model file" in out  # the end of Kaggle's log


def test_in_words() -> None:
    words = try_on_kaggle.in_words
    assert words({"in_stock": True, "price_inr": 220, "delivery": False, "alternative": None}) == (
        "in stock: yes; price: ₹220; delivery: no; other product offered: none")
    assert words({"in_stock": False, "price_inr": None, "delivery": None, "alternative": "32A MCB"}) == (
        "in stock: no; price: not said; delivery: not said; other product offered: 32A MCB")
    assert "price: ₹99.5" in words({"in_stock": True, "price_inr": 99.5, "delivery": None, "alternative": None})
    assert "price: ₹150;" in words({"in_stock": True, "price_inr": 150.0, "delivery": None, "alternative": None})
    assert words(None).startswith("no answer")


def run_stages(scope: dict) -> dict:
    """The cells after the model is made: ask(), the example calls, your questions, save."""
    for title in ("#@title 6.", "#@title 7.", "#@title 8.", "#@title 9."):
        exec(try_cell(title), scope)  # noqa: S102 - our own generated cell
    return json.loads((Path(scope["WORKING_DIR"]) / "try_results.json").read_text(encoding="utf-8"))


def test_notebook_asks_ollama_scores_and_saves(tmp_path: Path, fake_ollama: dict, monkeypatch, capsys) -> None:
    nb = try_on_kaggle.notebook_with([{"product": "Fevicol 1 kg", "said": "haan hai, 450 ka"}])
    scope = notebook_scope(tmp_path, nb)
    # What cells 3-5 (find the file, install Ollama, ollama create) would have set:
    scope.update(DEMO=ROWS, CPU_CHECK=2, PRODUCT="Test item", SAID="mil jayega, 300 ka", OLLAMA_URL=fake_ollama["url"],
                 MODEL_FILE="/kaggle/input/t/wimb-shop-model/wimb-shop-model.Q4_K_M.gguf")
    scope["RESULTS"].update(ollama_version="0.0.0-fake", gpu="Tesla T4", model_file=scope["MODEL_FILE"],
                            model_mb=2497, create_seconds=30.0)
    Path(scope["OLLAMA_LOG"]).write_text('time=2026-10-13T10:00:00Z level=INFO source=images.go:418 '
                                         'msg="template selection" model=wimb-shop selected=gguf_chat_template '
                                         'renderer="" parser=""\n', encoding="utf-8")
    bin_dir = tmp_path / "bin"  # a stand-in `ollama` for `ollama ps`
    bin_dir.mkdir()
    (bin_dir / "ollama").write_text('#!/bin/sh\necho "fake ollama $1"\n')
    (bin_dir / "ollama").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    results = run_stages(scope)
    out = capsys.readouterr().out

    assert results["errors"] == []
    assert results["demo_asked"] == 3 and results["demo_right"] == 2 and results["demo_total"] == 3
    assert [d["wrong"] for d in results["demo"]] == [[], [], ["price_inr", "delivery"]]
    assert results["load_seconds"] == LOAD and results["gpu_tokens_per_s"] == 20.0
    assert results["demo"][0]["tokens_per_s"] == 20.0 and results["gpu_avg_seconds"] >= 0
    assert results["processor"] == "100% GPU" and results["template"] == "gguf_chat_template"
    assert len(results["cpu"]) == 2
    assert CPU_SLEEP - LOAD <= results["cpu_avg_seconds"] < CPU_SLEEP - 0.05  # without the load time
    assert results["ollama_ps"] == results["cpu_ollama_ps"] == "fake ollama ps"
    assert results["model_mb"] == 2497 and results["gpu"] == "Tesla T4"  # kept from the earlier cells
    asked = [{"product": "Fevicol 1 kg", "said": "haan hai, 450 ka"}, {"product": "Test item", "said": "mil jayega, 300 ka"}]
    assert results["questions_asked"] == asked
    assert [(q["product"], q["parsed"]) for q in results["questions"]] == [
        ("Fevicol 1 kg", json.loads(OTHER)), ("Test item", json.loads(OTHER))]
    assert results["questions_processor"] == "100% GPU"
    assert try_on_kaggle.gaps(results, asked) == []
    assert "In words:   in stock: yes; price: ₹450; delivery: no; other product offered: none" in out
    assert "all 4 right" in out and "wrong: price_inr, delivery" in out

    sent = fake_ollama["requests"]
    asks = [b for b in sent if "prompt" in b]
    assert len(asks) == 1 + 3 + 2 + 2  # warm-up, examples, CPU check, questions
    for body in asks:
        assert body["model"] == "wimb-shop" and body["stream"] is False and "system" not in body
        assert body["options"]["temperature"] == 0 and body["options"]["num_predict"] == 96
    cpu = [b for b in asks if "num_gpu" in b["options"]]
    assert [b["options"]["num_gpu"] for b in cpu] == [0, 0]
    assert [b["prompt"] for b in cpu] == [user_message(ex["product"], ex["said"]) for ex in ROWS[:2]]
    # After the CPU check the model is unloaded and loaded again, so your questions run on the GPU.
    assert [b for b in sent if "prompt" not in b] == [{"model": "wimb-shop", "keep_alive": 0}, {"model": "wimb-shop"}]
    assert fake_ollama["ran_on"] == ["GPU"] * 4 + ["CPU"] * 2 + ["GPU"] * 2


def test_notebook_says_so_when_ollama_did_not_use_the_gpu(tmp_path: Path, fake_ollama: dict) -> None:
    fake_ollama["no_gpu"] = True
    scope = notebook_scope(tmp_path)
    scope.update(DEMO=ROWS[:1], CPU_CHECK=0, OLLAMA_URL=fake_ollama["url"])
    results = run_stages(scope)
    assert results["demo_total"] == 1 and results["processor"] == "100% CPU"
    assert [e.split(":", 1)[0] for e in results["errors"]] == ["GPU"] and "100% CPU" in results["errors"][0]


def test_notebook_records_errors_and_still_saves_when_ollama_is_down(tmp_path: Path) -> None:
    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    scope = notebook_scope(tmp_path)
    scope.update(DEMO=ROWS, SAID="haan hai", OLLAMA_URL=f"http://127.0.0.1:{port}")
    results = run_stages(scope)  # must not raise
    stages = [e.split(":", 1)[0] for e in results["errors"]]
    assert stages == ["warm-up", "example 1", "example 2", "example 3", "CPU check", "CPU check", "CPU check",
                      "back to the GPU", "your question"]
    assert results["demo_total"] == 0 and results["questions"] == [] and results["cpu_avg_seconds"] is None
    assert try_on_kaggle.gaps(results) == ["the model answered nothing", "1 of your 1 questions got no answer"]


def test_find_cell_picks_the_model_or_says_how_to_attach_it(tmp_path: Path, capsys) -> None:
    cell = try_cell("#@title 3.").replace("/kaggle/input", str(tmp_path / "input"))
    scope = notebook_scope(tmp_path)
    with pytest.raises(RuntimeError, match="Add Input"):
        exec(cell, scope)  # noqa: S102
    saved = json.loads((tmp_path / "try_results.json").read_text(encoding="utf-8"))
    assert len(saved["errors"]) == 1 and saved["errors"][0].startswith("find the model: no .gguf file")

    folder = tmp_path / "input" / "wimb-shop-model-training" / "wimb-shop-model"
    folder.mkdir(parents=True)
    (folder / "other.F16.gguf").write_bytes(b"x" * 3_000_000)
    (folder / "wimb-shop-model.Q4_K_M.gguf").write_bytes(b"x" * 2_000_000)
    scope = notebook_scope(tmp_path)
    exec(cell, scope)  # noqa: S102
    assert scope["MODEL_FILE"] == str(folder / "wimb-shop-model.Q4_K_M.gguf")
    assert scope["RESULTS"]["model_mb"] == 2 and "2 MB" in capsys.readouterr().out


FAKE_OLLAMA_CLI = """#!/bin/sh
# Stands in for "ollama create": fails on a linked file when FAIL_ON_LINK is set, or always when FAIL is set.
[ "$1" = "create" ] && [ "$2" = "wimb-shop" ] && [ "$3" = "-f" ] || exit 2
dir=$(dirname "$4")
grep -q '^FROM ./wimb-shop-model.Q4_K_M.gguf$' "$4" || exit 3
[ -n "$FAIL" ] && { echo "Error: no space left" >&2; exit 1; }
[ -n "$FAIL_ON_LINK" ] && [ -L "$dir/wimb-shop-model.Q4_K_M.gguf" ] && { echo "Error: invalid file" >&2; exit 1; }
printf '\\033[?25lgathering model components\\r\\033[Kcopying file sha256:abc 100%%\\nsuccess\\n' >&2
"""


def create_cell_scope(tmp_path: Path, monkeypatch, **env) -> tuple[dict, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ollama").write_text(FAKE_OLLAMA_CLI)
    (bin_dir / "ollama").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    model = tmp_path / "input" / "wimb-shop-model.Q4_K_M.gguf"
    model.parent.mkdir()
    model.write_bytes(b"gguf")
    scope = notebook_scope(tmp_path)
    scope["MODEL_FILE"] = str(model)
    return scope, try_cell("#@title 5.").replace("/tmp/wimb", str(tmp_path / "wimb"))


def test_create_cell_uses_the_repo_modelfile_and_retries_with_a_copy(tmp_path: Path, monkeypatch, capsys) -> None:
    scope, cell = create_cell_scope(tmp_path, monkeypatch, FAIL_ON_LINK="1")
    exec(cell, scope)  # noqa: S102 - our own generated cell
    made = tmp_path / "wimb"
    assert (made / "Modelfile").read_text(encoding="utf-8") == (HERE / "Modelfile").read_text(encoding="utf-8")
    assert not (made / "wimb-shop-model.Q4_K_M.gguf").is_symlink()  # the second try used a copy
    out = capsys.readouterr().out
    assert "trying again with a copy" in out and "success" in out and "\x1b" not in out
    assert scope["RESULTS"]["create_seconds"] is not None and scope["RESULTS"]["errors"] == []


def test_create_cell_stops_when_the_modelfile_differs_from_the_repo(tmp_path: Path, monkeypatch) -> None:
    scope, cell = create_cell_scope(tmp_path, monkeypatch)
    scope["MODELFILE_REPO"] += "PARAMETER num_ctx 4096\n"
    with pytest.raises(RuntimeError, match="differs from the repo's Modelfile"):
        exec(cell, scope)  # noqa: S102
    saved = json.loads((tmp_path / "try_results.json").read_text(encoding="utf-8"))
    assert saved["errors"][0].startswith("Modelfile:") and not (tmp_path / "wimb" / "Modelfile").exists()


def test_create_cell_stops_with_the_reason_when_ollama_create_fails(tmp_path: Path, monkeypatch) -> None:
    scope, cell = create_cell_scope(tmp_path, monkeypatch, FAIL="1")
    with pytest.raises(RuntimeError, match="ollama create failed"):
        exec(cell, scope)  # noqa: S102
    saved = json.loads((tmp_path / "try_results.json").read_text(encoding="utf-8"))
    assert "no space left" in saved["errors"][0]


def test_start_cell_uses_a_running_ollama(tmp_path: Path, fake_ollama: dict) -> None:
    cell = try_cell("#@title 4.")
    default = 'OLLAMA_URL = "http://127.0.0.1:11434"'
    assert cell.count(default) == 1
    scope = notebook_scope(tmp_path)
    exec(cell.replace(default, f'OLLAMA_URL = "{fake_ollama["url"]}"'), scope)  # noqa: S102
    saved = json.loads((tmp_path / "try_results.json").read_text(encoding="utf-8"))
    assert saved["ollama_version"] == "0.0.0-fake" and saved["errors"] == []

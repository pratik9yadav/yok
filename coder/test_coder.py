"""Checks coder_eval.ipynb (coding models fixing bugs, Ollama on Kaggle) and coder/run_eval.py, with a fake Ollama,
a tiny fake task set and a fake Kaggle.  Run:  python -m pytest"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import run_on_kaggle
from coder import build, fixeval, run_eval
from test_try import FakeApi, Status

# Two made-up tasks in the HumanEvalFix format: the fake model fixes the first, not the second.
TASKS = [
    {"task_id": "Fake/0", "entry_point": "add", "declaration": "def add(a, b):\n", "buggy_solution": "    return a - b\n",
     "test": "\n\ndef check(add):\n    assert add(2, 3) == 5\n\ncheck(add)\n"},
    {"task_id": "Fake/1", "entry_point": "halve", "declaration": "def halve(x):\n",
     "buggy_solution": "    return x * 2\n", "test": "def check(halve):\n    assert halve(8) == 4\n"},  # no check() call
]
FIXED = "Here is the fix:\n```python\ndef add(a, b):\n    return a + b\n```\nThe minus was wrong."
NOT_FIXED = "```python\ndef halve(x):\n    return x * 2\n```"
THOUGHT = "The bug: minus. Draft:\n```python\ndef add(a, b):\n    return 0\n```"  # must never be tested


def cell(title: str, nb: dict | None = None) -> str:
    nb = nb or json.loads((HERE / build.NOTEBOOK).read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith(title))["source"])


def notebook_scope(tmp_path: Path, nb: dict | None = None) -> dict:
    """The settings and helpers cells, run as the notebook would, saving into tmp_path."""
    scope: dict = {}
    exec(cell("#@title 1.", nb), scope)  # noqa: S102 - our own generated cell
    scope.update(WORKING_DIR=str(tmp_path), MODELS_DIR=str(tmp_path / "models"))
    exec(cell("#@title 2.", nb), scope)  # noqa: S102
    scope["OLLAMA_LOG"] = str(tmp_path / "ollama.log")
    return scope


@pytest.fixture
def fake_ollama():
    """A tiny stand-in for Ollama's API: version, ps, pull (streamed), generate (load/unload), chat, delete.

    The test fixtures in test_try.py answer shop questions through /api/generate; this one answers /api/chat.
    """
    state: dict = {"chats": [], "loads": [], "unloads": [], "deletes": [], "pulls": [], "loaded": None}
    replies = {"add": FIXED, "halve": NOT_FIXED}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def respond(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/api/ps":
                size = 5_000_000_000
                loaded = state["loaded"]
                self.respond({"models": [{"name": loaded, "model": loaded, "size": size, "size_vram": size}]
                              if loaded else []})
            else:
                self.respond({"version": "0.0.0-fake"})

        def body(self) -> dict:
            return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

        def do_DELETE(self) -> None:
            state["deletes"].append(self.body()["model"])
            self.respond({})

        def do_POST(self) -> None:
            body = self.body()
            model = body.get("model")
            if self.path == "/api/pull":
                state["pulls"].append(body)
                if model.startswith("missing"):
                    self.respond({"error": "pull model manifest: file does not exist"}, 500)
                    return
                steps = [{"status": "pulling manifest"}]
                steps += [{"status": "pulling 0123456789abcdef", "digest": "sha256:01", "total": 2_000_000_000,
                           "completed": done} for done in (0, 1_000_000_000, 2_000_000_000)]
                steps += [{"status": "verifying sha256 digest"}, {"status": "success"}]
                data = "".join(json.dumps(s) + "\n" for s in steps).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/generate":
                if body.get("keep_alive") == 0:
                    state["unloads"].append(model)
                    state["loaded"] = None
                else:
                    state["loads"].append(body)
                    state["loaded"] = model
                self.respond({"done": True})
            elif self.path == "/api/chat":
                state["chats"].append(body)
                state["loaded"] = model
                said = body["messages"][-1]["content"]
                if state.get("answer"):  # a test's own replies (coder/exam/test_exam.py): answer(prompt) -> text
                    content = state["answer"](said)
                else:
                    content = replies[said.split("Fix bugs in ", 1)[1].split(".", 1)[0]]
                message = {"role": "assistant", "content": content}
                if body.get("think"):
                    message["thinking"] = THOUGHT
                self.respond({"message": message, "done": True, "done_reason": "stop",
                              "eval_count": 40, "eval_duration": 2e9})
            else:
                self.respond({"error": "unknown"}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


# ---- the notebook file ----

def test_notebook_is_built_from_current_files() -> None:
    current = (HERE / build.NOTEBOOK).read_text(encoding="utf-8")
    assert current == build.text(build.notebook()), "coder_eval.ipynb is out of date: run python coder/build.py"


def test_notebook_settings_and_helpers(tmp_path: Path) -> None:
    scope = notebook_scope(tmp_path)
    assert scope["MODELS"] == ["qwen2.5-coder:7b", "qwen2.5-coder:14b", "qwen3-coder:30b", "gpt-oss:20b"]
    assert scope["LIMIT"] is None and scope["MINUTES_PER_MODEL"] == 45
    assert scope["OPTIONS"] == {"temperature": 0, "num_ctx": 4096, "num_predict": 1024}
    assert scope["RESULTS"]["licence"] == "MIT" and scope["RESULTS"]["dataset"] == "bigcode/humanevalpack"
    assert scope["RESULTS_FILE"] == "coder_eval.json" and "/kaggle/working" not in scope["MODELS_DIR"]
    # the shared Ollama start code (from build_notebook.py), with the downloads sent to the big disk
    ollama = cell("#@title 3.")
    assert build.START_OLLAMA.strip("\n") in ollama
    assert ollama.index('os.environ["OLLAMA_MODELS"] = MODELS_DIR') < ollama.index("ollama serve")


def test_start_cell_uses_a_running_ollama(tmp_path: Path, fake_ollama: dict) -> None:
    default = 'OLLAMA_URL = "http://127.0.0.1:11434"'
    scope = notebook_scope(tmp_path)
    exec(cell("#@title 3.").replace(default, f'OLLAMA_URL = "{fake_ollama["url"]}"'), scope)  # noqa: S102
    saved = json.loads((tmp_path / "coder_eval.json").read_text(encoding="utf-8"))
    assert saved["ollama_version"] == "0.0.0-fake" and saved["errors"] == []
    assert (tmp_path / "models").is_dir()


# ---- patching the settings ----

def test_models_and_limit_lines_round_trip() -> None:
    nb = run_eval.notebook_with(["qwen2.5-coder:7b", "deepseek-coder-v2:16b", "hf.co/x/y-GGUF:Q4_K_M"], 5)
    s = run_eval.settings(nb)
    assert s["MODELS"] == ["qwen2.5-coder:7b", "deepseek-coder-v2:16b", "hf.co/x/y-GGUF:Q4_K_M"] and s["LIMIT"] == 5
    s = run_eval.settings(run_eval.notebook_with(None, None))  # the notebook's own models, all tasks
    assert s["MODELS"] == run_eval.settings(json.loads(run_eval.NOTEBOOK.read_text("utf-8")))["MODELS"]
    assert s["LIMIT"] is None
    lines = "".join(run_eval.settings_cell(nb)["source"]).splitlines()
    assert sum(line.startswith("MODELS = ") for line in lines) == 1
    assert sum(line.startswith("LIMIT = ") for line in lines) == 1
    assert run_eval.hours_needed(run_eval.notebook_with(["a:1"], None)) == round((45 + 5 + 15) / 60, 1)


def test_model_names_are_checked() -> None:
    assert run_eval.models_from(" qwen2.5-coder:7b , gpt-oss:20b,") == ["qwen2.5-coder:7b", "gpt-oss:20b"]
    for bad in ("", " , ", "a b", 'x"]; import os', "qwen:7b\nLIMIT = 1", "a:b:c", ",".join(["m"] * 9)):
        with pytest.raises(ValueError):
            run_eval.models_from(bad)


# ---- reading the reply ----

@pytest.mark.parametrize(("reply", "code"), [
    (FIXED, "def add(a, b):\n    return a + b\n"),
    ("```python\nx = 1\n```\nor better:\n```python\nx = 2\n```", "x = 2\n"),  # the last block
    ("```python\nx = 1\n```\nexample:\n```text\nhello\n```", "x = 1\n"),  # the last *python* block
    ("```\nx = 3\n```", "x = 3\n"),  # a block with no language
    ("```bash\npip x\n```", "pip x\n"),  # no python block: the last block
    ("def f():\n    return 1", "def f():\n    return 1\n"),  # no block: the whole reply
    ("<think>try:\n```python\nx = 0\n```\n</think>\nFinal:\n```python\nx = 9\n```", "x = 9\n"),  # not the thinking
    ("first notes ```python\nx = 0\n``` done</think>```python\nx = 8\n```", "x = 8\n"),  # only the closing tag
    ("Sure:\n```python\ndef f():\n    return 1", "def f():\n    return 1\n"),  # cut off before the closing ```
    ("```Python  title\nx = 4\n```", "x = 4\n"),
])
def test_extract_code(reply: str, code: str) -> None:
    assert fixeval.extract_code(reply) == code


def test_prompt_shows_the_buggy_function_not_the_tests() -> None:
    prompt = fixeval.prompt_for(TASKS[0])
    assert prompt == ("```python\ndef add(a, b):\n    return a - b\n```\n\n"
                      "Fix bugs in add. Reply with the full corrected function in one ```python block.")
    assert "assert" not in prompt and "check" not in prompt


def test_run_program_passes_fails_and_times_out() -> None:
    fixed = fixeval.program_for(fixeval.extract_code(FIXED), TASKS[0])
    assert fixeval.run_program(fixed) == (True, "passed")
    assert fixed.count("\ncheck(add)") == 1  # the call already in the tests is not added again
    # The second task's tests have no check() call: it is added, so a wrong fix fails.
    wrong = fixeval.program_for(fixeval.extract_code(NOT_FIXED), TASKS[1])
    assert wrong.rstrip().endswith("check(halve)")
    assert fixeval.run_program(wrong) == (False, "AssertionError")
    assert fixeval.run_program(fixeval.program_for("def halve(x):\n    return x // 2\n", TASKS[1]))[0]
    assert fixeval.run_program("raise SystemExit(3)") == (False, "exit code 3")
    assert fixeval.run_program("x = (")[1].startswith("SyntaxError")
    assert fixeval.run_program("while True:\n    pass\n", timeout=1) == (False, "timeout after 1 s")
    assert fixeval.run_program("input()")[0] is False  # no keyboard: never waits


def test_load_tasks_checks_the_fields(monkeypatch) -> None:
    rows = [{"task_id": "Fake/0", "prompt": "p", **TASKS[0]}]

    class Resp:
        def __init__(self, text):
            self.text = text

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.text.encode()

    monkeypatch.setattr(fixeval.urllib.request, "urlopen",
                        lambda url, timeout: Resp("\n".join(json.dumps(r) for r in rows) + "\n"))
    assert fixeval.load_tasks() == (rows, fixeval.DATASET_URL)
    monkeypatch.setattr(fixeval.urllib.request, "urlopen", lambda url, timeout: Resp(json.dumps({"task_id": "x"})))
    with pytest.raises(RuntimeError, match="no \\['declaration'"):
        fixeval.load_tasks()


# ---- the whole run, against the fake Ollama ----

def run_notebook(tmp_path: Path, fake_ollama: dict, models: list[str], minutes: float = 45) -> tuple[dict, list]:
    """Cells 4-6 with the fake tasks and the fake Ollama. The results file, and a copy of it at every save."""
    scope = notebook_scope(tmp_path, run_eval.notebook_with(models, None))
    scope.update(OLLAMA_URL=fake_ollama["url"], MINUTES_PER_MODEL=minutes, load_tasks=lambda: (TASKS, "fake tasks"))
    saves = []
    save = scope["save_results"]

    def save_and_copy():
        save()
        saves.append(json.loads((tmp_path / "coder_eval.json").read_text(encoding="utf-8")))

    scope["save_results"] = save_and_copy
    for title in ("#@title 4.", "#@title 5.", "#@title 6."):
        exec(cell(title), scope)  # noqa: S102 - our own generated cell
    return json.loads((tmp_path / "coder_eval.json").read_text(encoding="utf-8")), saves


def test_notebook_scores_every_model_and_goes_on_after_a_failure(tmp_path, fake_ollama, capsys) -> None:
    results, saves = run_notebook(tmp_path, fake_ollama, ["qwen-fake:7b", "missing:1b", "gpt-oss:20b"])
    out = capsys.readouterr().out
    assert results["tasks_total"] == results["tasks_used"] == 2 and results["dataset_source"] == "fake tasks"
    assert results["fields"] == ["buggy_solution", "declaration", "entry_point", "task_id", "test"]
    first, missing, oss = results["models"]

    assert first["model"] == "qwen-fake:7b" and first["pulled"] and first["size_gb"] == 2.0
    assert (first["done"], first["passed"], first["total"], first["pass_at_1"]) == (2, 1, 2, 0.5)
    assert [(t["task_id"], t["passed"], t["why"]) for t in first["tasks"]] == [
        ("Fake/0", True, "passed"), ("Fake/1", False, "AssertionError")]
    assert first["tokens_per_s"] == 20.0 and first["avg_seconds"] is not None
    assert first["processor"] == "100% GPU" and first["load_seconds"] is not None and first["errors"] == []
    assert first["think"] is None and not first["stopped_by_time"]

    assert not missing["pulled"] and missing["done"] == 0 and "file does not exist" in missing["errors"][0]
    assert any(e.startswith("missing:1b:") for e in results["errors"])

    assert (oss["done"], oss["passed"], oss["think"]) == (2, 1, "low")  # the thinking's draft was not tested
    assert oss["tasks"][0]["thinking_chars"] == len(THOUGHT) and oss["tasks"][0]["reply"] == FIXED

    chats = fake_ollama["chats"]
    assert len(chats) == 4
    for body in chats:
        assert body["stream"] is False and len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
        assert body["options"]["temperature"] == 0 and body["options"]["num_ctx"] == 4096
    assert [b.get("think") for b in chats] == [None, None, "low", "low"]
    assert [b["options"]["num_predict"] for b in chats] == [1024, 1024, 3072, 3072]
    assert chats[0]["messages"][0]["content"] == fixeval.prompt_for(TASKS[0])
    assert [b["options"] for b in fake_ollama["loads"]] == [chats[0]["options"], chats[2]["options"]]  # same num_ctx
    # Each downloaded model is unloaded and deleted after its turn; the one that never downloaded is left alone.
    assert fake_ollama["unloads"] == fake_ollama["deletes"] == ["qwen-fake:7b", "gpt-oss:20b"]

    # Saved after each model: a copy where the first model is finished and the next has not started.
    assert any(len(s["models"]) == 1 and s["models"][0]["done"] == 2 and s["models"][0]["minutes"] is not None
               for s in saves)
    assert any(len(s["models"]) == 2 and s["models"][1]["errors"] for s in saves)
    assert saves[-1] == results
    for text in ("qwen-fake:7b", "1/2", "50%", "download failed", "think low", "100% GPU"):
        assert text in out, text


def test_a_model_stops_at_its_time_limit(tmp_path, fake_ollama) -> None:
    results, _ = run_notebook(tmp_path, fake_ollama, ["qwen-fake:7b"], minutes=0)
    m = results["models"][0]
    assert m["pulled"] and m["stopped_by_time"] and m["done"] == 0 and m["pass_at_1"] is None
    assert fake_ollama["chats"] == [] and fake_ollama["deletes"] == ["qwen-fake:7b"]
    assert "stopped at the time limit after 0 of 2 tasks" in "\n".join(fixeval.table(results))


def test_a_test_that_never_ends_is_a_fail(tmp_path, fake_ollama, monkeypatch) -> None:
    looping = {**TASKS[0], "test": "while True:\n    pass\n"}
    scope = notebook_scope(tmp_path)
    scope["TEST_TIMEOUT"] = 1
    entry: dict = {}
    scope["evaluate_model"](fake_ollama["url"], "qwen-fake:7b", [looping], entry, 45, scope["OPTIONS"])
    assert entry["done"] == 1 and entry["passed"] == 0 and entry["tasks"][0]["why"] == "timeout after 1 s"


def test_failed_requests_count_as_fails_and_stop_the_model(tmp_path, fake_ollama, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise TimeoutError("timed out")

    entry: dict = {}
    monkeypatch.setattr(fixeval, "ask", broken)
    fixeval.evaluate_model(fake_ollama["url"], "qwen-fake:7b", TASKS * 3, entry, 45, {"temperature": 0},
                           log=lambda *a: None)
    assert entry["done"] == 3 and entry["passed"] == 0 and entry["total"] == 6
    assert entry["errors"][-1] == "3 requests in a row failed: stopped this model"
    assert all(t["why"].startswith("request failed: TimeoutError") for t in entry["tasks"])
    assert "stopped after 3 of 6 tasks" in "\n".join(fixeval.table({"models": [entry]}))


# ---- the runner ----

RESULT = {
    "benchmark": "HumanEvalFix (Python)", "dataset": "bigcode/humanevalpack", "licence": "MIT", "tasks_total": 164,
    "tasks_used": 164, "ollama_version": "0.40.3", "gpus": "Tesla T4, 15360 MiB\nTesla T4, 15360 MiB",
    "setting": fixeval.SETTING, "options": {"temperature": 0},
    "models": [
        {"model": "qwen2.5-coder:7b", "pulled": True, "size_gb": 4.7, "pull_seconds": 60.0, "load_seconds": 9.0,
         "total": 164, "done": 164, "passed": 120, "pass_at_1": 0.7317, "rate_done": 0.7317, "avg_seconds": 4.2,
         "tokens_per_s": 41.0, "processor": "100% GPU", "minutes": 13.0, "errors": [], "tasks": []},
        {"model": "gpt-oss:20b", "pulled": True, "size_gb": 13.8, "pull_seconds": 200.0, "load_seconds": 30.0,
         "total": 164, "done": 90, "passed": 70, "pass_at_1": None, "rate_done": 0.7778, "avg_seconds": 28.0,
         "tokens_per_s": 30.0, "processor": "100% GPU", "stopped_by_time": True, "think": "low", "minutes": 45.0,
         "errors": [], "tasks": []},
        {"model": "missing:1b", "pulled": False, "total": 164, "done": 0, "passed": 0,
         "errors": ["RuntimeError('file does not exist')"]},
    ],
    "errors": ["missing:1b: RuntimeError('file does not exist')"],
}


class FakeKaggle(FakeApi):
    """test_try.py's fake Kaggle, handing out coder_eval.json instead."""

    def kernels_output(self, kernel, path, file_pattern=None, force=False):
        self.calls["pattern"] = file_pattern
        log = [{"stream_name": "stdout", "time": 1, "data": "RuntimeError: Could not download HumanEvalFix\n"}]
        (Path(path) / "wimb-coder-eval.log").write_text(json.dumps(log))
        if self.results is not None:
            (Path(path) / "coder_eval.json").write_text(json.dumps(self.results), encoding="utf-8")


def test_runner_pushes_a_private_gpu_run_with_internet() -> None:
    api = FakeKaggle(RESULT)
    run_eval.push(api, "me/wimb-coder-eval", run_eval.notebook_with(["qwen2.5-coder:7b"], 3))
    meta = api.calls["meta"]
    assert meta["id"] == "me/wimb-coder-eval" and meta["title"] == "wimb-coder-eval"
    assert meta["code_file"] == "coder_eval.ipynb" and meta["kernel_sources"] == []
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert api.calls["acc"] == "NvidiaTeslaT4"
    s = run_eval.settings(api.calls["notebook"])
    assert s["MODELS"] == ["qwen2.5-coder:7b"] and s["LIMIT"] == 3


def run_main(api, tmp_path: Path, monkeypatch, *argv: str) -> int:
    """run_eval.main() against the fake Kaggle: its exit code."""
    monkeypatch.setattr(run_eval, "OUT", tmp_path / "out")
    monkeypatch.setattr(run_eval, "connect", lambda slug: (api, f"me/{slug}"))
    monkeypatch.setattr(run_eval, "check_gpu", lambda api, need_hours: api.calls.update(need_hours=need_hours))
    monkeypatch.setattr(run_on_kaggle.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv])
    with pytest.raises(SystemExit) as stop:
        run_eval.main()
    return stop.value.code


def test_runner_main_prints_the_table(tmp_path, monkeypatch, capsys) -> None:
    api = FakeKaggle(RESULT)
    assert run_main(api, tmp_path, monkeypatch, "--models", "qwen2.5-coder:7b,gpt-oss:20b", "--limit", "20") == 0
    assert api.calls["need_hours"] == round((2 * 50 + 15) / 60, 1)
    assert run_eval.settings(api.calls["notebook"])["LIMIT"] == 20
    assert api.calls["pattern"] == r"(coder_eval\.json|.*\.log|.*\.ipynb)$"
    out = capsys.readouterr().out
    lines = out.splitlines()
    row = next(line for line in lines if line.startswith("qwen2.5-coder:7b "))
    assert "120/164" in row and "73%" in row and "4.2" in row and "100% GPU" in row
    row = next(line for line in lines if line.startswith("gpt-oss:20b "))
    assert "70/90 of 164" in row and "78%" in row and "stopped at the time limit after 90 of 164 tasks" in row
    assert "think low" in row
    row = next(line for line in lines if line.startswith("missing:1b "))
    assert "download failed" in row
    for text in ("164 of 164 tasks from bigcode/humanevalpack (licence MIT)", "Ollama 0.40.3",
                 "download 4.7 GB in 60.0 s", "error: RuntimeError('file does not exist')"):
        assert text in out, text
    assert "Kaggle's log" not in out and "No model produced a score" not in out


@pytest.mark.parametrize(("api", "says"), [
    (FakeKaggle(None), "No coder_eval.json"),
    (FakeKaggle({**RESULT, "models": RESULT["models"][2:]}), "No model produced a score"),
    (FakeKaggle({**RESULT, "models": []}, final=Status.ERROR), "The run did not finish. Kaggle's log:"),
])
def test_runner_fails_when_no_model_scored(api, says, tmp_path, monkeypatch, capsys) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "coder_eval.json").write_text(json.dumps(RESULT))  # an older run's file must not count
    assert run_main(api, tmp_path, monkeypatch) == 1
    out = capsys.readouterr().out
    assert says in out
    if "log" in says:
        assert "Could not download HumanEvalFix" in out


def test_runner_reports_a_partial_run_that_did_not_finish(tmp_path, monkeypatch, capsys) -> None:
    api = FakeKaggle(RESULT, final=Status.ERROR)
    assert run_main(api, tmp_path, monkeypatch, "--wait") == 0  # the scores it got are kept
    out = capsys.readouterr().out
    assert "120/164" in out and "The run did not finish" in out
    assert "meta" not in api.calls and "need_hours" not in api.calls  # --wait: nothing pushed


@pytest.mark.parametrize("argv", [("--models", "a b"), ("--models", ""), ("--limit", "0"), ("--limit", "x")])
def test_bad_options_stop_before_kaggle(argv, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_eval.py", *argv])
    monkeypatch.setattr(run_eval, "connect", lambda slug: pytest.fail("must not reach Kaggle"))
    with pytest.raises(SystemExit) as stop:
        run_eval.main()
    assert stop.value.code == 2

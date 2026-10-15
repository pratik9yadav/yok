"""Checks the coding agent (phase 3): coder/agent.py with a scripted fake model in a temporary git repository,
coder_agent.ipynb with a fake Ollama, coder/run_agent.py with a fake Kaggle, coder/agent_pr.py and agent.yml.
All offline.  Run:  python -m pytest"""

import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import run_on_kaggle
from coder import agent, agent_pr, build_agent, run_agent
from test_try import FakeApi, Status

WORKFLOW = HERE.parent / ".github" / "workflows" / "agent.yml"
NASTY = [
    "Fix the 'quote' and \"double\" bug",
    "Use '''triple''' and \"\"\"triple\"\"\" quotes",
    "back\\slash \\n not a newline \\",
    "line one\nline two\r\nline three\ttab",
    "हिंदी में README ठीक करो — ünïcödé ✓ 😀",
    "--help is not an option here",
    "TASK = 'x'\nimport os; os.system('echo hacked')",
    "$(rm -rf /) `whoami` ${HOME} && echo hi; | > /dev/null",
    "odd breaks:     \x85 \x0b \x0c end",
    "@owner please <!-- hidden --> <script>alert(1)</script> ``` fence ```` four",
]


# ---- a scripted fake model and a small git repository ----

def bash(command: str, thought: str = "THOUGHT: next step") -> str:
    return f"{thought}\n\n```bash\n{command}\n```"


DONE_REPLY = "I fixed add in calc.py (minus to plus) and added test_more.py. Tests pass.\n\n```bash\necho AGENT_DONE\n```"


class Scripted:
    """A fake model: hands out the scripted replies in turn (an exception is raised) and keeps every list of
    messages it was sent. After the script it keeps asking to run `echo again`."""

    name = "fake-coder"

    def __init__(self, *replies, clock: list | None = None, minutes_per_reply: float = 0) -> None:
        self.replies, self.seen, self.clock, self.step = list(replies), [], clock, minutes_per_reply
        self.last = {}

    def __call__(self, messages: list[dict]) -> str:
        self.seen.append(copy.deepcopy(messages))
        if self.clock is not None:
            self.clock[0] += self.step * 60
        reply = self.replies.pop(0) if self.replies else bash("echo again")
        if isinstance(reply, Exception):
            raise reply
        return reply


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A tiny repository with a bug (add subtracts), a test that shows it, a guide and a workflow."""
    path = tmp_path / "repo"
    (path / ".github" / "workflows").mkdir(parents=True)
    (path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (path / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    (path / "README.md").write_text("# calc\n")
    (path / "CLAUDE.md").write_text("Run python -m pytest -q.\n")
    (path / "old.txt").write_text("remove me\n")
    (path / ".github" / "workflows" / "ci.yml").write_text("name: ci\n")
    git(path, "init", "-q")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "start")
    return path


def run(model, repo: Path, tmp_path: Path, **kwargs) -> dict:
    kwargs.setdefault("say", lambda *a: None)
    kwargs.setdefault("sleep", lambda s: None)
    kwargs.setdefault("changed", lambda workdir: True)  # most tests are about other things; see the tests below
    return agent.run("Fix add in calc.py", model, str(repo), log_path=str(tmp_path / "agent_log.json"), **kwargs)


FIX = bash("python3 - <<'EOF'\nfrom pathlib import Path\np = Path('calc.py')\ntext = p.read_text()\n"
           "assert 'a - b' in text\np.write_text(text.replace('a - b', 'a + b'))\nEOF")
NEW_TEST = bash("cat > test_more.py <<'EOF'\nfrom calc import add\n\n\ndef test_zero():\n    assert add(0, 0) == 0\n"
                "EOF")


# ---- the loop ----

def test_agent_runs_commands_feeds_output_back_and_stops_on_done(repo, tmp_path) -> None:
    model = Scripted(bash("cat calc.py", "THOUGHT: read the code first"), FIX, NEW_TEST,
                     bash("python -m pytest -q"), bash("git rm -q old.txt"), DONE_REPLY)
    log = run(model, repo, tmp_path)
    assert log["status"] == "done" and len(log["steps"]) == 6 and len(model.seen) == 6
    assert log["summary"].startswith("I fixed add in calc.py") and "AGENT_DONE" not in log["summary"]
    first = model.seen[0]
    assert first[0]["role"] == "system" and "junior software engineer" in first[0]["content"]
    assert "Fix add in calc.py" in first[1]["content"] and "calc.py" in first[1]["content"]  # the task, the files
    assert "CLAUDE.md, README.md" in first[1]["content"]  # the guide is named
    # The output of each command comes back in the next request.
    second = model.seen[1]
    assert second[-2] == {"role": "assistant", "content": model_reply(log, 1)}
    assert second[-1]["role"] == "user" and second[-1]["content"].startswith("Exit code: 0\nOutput:\ndef add(a, b):")
    assert "(Step 1 of 40; about 45 minutes left.)" in second[-1]["content"]
    assert "passed" in model.seen[4][-1]["content"]
    step = log["steps"][0]
    assert step["thought"] == "read the code first" and step["command"] == "cat calc.py" and step["exit_code"] == 0
    assert "return a - b" in step["output"] and step["seconds"] >= 0 and "model_seconds" in step
    assert (repo / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    saved = json.loads((tmp_path / "agent_log.json").read_text(encoding="utf-8"))
    assert saved["status"] == "done" and len(saved["steps"]) == 6 and saved["model"] == "fake-coder"
    assert saved["steps"][3]["command"] == "python -m pytest -q" and saved["num_ctx"] == 32768


def model_reply(log: dict, n: int) -> str:
    return log["steps"][n - 1]["reply"]


def test_patch_has_new_changed_and_deleted_files_and_no_caches(repo, tmp_path) -> None:
    base = git(repo, "rev-parse", "HEAD").strip()
    log = run(Scripted(FIX, NEW_TEST, bash("python -m pytest -q && ruff --version >/dev/null 2>&1; true"),
                       bash("rm old.txt && mkdir -p pkg/__pycache__ && touch pkg/__pycache__/x.pyc z.pyc"),
                       bash("git add -A && git commit -qm 'agent commit' --no-verify || true"), DONE_REPLY),
              repo, tmp_path)
    assert log["status"] == "done"
    patch, files = agent.make_patch(str(repo), base)
    assert sorted(files) == ["calc.py", "old.txt", "test_more.py"]  # the commit is included; caches are not
    assert ".pytest_cache" not in patch and "__pycache__" not in patch and ".pyc" not in patch
    assert "new file mode" in patch and "deleted file mode" in patch
    fresh = tmp_path / "fresh"
    subprocess.run(["git", "clone", "-q", str(repo), str(fresh)], check=True)
    git(fresh, "checkout", "-q", base)
    (tmp_path / "changes.patch").write_text(patch)
    git(fresh, "apply", "--index", str(tmp_path / "changes.patch"))
    assert (fresh / "test_more.py").exists() and not (fresh / "old.txt").exists()
    assert agent.run_tests(str(fresh))["passed"]


def test_a_binary_file_survives_the_patch(repo, tmp_path) -> None:
    base = git(repo, "rev-parse", "HEAD").strip()
    (repo / "logo.png").write_bytes(bytes(range(256)) * 4)
    patch, files = agent.make_patch(str(repo), base)
    assert files == ["logo.png"] and "GIT binary patch" in patch
    stat = run_agent.diff_stat(patch)
    assert stat == [{"file": "logo.png", "added": 0, "removed": 0, "binary": True}]


def test_max_steps_stops_the_agent(repo, tmp_path) -> None:
    model = Scripted()  # never says it is done
    log = run(model, repo, tmp_path, max_steps=5)
    assert log["status"] == "stopped" and len(log["steps"]) == 5 and len(model.seen) == 5
    assert "all 5 steps" in log["summary"]
    assert "Finish soon" in model.seen[-1][-1]["content"]  # warned near the end


def test_the_time_limit_stops_the_agent(repo, tmp_path) -> None:
    now = [1000.0]
    model = Scripted(clock=now, minutes_per_reply=10)
    log = run(model, repo, tmp_path, minutes=25, clock=lambda: now[0])
    assert log["status"] == "stopped" and len(log["steps"]) == 3 and "25 minutes" in log["summary"]
    assert log["minutes"] == 30.0


def test_long_output_is_cut_in_the_middle(repo, tmp_path) -> None:
    model = Scripted(bash("python3 -c \"print('A' * 30000 + 'END')\""), DONE_REPLY)
    log = run(model, repo, tmp_path)
    seen = model.seen[1][-1]["content"]
    assert "characters cut here" in seen and seen.count("A") < 10_500 and "END" in seen
    assert len(log["steps"][0]["output"]) < agent.MAX_OUTPUT + 200
    assert agent.truncate("x" * 10) == "x" * 10


def test_commands_have_no_keyboard_no_secrets_and_a_time_limit(repo, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLE_USER_SECRETS_TOKEN", "s3cret")
    monkeypatch.setenv("GH_TOKEN", "s3cret")
    monkeypatch.setenv("MY_VISIBLE_SETTING", "shown")
    started = time.time()
    model = Scripted(bash("read line; echo \"exit $?\""), bash("sleep 30; echo late"), bash("env"), DONE_REPLY)
    log = run(model, repo, tmp_path, command_timeout=1)
    assert time.time() - started < 20
    assert log["steps"][0]["output"].strip() == "exit 1"  # stdin is closed: read gets nothing at once
    slow = log["steps"][1]
    assert slow["timed_out"] and slow["exit_code"] == -9 and "late" not in slow["output"]
    assert "longer than 1 seconds" in slow["output"] and "Exit code: timeout" in model.seen[2][-1]["content"]
    env = log["steps"][2]["output"]
    assert "s3cret" not in env and "MY_VISIBLE_SETTING=shown" in env and "GIT_PAGER=cat" in env


def test_a_reply_without_a_bash_block_is_asked_again_and_counts_as_a_step(repo, tmp_path) -> None:
    model = Scripted("I will look at the files now.", "```python\nprint('not bash')\n```", bash("ls"), DONE_REPLY)
    log = run(model, repo, tmp_path)
    assert log["status"] == "done" and len(log["steps"]) == 4
    assert [s.get("note") for s in log["steps"]] == ["no bash block", "no bash block", None, None]
    assert model.seen[1][-1] == {"role": "user", "content": agent.NO_BLOCK}
    assert log["steps"][0]["command"] is None and log["steps"][0]["exit_code"] is None


def test_three_replies_without_a_block_in_a_row_stop_the_agent(repo, tmp_path) -> None:
    model = Scripted("hmm", "```bash\ncat > big.py <<'EOF'\nx = 1\n", "still nothing", bash("ls"))
    log = run(model, repo, tmp_path)
    assert log["status"] == "error" and len(log["steps"]) == 3 and "3 times in a row" in log["summary"]
    assert model.seen[2][-1]["content"] == agent.CUT_OFF  # the second reply was cut off


def test_the_question_path(repo, tmp_path) -> None:
    question = ("QUESTION: The task says 'make it faster', but which function? calc.py has only add().\n\n"
                "```bash\necho AGENT_QUESTION\n```")
    log = run(Scripted(bash("ls"), question), repo, tmp_path)
    assert log["status"] == "question" and log["summary"].startswith("The task says 'make it faster'")
    assert len(log["steps"]) == 2 and git(repo, "status", "--porcelain") == ""


def test_other_ways_to_finish(repo, tmp_path) -> None:
    log = run(Scripted("All done: nothing to change.\nAGENT_DONE"), repo, tmp_path)  # no block, a bare marker
    assert log["status"] == "done" and log["summary"] == "All done: nothing to change.\nAGENT_DONE"
    # `tests && echo AGENT_DONE`: done only when the tests pass
    model = Scripted(bash("python -m pytest -q && echo AGENT_DONE"), FIX,
                     bash("python -m pytest -q && echo AGENT_DONE", "Fixed and tested."))
    log = run(model, repo, tmp_path)
    assert log["status"] == "done" and len(log["steps"]) == 3 and log["summary"].startswith("Fixed and tested.")
    log = run(Scripted(bash("echo AGENT_DONE\necho 'Changed calc.py.'", "")), repo, tmp_path)
    assert log["summary"] == "Changed calc.py."  # the text printed after the marker


def test_only_the_first_of_several_blocks_runs(repo, tmp_path) -> None:
    model = Scripted("```bash\necho one\n```\nthen\n```bash\necho two\n```", DONE_REPLY)
    log = run(model, repo, tmp_path)
    assert log["steps"][0]["output"].strip() == "one"
    assert "only the first ran" in model.seen[1][-1]["content"]


def test_the_same_command_three_times_gets_a_nudge(repo, tmp_path) -> None:
    model = Scripted(bash("ls"), bash("ls"), bash("ls"), DONE_REPLY)
    run(model, repo, tmp_path)
    assert "same command 3 times" in model.seen[3][-1]["content"]
    assert "same command" not in model.seen[2][-1]["content"]


def test_failed_model_requests_are_retried_then_stop(repo, tmp_path) -> None:
    sleeps = []
    model = Scripted(RuntimeError("HTTP 500: busy"), bash("ls"), DONE_REPLY)
    log = run(model, repo, tmp_path, sleep=sleeps.append)
    assert log["status"] == "done" and len(log["steps"]) == 2 and sleeps == [10]
    assert "HTTP 500: busy" in log["errors"][0]
    log = run(Scripted(*[TimeoutError("slow")] * 3), repo, tmp_path)
    assert log["status"] == "error" and log["steps"] == [] and "3 times in a row" in log["summary"]


def test_messages_are_trimmed_to_fit_num_ctx(repo, tmp_path) -> None:
    big = bash("python3 -c \"print('line of output ' * 200)\"")
    model = Scripted(*[big] * 9, DONE_REPLY)
    num_ctx, num_predict = 4000, 500
    log = run(model, repo, tmp_path, num_ctx=num_ctx, num_predict=num_predict)
    budget = num_ctx - num_predict - 512
    assert log["status"] == "done" and log["steps_cut_from_messages"] > 0
    for messages in model.seen:
        assert agent.tokens(messages) <= budget
        assert messages[0]["content"] == model.seen[0][0]["content"]  # the system prompt is never cut
        assert "Fix add in calc.py" in messages[1]["content"]  # nor the task
    last = model.seen[-1]
    assert "Earlier steps were cut to save memory" in last[1]["content"] and "step 1: `python3 -c" in last[1]["content"]
    assert len(last) >= 4 and last[-2]["content"] == big  # the newest step stays
    assert "(Step 9 of 40" in last[-1]["content"]
    # With more room the newest KEEP_LAST_STEPS steps always stay whole.
    model = Scripted(*[big] * 12, DONE_REPLY)
    log = run(model, repo, tmp_path, num_ctx=8000, num_predict=500)
    assert log["steps_cut_from_messages"] > 0
    assert all(agent.tokens(m) <= 8000 - 500 - 512 for m in model.seen)
    assert len(model.seen[-1]) >= 2 + 2 * agent.KEEP_LAST_STEPS
    assert all("characters cut" not in m["content"] for m in model.seen[-1])


@pytest.mark.parametrize(("reply", "command", "thought"), [
    (bash("ls -la", "THOUGHT: look"), "ls -la", "look"),
    ("**THOUGHT**: look\n```sh\nls\n```", "ls", "look"),
    ("Let me see.\n```\ngrep -rn add .\n```", "grep -rn add .", "Let me see."),
    ("<think>maybe ```bash\nrm -rf /\n```</think>\nOK.\n```bash\nls\n```", "ls", "OK."),
    ("Write it:\n```bash\ncat > a.md <<'EOF'\nuse ```python fences``` inline\nEOF\n```", None, None),
    ("```python\nx = 1\n```\nand\n```bash\necho hi\n```", "echo hi", "and"),
])
def test_parse_reply(reply, command, thought) -> None:
    parsed = agent.parse_reply(reply)
    if command is None:  # a ``` that is not alone on its line does not end the block
        assert parsed["command"] == "cat > a.md <<'EOF'\nuse ```python fences``` inline\nEOF"
        return
    assert parsed["command"] == command and parsed["thought"] == thought


def test_run_tests_reports_exit_and_last_lines(repo) -> None:
    before = agent.run_tests(str(repo))
    assert before["exit_code"] == 1 and not before["passed"] and "1 failed" in before["last_lines"]
    assert before["command"] == "python -m pytest -q"


# ---- the default model: Ollama ----

@pytest.fixture
def fake_ollama():
    """A tiny stand-in for Ollama: version, ps, pull, generate (load/unload) and chat. A model loaded with num_ctx
    32768 is shown part on the CPU when state["fits"] is False."""
    state: dict = {"chats": [], "loads": [], "unloads": [], "loaded": None, "fits": True}

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
                loaded = state["loaded"]
                if not loaded:
                    self.respond({"models": []})
                    return
                size = 20_000_000_000
                vram = size if state["fits"] or loaded[1] < 32768 else size * 3 // 4
                self.respond({"models": [{"name": loaded[0], "model": loaded[0], "size": size, "size_vram": vram}]})
            else:
                self.respond({"version": "0.0.0-fake"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/api/pull":
                steps = [{"status": "pulling manifest"},
                         {"status": "pulling 0123456789abcdef", "digest": "sha256:01", "total": 18_600_000_000},
                         {"status": "success"}]
                data = "".join(json.dumps(s) + "\n" for s in steps).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/api/generate":
                if body.get("keep_alive") == 0:
                    state["unloads"].append(body["model"])
                    state["loaded"] = None
                else:
                    state["loads"].append(body)
                    state["loaded"] = (body["model"], body["options"]["num_ctx"])
                self.respond({"done": True})
            elif self.path == "/api/chat":
                state["chats"].append(body)
                if body["model"] == "missing":
                    self.respond({"error": "model 'missing' not found"}, 404)
                    return
                self.respond({"message": {"role": "assistant", "content": DONE_REPLY}, "done": True,
                              "done_reason": "stop", "eval_count": 30, "prompt_eval_count": 900})
            else:
                self.respond({"error": "unknown"}, 404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()
    server.server_close()


def test_ollama_chat_sends_the_messages_and_options(fake_ollama) -> None:
    options = {"temperature": 0.7, "num_ctx": 32768, "num_predict": 4096}
    model = agent.OllamaChat(fake_ollama["url"], "qwen3-coder:30b", options)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert model(messages) == DONE_REPLY
    (body,) = fake_ollama["chats"]
    assert body == {"model": "qwen3-coder:30b", "messages": messages, "stream": False, "options": options,
                    "keep_alive": "90m"}
    assert model.last == {"tokens": 30, "prompt_tokens": 900, "done_reason": "stop"}
    with pytest.raises(RuntimeError, match="HTTP 404: .*not found"):
        agent.OllamaChat(fake_ollama["url"], "missing", options)(messages)


# ---- the notebook ----

def nb_cell(title: str, nb: dict | None = None) -> str:
    nb = nb or json.loads((HERE / build_agent.NOTEBOOK).read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith(title))["source"])


def notebook_scope(tmp_path: Path, nb: dict | None = None) -> dict:
    """The settings and helpers cells, run as the notebook would, saving into tmp_path/out."""
    scope: dict = {}
    exec(nb_cell("#@title 1.", nb), scope)  # noqa: S102 - our own generated cell
    out = tmp_path / "out"
    out.mkdir()
    scope.update(WORKING_DIR=str(out), MODELS_DIR=str(tmp_path / "models"), REPO_DIR=str(tmp_path / "work"))
    exec(nb_cell("#@title 2.", nb), scope)  # noqa: S102
    scope["OLLAMA_LOG"] = str(tmp_path / "ollama.log")
    return scope


def test_notebook_is_built_from_current_files() -> None:
    current = (HERE / build_agent.NOTEBOOK).read_text(encoding="utf-8")
    assert current == build_agent.text(build_agent.notebook()), \
        "coder_agent.ipynb is out of date: run python coder/build.py"


def test_notebook_settings_and_helpers(tmp_path: Path) -> None:
    settings: dict = {}
    exec(nb_cell("#@title 1."), settings)  # noqa: S102
    scope = notebook_scope(tmp_path)
    for name in ("TASK", "REPO_URL", "COMMIT", "MODEL", "MAX_STEPS", "AGENT_MINUTES", "NUM_CTX", "NUM_PREDICT",
                 "OPTIONS"):
        assert scope[name] == settings[name], f"the helpers cell changes {name}"
    assert settings["MODEL"] == "qwen3-coder:30b" and settings["MAX_STEPS"] == 40 and settings["AGENT_MINUTES"] == 45
    assert settings["NUM_CTX"] == 32768 and settings["NUM_CTX_FALLBACK"] == 16384
    assert settings["TASK"] == "" and settings["COMMIT"] == ""
    assert settings["REPO_URL"] == "https://github.com/pratik9yadav/yok.git"
    assert "/kaggle/working" not in settings["MODELS_DIR"] + settings["REPO_DIR"]
    assert scope["RESULTS"]["status"] == "error" and scope["RESULTS_FILE"] == "agent_result.json"
    for name in ("run", "OllamaChat", "make_patch", "run_tests", "pull", "post", "processor", "unload"):
        assert callable(scope[name]), name
    ollama = nb_cell("#@title 3.")
    assert build_agent.START_OLLAMA.strip("\n") in ollama
    assert ollama.index('os.environ["OLLAMA_MODELS"] = MODELS_DIR') < ollama.index("ollama serve")


@pytest.mark.parametrize("fits", [True, False])
def test_notebook_loads_the_model_and_falls_back_to_a_smaller_context(tmp_path, fake_ollama, fits) -> None:
    fake_ollama["fits"] = fits
    scope = notebook_scope(tmp_path)
    default = 'OLLAMA_URL = "http://127.0.0.1:11434"'
    exec(nb_cell("#@title 3.").replace(default, f'OLLAMA_URL = "{fake_ollama["url"]}"'), scope)  # noqa: S102
    exec(nb_cell("#@title 4."), scope)  # noqa: S102
    results = json.loads((tmp_path / "out" / "agent_result.json").read_text(encoding="utf-8"))
    assert results["size_gb"] == 18.6 and results["ollama_version"] == "0.0.0-fake"
    loads = [b["options"]["num_ctx"] for b in fake_ollama["loads"]]
    if fits:
        assert loads == [32768] and results["num_ctx"] == 32768 and fake_ollama["unloads"] == []
    else:
        assert loads == [32768, 16384] and results["num_ctx"] == 16384
        assert fake_ollama["unloads"] == ["qwen3-coder:30b"]
    assert results["processor"] == "100% GPU"


def test_notebook_end_to_end_with_a_fake_model(tmp_path, repo, monkeypatch) -> None:
    base = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "commit", "-q", "--allow-empty", "-m", "newer")  # the run must start at COMMIT, not the newest
    nb = run_agent.notebook_with("Fix add in calc.py", "o/r", base)
    scope = notebook_scope(tmp_path, nb)
    scope.update(REPO_URL=str(repo), OLLAMA_URL="http://unused")
    scope["RESULTS"]["num_ctx"] = 32768
    fake = Scripted(FIX, NEW_TEST, bash("python -m pytest -q"), DONE_REPLY)
    made = []

    def fake_chat(url, model, options):
        made.append((model, options))
        return fake

    exec(nb_cell("#@title 5.", nb), scope)  # noqa: S102
    scope["OllamaChat"] = fake_chat
    for title in ("#@title 6.", "#@title 7."):
        exec(nb_cell(title, nb), scope)  # noqa: S102
    out = tmp_path / "out"
    result = json.loads((out / "agent_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "done" and result["steps"] == 4 and result["commit"] == base
    assert result["task"] == "Fix add in calc.py" and result["summary"].startswith("I fixed add")
    assert result["tests_before"]["exit_code"] == 1 and result["tests_after"]["exit_code"] == 0
    assert sorted(result["files_changed"]) == ["calc.py", "test_more.py"] and result["errors"] == []
    assert made == [("qwen3-coder:30b", {**scope["OPTIONS"], "num_ctx": 32768, "num_predict": 4096})]
    patch = (out / "changes.patch").read_text(encoding="utf-8")
    assert result["patch_bytes"] == len(patch.encode()) and "+    return a + b" in patch
    log = json.loads((out / "agent_log.json").read_text(encoding="utf-8"))
    assert len(log["steps"]) == 4 and log["status"] == "done"


def test_notebook_keeps_what_it_has_when_the_agent_fails(tmp_path, repo) -> None:
    scope = notebook_scope(tmp_path)
    scope.update(REPO_URL=str(repo), TASK="Fix it", OLLAMA_URL="http://unused")
    scope["RESULTS"]["num_ctx"] = 32768
    exec(nb_cell("#@title 5."), scope)  # noqa: S102

    def broken(url, model, options):
        raise ConnectionError("Ollama is gone")

    scope["OllamaChat"] = broken
    (Path(scope["REPO_DIR"]) / "notes.txt").write_text("left behind\n")
    for title in ("#@title 6.", "#@title 7."):
        exec(nb_cell(title), scope)  # noqa: S102
    result = json.loads((tmp_path / "out" / "agent_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "error" and "Ollama is gone" in result["summary"]
    assert any(e.startswith("run the agent:") for e in result["errors"])
    assert result["files_changed"] == ["notes.txt"] and (tmp_path / "out" / "changes.patch").exists()


# ---- the runner ----

@pytest.mark.parametrize("task", NASTY)
def test_task_line_round_trip(task: str) -> None:
    nb = run_agent.notebook_with(task, "someone/some-repo", "0123abc")
    s = run_agent.settings(nb)
    assert s["TASK"] == task.strip() and s["REPO_URL"] == "https://github.com/someone/some-repo.git"
    assert s["COMMIT"] == "0123abc"
    lines = "".join(run_agent.settings_cell(nb)["source"]).splitlines()
    assert sum(line.startswith("TASK = ") for line in lines) == 1
    assert sum(line.startswith("COMMIT = ") for line in lines) == 1
    original = "".join(run_agent.settings_cell(json.loads(run_agent.NOTEBOOK.read_text("utf-8")))["source"])
    assert len(lines) == len(original.splitlines())  # the task stays on its one line
    assert run_agent.settings(json.loads(json.dumps(nb)))["TASK"] == task.strip()  # survives the notebook file


@pytest.mark.parametrize(("task", "repo", "commit"), [
    ("", "o/r", None), ("   \n", "o/r", None), ("x" * 4001, "o/r", None), ("ok", "o/r; rm", None),
    ("ok", "o/r\nTASK = 1", None), ("ok", "o", None), ("ok", "o/r", "main; rm"), ("ok", "o/r", "XYZ1234"),
])
def test_bad_settings_are_refused(task, repo, commit) -> None:
    with pytest.raises(ValueError):
        run_agent.notebook_with(task, repo, commit)


PATCH = """diff --git a/calc.py b/calc.py
index 1111111..2222222 100644
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
diff --git a/test_more.py b/test_more.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/test_more.py
@@ -0,0 +1,2 @@
+def test_zero():
+    assert True
"""
RESULT = {"status": "done", "summary": "Fixed add.\nTests pass.", "task": "Fix add", "repo": "https://x/o/r.git",
          "commit": "0123abc", "model": "qwen3-coder:30b", "num_ctx": 32768, "steps": 7, "minutes": 12.5,
          "processor": "100% GPU", "files_changed": ["calc.py", "test_more.py"],
          "tests_before": {"exit_code": 1, "seconds": 2.0, "last_lines": "1 failed in 0.1s"},
          "tests_after": {"exit_code": 0, "seconds": 2.1, "last_lines": "2 passed in 0.1s"}, "errors": []}


class FakeKaggle(FakeApi):
    """test_try.py's fake Kaggle, handing out the agent's files instead."""

    def __init__(self, result: dict | None = RESULT, patch: str | None = PATCH, **kwargs) -> None:
        super().__init__(result, **kwargs)
        self.patch = patch

    def kernels_output(self, kernel, path, file_pattern=None, force=False):
        self.calls["pattern"] = file_pattern
        log = [{"stream_name": "stdout", "time": 1, "data": "RuntimeError: Ollama did not start\n"}]
        (Path(path) / "wimb-coder-agent.log").write_text(json.dumps(log))
        if self.results is not None:
            (Path(path) / "agent_result.json").write_text(json.dumps(self.results), encoding="utf-8")
            (Path(path) / "agent_log.json").write_text(json.dumps({"steps": []}), encoding="utf-8")
        if self.patch is not None:
            (Path(path) / "changes.patch").write_text(self.patch, encoding="utf-8")


def test_runner_pushes_a_private_gpu_run_with_internet() -> None:
    api = FakeKaggle()
    run_agent.push(api, "me/wimb-coder-agent", run_agent.notebook_with("Fix add", "o/r", "abcdef1"))
    meta = api.calls["meta"]
    assert meta["id"] == "me/wimb-coder-agent" and meta["title"] == "wimb-coder-agent"
    assert meta["code_file"] == "coder_agent.ipynb" and meta["kernel_sources"] == []
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert api.calls["acc"] == "NvidiaTeslaT4" and run_agent.SLUG == "wimb-coder-agent"
    s = run_agent.settings(api.calls["notebook"])
    assert (s["TASK"], s["COMMIT"], s["REPO_URL"]) == ("Fix add", "abcdef1", "https://github.com/o/r.git")


def run_main(api, tmp_path: Path, monkeypatch, *argv: str) -> int:
    """run_agent.main() against the fake Kaggle: its exit code."""
    monkeypatch.setattr(run_agent, "OUT", tmp_path / "out")
    monkeypatch.setattr(run_agent, "connect", lambda slug: (api, f"me/{slug}"))
    monkeypatch.setattr(run_agent, "check_gpu", lambda api, need_hours: api.calls.update(need_hours=need_hours))
    monkeypatch.setattr(run_on_kaggle.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["run_agent.py", *argv])
    with pytest.raises(SystemExit) as stop:
        run_agent.main()
    return stop.value.code


def test_runner_prints_the_summary_tests_and_diff_stat(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/fork")
    api = FakeKaggle()
    assert run_main(api, tmp_path, monkeypatch, "--task=--help me fix add", "--commit", "0123abc") == 0
    s = run_agent.settings(api.calls["notebook"])
    assert s["TASK"] == "--help me fix add" and s["REPO_URL"] == "https://github.com/someone/fork.git"
    assert api.calls["need_hours"] == run_agent.NEED_HOURS
    assert api.calls["pattern"] == r"(agent_result\.json|agent_log\.json|changes\.patch|.*\.log)$"
    out = capsys.readouterr().out
    for text in ("Status: done  (7 steps, 12.5 min; qwen3-coder:30b, num_ctx 32768, runs on 100% GPU)",
                 "   Fixed add.\n   Tests pass.", "Tests before: failed (exit 1, 2.0 s)", "1 failed in 0.1s",
                 "Tests after: passed (exit 0, 2.1 s)", "calc.py  +1 -1", "test_more.py  +2 -0",
                 "2 file(s) changed, +3 -1"):
        assert text in out, text
    assert (tmp_path / "out" / "changes.patch").read_text() == PATCH


def test_runner_reads_the_task_from_a_file_and_shows_a_question(tmp_path, monkeypatch, capsys) -> None:
    (tmp_path / "task.txt").write_text("Make it faster\n", encoding="utf-8")
    api = FakeKaggle({**RESULT, "status": "question", "summary": "Which function?"}, patch="")
    assert run_main(api, tmp_path, monkeypatch, "--task-file", str(tmp_path / "task.txt")) == 0
    assert run_agent.settings(api.calls["notebook"])["TASK"] == "Make it faster"
    out = capsys.readouterr().out
    assert "The agent asks:\n   Which function?" in out and "no change" in out and "No change was made" in out


@pytest.mark.parametrize(("patch", "says"), [
    (PATCH.replace("calc.py", ".github/workflows/ci.yml"), "touches .github/"),
    ('diff --git "a/\\056github/x.yml" "b/\\056github/x.yml"\n', "touches .github/"),
    (PATCH + "+" + "x" * 200_000 + "\n", "too large"),
])
def test_runner_refuses_a_bad_patch(patch, says, tmp_path, monkeypatch, capsys) -> None:
    assert run_main(FakeKaggle(patch=patch), tmp_path, monkeypatch, "--task", "Fix add") == 1
    assert says in capsys.readouterr().out


def test_patch_checks() -> None:
    assert run_agent.patch_problems(PATCH) == [] and run_agent.patch_problems("") == []
    for fine in ("my.github.io/x", ".githubrc", "docs/github/x"):
        assert run_agent.patch_problems(PATCH.replace("calc.py", fine)) == [], fine
    assert run_agent.patch_problems(PATCH.replace("b/calc.py", "b/.github")) != []  # a rename into it
    assert run_agent.paths_in('--- "a/\\303\\251 x"') == ['"a/é x"']


def test_runner_fails_without_a_result(tmp_path, monkeypatch, capsys) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "agent_result.json").write_text(json.dumps(RESULT))  # an older run's file must not count
    assert run_main(FakeKaggle(None, None, final=Status.ERROR), tmp_path, monkeypatch, "--wait") == 1
    out = capsys.readouterr().out
    assert "No agent_result.json" in out and "Ollama did not start" in out


@pytest.mark.parametrize("argv", [(), ("--task", ""), ("--task", "x", "--task-file", "y"), ("--task-file", "/no/such"),
                                  ("--task", "x", "--commit", "zzz"), ("--task", "x", "--repo", "bad repo")])
def test_bad_options_stop_before_kaggle(argv, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_agent.py", *argv])
    monkeypatch.setattr(run_agent, "connect", lambda slug: pytest.fail("must not reach Kaggle"))
    with pytest.raises(SystemExit) as stop:
        run_agent.main()
    assert stop.value.code == 2


# ---- the pull request text ----

def outside_fences(markdown: str) -> str:
    """The markdown without its fenced blocks (what GitHub would render as markdown)."""
    return re.sub(r"^(`{3,})text\n.*?\n\1$", "", markdown, flags=re.DOTALL | re.MULTILINE)


def write_outputs(folder: Path, result: dict | None = RESULT, patch: str = PATCH, checks: bool = True) -> Path:
    folder.mkdir(exist_ok=True)
    if result is not None:
        (folder / "agent_result.json").write_text(json.dumps(result), encoding="utf-8")
    (folder / "changes.patch").write_text(patch, encoding="utf-8")
    if checks:
        (folder / "runner_pytest.txt").write_text("....\n2 passed in 0.3s\n")
        (folder / "runner_pytest.exit").write_text("0\n")
        (folder / "runner_ruff.txt").write_text("calc.py:1:1: F401 unused\nFound 1 error.\n")
        (folder / "runner_ruff.exit").write_text("1\n")
    return folder


def test_pr_title_and_body_keep_the_task_as_plain_text(tmp_path, monkeypatch) -> None:
    task = NASTY[-1] + "\n" + NASTY[7]
    monkeypatch.setenv("TASK", task)
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    out = write_outputs(tmp_path / "out", {**RESULT, "summary": "Done. @owner see <script>x</script> ```` ok"})
    assert agent_pr.main(["pr"], out) == 0
    title = (out / "pr_title.txt").read_text(encoding="utf-8")
    assert title == "Agent: " + " ".join(task.split())[:60] + "…" and "\n" not in title
    body = (out / "pr_body.md").read_text(encoding="utf-8")
    rendered = outside_fences(body)
    for danger in ("@owner", "<script>", "<!--", "$(rm", "`whoami`"):
        assert danger not in rendered, danger
    assert task.split("\n")[0][:30] in body and "<script>x</script> ```` ok" in body
    for text in ("Status: **done** (7 steps, 12.5 minutes; `qwen3-coder:30b`, num_ctx 32768",
                 "before failed (exit 1, 2.0 s) → after passed (exit 0, 2.1 s)",
                 "`python -m pytest -q`: **passed** (exit 0)", "2 passed in 0.3s",
                 "`ruff check .`: **failed** (exit 1)", "F401 unused", "calc.py  +1 -1",
                 "https://github.com/o/r/actions/runs/42", "> " + agent_pr.NOTE):
        assert text in body, text
    assert agent_pr.NOTE == "Made by the free coding agent (qwen3-coder:30b on Kaggle). Review before merging."
    message = (out / "commit_message.txt").read_text(encoding="utf-8")
    assert message.startswith(title + "\n\nTask:\n") and agent_pr.NOTE in message


def test_fence_is_longer_than_any_backticks_inside() -> None:
    assert agent_pr.fence("a ```` b") == "`````text\na ```` b\n`````"
    assert agent_pr.fence("") == "```text\n(none)\n```"
    assert "\x00" not in agent_pr.fence("a\x00b c") and agent_pr.word("x; rm -rf /") == "?"
    assert agent_pr.title_for("short") == "Agent: short"


@pytest.mark.parametrize(("result", "patch", "opened", "says"), [
    (RESULT, PATCH, True, "A pull request will be opened"),
    ({**RESULT, "status": "question", "summary": "Which file?"}, "", False, "The agent asks a question"),
    ({**RESULT, "status": "question", "summary": "Which file?"}, PATCH, False, "The question:"),
    (RESULT, "", False, "the agent changed nothing"),
    (RESULT, PATCH.replace("calc.py", ".github/workflows/x.yml"), False, "touches .github/"),
    ({**RESULT, "status": "stopped"}, PATCH, True, "Coding agent: stopped"),
    (None, PATCH, False, "no result"),
])
def test_decide_writes_the_job_summary_and_the_output(result, patch, opened, says, tmp_path, monkeypatch) -> None:
    summary, output = tmp_path / "summary.md", tmp_path / "output.txt"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("TASK", "Fix add")
    out = write_outputs(tmp_path / "out", result, patch, checks=False)
    assert agent_pr.main(["decide"], out) == 0
    assert output.read_text() == f"open_pr={'true' if opened else 'false'}\n"
    assert says in summary.read_text(encoding="utf-8")
    if result and result["status"] == "question":
        assert "Which file?" in summary.read_text(encoding="utf-8")
    if not opened:
        assert agent_pr.main(["pr"], out) == 1 and not (out / "pr_body.md").exists()


# ---- the workflow ----

def workflow() -> dict:
    yaml = pytest.importorskip("yaml")  # comes with kaggle (through jupytext)
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def step_named(job: str, needle: str) -> dict:
    (step,) = [s for s in workflow()["jobs"][job]["steps"] if needle in s.get("run", "")]
    return step


def run_step(tmp_path: Path, step: dict, env: dict, *stubs: str) -> list[list[str]]:
    """Run a workflow step's bash with stand-ins for the commands in stubs; each call's arguments, in order."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "calls"
    for name in stubs:  # writes its name and arguments as one JSON line
        (bin_dir / name).write_text(f"#!{sys.executable}\nimport json, sys\n"
                                    f"open({str(calls)!r}, 'a').write(json.dumps([{name!r}, *sys.argv[1:]]) + '\\n')\n")
        (bin_dir / name).chmod(0o755)
    run_env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", **env}
    script = step["run"].replace("${{ github.run_id }}", "42").replace("${{ github.repository }}", "o/r")  # GitHub's
    subprocess.run(["bash", "-e", "-c", script], env=run_env, cwd=tmp_path, capture_output=True, check=True)
    return [json.loads(line) for line in calls.read_text().splitlines()]


def test_workflow_passes_the_task_only_through_the_environment(tmp_path: Path) -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for line in text.splitlines():  # inputs reach the scripts only as environment variables
        if "${{ inputs." in line:
            assert re.fullmatch(r" +[A-Z_]+: \$\{\{ inputs\.\w+ \}\}", line), line
    for job in workflow()["jobs"].values():
        for step in job["steps"]:
            used = set(re.findall(r"\$\{\{\s*([^}]*?)\s*\}\}", step.get("run", "")))
            assert used <= {"github.run_id", "github.repository"}, (step.get("name"), used)
    task = NASTY[7] + "\n" + NASTY[5]
    calls = run_step(tmp_path, step_named("agent", "run_agent.py"),
                     {"TASK": task, "GITHUB_SHA": "0123abc", "GITHUB_REPOSITORY": "o/r"}, "python")
    assert calls == [["python", "coder/run_agent.py", f"--task={task}", "--commit", "0123abc", "--repo", "o/r"]]


def test_workflow_opens_the_pull_request_with_the_files_as_they_are(tmp_path: Path) -> None:
    out = tmp_path / "kaggle_output_agent"
    out.mkdir()
    title = "Agent: $(touch /tmp/hacked) `id` \"quoted\" -x --body evil"
    (out / "pr_title.txt").write_text(title, encoding="utf-8")
    summary = tmp_path / "summary.md"
    calls = run_step(tmp_path, step_named("pull-request", "gh pr create"),
                     {"GH_TOKEN": "t", "GITHUB_REF_NAME": "main", "GITHUB_REPOSITORY": "o/r",
                      "GITHUB_STEP_SUMMARY": str(summary)}, "git", "gh")
    branch = "agent/run-42"
    assert ["git", "switch", "-c", branch] in calls
    assert ["git", "apply", "--index", "kaggle_output_agent/changes.patch"] in calls
    assert ["git", "commit", "-q", "--no-verify", "-F", "kaggle_output_agent/commit_message.txt"] in calls
    assert ["git", "push", "origin", f"HEAD:refs/heads/{branch}"] in calls
    assert calls[-1] == ["gh", "pr", "create", "--base", "main", "--head", branch, "--title", title,
                         "--body-file", "kaggle_output_agent/pr_body.md"]
    assert not Path("/tmp/hacked").exists() and "Opened a pull request" in summary.read_text()


def test_workflow_jobs_permissions_and_limits() -> None:
    w = workflow()
    text = WORKFLOW.read_text(encoding="utf-8")
    assert w["name"] == "Coding agent (free model)"
    on = w.get("on") or w.get(True)  # YAML 1.1 reads the key `on` as True
    task = on["workflow_dispatch"]["inputs"]["task"]
    assert task["required"] is True and task["type"] == "string"
    assert w["permissions"] == {"contents": "read"} and w["concurrency"]["group"] == "kaggle-agent"
    agent_job, pr_job = w["jobs"]["agent"], w["jobs"]["pull-request"]
    assert agent_job["permissions"] == {"contents": "read"} and agent_job["timeout-minutes"] == 120
    assert agent_job["steps"][0]["with"]["persist-credentials"] is False  # agent code runs in this job
    assert pr_job["permissions"] == {"contents": "write", "pull-requests": "write"}
    assert pr_job["needs"] == "agent" and pr_job["if"] == "needs.agent.outputs.open_pr == 'true'"
    assert not any("pytest" in s.get("run", "") for s in pr_job["steps"])  # the job that can write runs no agent code
    upload = next(s for s in agent_job["steps"] if s.get("uses", "").startswith("actions/upload-artifact"))
    assert upload["if"] == "always()" and upload["with"]["path"] == "kaggle_output_agent/"
    assert "refs/heads/main" not in text and "--force" not in text
    assert "kaggle_output_agent/" in (HERE.parent / ".gitignore").read_text().splitlines()


def test_saying_done_with_nothing_changed_is_sent_back(repo, tmp_path) -> None:
    """The first real run said "done" but had changed no file: the loop now sends it back to make the change."""
    log = run(Scripted(bash("echo AGENT_DONE"), FIX, bash("echo AGENT_DONE")), repo, tmp_path,
              changed=agent.repo_changed)
    assert log["status"] == "done" and len(log["steps"]) == 3
    assert log["steps"][0]["note"] == "said done, but nothing changed: sent back"
    assert "def add(a, b):\n    return a + b" in (repo / "calc.py").read_text()


def test_saying_done_again_and_again_with_nothing_changed_is_taken_in_the_end(repo, tmp_path) -> None:
    log = run(Scripted(*[bash("echo AGENT_DONE")] * 3), repo, tmp_path, changed=agent.repo_changed)
    assert log["status"] == "done" and len(log["steps"]) == 3
    assert [s.get("note") for s in log["steps"]][:2] == ["said done, but nothing changed: sent back"] * 2


def test_repo_changed_sees_edits_and_new_files(repo) -> None:
    assert not agent.repo_changed(str(repo))
    (repo / "new.py").write_text("x = 1\n")
    assert agent.repo_changed(str(repo))

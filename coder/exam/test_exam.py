"""Checks the engineer exam (coder phase 4): grade.py (reading replies, grading them, the reference check),
exam.ipynb (run against a fake Ollama with the 2 sample tasks), coder/exam/run_exam.py (a fake Kaggle) and the
workflow. Offline.  Run:  python -m pytest"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

import run_on_kaggle
from coder.exam import build_exam, exam, grade, run_exam
from coder.test_coder import fake_ollama  # noqa: F401 - a pytest fixture: the fake Ollama, reused
from coder.test_train import workflow_command
from test_try import FakeApi, Status

SAMPLES = HERE / "sample_tasks"
WORDS, AVERAGE = grade.load_tasks(SAMPLES)
HAS_BANDIT = grade.program("bandit", "bandit") is not None
EXPLANATION = ("I added count_words in words.py. It lowercases the text and finds the words with a regular expression "
               "that keeps letters and digits, so punctuation and extra spaces never count as words. Anything that "
               "is not a string raises TypeError with a clear message, because a silent empty result would hide bugs "
               "in the caller. The tests cover the example, empty text and bad input.")
PERFECT = (EXPLANATION + "\n\n```words.py\n" + WORDS["reference"]["words.py"] + "```\n\n```test_words.py\n"
           + WORDS["reference"]["test_words.py"] + "```\n")
# Right answers, but no tests, no explanation, 5 lint issues and eval() (bandit: medium).
SLOPPY = """```python words.py
import os, sys
import re
def count_words(text):
    if type(text) != str: raise TypeError("no")
    counts = {}
    for w in re.findall(r"[^\\W_]+", text.lower()):
        counts[w] = eval(str(counts.get(w, 0)) + "+1")
    return counts
```"""
BROKEN = "Done.\n```words.py\ndef count_words(text)\n    return {}\n```"  # a syntax error
FIXED_AVERAGE = "```stats.py\n" + AVERAGE["reference"]["stats.py"] + "```"
HALF_FIXED = "```stats.py\ndef average(numbers):\n    return sum(numbers) / len(numbers)\n```"  # [] → ZeroDivisionError


def graded(task: dict, reply: str, tmp_path: Path) -> dict:
    return grade.grade_reply(task, reply, tmp_path)


# ---- reading a reply ----

@pytest.mark.parametrize(("reply", "files"), [
    ("```inventory.py\nx = 1\n```", {"inventory.py": "x = 1\n"}),
    ("```python inventory.py\nx = 1\n```", {"inventory.py": "x = 1\n"}),
    ("```python:inventory.py\nx = 1\n```", {"inventory.py": "x = 1\n"}),
    ("```py title=\"shop/cart.py\"\nx = 1\n```", {"shop/cart.py": "x = 1\n"}),
    ("```python\n# file: inventory.py\nx = 1\n```", {"inventory.py": "x = 1\n"}),
    ("```python\n# File: shop/a.py\nx = 1\n```", {"shop/a.py": "x = 1\n"}),
    ("**inventory.py**\n```python\nx = 1\n```", {"inventory.py": "x = 1\n"}),
    ("### `shop/cart.py`\n\n```python\nx = 1\n```", {"shop/cart.py": "x = 1\n"}),
    ("File: inventory.py\n```python\nx = 1\n```", {"inventory.py": "x = 1\n"}),
    ("1. `inventory.py` (new file):\n```python\nx = 1\n```", {"inventory.py": "x = 1\n"}),
    ("Here is the new `inventory.py`:\n```python\nx = 1\n```", {"inventory.py": "x = 1\n"}),
    ("Some code:\n```python\nprint(1)\n```", {}),  # no name: not a file
    ("```python 3.11\nx\n```\n```text\nx\n```", {}),  # a version or a language is not a path
    ("<think>draft\n```a.py\nbad\n```\n</think>\n```a.py\ngood\n```", {"a.py": "good\n"}),  # not the thinking
    ("notes ```a.py\nbad\n``` </think>```a.py\ngood\n```", {"a.py": "good\n"}),  # only the closing tag
    ("```a.py\nfirst\n```\nBetter:\n```a.py\nsecond\n```", {"a.py": "second\n"}),  # the later one wins
    ("```a.py\nreal = 1\n```\nUsage of `a.py`:\n```python\nimport a\n```", {"a.py": "real = 1\n"}),  # not an example
    ("```README.md\n# Shop\n```bash\npip install x\n```\nThen run it.\n```\n```a.py\nx = 1\n```",
     {"README.md": "# Shop\n```bash\npip install x\n```\nThen run it.\n", "a.py": "x = 1\n"}),  # nested fences
    ("````README.md\n```py\nx\n```\n````", {"README.md": "```py\nx\n```\n"}),
    ("~~~a.py\nx = 1\n~~~", {"a.py": "x = 1\n"}),
    ("  ```a.py\n  x = 1\n  ```", {"a.py": "  x = 1\n"}),
    ("```a.py\n\n\ndef f():\n    pass\n\n\n```", {"a.py": "\n\ndef f():\n    pass\n"}),
    ("```.\\shop\\a.py\nx\n```", {"shop/a.py": "x\n"}),
    ("```a.py\nx = 1", {"a.py": "x = 1\n"}),  # cut off before the closing fence
    ("", {}),
    (None, {}),
])
def test_parse_files(reply, files) -> None:
    assert grade.parse_files(reply) == files


@pytest.mark.parametrize("path", ["/etc/passwd", "/tmp/x.py", "../x.py", "a/../../b.py", "hidden/test_hidden.py",
                                  "reference/a.py", "C:\\x.py", "~/.bashrc.py", ".git/config.py", "a//b.py",
                                  "x" * 300 + ".py"])
def test_unsafe_paths_are_never_written(path) -> None:
    assert grade.safe_path(path) is None
    for reply in (f"```{path}\nx = 1\n```", f"```python\n# file: {path}\nx = 1\n```", f"**{path}**\n```\nx\n```"):
        assert grade.parse_files(reply) == {}, reply
    if path.endswith(".py") and not path.startswith(("~", "C:")):  # read as a path, then refused
        assert [p for p, _ in grade.read_reply(f"```{path}\nx = 1\n```")["rejected"]] == [path]


def test_safe_paths_are_cleaned() -> None:
    assert grade.safe_path("./shop/a.py") == "shop/a.py"
    assert grade.safe_path("shop\\a.py") == "shop/a.py"
    assert grade.safe_path(" a.py ") == "a.py"
    assert grade.safe_path("hidden_helpers.py") == "hidden_helpers.py"  # only the folder hidden/ is out
    assert grade.safe_path("a b.py") is None and grade.safe_path(None) is None


def test_read_reply_counts_the_explanation_and_cut_off() -> None:
    parsed = grade.read_reply(PERFECT)
    assert parsed["words"] > grade.EXPLAIN_WORDS and parsed["named_by"] == {"words.py": "fence", "test_words.py": "fence"}
    assert not parsed["cut_off"] and parsed["unnamed_blocks"] == 0
    parsed = grade.read_reply("Short.\n```python\nx\n```\n```a.py\nx = (")
    assert parsed["words"] == 1 and parsed["cut_off"] and parsed["unnamed_blocks"] == 1
    assert grade.read_reply("<think>" + "word " * 100 + "</think>ok")["words"] == 1  # thinking is not explaining


# ---- the prompt ----

def test_prompt_is_the_request_and_one_line_on_the_format() -> None:
    assert grade.prompt_for(WORDS) == WORDS["task_md"] + "\n\n" + grade.REPLY_FORMAT
    prompt = grade.prompt_for(AVERAGE)
    assert prompt == (AVERAGE["task_md"] + "\n\nThe project's files:\n\n```stats.py\n"
                      + AVERAGE["starter"]["stats.py"] + "```\n\n" + grade.REPLY_FORMAT)
    assert grade.REPLY_FORMAT == ("Reply with every file you create or change as a fenced block whose first line is "
                                  "```path/to/file.ext (full file contents). You may write a short explanation first.")


@pytest.mark.parametrize("task", [WORDS, AVERAGE], ids=["build", "bugfix"])
def test_prompt_has_no_checklist_and_no_hidden_parts(task) -> None:
    prompt = grade.prompt_for(task).lower()
    given = (task["task_md"] + "".join(task["starter"].values())).lower()
    for word in ("production", "error handling", "tests", "test", "edge case", "validate", "lint", "security",
                 "type hint", "docstring", "readme", "logging", "typeerror", "valueerror", "robust", "clean"):
        assert (word in prompt) == (word in given), word
    for secret in [task["hidden"], *task["reference"].values()]:
        for line in secret.splitlines():
            if len(line.strip()) > 25 and line not in given:
                assert line.lower() not in prompt


def test_starter_files_with_fences_get_a_longer_fence() -> None:
    task = {**AVERAGE, "starter": {"README.md": "Run:\n```\npython x.py\n```\n"}}
    prompt = grade.prompt_for(task)
    assert "````README.md\nRun:\n```\npython x.py\n```\n````" in prompt
    assert grade.parse_files(prompt) == {"README.md": "Run:\n```\npython x.py\n```\n"}  # it reads back the same


# ---- the tasks ----

def test_sample_tasks_load() -> None:
    assert [t["id"] for t in (WORDS, AVERAGE)] == ["01-word-count", "02-fix-average"]
    assert WORDS["meta"] == {"kind": "build", "difficulty": 1, "files": ["words.py"], "timeout": 30}
    assert WORDS["starter"] == {} and set(WORDS["reference"]) == {"words.py", "test_words.py"}
    assert AVERAGE["starter"].keys() == AVERAGE["reference"].keys() == {"stats.py"}
    assert "def test_" in AVERAGE["hidden"] and AVERAGE["task_md"].startswith("`average([2, 4])`")


def write_task(root: Path, name: str, meta, hidden="def test_x():\n    pass\n", reference=None, starter=None,
               task_md="Do it.") -> None:
    folder = root / name
    for sub, files in (("reference", reference if reference is not None else {"a.py": "x = 1\n"}),
                       ("starter", starter or {}), ("hidden", {"test_hidden.py": hidden} if hidden else {})):
        for path, text in files.items():
            (folder / sub / path).parent.mkdir(parents=True, exist_ok=True)
            (folder / sub / path).write_text(text, encoding="utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    if task_md is not None:
        (folder / "task.md").write_text(task_md, encoding="utf-8")
    if meta is not None:
        (folder / "meta.json").write_text(meta if isinstance(meta, str) else json.dumps(meta), encoding="utf-8")


def test_load_tasks_says_what_is_wrong(tmp_path: Path) -> None:
    good = {"kind": "build", "difficulty": 2, "files": ["a.py"]}
    write_task(tmp_path, "01-good", good)
    write_task(tmp_path, "_drafts", None)  # ignored
    (tmp_path / "README.md").write_text("notes")  # ignored
    assert [t["id"] for t in grade.load_tasks(tmp_path)] == ["01-good"]
    assert grade.load_tasks(tmp_path)[0]["meta"]["timeout"] == grade.DEFAULT_TIMEOUT
    bad = {
        "Bad_Name": (good, {}, "folder name"),
        "02-no-meta": (None, {}, "meta.json is missing"),
        "03-bad-json": ("{nope", {}, "not valid JSON"),
        "04-kind": ({**good, "kind": "essay"}, {}, "kind must be one of"),
        "05-difficulty": ({**good, "difficulty": 4}, {}, "difficulty must be 1, 2 or 3"),
        "06-files": ({**good, "files": "a.py"}, {}, "files must be a list"),
        "07-timeout": ({**good, "timeout": 0}, {}, "timeout must be whole seconds"),
        "08-missing-file": ({**good, "files": ["b.py"]}, {}, "neither starter/ nor reference/ has it"),
        "09-no-reference": (good, {"reference": {}}, "reference/ is missing or empty"),
        "10-no-tests": (good, {"hidden": "x = 1\n"}, "has no test functions"),
        "11-no-hidden": (good, {"hidden": ""}, "is missing or empty"),
        "12-no-task": (good, {"task_md": None}, "task.md is missing"),
        "13-unsafe": ({**good, "files": ["../a.py"]}, {}, "files must be a list of relative paths"),
    }
    for name, (meta, extra, _) in bad.items():
        write_task(tmp_path, name, meta, **extra)
    (tmp_path / "01-good" / "hidden" / "notes.txt").write_text("x")
    with pytest.raises(ValueError) as error:
        grade.load_tasks(tmp_path)
    message = str(error.value)
    assert f"{len(bad) + 1} task(s)" in message
    for name, (_, _, says) in bad.items():
        line = next(line for line in message.splitlines() if line.startswith(f" - {name}:"))
        assert says in line, (name, line)
    assert "hidden/ must hold only test_hidden.py" in message
    with pytest.raises(FileNotFoundError):
        grade.load_tasks(tmp_path / "missing")


# ---- grading ----

def test_a_perfect_reply(tmp_path: Path) -> None:
    g = graded(WORDS, PERFECT, tmp_path)
    assert (g["hidden"], g["hidden_passed"], g["hidden_total"], g["solved"]) == (1.0, 4, 4, True)
    assert g["own_tests"] == 1.0 and g["own_test_files"] == ["test_words.py"] and g["own_tests_passed"] == 3
    assert g["lint"] == 1.0 and g["lint_issues"] == 0
    assert g["security"] == (1.0 if HAS_BANDIT else None)
    assert g["explains"] == 1.0 and not g["junk"] and g["total"] == 100.0
    assert g["files"] == ["test_words.py", "words.py"] and g["named_by"] == {"words.py": "fence",
                                                                             "test_words.py": "fence"}
    assert set(g["outputs"]) == {"hidden", "own_tests", "lint", "security"}
    assert list(tmp_path.iterdir()) == []  # the workspaces are gone


def test_a_sloppy_reply_scores_lower_in_each_part(tmp_path: Path) -> None:
    good, g = graded(WORDS, PERFECT, tmp_path), graded(WORDS, SLOPPY, tmp_path)
    assert g["hidden"] == 1.0 and g["solved"]  # it works...
    assert g["own_tests"] == 0 < good["own_tests"]
    assert g["lint_issues"] >= 4 and g["lint"] <= 0.6 < good["lint"]
    assert "F401" in g["outputs"]["lint"] and "E721" in g["outputs"]["lint"]
    assert g["explains"] == 0 < good["explains"] and g["explanation_words"] == 0
    if HAS_BANDIT:
        assert g["security"] == 0.0 and any(s["test_id"] == "B307" for s in g["security_issues"])
    assert g["total"] <= good["total"] - 30


@pytest.mark.skipif(not HAS_BANDIT, reason="bandit is not installed")
def test_security_levels(tmp_path: Path) -> None:
    low = "```words.py\nimport subprocess\n\n\ndef count_words(text):\n    assert isinstance(text, str)\n    return {}\n```"
    assert graded(WORDS, low, tmp_path)["security"] == 0.5  # assert and subprocess: low
    tests_only = "```test_words.py\ndef test_x():\n    assert 1\n```"
    assert graded(WORDS, tests_only, tmp_path)["security"] == 0.0  # no code at all: nothing earned


def test_without_bandit_security_is_left_out(tmp_path: Path, monkeypatch) -> None:
    real = grade.program
    monkeypatch.setattr(grade, "program", lambda module, name: None if module == "bandit" else real(module, name))
    g = graded(WORDS, PERFECT, tmp_path)
    assert g["security"] is None and g["total"] == 100.0 and "not installed" in g["outputs"]["security"]
    g = graded(WORDS, SLOPPY, tmp_path)
    expected = 100 * (55 * 1 + 15 * 0 + 10 * g["lint"] + 10 * 0) / 90
    assert g["security"] is None and g["total"] == round(expected, 1)


def test_a_broken_reply_scores_hidden_0(tmp_path: Path) -> None:
    g = graded(WORDS, BROKEN, tmp_path)
    assert g["hidden"] == 0 and not g["solved"] and g["hidden_passed"] == 0
    assert "SyntaxError" in g["outputs"]["hidden"] or "error" in g["outputs"]["hidden"].lower()
    assert g["lint_issues"] >= 1  # ruff reports the syntax error too
    empty = graded(WORDS, "I can't do this.", tmp_path)
    assert empty["files"] == [] and empty["total"] == 0.0 and empty["lint"] == 0.0
    assert empty["security"] == (0.0 if HAS_BANDIT else None) and empty["hidden_total"] >= 1


def test_bugfix_partial_and_starter_alone(tmp_path: Path) -> None:
    assert graded(AVERAGE, FIXED_AVERAGE, tmp_path)["solved"]
    half = graded(AVERAGE, HALF_FIXED, tmp_path)
    assert (half["hidden_passed"], half["hidden_total"], half["hidden"]) == (3, 4, 0.75)
    nothing = grade.grade(AVERAGE, {}, tmp_path)  # the starter as it is
    assert nothing["hidden_passed"] == 0 and nothing["explains"] is None


def test_own_tests_that_fail_score_half(tmp_path: Path) -> None:
    reply = PERFECT.replace('{"hi": 2, "there": 1}', '{"hi": 3}')
    g = graded(WORDS, reply, tmp_path)
    assert g["own_tests"] == 0.5 and g["hidden"] == 1.0 and g["own_tests_passed"] == 2
    tests_in_folder = (FIXED_AVERAGE + "\n```tests/test_stats.py\nfrom stats import average\n\n\n"
                       "def test_mean():\n    assert average([2, 4]) == 3\n```")
    g = graded(AVERAGE, tests_in_folder, tmp_path)
    assert g["own_tests"] == 1.0 and g["own_test_files"] == ["tests/test_stats.py"] and not g["junk"]


def test_the_hidden_tests_cannot_be_steered_or_replaced(tmp_path: Path) -> None:
    cheat = (BROKEN.replace("def count_words(text)", "def count_words(text):")  # wrong, but it runs
             + "\n```conftest.py\nimport pytest\n\n\n@pytest.hookimpl(hookwrapper=True)\n"
             "def pytest_runtest_makereport(item, call):\n    outcome = yield\n    outcome.get_result().outcome = "
             "\"passed\"\n```\n```pytest.ini\n[pytest]\naddopts = -k nothing_at_all\n```\n"
             "```test_hidden.py\ndef test_fine():\n    pass\n```\n```hidden/test_hidden.py\ndef test_x():\n    pass\n```")
    g = graded(WORDS, cheat, tmp_path)
    assert g["hidden_total"] == 4 and g["hidden_passed"] == 1  # only the empty-text test passes
    assert g["rejected"] == [("hidden/test_hidden.py", g["rejected"][0][1])]


def test_a_reply_that_never_ends_times_out(tmp_path: Path) -> None:
    task = {**WORDS, "meta": {**WORDS["meta"], "timeout": 2}}
    hang = ("```words.py\nimport subprocess\nimport time\n\nsubprocess.Popen(['sleep', '60'])\n\n\n"
            "def count_words(text):\n    while True:\n        time.sleep(0.1)\n```\n"
            "```test_words.py\nfrom words import count_words\n\n\ndef test_x():\n    count_words('a')\n```")
    started = time.time()
    g = graded(task, hang, tmp_path)
    assert time.time() - started < 20  # the whole process group was stopped (the child sleep too)
    assert g["hidden"] == 0 and g["hidden_timed_out"] and "timeout after 2 s" in g["outputs"]["hidden"]
    assert g["own_tests"] == 0.5


def test_junk_files(tmp_path: Path) -> None:
    helper = ("```words.py\nfrom textutil import split\n\n\ndef count_words(text):\n    return split(text)\n```\n"
              "```textutil.py\ndef split(text):\n    return {}\n```\n```README.md\n# Words\n```\n"
              "```test_words.py\ndef test_x():\n    pass\n```\n```scratch.py\nprint('trying')\n```\n"
              "```notes.txt\ntodo\n```")
    g = graded(WORDS, helper, tmp_path)
    assert g["junk"] and g["junk_files"] == ["notes.txt", "scratch.py"]
    assert g["total"] == round(grade.total_of(g, False) - grade.JUNK_PENALTY, 1)
    task = {**WORDS, "meta": {**WORDS["meta"], "files": ["shop/words.py"]}}
    files = {"shop/words.py": "from . import util\nfrom .sub.deep import x\n", "shop/extra.py": "", "util.py": "",
             "shop/util.py": "", "pyproject.toml": "", "docs/usage.md": "", "conftest.py": "", "other/thing.py": ""}
    assert grade.junk_files(task, files) == ["other/thing.py", "util.py"]


def test_total_of_renormalises_and_never_goes_below_0() -> None:
    full = {"hidden": 1.0, "own_tests": 1.0, "lint": 1.0, "security": 1.0, "explains": 1.0}
    assert grade.total_of(full, False) == 100.0 and grade.total_of(full, True) == 95.0
    assert grade.total_of({**full, "security": None, "hidden": 0.5}, False) == round(100 * 62.5 / 90, 1)
    assert grade.total_of(dict.fromkeys(full, 0.0), True) == 0.0
    assert sum(grade.WEIGHTS.values()) == 100


# ---- checking the tasks ----

def test_check_references_on_the_sample_tasks() -> None:
    rows, ok = grade.check_references([WORDS, AVERAGE])
    assert ok and [r["task"] for r in rows] == ["01-word-count", "02-fix-average"]
    assert rows[0]["grade"]["total"] == 100.0 and rows[1]["starter_passed"] == 0
    lines = grade.reference_table(rows)
    assert lines[1].startswith("01-word-count") and lines[1].endswith("ok") and "4/4" in lines[1]


def test_check_references_finds_broken_tasks() -> None:
    wrong_reference = {**AVERAGE, "id": "98-wrong", "reference": AVERAGE["starter"]}
    nothing_to_do = {**AVERAGE, "id": "99-done", "starter": AVERAGE["reference"]}
    rows, ok = grade.check_references([wrong_reference, nothing_to_do])
    assert not ok
    assert rows[0]["problems"][0].startswith("the reference passes 0 of 4 hidden tests")
    assert rows[1]["problems"] == ["the starter files alone already pass every hidden test"]
    assert all(line.endswith("FAIL") for line in grade.reference_table(rows)[1:3])


def test_check_references_command_line(tmp_path: Path, capsys) -> None:
    assert grade.main(["--check-references", "--tasks", str(SAMPLES)]) == 0
    assert "All references pass." in capsys.readouterr().out
    write_task(tmp_path, "01-bad", {"kind": "build", "difficulty": 1, "files": ["a.py"]},
               hidden="from a import x\n\n\ndef test_x():\n    assert x == 2\n")
    assert grade.main(["--check-references", "--tasks", str(tmp_path)]) == 1
    assert "1 task(s) FAIL" in capsys.readouterr().out
    write_task(tmp_path, "02-worse", {"kind": "poem"})
    assert grade.main(["--check-references", "--tasks", str(tmp_path)]) == 1
    assert "02-worse: " in capsys.readouterr().out
    assert grade.main(["--check-references", "--tasks", str(tmp_path / "none")]) == 1
    with pytest.raises(SystemExit) as stop:
        grade.main([])
    assert stop.value.code == 2


def test_every_exam_task_reference_passes() -> None:
    """CI: the tasks in coder/exam/tasks are in the format and each reference passes its hidden tests."""
    folder = HERE / "tasks"
    if not folder.is_dir() or not any(p.is_dir() and not p.name.startswith((".", "_")) for p in folder.iterdir()):
        pytest.skip("no exam tasks yet")
    rows, ok = grade.check_references(grade.load_tasks(folder))
    assert ok, "\n".join(grade.reference_table(rows))


def test_check_references_from_a_shell() -> None:
    done = subprocess.run([sys.executable, "-m", "coder.exam.grade", "--check-references", "--tasks", str(SAMPLES)],
                          cwd=HERE.parent.parent, capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "01-word-count" in done.stdout


# ---- the notebook ----

def cell(title: str, nb: dict | None = None) -> str:
    nb = nb or json.loads((HERE / build_exam.NOTEBOOK).read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith(title))["source"])


def notebook_scope(tmp_path: Path, nb: dict | None = None) -> dict:
    """The settings and helpers cells, run as the notebook would, saving into tmp_path."""
    scope: dict = {}
    exec(cell("#@title 1.", nb), scope)  # noqa: S102 - our own generated cell
    scope.update(WORKING_DIR=str(tmp_path), MODELS_DIR=str(tmp_path / "models"), GRADE_DIR=str(tmp_path / "grading"))
    exec(cell("#@title 2.", nb), scope)  # noqa: S102
    scope["OLLAMA_LOG"] = str(tmp_path / "ollama.log")
    return scope


def test_notebook_is_built_from_current_files() -> None:
    current = (HERE / build_exam.NOTEBOOK).read_text(encoding="utf-8")
    assert current == build_exam.text(build_exam.notebook()), "exam.ipynb is out of date: run python coder/build.py"


def test_notebook_settings_and_helpers(tmp_path: Path) -> None:
    scope = notebook_scope(tmp_path)
    assert scope["MODELS"] == ["qwen3-coder:30b", "qwen2.5-coder:14b", "gpt-oss:20b", "qwen2.5-coder:7b"]
    assert scope["LIMIT"] is None and scope["MINUTES_PER_MODEL"] == 60 and scope["SAMPLES"] == 1
    assert scope["OPTIONS"] == {"temperature": 0, "num_ctx": 16384, "num_predict": 6144}
    assert scope["THINK"] == {"gpt-oss": "medium"}
    assert scope["RESULTS_FILE"] == "exam.json" and "/kaggle/working" not in scope["MODELS_DIR"]
    assert scope["RESULTS"]["reply_format"] == grade.REPLY_FORMAT and scope["RESULTS"]["weights"] == grade.WEIGHTS
    helpers = cell("#@title 2.")
    assert not any(line.startswith(("from coder", "import coder")) for line in helpers.splitlines())
    assert "def main(" not in helpers and "__main__" not in helpers  # not grade.py's command line
    for name in ("grade_reply", "check_references", "take_exam", "table", "weakest", "pull", "tidy_up"):
        assert callable(scope[name]), name
    ollama = cell("#@title 4.")  # the shared Ollama start code, with the downloads sent to the big disk
    assert build_exam.START_OLLAMA.strip("\n") in ollama
    assert ollama.index('os.environ["OLLAMA_MODELS"] = MODELS_DIR') < ollama.index("ollama serve")


def test_notebook_holds_the_tasks(tmp_path: Path) -> None:
    scope: dict = {"json": json}
    exec(cell("#@title 5."), scope)  # noqa: S102
    folder = build_exam.TASKS_DIR
    assert scope["EXAM_TASKS"] == (grade.load_tasks(folder) if folder.is_dir() else [])
    nb = build_exam.notebook(SAMPLES)
    exec(cell("#@title 5.", nb), scope)  # noqa: S102
    assert scope["EXAM_TASKS"] == [WORDS, AVERAGE]
    assert run_exam.task_count(nb) == 2


def test_checkers_cell(tmp_path: Path, monkeypatch) -> None:
    scope = notebook_scope(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append(command)
                        or subprocess.CompletedProcess(command, 0, "", ""))
    exec(cell("#@title 3."), scope)  # noqa: S102
    assert calls[0][1:5] == ["-m", "pip", "install", "-q"] and calls[0][-3:] == ["pytest", "ruff", "bandit"]
    saved = json.loads((tmp_path / "exam.json").read_text(encoding="utf-8"))
    assert saved["checkers"]["pytest"].startswith("pytest") and saved["checkers"]["python"]


def test_start_cell_uses_a_running_ollama(tmp_path: Path, fake_ollama: dict) -> None:  # noqa: F811
    default = 'OLLAMA_URL = "http://127.0.0.1:11434"'
    scope = notebook_scope(tmp_path)
    exec(cell("#@title 4.").replace(default, f'OLLAMA_URL = "{fake_ollama["url"]}"'), scope)  # noqa: S102
    saved = json.loads((tmp_path / "exam.json").read_text(encoding="utf-8"))
    assert saved["ollama_version"] == "0.0.0-fake" and saved["errors"] == []


def answer(prompt: str) -> str:
    """The fake model: a perfect word counter, a bare fix of average()."""
    return PERFECT if "count_words" in prompt else FIXED_AVERAGE


def run_notebook(tmp_path: Path, fake: dict, models: list[str], minutes: float = 60,
                 tasks_dir: Path = SAMPLES) -> tuple[dict, list]:
    """Cells 5-8 with the sample tasks and the fake Ollama. The results file, and a copy of it at every save."""
    fake["answer"] = answer
    nb = build_exam.notebook(tasks_dir)
    scope = notebook_scope(tmp_path, nb)
    scope.update(OLLAMA_URL=fake["url"], MODELS=models, MINUTES_PER_MODEL=minutes)
    saves = []
    save = scope["save_results"]

    def save_and_copy():
        save()
        saves.append(json.loads((tmp_path / "exam.json").read_text(encoding="utf-8")))

    scope["save_results"] = save_and_copy
    for title in ("#@title 5.", "#@title 6.", "#@title 7.", "#@title 8."):
        exec(cell(title, nb), scope)  # noqa: S102 - our own generated cell
    return json.loads((tmp_path / "exam.json").read_text(encoding="utf-8")), saves


def test_notebook_examines_every_model_and_goes_on_after_a_failure(tmp_path, fake_ollama, capsys) -> None:  # noqa: F811
    results, saves = run_notebook(tmp_path, fake_ollama, ["qwen-fake:7b", "missing:1b", "gpt-oss:20b"])
    out = capsys.readouterr().out
    assert results["tasks_total"] == results["tasks_used"] == 2 and results["tasks_left_out"] == []
    assert results["task_ids"] == ["01-word-count", "02-fix-average"] and results["kinds"] == {"build": 1, "bugfix": 1}
    assert [r["problems"] for r in results["reference_check"]] == [[], []]
    first, missing, oss = results["models"]

    assert first["pulled"] and first["size_gb"] == 2.0 and first["processor"] == "100% GPU"
    assert (first["done"], first["total"], first["solved"]) == (2, 2, 2) and first["errors"] == []
    words, average = first["tasks"]
    assert (words["task"], words["total"], words["solved"]) == ("01-word-count", 100.0, True)
    assert words["attempts"][0]["reply"] == PERFECT and words["attempts"][0]["grade"]["hidden_passed"] == 4
    assert average["solved"] and average["own_tests"] == 0 and average["explains"] == 0
    assert first["score"] == round((100 + average["total"]) / 2, 1) and first["hidden"] == 1.0
    assert first["by_kind"] == {"bugfix": {"tasks": 1, "total": average["total"], "solved": 1},
                                "build": {"tasks": 1, "total": 100.0, "solved": 1}}
    assert first["tokens_per_s"] == 20.0 and first["think"] is None

    assert not missing["pulled"] and missing["done"] == 0 and "file does not exist" in missing["errors"][0]
    assert any(e.startswith("missing:1b:") for e in results["errors"])
    assert oss["think"] == "medium" and oss["solved"] == 2

    chats = fake_ollama["chats"]
    assert len(chats) == 4
    for body in chats:
        assert body["stream"] is False and len(body["messages"]) == 1 and body["messages"][0]["role"] == "user"
        assert body["options"] == {"temperature": 0, "num_ctx": 16384, "num_predict": 6144}
    assert [b.get("think") for b in chats] == [None, None, "medium", "medium"]
    assert chats[0]["messages"][0]["content"] == grade.prompt_for(WORDS)
    assert chats[1]["messages"][0]["content"] == grade.prompt_for(AVERAGE)
    assert fake_ollama["unloads"] == fake_ollama["deletes"] == ["qwen-fake:7b", "gpt-oss:20b"]

    # Saved after every task: a copy with one task done, and one where the first model is done and no other started.
    assert any(len(s["models"]) == 1 and s["models"][0]["done"] == 1 for s in saves)
    assert any(len(s["models"]) == 1 and s["models"][0]["done"] == 2 and s["models"][0]["minutes"] is not None
               for s in saves)
    assert saves[-1] == results
    for text in ("01-word-count", "qwen-fake:7b", "2/2", "download failed", "think medium",
                 "by kind: bugfix ", "build 100.0 (1/1 solved)", "the 5 weakest tasks", "no tests", "--- What a model is sent"):
        assert text in out, text


def test_a_model_stops_at_its_time_limit(tmp_path, fake_ollama) -> None:  # noqa: F811
    results, _ = run_notebook(tmp_path, fake_ollama, ["qwen-fake:7b"], minutes=0)
    m = results["models"][0]
    assert m["pulled"] and m["stopped_by_time"] and m["done"] == 0 and m["score"] is None
    assert fake_ollama["chats"] == [] and fake_ollama["deletes"] == ["qwen-fake:7b"]
    assert "stopped at the time limit after 0 of 2 tasks" in "\n".join(exam.table(results))


def test_a_task_whose_reference_fails_here_is_left_out(tmp_path, fake_ollama) -> None:  # noqa: F811
    tasks = tmp_path / "tasks"
    for task in (WORDS, AVERAGE):
        broken = task is AVERAGE
        write_task(tasks, task["id"], task["meta"], hidden=task["hidden"], starter=task["starter"],
                   reference=task["starter"] if broken else task["reference"], task_md=task["task_md"])
    results, _ = run_notebook(tmp_path, fake_ollama, ["qwen-fake:7b"], tasks_dir=tasks)
    assert results["tasks_left_out"] == ["02-fix-average"] and results["tasks_used"] == 1
    assert [t["task"] for t in results["models"][0]["tasks"]] == ["01-word-count"]


def test_failed_requests_and_grading_count_as_0(tmp_path, fake_ollama, monkeypatch) -> None:  # noqa: F811
    fake_ollama["answer"] = answer
    monkeypatch.setattr(exam, "grade_reply", lambda *a: 1 / 0)
    entry: dict = {}
    exam.take_exam(fake_ollama["url"], "qwen-fake:7b", [WORDS], entry, 60, {"temperature": 0}, log=lambda *a: None)
    assert entry["done"] == 1 and entry["tasks"][0]["total"] == 0.0
    assert entry["tasks"][0]["attempts"][0]["error"].startswith("grading failed: ZeroDivisionError")

    def broken(*args, **kwargs):
        raise TimeoutError("timed out")

    entry = {}
    monkeypatch.setattr(exam, "ask_model", broken)
    exam.take_exam(fake_ollama["url"], "qwen-fake:7b", [WORDS, AVERAGE] * 3, entry, 60, {"temperature": 0},
                   log=lambda *a: None)
    assert entry["done"] == 3 and entry["solved"] == 0 and entry["total"] == 6 and entry["score_done"] == 0.0
    assert entry["errors"][-1] == "3 requests in a row failed: stopped this model"
    assert "stopped after 3 of 6 tasks" in "\n".join(exam.table({"models": [entry]}))
    assert "request failed: TimeoutError" in "\n".join(exam.weakest(entry))


def test_more_samples_average_and_send_a_seed(tmp_path, fake_ollama) -> None:  # noqa: F811
    replies = iter([PERFECT, BROKEN])
    fake_ollama["answer"] = lambda prompt: next(replies)
    entry: dict = {}
    exam.take_exam(fake_ollama["url"], "qwen-fake:7b", [WORDS], entry, 60, {"temperature": 0.7}, samples=2,
                   log=lambda *a: None, workdir=tmp_path)
    record = entry["tasks"][0]
    assert [c["options"].get("seed") for c in fake_ollama["chats"]] == [0, 1]
    assert record["solve_rate"] == 0.5 and not record["solved"] and entry["solved"] == 0
    assert record["total"] == round((100 + record["attempts"][1]["grade"]["total"]) / 2, 1)


# ---- the runner ----

def model_result(name: str, totals: list[float], **extra) -> dict:
    """A model's entry as the notebook leaves it, from made-up task totals."""
    tasks = []
    for i, total in enumerate(totals):
        g = {"total": total, "solved": total >= 90, "hidden": 1.0 if total >= 90 else 0.25, "hidden_passed": 1,
             "hidden_total": 4, "own_tests": 1.0, "lint": 0.9, "security": 1.0, "explains": 0.0,
             "files": ["a.py"], "own_test_files": ["test_a.py"], "lint_issues": 1, "security_issues": [],
             "outputs": {"hidden": "FAILED test_hidden.py::test_empty - KeyError: 'x'"}, "junk_files": []}
        tasks.append({"task": f"{i:02d}-task", "kind": ("build", "bugfix")[i % 2], "difficulty": 2,
                      "attempts": [{"reply": "...", "seconds": 4.0, "tokens": 400, "token_seconds": 10.0,
                                    "done_reason": "stop", "grade": g}]})
    for t in tasks:
        exam.score_task(t)
    entry = {"model": name, "pulled": True, "size_gb": 18.6, "pull_seconds": 90.0, "load_seconds": 20.0,
             "total": len(totals), "processor": "100% GPU", "minutes": 30.0, "errors": [], "tasks": tasks, **extra}
    exam.summarise(entry)
    return entry


RESULT = {
    "exam": "engineer exam (coder phase 4)", "tasks_total": 7, "tasks_used": 6, "kinds": {"build": 3, "bugfix": 3},
    "ollama_version": "0.40.4", "gpus": "Tesla T4, 15360 MiB\nTesla T4, 15360 MiB", "setting": exam.SETTING,
    "options": {"temperature": 0}, "think": {"gpt-oss": "medium"}, "tasks_left_out": ["07-broken"],
    "checkers": {"python": "3.11.13", "pytest": "pytest 8.4.0", "ruff": "ruff 0.12.0", "bandit": None},
    "models": [model_result("qwen3-coder:30b", [100, 95, 40, 20, 90, 60]),
               model_result("gpt-oss:20b", [100, 10], total=6, stopped_by_time=True, think="medium"),
               {"model": "missing:1b", "pulled": False, "total": 6, "done": 0, "errors": ["RuntimeError('no')"]}],
    "errors": ["missing:1b: RuntimeError('no')"],
}


class FakeKaggle(FakeApi):
    """test_try.py's fake Kaggle, handing out exam.json instead."""

    def kernels_output(self, kernel, path, file_pattern=None, force=False):
        self.calls["pattern"] = file_pattern
        log = [{"stream_name": "stdout", "time": 1, "data": "RuntimeError: Ollama did not start\n"}]
        (Path(path) / "wimb-coder-exam.log").write_text(json.dumps(log))
        if self.results is not None:
            (Path(path) / "exam.json").write_text(json.dumps(self.results), encoding="utf-8")


def test_models_and_limit_lines_round_trip() -> None:
    nb = run_exam.notebook_with(["qwen3-coder:30b", "hf.co/x/y-GGUF:Q4_K_M"], 5)
    s = run_exam.settings(nb)
    assert s["MODELS"] == ["qwen3-coder:30b", "hf.co/x/y-GGUF:Q4_K_M"] and s["LIMIT"] == 5
    s = run_exam.settings(run_exam.notebook_with(None, None))
    assert s["MODELS"] == run_exam.settings(json.loads(run_exam.NOTEBOOK.read_text("utf-8")))["MODELS"]
    assert s["LIMIT"] is None
    lines = "".join(run_exam.settings_cell(nb)["source"]).splitlines()
    assert sum(line.startswith("MODELS = ") for line in lines) == 1 and sum(line.startswith("LIMIT = ") for line in
                                                                               lines) == 1
    assert run_exam.hours_needed(run_exam.notebook_with(["a:1"], None)) == round((60 + 5 + 20) / 60, 1)
    assert run_exam.hours_needed(run_exam.notebook_with(None, None)) <= run_exam.WAIT_MINUTES / 60


def test_runner_pushes_a_private_gpu_run_with_internet() -> None:
    api = FakeKaggle(RESULT)
    run_exam.push(api, "me/wimb-coder-exam", run_exam.notebook_with(["qwen3-coder:30b"], 3))
    meta = api.calls["meta"]
    assert meta["id"] == "me/wimb-coder-exam" and meta["title"] == "wimb-coder-exam"
    assert meta["code_file"] == "exam.ipynb" and meta["kernel_sources"] == []
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"] and api.calls["acc"] == "NvidiaTeslaT4"
    s = run_exam.settings(api.calls["notebook"])
    assert s["MODELS"] == ["qwen3-coder:30b"] and s["LIMIT"] == 3


def run_main(api, tmp_path: Path, monkeypatch, *argv: str, nb_path: Path | None = None) -> int:
    """run_exam.main() against the fake Kaggle: its exit code."""
    monkeypatch.setattr(run_exam, "OUT", tmp_path / "out")
    if nb_path is None:  # a notebook with the 2 sample tasks (the real one may have none yet)
        nb_path = tmp_path / "exam.ipynb"
        nb_path.write_text(build_exam.text(build_exam.notebook(SAMPLES)), encoding="utf-8")
    monkeypatch.setattr(run_exam, "NOTEBOOK", nb_path)
    monkeypatch.setattr(run_exam, "connect", lambda slug: (api, f"me/{slug}"))
    monkeypatch.setattr(run_exam, "check_gpu", lambda api, need_hours: api.calls.update(need_hours=need_hours))
    monkeypatch.setattr(run_on_kaggle.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["run_exam.py", *argv])
    with pytest.raises(SystemExit) as stop:
        run_exam.main()
    return stop.value.code


def test_runner_main_prints_the_report(tmp_path, monkeypatch, capsys) -> None:
    api = FakeKaggle(RESULT)
    assert run_main(api, tmp_path, monkeypatch, "--models", "qwen3-coder:30b,gpt-oss:20b", "--limit", "2") == 0
    assert api.calls["need_hours"] == round((2 * 65 + 20) / 60, 1)
    assert api.calls["pattern"] == r"(exam\.json|.*\.log|.*\.ipynb)$"
    out = capsys.readouterr().out
    lines = out.splitlines()
    row = next(line for line in lines if line.startswith("qwen3-coder:30b "))
    assert " 67.5 " in row and "3/6" in row and "100% GPU" not in row  # only a processor that is not all GPU shows
    row = next(line for line in lines if line.startswith("gpt-oss:20b "))
    assert "1/2 of 6" in row and "stopped at the time limit after 2 of 6 tasks" in row and "think medium" in row
    assert "download failed" in next(line for line in lines if line.startswith("missing:1b "))
    for text in ("6 of 7 tasks (build 3, bugfix 3)", "Ollama 0.40.4",
                 "Left out (their reference failed on Kaggle): 07-broken", "qwen3-coder:30b by kind: bugfix ",
                 "the 5 weakest tasks",
                 "03-task (bugfix, difficulty 2): total 20.0, hidden 1/4: FAILED test_hidden.py::test_empty",
                 "no explanation", "1 lint issues", "download 18.6 GB in 90.0 s", "error: RuntimeError('no')",
                 "Models: qwen3-coder:30b, gpt-oss:20b; tasks: 2 of 2"):
        assert text in out, text
    assert sum("difficulty 2): total" in line for line in lines) == 5 + 2  # 5 of the first model, both of the 2nd
    assert "Kaggle's log" not in out and "No model got a score" not in out


@pytest.mark.parametrize(("api", "says"), [
    (FakeKaggle(None), "No exam.json"),
    (FakeKaggle({**RESULT, "models": RESULT["models"][2:]}), "No model got a score"),
    (FakeKaggle({**RESULT, "models": []}, final=Status.ERROR), "The run did not finish. Kaggle's log:"),
])
def test_runner_fails_when_no_model_scored(api, says, tmp_path, monkeypatch, capsys) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "exam.json").write_text(json.dumps(RESULT))  # an older run's file must not count
    assert run_main(api, tmp_path, monkeypatch) == 1
    out = capsys.readouterr().out
    assert says in out
    if "log" in says:
        assert "Ollama did not start" in out


def test_runner_reports_a_partial_run_and_a_run_still_going(tmp_path, monkeypatch, capsys) -> None:
    api = FakeKaggle(RESULT, final=Status.ERROR)
    assert run_main(api, tmp_path, monkeypatch, "--wait") == 0  # the scores it got are kept
    out = capsys.readouterr().out
    assert "3/6" in out and "The run did not finish" in out
    assert "meta" not in api.calls and "need_hours" not in api.calls  # --wait: nothing pushed
    api = FakeKaggle(RESULT, final=Status.RUNNING)
    assert run_main(api, tmp_path, monkeypatch, "--wait", "--wait-minutes", "1") == 0
    out = capsys.readouterr().out
    assert run_exam.STILL_RUNNING in out and "pattern" not in api.calls


def test_runner_stops_when_the_notebook_has_no_tasks(tmp_path, monkeypatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    nb_path = tmp_path / "none.ipynb"
    nb_path.write_text(build_exam.text(build_exam.notebook(empty)), encoding="utf-8")
    monkeypatch.setattr(run_exam, "NOTEBOOK", nb_path)
    monkeypatch.setattr(run_exam, "connect", lambda slug: pytest.fail("must not reach Kaggle"))
    monkeypatch.setattr(sys, "argv", ["run_exam.py"])
    with pytest.raises(SystemExit, match="has no tasks"):
        run_exam.main()


@pytest.mark.parametrize("argv", [("--models", "a b"), ("--models", ""), ("--limit", "0"), ("--limit", "x"),
                                  ("--wait-minutes", "0")])
def test_bad_options_stop_before_kaggle(argv, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_exam.py", *argv])
    monkeypatch.setattr(run_exam, "connect", lambda slug: pytest.fail("must not reach Kaggle"))
    with pytest.raises(SystemExit) as stop:
        run_exam.main()
    assert stop.value.code == 2


# ---- the workflow ----

def test_workflow_runs_the_exam(tmp_path: Path) -> None:
    text = (HERE.parent.parent / ".github" / "workflows" / "coder.yml").read_text(encoding="utf-8")
    assert "kaggle_output_exam/" in text
    exam_run = ["coder/exam/run_exam.py", "--wait-minutes", "330"]
    assert workflow_command(tmp_path, ACTION="exam", MODELS="qwen3-coder:30b,gpt-oss:20b", LIMIT="5") == [
        *exam_run, "--models", "qwen3-coder:30b,gpt-oss:20b", "--limit", "5"]
    assert workflow_command(tmp_path, ACTION="exam") == exam_run
    assert workflow_command(tmp_path, ACTION="fetch-exam", MODELS="a:1", LIMIT="3") == [*exam_run, "--wait"]
    assert "kaggle_output_exam/" in (HERE.parent.parent / ".gitignore").read_text(encoding="utf-8")
    assert os.path.exists(HERE / "run_exam.py")

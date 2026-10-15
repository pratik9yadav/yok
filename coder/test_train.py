"""Checks coder_train.ipynb (phase 2: train a coding model on open bug-fix commits) and coder/run_train.py, offline:
tiny fake commits, fake HumanEvalFix tasks, a fake model and a fake Kaggle.  Run:  python -m pytest"""

import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import run_on_kaggle
from coder import build, build_train, fixeval, fixtrain, run_train
from coder.test_coder import FIXED, NOT_FIXED, TASKS
from test_try import FakeApi, Status

VOWELS = {
    "task_id": "Fake/2", "entry_point": "count_vowels", "declaration": "def count_vowels(text: str) -> int:\n",
    "prompt": 'def count_vowels(text: str) -> int:\n    """Count the vowels a e i o u in the given text and return '
              'how many there are."""\n',
    "canonical_solution": "    return sum(1 for ch in text.lower() if ch in 'aeiou')\n",
    "buggy_solution": "    return sum(1 for ch in text if ch in 'aeiou')\n",
    "test": "def check(count_vowels):\n    assert count_vowels('hello') == 2\n    assert count_vowels('AEIOU') == 5\n",
}
HUMANEVAL = [*TASKS, VOWELS]
VOWELS_FIXED = "```python\ndef count_vowels(text: str) -> int:\n    return sum(1 for ch in text.lower() if ch in 'aeiou')\n```"
HALVE_FIXED = "```python\ndef halve(x):\n    return x // 2\n```"
BROKEN_ADD = "```python\ndef add(a, b):\n    return a * b\n```"


def commit(n: int = 0, **changes) -> dict:
    """A small made-up bug-fix commit in the CommitPackFT format."""
    row = {"commit": f"c{n}", "old_file": f"pkg/mod{n}.py", "new_file": f"pkg/mod{n}.py",
           "old_contents": f"def f{n}(x):\n    return x - 1\n", "new_contents": f"def f{n}(x):\n    return x + 1\n",
           "subject": f"Fix off-by-one in f{n}.", "message": f"Fix off-by-one in f{n}.\n", "lang": "Python",
           "license": "mit", "repos": "someone/something"}
    return {**row, **changes}


def cell(title: str, nb: dict | None = None) -> str:
    nb = nb or json.loads((HERE / build_train.NOTEBOOK).read_text(encoding="utf-8"))
    return "".join(next(c for c in nb["cells"] if "".join(c["source"]).startswith(title))["source"])


# ---- the notebook file and its settings ----

def test_notebook_is_built_from_current_files() -> None:
    current = (HERE / build_train.NOTEBOOK).read_text(encoding="utf-8")
    assert current == build_train.text(build_train.notebook()), "coder_train.ipynb is out of date: python coder/build.py"
    assert "build_train.main()" in (HERE / "build.py").read_text(encoding="utf-8")  # one command writes both
    nb = build_train.notebook()
    sources = ["".join(c["source"]) for c in nb["cells"]]
    # The shop notebook's GPU check and install cell, and phase 1's scoring code: one source each.
    assert build_train.CHECK_GPU.strip("\n") in sources and build_train.INSTALL.strip("\n") in sources
    helpers = next(s for s in sources if s.startswith("#@title 4."))
    assert (HERE / "fixeval.py").read_text(encoding="utf-8") in helpers
    assert (HERE / "fixtrain.py").read_text(encoding="utf-8") in helpers
    assert "OLLAMA_LOG" not in helpers and "def save_results" in helpers
    assert not any("save_pretrained_gguf" in s or "save_pretrained_merged" in s for s in sources)  # no export
    train = cell("#@title 8.", nb)
    for text in ('instruction_part="<|im_start|>user\\n"', 'response_part="<|im_start|>assistant\\n"',
                 "per_device_train_batch_size=1, gradient_accumulation_steps=8", 'optim="adamw_8bit"',
                 "learning_rate=2e-4, warmup_steps=10", 'lr_scheduler_type="linear"', "logging_steps=LOGGING_STEPS",
                 "seed=SEED", "control.should_training_stop = True", "model.save_pretrained(LORA_DIR)"):
        assert text in train, text
    load = cell("#@title 5.", nb)
    for text in ("r=16, lora_alpha=16", '"gate_proj", "up_proj", "down_proj"', 'use_gradient_checkpointing="unsloth"',
                 "do_sample=False", "repetition_penalty=1.0", "max_new_tokens=MAX_NEW_TOKENS"):
        assert text in load, text
    assert load.index("from unsloth import") < load.index("import torch")


def settings_of(nb: dict) -> dict:
    scope: dict = {}
    exec(cell("#@title 3.", nb), scope)  # noqa: S102 - our own generated cell
    return scope


def test_settings_and_the_smoke_check() -> None:
    s = settings_of(json.loads(run_train.NOTEBOOK.read_text("utf-8")))
    assert s["BASE"] == "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
    assert (s["MAX_TRAIN"], s["MAX_LEN"], s["EPOCHS"], s["EVAL_LIMIT"], s["SMOKE"]) == (2000, 2048, 1, None, False)
    assert (s["TRAIN_MINUTES"], s["MAX_STEPS"], s["LORA_DIR"]) == (240, None, "/kaggle/working/coder-lora")
    smoke = run_train.settings(run_train.notebook_with(smoke=True))
    assert (smoke["SMOKE"], smoke["MAX_TRAIN"], smoke["MAX_STEPS"], smoke["EVAL_LIMIT"]) == (True, 40, 8, 8)


def test_settings_lines_round_trip() -> None:
    nb = run_train.notebook_with(False, 3000, 2)
    s = run_train.settings(nb)
    assert (s["SMOKE"], s["MAX_TRAIN"], s["EPOCHS"]) == (False, 3000, 2)
    lines = "".join(run_train.settings_cell(nb)["source"]).splitlines()
    for name in ("SMOKE", "MAX_TRAIN", "EPOCHS"):
        assert sum(line.startswith(f"{name} = ") for line in lines) == 1, name
    assert run_train.settings(run_train.notebook_with())["MAX_TRAIN"] == 2000  # the notebook's own
    assert run_train.settings(run_train.notebook_with(True, 3000))["MAX_TRAIN"] == 40  # smoke wins


# ---- the training data ----

@pytest.mark.parametrize(("subject", "message", "fix"), [
    ("Fix off-by-one in parse", "", True),
    ("fixes #12: wrong default", "", True),
    ("Handle None", "This fixed a crash when x is None", True),  # the message says it
    ("Correct typo in variable name", "", True),
    ("Raise a clearer exception", "", True),
    ("Fixed incorrect rounding", "", True),
    ("Add a fixture for the db", "", False),  # "fixture" is not "fix"
    ("Merge pull request #3 from x/fix-bug", "", False),
    ("Revert \"Fix parser\"", "", False),
    ("Bump version to 1.2 (fixes build)", "", False),
    ("Fix typo in README", "", False),
    ("docs: fix wrong link", "", False),
    ("Add a new feature", "Adds feature", False),
])
def test_bug_fix_messages(subject, message, fix) -> None:
    assert fixtrain.is_bug_fix({"subject": subject, "message": message}) is fix


def test_small_filters() -> None:
    assert fixtrain.changed_lines("a\nb\nc\n", "a\nB\nc\nd\n") == 3  # one line changed (2) + one added
    assert fixtrain.changed_lines("x\n", "x\n") == 0
    assert fixtrain.row_licence_ok({"license": "MIT"}) and fixtrain.row_licence_ok({})  # no field: kept
    assert fixtrain.row_licence_ok({"license": ["bsd-3-clause", "isc"]})
    for bad in ("gpl-3.0", "agpl-3.0", "", None, ["mit", "gpl-2.0"], "unknown"):
        assert not fixtrain.row_licence_ok({"license": bad}), bad
    for path, test in (("tests/test_x.py", True), ("pkg/test_io.py", True), ("io_test.py", True),
                       ("conftest.py", True), ("pkg/testing.py", False), ("pkg/contest.py", False)):
        assert fixtrain.is_test_file({"new_file": path}) is test, path


@pytest.mark.parametrize(("value", "names"), [
    ("mit", ["mit"]), (["MIT"], ["mit"]), ("apache-2.0", ["apache-2.0"]), (["bsd-3-clause", "cc0-1.0"],
                                                                          ["bsd-3-clause", "cc0-1.0"]),
    ("cc-by-4.0", ["cc-by-4.0"]), ("odc-by", ["odc-by"]),
])
def test_open_licences_pass(value, names) -> None:
    assert fixtrain.check_licence(value) == names


@pytest.mark.parametrize("value", [None, "", [], "other", "cc-by-nc-4.0", "gpl-3.0", ["mit", "cc-by-sa-4.0"]])
def test_closed_or_missing_licences_stop_the_run(value) -> None:
    with pytest.raises(RuntimeError, match="licence"):
        fixtrain.check_licence(value)


def fake_hub(monkeypatch, tmp_path: Path, rows=None, licence="mit", download_fails=False) -> None:
    """A stand-in huggingface_hub (and datasets) module."""
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows or []), encoding="utf-8")

    def hf_hub_download(repo, file, repo_type=None):
        assert (repo, file, repo_type) == ("bigcode/commitpackft", "data/python/data.jsonl", "dataset")
        if download_fails:
            raise OSError("no network")
        return str(path)

    card = SimpleNamespace(to_dict=lambda: {"license": licence})
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = hf_hub_download
    hub.HfApi = lambda: SimpleNamespace(dataset_info=lambda repo: SimpleNamespace(card_data=card))
    hub.DatasetCard = SimpleNamespace(load=lambda repo, repo_type=None: pytest.fail("card_data was enough"))
    datasets = types.ModuleType("datasets")

    def load_dataset(repo, config, split, revision=None):
        if revision is not None:
            raise ValueError("no parquet")
        return rows

    datasets.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "datasets", datasets)


def test_load_commits_and_the_licence(monkeypatch, tmp_path: Path) -> None:
    rows = [commit(0), commit(1)]
    fake_hub(monkeypatch, tmp_path, rows)
    assert fixtrain.load_commits() == (rows, "bigcode/commitpackft/data/python/data.jsonl (hf_hub_download)")
    assert fixtrain.dataset_licence() == {"dataset": "bigcode/commitpackft", "licence": ["mit"],
                                          "from": "dataset_info card_data"}
    fake_hub(monkeypatch, tmp_path, rows, download_fails=True)  # then the datasets library
    loaded, source = fixtrain.load_commits()
    assert loaded == rows and "revision=None" in source
    fake_hub(monkeypatch, tmp_path, [{"commit": "x", "subject": "y"}])
    with pytest.raises(RuntimeError, match="no \\['old_contents', 'new_contents'\\]"):
        fixtrain.load_commits()
    fake_hub(monkeypatch, tmp_path, rows, licence="other")
    with pytest.raises(RuntimeError, match="not an open licence"):
        fixtrain.dataset_licence()


def test_question_and_answer_format() -> None:
    row = commit(0, old_contents="\ndef f0(x):\n    return x - 1\n\n")
    assert fixtrain.question(row) == ("```python\ndef f0(x):\n    return x - 1\n```\n\n"
                                      "Fix bugs: Fix off-by-one in f0. Reply with the full corrected code in one "
                                      "```python block.")
    assert fixtrain.answer(row) == "```python\ndef f0(x):\n    return x + 1\n```"
    assert [m["role"] for m in fixtrain.messages_for(row)] == ["user", "assistant"]
    # The same style as phase 1's prompt, and the answer is what fixeval's extraction takes out.
    assert fixeval.prompt_for(TASKS[0]).split("\n```\n\n")[0] == "```python\ndef add(a, b):\n    return a - b"
    assert fixeval.extract_code(fixtrain.answer(row)) == "def f0(x):\n    return x + 1\n"


def tokens(row) -> int:
    """A stand-in tokenizer: one token per word of the whole example."""
    return sum(len(m["content"].split()) for m in fixtrain.messages_for(row))


def test_select_rows_counts_every_filter() -> None:
    rows = [
        commit(0),  # kept
        commit(1, license="gpl-3.0"),
        commit(2, subject="Add feature", message="Add feature"),
        commit(3, new_file="tests/test_mod.py"),
        commit(4, new_contents=commit(4)["old_contents"]),  # no change
        commit(5, old_contents=""),
        commit(6, old_contents="x = '```'\n", new_contents="x = '``'\n"),
        commit(7, new_contents="".join(f"y{i} = {i}\n" for i in range(21))),  # 22 lines changed
        commit(8),  # kept
        commit(0, commit="c0-copy"),  # the same before and after as commit 0
        commit(9, new_contents="def f9(x):\n    return x + 1\n" + "# long " * 300),
        commit(10, new_contents='def f10(x):\n    """Count the vowels a e i o u in the given text."""\n    return 1\n'),
        commit(11, old_contents="def count_vowels( text:str )->int:\n    return 1\n"),  # its signature
        commit(12, new_contents="def f12(x):\n    # count the vowels a e i o u in\n    return x + 1\n"),  # 9 words
        commit(13, old_contents="def count_vowels(s):\n    return 0\n", new_contents="def count_vowels(s):\n"
               "    return 1\n"),  # same name, another signature: kept
    ]
    logged = []
    kept, counts = fixtrain.select_rows(rows, tokens, 200, HUMANEVAL, log=logged.append)
    assert [r["commit"] for r in kept] == ["c0", "c8", "c12", "c13"]
    expected = {"raw": 15, "licence": 1, "not_a_fix": 1, "test_file": 1, "empty_or_same": 2, "code_fence": 1,
                "too_many_changes": 1, "duplicate": 1, "too_long": 1, "contaminated": 2, "kept": 4}
    assert {k: counts[k] for k in expected} == expected
    assert counts["contaminated_by"] == {"def_line": 1, "10_words": 1}
    assert {(e["commit"], e["task_id"]) for e in counts["contaminated_examples"]} == {("c10", "Fake/2"),
                                                                                    ("c11", "Fake/2")}
    assert len(logged) == 2
    text = "\n".join(fixtrain.counts_lines(counts))
    for words in ("15 commits", "-1 repository licence", "-2 decontamination: shares text with HumanEvalFix (1 by 10 words, 1 by a def",
                  "4 kept"):
        assert words in text, words


def test_decontamination_ignores_case_and_punctuation() -> None:
    check = fixtrain.Contamination([VOWELS])
    shouty = {"old_contents": "# COUNT the vowels: A, E, I, O, U -- in the GIVEN text!\n", "new_contents": ""}
    assert check.match(shouty)[0] == "Fake/2"
    assert check.match({"old_contents": "def other(text: str) -> int:\n    return 0\n"}) is None
    # The tests are checked too (10 words in a row)
    tests = "def check(count_vowels):\n    assert count_vowels('hello') == 2\n    assert count_vowels('AEIOU')\n"
    assert check.match({"new_contents": tests})[1].startswith("shares 10 words: def check count_vowels")


def test_pick_is_the_same_every_time() -> None:
    rows = [commit(i) for i in range(50)]
    first = fixtrain.pick(rows, 10, 3407)
    assert first == fixtrain.pick(rows, 10, 3407) and len(first) == 10 and first != rows[:10]
    assert len(fixtrain.pick(rows, None)) == 50


# ---- scoring ----

def fake_generate(replies: dict, calls: list | None = None, oom_above: int = 0):
    """generate(prompts) as the notebook's: answers by the function name in the prompt."""

    def generate(prompts):
        if calls is not None:
            calls.append(len(prompts))
        if oom_above and len(prompts) > oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        names = [p.split("Fix bugs in ", 1)[1].split(".", 1)[0] for p in prompts]
        return [{"reply": replies.get(n, ""), "tokens": 30, "cut_off": False} for n in names]

    return generate


def test_score_runs_the_hidden_tests_and_halves_the_batch() -> None:
    calls, saves = [], []
    entry: dict = {}
    fixtrain.score(fake_generate({"add": FIXED, "halve": NOT_FIXED}, calls, oom_above=1), HUMANEVAL, entry, 60,
                   batch=4, save=lambda: saves.append(entry["done"]), log=lambda *a: None)
    assert calls == [3, 2, 1, 1, 1]  # out of memory with 3: halved to 2, still too many, then one at a time
    assert entry["batch"] == 1 and entry["errors"] == ["out of memory: batch now 2", "out of memory: batch now 1"]
    assert (entry["done"], entry["passed"], entry["total"], entry["pass_at_1"]) == (3, 1, 3, 0.3333)
    assert [(t["task_id"], t["passed"], t["why"]) for t in entry["tasks"]] == [
        ("Fake/0", True, "passed"), ("Fake/1", False, "AssertionError"), ("Fake/2", False, "no answer")]
    assert entry["tokens"] == 90 and entry["no_code_block"] == 1 and saves[-1] == 3


def test_score_stops_on_time_and_after_three_failures() -> None:
    entry: dict = {}
    fixtrain.score(fake_generate({}), HUMANEVAL, entry, 0, log=lambda *a: None)
    assert entry["stopped_by_time"] and entry["done"] == 0 and entry["pass_at_1"] is None

    def broken(prompts):
        raise ValueError("bad input")

    entry = {}
    fixtrain.score(broken, HUMANEVAL * 2, entry, 60, batch=1, log=lambda *a: None)
    assert entry["done"] == 3 and entry["passed"] == 0 and entry["errors"][-1] == "3 batches in a row failed: stopped"
    assert entry["tasks"][0]["why"].startswith("generation failed: ValueError")


def test_flips_and_the_report() -> None:
    before, after = {}, {}
    quiet = {"log": lambda *a: None}
    fixtrain.score(fake_generate({"add": FIXED}), HUMANEVAL, before, 60, **quiet)
    fixtrain.score(fake_generate({"add": BROKEN_ADD, "halve": HALVE_FIXED, "count_vowels": VOWELS_FIXED}),
                   HUMANEVAL, after, 60, **quiet)
    f = fixtrain.flips(before, after)
    assert f == {"both_done": 3, "before_passed": 1, "after_passed": 2, "now_fixed": ["Fake/1", "Fake/2"],
                 "now_broken": ["Fake/0"]}
    text = "\n".join(fixtrain.report_lines({"base": "fake", "before": before, "after": after, "flips": f}))
    for words in ("before             1/3   33%", "after              2/3   67%", "1 → 2 (+1); now fixed 2, now broken 1",
                  "now fixed:  Fake/1, Fake/2", "now broken: Fake/0", "Beats phase 1's best (qwen3-coder:30b, 73%",
                  "? no: 66.7%"):
        assert words in text, words
    text = "\n".join(fixtrain.report_lines({"before": before, "after": {}}))
    assert "not scored" in text and "Beats" not in text
    for partial in ({"smoke": True}, {"settings": {"EVAL_LIMIT": 8}}):  # a smoke or limited run is not compared
        text = "\n".join(fixtrain.report_lines({"before": before, "after": after, "flips": f, **partial}))
        assert "Beats" not in text


def test_progress_notes_the_loss_and_the_projected_time() -> None:
    train: dict = {}
    fixtrain.note_progress(train, 10, 250, {"loss": 0.81234, "learning_rate": 0.0002}, 600)
    assert train["log"] == [{"step": 10, "loss": 0.8123, "learning_rate": 0.0002}]
    assert (train["steps"], train["planned_steps"], train["last_loss"], train["projected_minutes"]) == (
        10, 250, 0.8123, 250.0)


# ---- the notebook's cells, run with fakes in place of the GPU parts ----

class FakeTokenizer:
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_notebook_cells_from_data_to_report(tmp_path: Path, capsys) -> None:
    scope: dict = {}
    exec(cell("#@title 3."), scope)  # noqa: S102 - our own generated cells
    scope["WORKING_DIR"] = str(tmp_path)
    exec(cell("#@title 4."), scope)  # noqa: S102
    commits = [commit(i) for i in range(5)] + [commit(9, new_contents=VOWELS["prompt"]), commit(10, license="gpl-3.0")]
    switched = {"inference": 0, "training": 0}
    scope.update(
        load_tasks=lambda: (HUMANEVAL, "fake tasks"), load_commits=lambda: (commits, "fake commits"),
        dataset_licence=lambda: {"dataset": "bigcode/commitpackft", "licence": ["mit"], "from": "fake"},
        tokenizer=FakeTokenizer(), model=object(), torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        FastLanguageModel=SimpleNamespace(for_inference=lambda m: switched.update(inference=switched["inference"] + 1),
                                          for_training=lambda m: switched.update(training=switched["training"] + 1)),
        generate_replies=fake_generate({"add": FIXED}),
    )
    exec(cell("#@title 6."), scope)  # noqa: S102
    assert len(scope["TRAIN_ROWS"]) == 5 and scope["EVAL_TASKS"] == HUMANEVAL
    exec(cell("#@title 7."), scope)  # noqa: S102
    scope["generate_replies"] = fake_generate({"add": FIXED, "halve": HALVE_FIXED})  # "trained"
    exec(cell("#@title 9."), scope)  # noqa: S102
    exec(cell("#@title 10."), scope)  # noqa: S102
    saved = json.loads((tmp_path / "coder_train.json").read_text(encoding="utf-8"))
    assert saved["licence"]["licence"] == ["mit"] and saved["data"]["source"] == "fake commits"
    assert saved["data"]["counts"]["contaminated"] == 1 and saved["data"]["counts"]["licence"] == 1
    assert saved["data"]["used"] == 5 and "old_contents" in saved["data"]["fields"]
    assert (saved["before"]["passed"], saved["after"]["passed"]) == (1, 2)
    assert saved["flips"]["now_fixed"] == ["Fake/1"] and saved["errors"] == []
    assert saved["settings"]["BASE"] == "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit" and saved["smoke"] is False
    assert "No GGUF" in saved["export"] and saved["eval_source"] == "fake tasks"
    assert switched == {"inference": 2, "training": 1}
    out = capsys.readouterr().out
    for words in ("Licence of bigcode/commitpackft", "Fields: commit, lang, license", "decontamination dropped 1",
                  "[assistant]", "after              2/3   67%"):
        assert words in out, words


def test_a_failed_data_step_is_recorded(tmp_path: Path) -> None:
    scope: dict = {}
    exec(cell("#@title 3."), scope)  # noqa: S102
    scope["WORKING_DIR"] = str(tmp_path)
    exec(cell("#@title 4."), scope)  # noqa: S102

    def closed():
        raise RuntimeError("bigcode/commitpackft: licence ['other'] is not an open licence")

    scope.update(load_tasks=lambda: (HUMANEVAL, "fake"), dataset_licence=closed, tokenizer=FakeTokenizer())
    with pytest.raises(RuntimeError):
        exec(cell("#@title 6."), scope)  # noqa: S102
    saved = json.loads((tmp_path / "coder_train.json").read_text(encoding="utf-8"))
    assert saved["errors"][0].startswith("training data: RuntimeError(\"bigcode/commitpackft: licence ['other']")


# ---- the runner ----

RESULT = {
    "phase": 2, "base": "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit", "smoke": False, "benchmark": fixeval.BENCHMARK,
    "eval_setting": fixtrain.EVAL_SETTING, "train_dataset": "bigcode/commitpackft", "versions": {"unsloth": "2026.10"},
    "gpu": "Tesla T4", "licence": {"dataset": "bigcode/commitpackft", "licence": ["mit"], "from": "fake"},
    "data": {"source": "bigcode/commitpackft/data/python/data.jsonl", "used": 2000,
             "counts": {"raw": 56025, "licence": 10, "not_a_fix": 40000, "test_file": 500, "empty_or_same": 5,
                        "code_fence": 3, "too_many_changes": 4000, "duplicate": 20, "too_long": 3000,
                        "contaminated": 37, "contaminated_by": {"def_line": 7, "10_words": 30},
                        "contaminated_examples": [{"subject": "Fix add", "task_id": "Python/53", "why": "defines x"}],
                        "kept": 8451}},
    "train": {"examples": 2000, "steps": 250, "planned_steps": 250, "loss": 0.4123, "minutes": 201.5,
              "stopped_by_time": False, "log": []},
    "before": {"total": 164, "done": 164, "passed": 103, "pass_at_1": 0.628, "tokens_per_s": 31.0, "errors": [],
               "tasks": [{"task_id": "Python/0", "passed": True}, {"task_id": "Python/1", "passed": False}]},
    "after": {"total": 164, "done": 164, "passed": 112, "pass_at_1": 0.6829, "tokens_per_s": 29.5, "errors": [],
              "tasks": [{"task_id": "Python/0", "passed": False}, {"task_id": "Python/1", "passed": True}]},
    "flips": {"both_done": 164, "before_passed": 103, "after_passed": 112, "now_fixed": ["Python/1"],
              "now_broken": ["Python/0"]},
    "lora_dir": "/kaggle/working/coder-lora", "export": fixtrain.NO_EXPORT, "errors": [],
}


class FakeKaggle(FakeApi):
    """test_try.py's fake Kaggle, handing out coder_train.json instead."""

    def kernels_output(self, kernel, path, file_pattern=None, force=False):
        self.calls["pattern"] = file_pattern
        log = [{"stream_name": "stdout", "time": 1, "data": "RuntimeError: Could not download the commits\n"}]
        (Path(path) / "wimb-coder-train.log").write_text(json.dumps(log))
        if self.results is not None:
            (Path(path) / "coder_train.json").write_text(json.dumps(self.results), encoding="utf-8")


def test_runner_pushes_a_private_gpu_run_with_internet() -> None:
    api = FakeKaggle(RESULT)
    run_train.push(api, "me/wimb-coder-train", run_train.notebook_with(True))
    meta = api.calls["meta"]
    assert meta["id"] == "me/wimb-coder-train" and meta["title"] == "wimb-coder-train"
    assert meta["code_file"] == "coder_train.ipynb" and meta["kernel_sources"] == []
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert api.calls["acc"] == "NvidiaTeslaT4"
    assert run_train.settings(api.calls["notebook"])["SMOKE"] is True


def run_main(api, tmp_path: Path, monkeypatch, *argv: str) -> int:
    """run_train.main() against the fake Kaggle: its exit code."""
    monkeypatch.setattr(run_train, "OUT", tmp_path / "out")
    monkeypatch.setattr(run_train, "connect", lambda slug: (api, f"me/{slug}"))
    monkeypatch.setattr(run_train, "check_gpu", lambda api, need_hours: api.calls.update(need_hours=need_hours))
    real_wait = run_train.wait
    monkeypatch.setattr(run_train, "wait", lambda api, kernel, limit_min: api.calls.update(wait=limit_min)
                        or real_wait(api, kernel, limit_min))
    monkeypatch.setattr(run_on_kaggle.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", ["run_train.py", *argv])
    with pytest.raises(SystemExit) as stop:
        run_train.main()
    return stop.value.code


def test_runner_prints_before_and_after(tmp_path, monkeypatch, capsys) -> None:
    api = FakeKaggle(RESULT)
    assert run_main(api, tmp_path, monkeypatch, "--max-train", "3000", "--epochs", "2") == 0
    s = run_train.settings(api.calls["notebook"])
    assert (s["MAX_TRAIN"], s["EPOCHS"], s["SMOKE"]) == (3000, 2, False)
    assert api.calls["need_hours"] == 7 and api.calls["wait"] == 690
    assert api.calls["pattern"] == r"(coder_train\.json|.*\.log|.*\.ipynb)$"
    out = capsys.readouterr().out
    for words in ("licence ['mit']", "56,025 commits", "-37 decontamination: shares text with HumanEvalFix (30 by 10 words, 7 by a def",
                  "8,451 kept", "decontamination dropped: 'Fix add' (Python/53: defines x)",
                  "Training: 2000 examples, 250 of 250 steps, 201.5 min, loss 0.4123",
                  "before         103/164   63%", "after          112/164   68%", "103 → 112 (+9)",
                  "now fixed:  Python/1", "Beats phase 1's best (qwen3-coder:30b, 73% with Ollama)? no: 68.3%",
                  "LoRA: /kaggle/working/coder-lora", "No GGUF"):
        assert words in out, words
    assert "Kaggle's log" not in out and "still running" not in out


def test_runner_smoke_needs_less_time(tmp_path, monkeypatch) -> None:
    api = FakeKaggle({**RESULT, "smoke": True})
    assert run_main(api, tmp_path, monkeypatch, "--smoke") == 0
    assert api.calls["need_hours"] == 0.5 and api.calls["wait"] == 90
    assert run_train.settings(api.calls["notebook"])["SMOKE"] is True


def test_runner_says_still_running_and_exits_0(tmp_path, monkeypatch, capsys) -> None:
    api = FakeKaggle(RESULT, final=Status.RUNNING)
    assert run_main(api, tmp_path, monkeypatch, "--wait-minutes", "1") == 0
    out = capsys.readouterr().out
    assert "still running; run again with action fetch-train" in out
    assert "pattern" not in api.calls  # nothing fetched: Kaggle would hand out the previous run's output
    api = FakeKaggle(RESULT, final=Status.RUNNING)
    assert run_main(api, tmp_path, monkeypatch, "--wait", "--wait-minutes", "1") == 0
    assert "meta" not in api.calls and "need_hours" not in api.calls and api.calls["wait"] == 1


@pytest.mark.parametrize(("api", "says"), [
    (FakeKaggle(None), "No coder_train.json"),
    (FakeKaggle({**RESULT, "after": {}}), "No AFTER score"),
    (FakeKaggle({**RESULT, "after": {}}, final=Status.ERROR), "The run did not finish. Kaggle's log:"),
])
def test_runner_fails_without_an_after_score(api, says, tmp_path, monkeypatch, capsys) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "coder_train.json").write_text(json.dumps(RESULT))  # an older run's file must not count
    assert run_main(api, tmp_path, monkeypatch, "--wait") == 1
    out = capsys.readouterr().out
    assert says in out
    if "log" in says:
        assert "Could not download the commits" in out


@pytest.mark.parametrize("argv", [("--max-train", "0"), ("--max-train", "x"), ("--epochs", "9"),
                                  ("--wait-minutes", "0")])
def test_bad_options_stop_before_kaggle(argv, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_train.py", *argv])
    monkeypatch.setattr(run_train, "connect", lambda slug: pytest.fail("must not reach Kaggle"))
    with pytest.raises(SystemExit) as stop:
        run_train.main()
    assert stop.value.code == 2


# ---- the workflow ----

def workflow_command(tmp_path: Path, **env: str) -> list[str]:
    """What coder.yml's main step runs with these inputs: the arguments it gives python."""
    yaml = pytest.importorskip("yaml")  # comes with kaggle (through jupytext)
    workflow = yaml.safe_load((HERE.parent / ".github" / "workflows" / "coder.yml").read_text(encoding="utf-8"))
    (step,) = [s for s in workflow["jobs"]["eval"]["steps"] if "run_train.py" in s.get("run", "")]
    assert set(env) <= set(step["env"])
    bin_dir = tmp_path / "bin"  # a stand-in `python` that writes down its arguments, one per line
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python").write_text(f'#!/bin/sh\nfor a in "$@"; do echo "$a"; done > {tmp_path}/args\n')
    (bin_dir / "python").chmod(0o755)
    run_env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "ACTION": "", "MODELS": "", "LIMIT": "",
               "SMOKE": "", **env}
    subprocess.run(["bash", "-e", "-c", step["run"]], env=run_env, capture_output=True, check=True)
    return (tmp_path / "args").read_text().splitlines()


def test_workflow_trains_fetches_and_still_evaluates(tmp_path: Path) -> None:
    text = (HERE.parent / ".github" / "workflows" / "coder.yml").read_text(encoding="utf-8")
    for line in text.splitlines():  # inputs reach the scripts only as environment variables
        if "${{ inputs." in line:
            assert re.fullmatch(r" +[A-Z_]+: \$\{\{ inputs\.\w+ \}\}", line), line
    assert "options: [eval, fetch, train, fetch-train]" in text and "kaggle_output_coder_train/" in text
    assert int(re.search(r"timeout-minutes: (\d+)", text).group(1)) <= 360  # GitHub's limit
    train = ["coder/run_train.py", "--wait-minutes", "330"]
    assert workflow_command(tmp_path, ACTION="train", SMOKE="no") == train
    assert workflow_command(tmp_path, ACTION="train", SMOKE="yes") == [*train, "--smoke"]
    assert workflow_command(tmp_path, ACTION="fetch-train", SMOKE="yes") == [*train, "--wait"]
    assert workflow_command(tmp_path, ACTION="eval", MODELS="qwen2.5-coder:7b", LIMIT="5", SMOKE="yes") == [
        "coder/run_eval.py", "--models", "qwen2.5-coder:7b", "--limit", "5"]
    assert workflow_command(tmp_path, ACTION="fetch", MODELS="a:1") == ["coder/run_eval.py", "--wait"]
    assert 330 < int(re.search(r"timeout-minutes: (\d+)", text).group(1))  # the job outlasts the wait


def test_build_writes_both_notebooks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(build, "HERE", tmp_path)
    monkeypatch.setattr(build_train, "HERE", tmp_path)
    for name in ("fixeval.py", "fixtrain.py"):
        (tmp_path / name).write_text((HERE / name).read_text(encoding="utf-8"), encoding="utf-8")
    build.main()
    assert (tmp_path / build.NOTEBOOK).read_text("utf-8") == (HERE / build.NOTEBOOK).read_text("utf-8")
    assert (tmp_path / build_train.NOTEBOOK).read_text("utf-8") == (HERE / build_train.NOTEBOOK).read_text("utf-8")

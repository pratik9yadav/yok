"""Build coder/exam/exam.ipynb: the engineer exam (coder phase 4). Free coding models in Ollama on Kaggle's free GPU
get real-world tasks written as a user would write them (no checklist, no style guide) and their replies are graded
on hidden tests, their own tests, lint, security and an explanation.

`python coder/build.py` writes it (with the other coder notebooks); run it after changing a task in
coder/exam/tasks, or coder/exam/grade.py, coder/exam/exam.py or this file. The tasks are put into the notebook, so
Kaggle needs no access to the repository. The notebook reuses the Ollama install/start code of the shop-model
notebooks (build_notebook.py) and the Ollama helpers of coder/fixeval.py.
"""

import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

from build_notebook import METADATA, OLLAMA_TITLE, START_OLLAMA, code, md, run_helpers, text, titled
from coder import fixeval
from coder.exam import grade

NOTEBOOK = "exam.ipynb"
RESULTS_FILE = "exam.json"
SETTINGS_TITLE = "1. Settings"
TASKS_DIR = HERE / "tasks"
CLI_PART = "# ---- the command line ----"  # grade.py from here on is not put into the notebook
# From coder/fixeval.py: talk to Ollama (download, load, where it runs, unload, delete) and read a command's output.
OLLAMA_HELPERS = (fixeval.post, fixeval.pull, fixeval.in_memory, fixeval.processor, fixeval.unload, fixeval.delete,
                  fixeval.text_of, fixeval.think_for, fixeval.disk_free_gb, fixeval.tidy_up)

INTRO = """
# Coder, phase 4: the engineer exam (Ollama on Kaggle's free GPU)

**The question:** does a free coding model work like an engineer *on its own*, with no style guide or checklist in
the prompt? Each model in `MODELS` gets every task as a user would write it (plus the project's files for a bug
fix, review or refactor, and one line on how to send files back). Nothing else: no system prompt.

**How a reply is graded (0-100):** the task's hidden tests, never shown (55); did it write its own tests, and do
they pass (15); lint with a fixed ruff rule set (10); security with bandit (10); does it explain what it did (10).
Stray files (scratch scripts nothing uses) cost 5. "Solved" = every hidden test passes.

It saves everything to `exam.json` (Output tab) after every task: every reply, every grade and the raw test output.
Time: at most `MINUTES_PER_MODEL` minutes per model (download included), so about 4½ hours for 4 models.

**Settings (right side):** Accelerator **GPU T4 x2**, Internet **on**. Then **Run all**.

**Or with no clicks:** on GitHub, **Actions → Coder model on Kaggle → Run workflow → action `exam`**, or ask Claude.
"""

SETTINGS = """
import glob, json, os, re, shutil, subprocess, sys, time, urllib.error, urllib.request
from collections import Counter

# The models: names from ollama.com/library. Claude and GitHub Actions fill this line.
MODELS = ["qwen3-coder:30b", "qwen2.5-coder:14b", "gpt-oss:20b", "qwen2.5-coder:7b"]
# How many tasks to use (None = all). Claude and GitHub Actions fill this line.
LIMIT = None
MINUTES_PER_MODEL = 60  # a model stops after this (download, loading and answers), so the run ends in time
SAMPLES = 1  # answers per task (with temperature 0 they are all the same)
OPTIONS = {"temperature": 0, "num_ctx": 16384, "num_predict": 6144}  # room for the task, its files and whole files back
THINK = {"gpt-oss": "medium"}  # thinking models: how much to think first
# Downloads go to the big disk. Not /kaggle/working: it holds only about 20 GB, and all of it is saved as output.
MODELS_DIR = "/tmp/ollama-models"
GRADE_DIR = "/tmp/exam-grading"  # each reply is tested in a fresh folder here (deleted after)
DELETE_AFTER = True  # delete each model after its turn, so the disk holds only one at a time
WORKING_DIR = "/kaggle/working"  # exam.json is saved here (the notebook's Output tab)
"""

STATE = """
RESULTS = {"exam": "engineer exam (coder phase 4)", "setting": SETTING, "reply_format": REPLY_FORMAT,
           "weights": WEIGHTS, "junk_penalty": JUNK_PENALTY, "ruff_config": RUFF_CONFIG, "options": OPTIONS,
           "think": THINK, "samples": SAMPLES, "minutes_per_model": MINUTES_PER_MODEL, "limit": LIMIT,
           "models_asked": MODELS, "models_dir": MODELS_DIR, "ollama_version": None, "gpu": None, "gpus": None,
           "disk": None, "pip": None, "checkers": None, "tasks_total": None, "tasks_used": None, "task_ids": None,
           "kinds": None, "reference_check": None, "tasks_left_out": [], "models": [], "errors": []}
"""

CHECKERS = """
# pytest runs the tests, ruff lints, bandit looks for security problems (optional: without it, security is left
# out of the total).
pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "pytest", "ruff",
                      "bandit"], capture_output=True, text=True, timeout=900)
RESULTS["pip"] = (pip.stdout + pip.stderr).strip()[-1500:] or "ok"
importlib.invalidate_caches()
RESULTS["checkers"] = checker_versions()
print(RESULTS["checkers"])
if not RESULTS["checkers"]["pytest"]:
    failed("install pytest", RESULTS["pip"])
    raise RuntimeError("pytest did not install, so no reply can be tested (see above)")
for name in ("ruff", "bandit"):
    if not RESULTS["checkers"][name]:
        print(f"{name} did not install: its part is left out of the total")
save_results()
"""

OLLAMA = """
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["OLLAMA_MODELS"] = MODELS_DIR  # where `ollama serve` keeps the downloads
os.environ["OLLAMA_NUM_PARALLEL"] = "1"  # one question at a time: less GPU memory
""" + START_OLLAMA + """
RESULTS["gpus"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader") or None
RESULTS["disk"] = sh(f"df -h {MODELS_DIR} {WORKING_DIR} 2>/dev/null") or None
print("GPUs:", RESULTS["gpus"])
print(RESULTS["disk"])
save_results()
"""

PICK_TASKS = """
os.makedirs(GRADE_DIR, exist_ok=True)
TASKS = EXAM_TASKS[:LIMIT] if LIMIT else list(EXAM_TASKS)
print(f"{len(EXAM_TASKS)} tasks; using {len(TASKS)}:", dict(Counter(t["meta"]["kind"] for t in TASKS)))
# Check the checker here, before any model: every task's reference solution must pass its hidden tests on this
# machine. A task whose reference fails is left out (it would only measure this machine).
rows, _ = check_references(TASKS, GRADE_DIR)
print("\\n".join(reference_table(rows)))
RESULTS["reference_check"] = [{"task": r["task"], "total": r["grade"]["total"], "problems": r["problems"],
                               "hidden": f"{r['grade']['hidden_passed']}/{r['grade']['hidden_total']}"} for r in rows]
RESULTS["tasks_left_out"] = sorted(r["task"] for r in rows if r["problems"])
if RESULTS["tasks_left_out"]:
    print("Left out (their reference fails here):", ", ".join(RESULTS["tasks_left_out"]))
TASKS = [t for t in TASKS if t["id"] not in RESULTS["tasks_left_out"]]
RESULTS.update(tasks_total=len(EXAM_TASKS), tasks_used=len(TASKS), task_ids=[t["id"] for t in TASKS],
               kinds=dict(Counter(t["meta"]["kind"] for t in TASKS)))
save_results()
if not TASKS:
    raise RuntimeError("No task left to give the models (see above)")
print("\\n--- What a model is sent (the first task):\\n" + prompt_for(TASKS[0]))
"""

EVALUATE = """
for model in MODELS:
    entry = {"model": model}
    RESULTS["models"].append(entry)
    try:
        take_exam(OLLAMA_URL, model, TASKS, entry, MINUTES_PER_MODEL, OPTIONS, THINK, SAMPLES, save=save_results,
                  models_dir=MODELS_DIR, workdir=GRADE_DIR)
    except Exception as e:  # record it and go on to the next model
        entry.setdefault("errors", []).append(f"{e!r}"[:500])
        failed(model, e)
    if entry.get("pulled"):
        tidy_up(OLLAMA_URL, model, entry, DELETE_AFTER)
    else:  # a download that stopped part-way leaves its pieces: remove them, so the next model has the disk
        for part in glob.glob(os.path.join(MODELS_DIR, "blobs", "*-partial*")):
            os.remove(part)
    save_results()
    print("\\n".join(table(RESULTS)))
"""

REPORT = """
print("\\n".join(table(RESULTS)))
for m in RESULTS["models"]:
    if m.get("tasks"):
        print(f"\\n{m['model']}: the 5 weakest tasks")
        print("\\n".join(weakest(m)))
    for error in m.get("errors") or []:
        print(f"{m['model']}: {error}")
for error in RESULTS["errors"]:
    print("Error:", error)
save_results()
print("Saved:", os.path.join(WORKING_DIR, RESULTS_FILE))
"""

NEXT = """
## What next
`exam.json` keeps every reply and every grade (with the test, lint and bandit output), so each weak task can be
read. The weakest parts say what to teach next: if models rarely write tests or explain themselves on their own,
that is what a fine-tune (or a better default prompt in the coding agent) should add.
"""


def grade_source() -> str:
    """coder/exam/grade.py without its command line part."""
    source = (HERE / "grade.py").read_text(encoding="utf-8")
    if CLI_PART not in source:
        raise RuntimeError(f"coder/exam/grade.py has no line {CLI_PART!r}")
    return source.split(CLI_PART, 1)[0].rstrip("\n") + "\n"


def exam_source() -> str:
    """coder/exam/exam.py without its imports from the repository (their code is earlier in the same cell)."""
    lines = (HERE / "exam.py").read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(line for line in lines if not line.startswith("from coder."))


def embedded_tasks(tasks_dir: Path) -> str:
    """The tasks as notebook code: one json.loads(...) line per task (repr keeps any text safe on one line)."""
    tasks = grade.load_tasks(tasks_dir) if tasks_dir.is_dir() else []
    rows = "".join(f"    json.loads({json.dumps(t, ensure_ascii=False, sort_keys=True)!r}),\n" for t in tasks)
    return f"EXAM_TASKS = [\n{rows}]\n"


def notebook(tasks_dir: Path | None = None) -> dict:
    """The exam notebook, with the tasks in tasks_dir (default coder/exam/tasks)."""
    tasks_dir = tasks_dir or TASKS_DIR
    helpers = (
        "#@title 2. Helpers (no need to read; built from coder/fixeval.py, coder/exam/grade.py and coder/exam/exam.py)\n"
        + "\n\n".join(inspect.getsource(f) for f in OLLAMA_HELPERS)
        + "\n\n# ---- coder/exam/grade.py ----\n"
        + grade_source()
        + "\n\n# ---- coder/exam/exam.py ----\n"
        + exam_source()
        + f"\n\nRESULTS_FILE = {RESULTS_FILE!r}\n"
        + STATE
        + run_helpers(RESULTS_FILE, 4)
    )
    tasks = (f"#@title 5. The tasks (no need to read; built from coder/exam/{tasks_dir.name})\n"
             "# Each: id, task_md (what the model is asked), meta, starter (files it is shown), hidden (the tests it\n"
             "# never sees) and reference (a solution, used only to check the checker).\n"
             + embedded_tasks(tasks_dir))
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": [
            md(INTRO),
            code(titled(f"{SETTINGS_TITLE}: which models, how many tasks, time per model", SETTINGS)),
            code(helpers, hidden=True, folded=True),
            code(titled("3. Install the checkers: pytest, ruff, bandit (about 1 minute)", CHECKERS)),
            code(titled(f"4. {OLLAMA_TITLE}", OLLAMA)),
            code(tasks, hidden=True, folded=True),
            code(titled("6. Pick the tasks, and check every reference solution passes here", PICK_TASKS)),
            code(titled("7. Each model: download, load, do every task, grade each reply", EVALUATE)),
            code(titled("8. Results", REPORT)),
            md(NEXT),
        ],
    }


def main() -> None:
    out = HERE / NOTEBOOK
    out.write_text(text(notebook()), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

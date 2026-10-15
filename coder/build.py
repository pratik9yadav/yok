"""Build coder/coder_eval.ipynb: measure free open coding models at fixing bugs (HumanEvalFix), with Ollama on Kaggle.
Also writes coder/coder_train.ipynb (phase 2, training; built by coder/build_train.py) and coder/coder_agent.ipynb
(phase 3, the coding agent; built by coder/build_agent.py).

Run after changing this file, coder/build_train.py, coder/build_agent.py, coder/fixeval.py, coder/fixtrain.py or
coder/agent.py:  python coder/build.py
The notebook reuses the Ollama install/start code and helpers of the shop-model notebooks (build_notebook.py).
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from build_notebook import METADATA, OLLAMA_TITLE, START_OLLAMA, code, md, run_helpers, text, titled

NOTEBOOK = "coder_eval.ipynb"
RESULTS_FILE = "coder_eval.json"
SETTINGS_TITLE = "1. Settings"

INTRO = """
# Coder: which free coding model fixes bugs best? (Ollama on Kaggle's free GPU)

**What it does:** downloads each model in `MODELS` with Ollama and gives it the 164 tasks of
**HumanEvalFix (Python)**: a small Python function with a bug in it. The model must send back the fixed
function. The fix is then run against the task's hidden tests: all pass = the task is fixed.
The score is the share of tasks fixed with one try (pass@1, greedy: temperature 0).

It saves everything to `coder_eval.json` (Output tab) after every model, so nothing is lost if it stops.
Time: about `MINUTES_PER_MODEL` minutes per model at most (download + answers), so about 3½ hours for 4 models.

**Settings (right side):** Accelerator **GPU T4 x2**, Internet **on**. Then **Run all**.

**Or with no clicks:** on GitHub, **Actions → Coder model on Kaggle → Run workflow**, or ask Claude.

The tasks: HumanEvalPack by BigCode (`bigcode/humanevalpack`, MIT licence).
"""

SETTINGS = """
import glob, json, os, re, shutil, subprocess, sys, time, urllib.request

# The models: names from ollama.com/library. Claude and GitHub Actions fill this line.
MODELS = ["qwen2.5-coder:7b", "qwen2.5-coder:14b", "qwen3-coder:30b", "gpt-oss:20b"]
# How many tasks to use (None = all 164). Claude and GitHub Actions fill this line.
LIMIT = None
MINUTES_PER_MODEL = 45  # a model stops after this (download, loading and answers), so the run ends in time
OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 1024}  # temperature 0: the same answer every time
THINK = {"gpt-oss": "low"}  # thinking models: how much to think first ("low" fits 164 tasks in the time)
THINKING_NUM_PREDICT = 3072  # thinking models: room for the thinking and the answer (inside num_ctx)
# Downloads go to the big disk. Not /kaggle/working: it holds only about 20 GB, and all of it is saved as output.
MODELS_DIR = "/tmp/ollama-models"
DELETE_AFTER = True  # delete each model after its turn, so the disk holds only one at a time
WORKING_DIR = "/kaggle/working"  # coder_eval.json is saved here (the notebook's Output tab)
"""

STATE = """
RESULTS = {"benchmark": BENCHMARK, "dataset": DATASET, "config": "python", "split": "test", "licence": LICENCE,
           "setting": SETTING, "instruction": INSTRUCTION, "test_timeout_s": TEST_TIMEOUT, "options": OPTIONS,
           "think": THINK, "thinking_num_predict": THINKING_NUM_PREDICT, "minutes_per_model": MINUTES_PER_MODEL,
           "limit": LIMIT, "models_asked": MODELS, "models_dir": MODELS_DIR, "ollama_version": None, "gpu": None,
           "gpus": None, "disk": None, "dataset_source": None, "fields": None, "tasks_total": None, "tasks_used": None,
           "models": [], "errors": []}
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

LOAD_TASKS = """
try:
    ALL_TASKS, RESULTS["dataset_source"] = load_tasks()
except Exception as e:
    failed("download the tasks", e)
    raise
RESULTS["fields"] = sorted(ALL_TASKS[0])
TASKS = ALL_TASKS[:LIMIT] if LIMIT else ALL_TASKS
RESULTS.update(tasks_total=len(ALL_TASKS), tasks_used=len(TASKS))
print(f"{len(ALL_TASKS)} tasks from {RESULTS['dataset_source']}; using {len(TASKS)}")
print("Fields:", ", ".join(RESULTS["fields"]))
for field in RESULTS["fields"]:  # the first task, field by field
    print(f"--- {field}:\\n{str(TASKS[0][field])[:400]}")
print("--- What a model is sent:\\n" + prompt_for(TASKS[0]))
save_results()
"""

EVALUATE = """
for model in MODELS:
    entry = {"model": model}
    RESULTS["models"].append(entry)
    try:
        evaluate_model(OLLAMA_URL, model, TASKS, entry, MINUTES_PER_MODEL, OPTIONS, THINK, THINKING_NUM_PREDICT,
                       save=save_results, models_dir=MODELS_DIR)
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
    for error in m.get("errors") or []:
        print(f"{m['model']}: {error}")
for error in RESULTS["errors"]:
    print("Error:", error)
save_results()
print("Saved:", os.path.join(WORKING_DIR, RESULTS_FILE))
"""

NEXT = """
## What next
The best model here is the one to train next (phase 2), if it runs fast enough on this GPU.
`coder_eval.json` keeps every reply, so the failed tasks can be read later.
"""


def notebook() -> dict:
    """The evaluation notebook."""
    helpers = (
        "#@title 2. Helpers (no need to read; built from coder/fixeval.py)\n"
        + (HERE / "fixeval.py").read_text(encoding="utf-8")
        + f"\n\nRESULTS_FILE = {RESULTS_FILE!r}\n"
        + STATE
        + run_helpers(RESULTS_FILE, 3)
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": [
            md(INTRO),
            code(titled(f"{SETTINGS_TITLE}: which models, how many tasks, time per model", SETTINGS)),
            code(helpers, hidden=True, folded=True),
            code(titled(f"3. {OLLAMA_TITLE}", OLLAMA)),
            code(titled("4. The tasks: HumanEvalFix (Python), buggy functions with hidden tests", LOAD_TASKS)),
            code(titled("5. Each model: download, load, fix every task, run the tests", EVALUATE)),
            code(titled("6. Results", REPORT)),
            md(NEXT),
        ],
    }


def main() -> None:
    from coder import build_agent, build_train

    out = HERE / NOTEBOOK
    out.write_text(text(notebook()), encoding="utf-8")
    print(f"wrote {out}")
    build_train.main()
    build_agent.main()


if __name__ == "__main__":
    main()

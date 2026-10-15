"""Build coder/coder_agent.ipynb: a free "junior engineer" coding agent. qwen3-coder:30b in Ollama on Kaggle's free
GPU works on a copy of a repository and leaves its change as changes.patch (phase 3).

`python coder/build.py` writes it (with the other coder notebooks). The notebook reuses the Ollama install/start code
of the shop-model notebooks (build_notebook.py), the Ollama helpers of coder/fixeval.py, and coder/agent.py (the loop).
"""

import inspect
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from build_notebook import METADATA, OLLAMA_TITLE, REPO, START_OLLAMA, code, md, run_helpers, text, titled
from coder import fixeval

NOTEBOOK = "coder_agent.ipynb"
RESULTS_FILE = "agent_result.json"
LOG_FILE = "agent_log.json"
PATCH_FILE = "changes.patch"
SETTINGS_TITLE = "1. Settings"
# From coder/fixeval.py: talk to Ollama (download, load, where it runs, unload) and read a command's output.
OLLAMA_HELPERS = (fixeval.post, fixeval.pull, fixeval.in_memory, fixeval.processor, fixeval.unload, fixeval.text_of)

INTRO = """
# Coder, phase 3: a free coding agent ("junior engineer") on Kaggle's free GPU

**What it does:** gets a copy of the repository at `COMMIT`, starts **qwen3-coder:30b** in Ollama (on both T4s), and
lets it work on `TASK` like a junior engineer: it reads the repo guide, finds the code, makes the smallest change,
and runs the tests and the linter, one bash command at a time (at most `MAX_STEPS` commands and `AGENT_MINUTES`
minutes). It never commits or pushes: its change is saved as `changes.patch`.

Output tab: `changes.patch` (the change), `agent_log.json` (every step: thought, command, exit code, output, seconds)
and `agent_result.json` (status: done / question / stopped / error, the agent's summary, the tests before and
after). Saved as it goes, so nothing is lost if it stops.

Time: about 10 minutes to start (Ollama, the 19 GB model, the repository), then up to `AGENT_MINUTES` minutes.

**Settings (right side):** Accelerator **GPU T4 x2**, Internet **on**. Write the task in cell 1, then **Run all**.

**Or with no clicks:** on GitHub, **Actions → Coding agent (free model) → Run workflow**, type the task: about
30–60 minutes later the change comes back as a pull request to review.
"""

SETTINGS = f"""
import json, os, re, shutil, signal, subprocess, sys, time, urllib.error, urllib.request

# The task, in plain words. Claude and GitHub Actions fill this line.
TASK = ''
# The repository (public: no password on this machine). Claude and GitHub Actions fill this line.
REPO_URL = "https://github.com/{REPO}.git"
# The commit to start from ("" = the newest one). Claude and GitHub Actions fill this line.
COMMIT = ""
MODEL = "qwen3-coder:30b"  # 73% on HumanEvalFix in phase 1, 2.2 s per task on 2x T4
MAX_STEPS = 40  # at most this many commands
AGENT_MINUTES = 45  # the agent stops after this
NUM_CTX = 32768  # tokens the model sees at once (the task and the steps so far)
NUM_CTX_FALLBACK = 16384  # used when NUM_CTX does not fit on the GPUs (Ollama would move part of it to the CPU)
NUM_PREDICT = 4096  # longest reply (a reply can write a file)
OPTIONS = {{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05, "seed": 0}}  # Qwen's advice
# Downloads go to the big disk. Not /kaggle/working: it holds only about 20 GB, and all of it is saved as output.
MODELS_DIR = "/tmp/ollama-models"
REPO_DIR = "/tmp/repo"  # the copy of the repository the agent works in
WORKING_DIR = "/kaggle/working"  # {PATCH_FILE}, {LOG_FILE} and {RESULTS_FILE} are saved here (the Output tab)
"""

STATE = """
RESULTS = {"status": "error", "summary": "The run stopped before the agent finished (see errors).", "task": TASK,
           "repo": REPO_URL, "commit": COMMIT or None, "model": MODEL, "num_ctx": None, "options": OPTIONS,
           "max_steps": MAX_STEPS, "agent_minutes": AGENT_MINUTES, "steps": 0, "minutes": None,
           "tests_before": None, "tests_after": None, "files_changed": [], "patch_bytes": 0, "ollama_version": None,
           "gpu": None, "gpus": None, "size_gb": None, "pull_seconds": None, "load_seconds": None, "processor": None,
           "gpu_memory": None, "ollama_ps": None, "pip": None, "errors": []}
"""

OLLAMA = """
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["OLLAMA_MODELS"] = MODELS_DIR  # where `ollama serve` keeps the downloads
os.environ["OLLAMA_NUM_PARALLEL"] = "1"  # one request at a time: memory for one NUM_CTX only
""" + START_OLLAMA + """
RESULTS["gpus"] = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader") or None
print("GPUs:", RESULTS["gpus"])
save_results()
"""

MODEL_CELL = """
try:
    started = time.time()
    RESULTS["size_gb"] = pull(OLLAMA_URL, MODEL)
    RESULTS["pull_seconds"] = round(time.time() - started, 1)
    print(f"{MODEL}: {RESULTS['size_gb']} GB in {RESULTS['pull_seconds']} s")
    for num_ctx in (NUM_CTX, NUM_CTX_FALLBACK):
        loading = time.time()
        post(OLLAMA_URL, "/api/generate", {"model": MODEL, "options": {**OPTIONS, "num_ctx": num_ctx},
                                           "keep_alive": "90m"}, timeout=900)  # only load it
        RESULTS.update(num_ctx=num_ctx, load_seconds=round(time.time() - loading, 1),
                       processor=processor(OLLAMA_URL, MODEL), ollama_ps=text_of("ollama ps") or None,
                       gpu_memory=sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
                       .replace("\\n", "; ") or None)
        print(f"num_ctx {num_ctx}: loaded in {RESULTS['load_seconds']} s, runs on {RESULTS['processor']}; "
              f"GPU memory {RESULTS['gpu_memory']}")
        if RESULTS["processor"] == "100% GPU" or num_ctx == NUM_CTX_FALLBACK:
            break
        print(f"num_ctx {num_ctx} does not fit on the GPUs: trying {NUM_CTX_FALLBACK}")
        unload(OLLAMA_URL, MODEL)
except Exception as e:
    failed("download or load the model", e)
    raise
print(RESULTS["ollama_ps"])
save_results()
"""

REPO_CELL = """
try:
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    clone = subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO_DIR], capture_output=True, text=True,
                           timeout=900)
    if clone.returncode:
        raise RuntimeError(f"git clone failed: {clone.stderr[-1500:]}")
    if COMMIT:
        git(REPO_DIR, "checkout", "--quiet", "--detach", COMMIT)
    RESULTS["commit"] = git(REPO_DIR, "rev-parse", "HEAD").strip()
    print(f"{REPO_URL} at {RESULTS['commit']}")
    if os.path.exists(os.path.join(REPO_DIR, "requirements-dev.txt")):
        pip = run_command(f"{sys.executable} -m pip install -q -r requirements-dev.txt", REPO_DIR, timeout=900)
        RESULTS["pip"] = "installed" if pip["exit_code"] == 0 else pip["output"][-1500:]
        print("pip install -r requirements-dev.txt:", RESULTS["pip"][-500:])
    RESULTS["tests_before"] = run_tests(REPO_DIR)
    print(f"Tests before (exit {RESULTS['tests_before']['exit_code']}):\\n{RESULTS['tests_before']['last_lines']}")
except Exception as e:
    failed("get the repository", e)
    raise
save_results()
"""

AGENT_CELL = """
try:
    if not TASK.strip():
        raise RuntimeError("TASK is empty: write the task in cell 1")
    model = OllamaChat(OLLAMA_URL, MODEL, {**OPTIONS, "num_ctx": RESULTS["num_ctx"], "num_predict": NUM_PREDICT})
    LOG = run(TASK, model, REPO_DIR, max_steps=MAX_STEPS, minutes=AGENT_MINUTES, num_ctx=RESULTS["num_ctx"],
              num_predict=NUM_PREDICT, log_path=os.path.join(WORKING_DIR, LOG_FILE))
    RESULTS.update(status=LOG["status"], summary=LOG["summary"], steps=len(LOG["steps"]), minutes=LOG["minutes"])
except Exception as e:
    RESULTS.update(status="error", summary=f"The agent stopped with an error: {e!r}"[:1000])
    failed("run the agent", e)
save_results()
"""

RESULT_CELL = """
try:  # the change first: the tests below must not add anything to it
    patch, RESULTS["files_changed"] = make_patch(REPO_DIR, RESULTS["commit"])
    with open(os.path.join(WORKING_DIR, PATCH_FILE), "w", encoding="utf-8") as fh:
        fh.write(patch)
    RESULTS["patch_bytes"] = len(patch.encode("utf-8"))
except Exception as e:
    failed("make the patch", e)
save_results()
try:
    RESULTS["tests_after"] = run_tests(REPO_DIR)
except Exception as e:
    failed("run the tests after", e)
save_results()
print(f"Status: {RESULTS['status']} after {RESULTS['steps']} steps, {RESULTS['minutes']} min "
      f"(num_ctx {RESULTS['num_ctx']})\\n\\n{RESULTS['summary']}\\n")
for name in ("tests_before", "tests_after"):
    t = RESULTS[name] or {}
    print(f"{name}: exit {t.get('exit_code')}\\n{t.get('last_lines', '')}\\n")
print(f"Files changed ({RESULTS['patch_bytes']} bytes of patch):", ", ".join(RESULTS["files_changed"]) or "none")
for error in RESULTS["errors"]:
    print("Error:", error)
print("Saved:", sorted(os.listdir(WORKING_DIR)))
"""

NEXT = """
## What next
Read `changes.patch` and `agent_log.json` (every step) in the Output tab. To use the change: `git apply changes.patch`
in your copy of the repository, or let the GitHub workflow **Coding agent (free model)** open a pull request with it.
"""


def notebook() -> dict:
    """The agent notebook."""
    helpers = (
        "#@title 2. Helpers (no need to read; built from coder/agent.py and coder/fixeval.py)\n"
        + (HERE / "agent.py").read_text(encoding="utf-8")
        + "\n\n# ---- Ollama: from coder/fixeval.py ----\n\n"
        + "\n\n".join(inspect.getsource(f) for f in OLLAMA_HELPERS)
        + f"\n\nRESULTS_FILE, LOG_FILE, PATCH_FILE = {RESULTS_FILE!r}, {LOG_FILE!r}, {PATCH_FILE!r}\n"
        + STATE
        + run_helpers(RESULTS_FILE, 3)
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": [
            md(INTRO),
            code(titled(f"{SETTINGS_TITLE}: the task, the repository, the model and the limits", SETTINGS)),
            code(helpers, hidden=True, folded=True),
            code(titled(f"3. {OLLAMA_TITLE}", OLLAMA)),
            code(titled("4. Download qwen3-coder:30b (19 GB, a few minutes) and load it on the GPUs", MODEL_CELL)),
            code(titled("5. Get the repository at COMMIT, install its tools, run its tests (before)", REPO_CELL)),
            code(titled("6. The agent works on the task (up to AGENT_MINUTES minutes)", AGENT_CELL)),
            code(titled("7. Save the change (changes.patch) and run the tests again (after)", RESULT_CELL)),
            md(NEXT),
        ],
    }


def main() -> None:
    out = HERE / NOTEBOOK
    out.write_text(text(notebook()), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

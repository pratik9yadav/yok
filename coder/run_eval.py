"""Measure free coding models at fixing bugs on Kaggle's free GPU (coder_eval.ipynb), wait, print the table. No clicks.

Starts coder/coder_eval.ipynb on Kaggle (2× T4, internet on), waits (about 3½ hours for 4 models; each model
stops at MINUTES_PER_MODEL) and prints, per model: tasks fixed of HumanEvalFix (Python), seconds per task, and
whether Ollama ran it on the GPU.

    python coder/run_eval.py                                          the 4 models in the notebook, all 164 tasks
    python coder/run_eval.py --models qwen2.5-coder:7b --limit 10     a quick check
    python coder/run_eval.py --wait                                   wait for the run already going, then print it

Needs what run_on_kaggle.py needs (KAGGLE_API_TOKEN, pip install kaggle).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import run_on_kaggle
from coder import build
from coder.fixeval import table
from run_on_kaggle import check_gpu, connect, print_log, wait

SLUG = "wimb-coder-eval"
NOTEBOOK = HERE / build.NOTEBOOK
OUT = HERE.parent / "kaggle_output_coder"
WAIT_MINUTES = 300
MODELS_LINE = re.compile(r"^MODELS = \[.*\]$", re.MULTILINE)
LIMIT_LINE = re.compile(r"^LIMIT = .*$", re.MULTILINE)
MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(:[A-Za-z0-9._-]+)?$")  # e.g. qwen2.5-coder:7b
MAX_MODELS = 8


def models_from(text: str) -> list[str]:
    """--models "a,b" as a list of Ollama model names (ValueError if a name is not one)."""
    models = [m.strip() for m in text.split(",") if m.strip()]
    if not models:
        raise ValueError("--models needs at least one model name, e.g. qwen2.5-coder:7b")
    if len(models) > MAX_MODELS:
        raise ValueError(f"at most {MAX_MODELS} models in one run (got {len(models)})")
    for m in models:
        if not MODEL_NAME.match(m) or len(m) > 100:
            raise ValueError(f"not an Ollama model name: {m!r} (like qwen2.5-coder:7b)")
    return models


def settings_cell(nb: dict) -> dict:
    for cell in nb["cells"]:
        if "".join(cell["source"]).startswith(f"#@title {build.SETTINGS_TITLE}"):
            return cell
    raise RuntimeError(f"{build.NOTEBOOK}: the settings cell is missing; run python coder/build.py")


def notebook_with(models: list[str] | None = None, limit: int | None = None) -> dict:
    """coder_eval.ipynb with these models (None: the notebook's own) and this many tasks (None: all)."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = settings_cell(nb)
    source = "".join(cell["source"])
    if models is not None:
        source, n = MODELS_LINE.subn(lambda _: f"MODELS = {json.dumps(models)}", source)
        if n != 1:
            raise RuntimeError(f"{build.NOTEBOOK}: the MODELS line is missing; run python coder/build.py")
    source, n = LIMIT_LINE.subn(lambda _: f"LIMIT = {limit!r}", source)
    if n != 1:
        raise RuntimeError(f"{build.NOTEBOOK}: the LIMIT line is missing; run python coder/build.py")
    cell["source"] = source.splitlines(keepends=True)
    return nb


def settings(nb: dict) -> dict:
    """The settings cell's values (MODELS, LIMIT, MINUTES_PER_MODEL, ...)."""
    scope: dict = {}
    exec("".join(settings_cell(nb)["source"]), scope)  # noqa: S102 - our own generated cell
    return scope


def hours_needed(nb: dict) -> float:
    """GPU hours a run can take: every model at its time limit, plus a few minutes each to start and to install."""
    s = settings(nb)
    return round((len(s["MODELS"]) * (s["MINUTES_PER_MODEL"] + 5) + 15) / 60, 1)


def push(api, kernel: str, nb: dict) -> None:
    s = settings(nb)
    print(f"Models: {', '.join(s['MODELS'])}; tasks: {s['LIMIT'] or 'all'}; "
          f"at most {s['MINUTES_PER_MODEL']} minutes per model", flush=True)
    run_on_kaggle.push_notebook(api, kernel, nb, build.NOTEBOOK)


def scored(r: dict | None) -> list[dict]:
    """The models that answered at least one task."""
    return [m for m in (r or {}).get("models") or [] if m.get("done")]


def report(r: dict) -> None:
    """Print coder_eval.json for a person."""
    print(f"\n{r.get('benchmark', 'HumanEvalFix')}: {r.get('tasks_used') or '?'} of {r.get('tasks_total') or '?'} tasks "
          f"from {r.get('dataset', '?')} (licence {r.get('licence', '?')})")
    gpus = (r.get("gpus") or r.get("gpu") or "none found").replace("\n", "; ")
    print(f"Ollama {r.get('ollama_version') or '?'}; GPUs: {gpus}")
    print(f"Setting: {r.get('setting', '?')}; options {r.get('options')}\n")
    for line in table(r):
        print(line)
    for m in r.get("models") or []:
        details = [f"download {m['size_gb']} GB in {m['pull_seconds']} s" if m.get("pulled") else None,
                   f"loading {m['load_seconds']} s" if m.get("load_seconds") is not None else None,
                   f"GPU memory {m['gpu_memory']}" if m.get("gpu_memory") else None,
                   f"{m['no_code_block']} replies without a code block" if m.get("no_code_block") else None,
                   f"{m['minutes']} min" if m.get("minutes") is not None else None]
        print(f"\n{m.get('model')}: " + ("; ".join(d for d in details if d) or "not downloaded"))
        for error in (m.get("errors") or [])[:10]:
            print("   error:", error[:300])
        if len(m.get("errors") or []) > 10:
            print(f"   ... and {len(m['errors']) - 10} more errors")
    if r.get("errors"):
        print("\nErrors:")
        for error in r["errors"]:
            print("   ", error[:500])


def fetch(api, kernel: str, ok: bool = True) -> dict | None:
    """Download coder_eval.json (and Kaggle's log) and print it. None when the run left none."""
    OUT.mkdir(exist_ok=True)
    path = OUT / build.RESULTS_FILE
    path.unlink(missing_ok=True)  # never report an older run's results
    api.kernels_output(kernel, str(OUT), file_pattern=r"(coder_eval\.json|.*\.log|.*\.ipynb)$", force=True)
    r = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if r is not None:
        report(r)
    if r is None or not ok:
        print(f"\nNo {build.RESULTS_FILE}: the run stopped early." if r is None else "\nThe run did not finish.",
              "Kaggle's log:")
        for log in OUT.glob("*.log"):
            print_log(log)
    print(f"\nDownloaded to {OUT}: {sorted(p.name for p in OUT.iterdir())}")
    print(f"The run on Kaggle: https://www.kaggle.com/code/{kernel}")
    return r


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--models", metavar="A,B", help="Ollama model names, comma-separated (default: the notebook's 4)")
    args.add_argument("--limit", type=int, metavar="N", help="use only the first N tasks (default: all 164)")
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    opts = args.parse_args()
    try:
        models = models_from(opts.models) if opts.models is not None else None
    except ValueError as e:
        args.error(str(e))
    if opts.limit is not None and opts.limit < 1:
        args.error("--limit must be 1 or more")
    nb = notebook_with(models, opts.limit)
    api, kernel = connect(SLUG)
    if not opts.wait:
        check_gpu(api, need_hours=hours_needed(nb))
        push(api, kernel, nb)
        time.sleep(30)
    ok = wait(api, kernel, limit_min=WAIT_MINUTES)
    r = fetch(api, kernel, ok)
    if not scored(r):
        print("\nNo model produced a score.")
    sys.exit(0 if scored(r) else 1)


if __name__ == "__main__":
    main()

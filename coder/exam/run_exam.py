"""The engineer exam on Kaggle's free GPU (coder/exam/exam.ipynb): start it, wait, print the report. No clicks.

Starts exam.ipynb on Kaggle (2× T4, internet on), waits (each model stops at MINUTES_PER_MODEL, 60: about 4½ hours
for 4 models) and prints, per model: the total (0-100), tasks solved, each part (hidden tests, own tests, lint,
security, explanation), the scores by kind of task, and the 5 weakest tasks.

    python coder/exam/run_exam.py                                       the notebook's 4 models, every task
    python coder/exam/run_exam.py --models qwen3-coder:30b --limit 3    a quick check
    python coder/exam/run_exam.py --wait                                wait for the run already going, then print it

If the run is still going when the wait ends (--wait-minutes, default 330), it says so and exits 0: run again with
--wait. Exit 1 when no model got a score. Needs what run_on_kaggle.py needs (KAGGLE_API_TOKEN, pip install kaggle).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

import run_on_kaggle
from coder.exam import build_exam
from coder.exam.exam import table, weakest
from coder.run_eval import models_from
from coder.run_train import still_running
from run_on_kaggle import check_gpu, connect, print_log, wait

SLUG = "wimb-coder-exam"
NOTEBOOK = HERE / build_exam.NOTEBOOK
OUT = HERE.parent.parent / "kaggle_output_exam"
WAIT_MINUTES = 330  # 5½ hours: a GitHub job may last 6
MODELS_LINE = re.compile(r"^MODELS = \[.*\]$", re.MULTILINE)
LIMIT_LINE = re.compile(r"^LIMIT = .*$", re.MULTILINE)
STILL_RUNNING = ("The Kaggle run is still going: still running; run again with action fetch-exam "
                 "(or python coder/exam/run_exam.py --wait).")


def settings_cell(nb: dict) -> dict:
    for cell in nb["cells"]:
        if "".join(cell["source"]).startswith(f"#@title {build_exam.SETTINGS_TITLE}"):
            return cell
    raise RuntimeError(f"{build_exam.NOTEBOOK}: the settings cell is missing; run python coder/build.py")


def notebook_with(models: list[str] | None = None, limit: int | None = None) -> dict:
    """exam.ipynb with these models (None: the notebook's own) and this many tasks (None: all)."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = settings_cell(nb)
    source = "".join(cell["source"])
    if models is not None:
        source, n = MODELS_LINE.subn(lambda _: f"MODELS = {json.dumps(models)}", source)
        if n != 1:
            raise RuntimeError(f"{build_exam.NOTEBOOK}: the MODELS line is missing; run python coder/build.py")
    source, n = LIMIT_LINE.subn(lambda _: f"LIMIT = {limit!r}", source)
    if n != 1:
        raise RuntimeError(f"{build_exam.NOTEBOOK}: the LIMIT line is missing; run python coder/build.py")
    cell["source"] = source.splitlines(keepends=True)
    return nb


def settings(nb: dict) -> dict:
    """The settings cell's values (MODELS, LIMIT, MINUTES_PER_MODEL, ...)."""
    scope: dict = {}
    exec("".join(settings_cell(nb)["source"]), scope)  # noqa: S102 - our own generated cell
    return scope


def task_count(nb: dict) -> int:
    """How many tasks are in the notebook."""
    return sum("".join(c["source"]).count("\n    json.loads(") for c in nb["cells"]
               if "".join(c["source"]).startswith("#@title 5."))


def hours_needed(nb: dict) -> float:
    """GPU hours a run can take: every model at its time limit, plus a few minutes each to start, and to install."""
    s = settings(nb)
    return round((len(s["MODELS"]) * (s["MINUTES_PER_MODEL"] + 5) + 20) / 60, 1)


def push(api, kernel: str, nb: dict) -> None:
    s = settings(nb)
    print(f"Models: {', '.join(s['MODELS'])}; tasks: {s['LIMIT'] or 'all'} of {task_count(nb)}; "
          f"at most {s['MINUTES_PER_MODEL']} minutes per model", flush=True)
    run_on_kaggle.push_notebook(api, kernel, nb, build_exam.NOTEBOOK)


def scored(r: dict | None) -> list[dict]:
    """The models that did at least one task."""
    return [m for m in (r or {}).get("models") or [] if m.get("done")]


def report(r: dict) -> None:
    """Print exam.json for a person."""
    kinds = ", ".join(f"{k} {n}" for k, n in (r.get("kinds") or {}).items())
    print(f"\n{r.get('exam', 'engineer exam')}: {r.get('tasks_used') or '?'} of {r.get('tasks_total') or '?'} tasks"
          + (f" ({kinds})" if kinds else ""))
    gpus = (r.get("gpus") or r.get("gpu") or "none found").replace("\n", "; ")
    print(f"Ollama {r.get('ollama_version') or '?'}; GPUs: {gpus}")
    print(f"Checkers: {r.get('checkers')}")
    print(f"Setting: {r.get('setting', '?')}; options {r.get('options')}; think {r.get('think')}")
    if r.get("tasks_left_out"):
        print(f"Left out (their reference failed on Kaggle): {', '.join(r['tasks_left_out'])}")
    print()
    for line in table(r):
        print(line)
    for m in r.get("models") or []:
        details = [f"download {m['size_gb']} GB in {m['pull_seconds']} s" if m.get("pulled") else None,
                   f"loading {m['load_seconds']} s" if m.get("load_seconds") is not None else None,
                   f"runs on {m['processor']}" if m.get("processor") else None,
                   f"{m['tokens_per_s']} tokens/s" if m.get("tokens_per_s") else None,
                   f"{m['minutes']} min" if m.get("minutes") is not None else None]
        print(f"\n{m.get('model')}: " + ("; ".join(d for d in details if d) or "not downloaded"))
        if m.get("tasks"):
            print("  the 5 weakest tasks:")
            for line in weakest(m):
                print(line)
        for error in (m.get("errors") or [])[:10]:
            print("   error:", error[:300])
        if len(m.get("errors") or []) > 10:
            print(f"   ... and {len(m['errors']) - 10} more errors")
    if r.get("errors"):
        print("\nErrors:")
        for error in r["errors"]:
            print("   ", error[:500])


def fetch(api, kernel: str, ok: bool = True) -> dict | None:
    """Download exam.json (and Kaggle's log and the run notebook) and print it. None when the run left none."""
    OUT.mkdir(exist_ok=True)
    path = OUT / build_exam.RESULTS_FILE
    path.unlink(missing_ok=True)  # never report an older run's results
    api.kernels_output(kernel, str(OUT), file_pattern=r"(exam\.json|.*\.log|.*\.ipynb)$", force=True)
    r = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if r is not None:
        report(r)
    if r is None or not ok:
        print(f"\nNo {build_exam.RESULTS_FILE}: the run stopped early." if r is None else "\nThe run did not finish.",
              "Kaggle's log:")
        for log in OUT.glob("*.log"):
            print_log(log)
    print(f"\nDownloaded to {OUT}: {sorted(p.name for p in OUT.iterdir())}")
    print(f"The run on Kaggle: https://www.kaggle.com/code/{kernel}")
    return r


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--models", metavar="A,B", help="Ollama model names, comma-separated (default: the notebook's 4)")
    args.add_argument("--limit", type=int, metavar="N", help="use only the first N tasks (default: all)")
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    args.add_argument("--wait-minutes", type=int, metavar="M",
                      help=f"stop waiting after M minutes (default {WAIT_MINUTES}; the run goes on)")
    opts = args.parse_args()
    try:
        models = models_from(opts.models) if opts.models is not None else None
    except ValueError as e:
        args.error(str(e))
    if opts.limit is not None and opts.limit < 1:
        args.error("--limit must be 1 or more")
    if opts.wait_minutes is not None and opts.wait_minutes < 1:
        args.error("--wait-minutes must be 1 or more")
    nb = notebook_with(models, opts.limit)
    if not opts.wait and not task_count(nb):
        sys.exit(f"{build_exam.NOTEBOOK} has no tasks: add them to coder/exam/tasks, then run python coder/build.py")
    api, kernel = connect(SLUG)
    if not opts.wait:
        check_gpu(api, need_hours=hours_needed(nb))
        push(api, kernel, nb)
        time.sleep(30)
    ok = wait(api, kernel, limit_min=opts.wait_minutes or WAIT_MINUTES)
    if not ok and still_running(api, kernel):
        print("\n" + STILL_RUNNING)
        print(f"The run on Kaggle: https://www.kaggle.com/code/{kernel}")
        sys.exit(0)
    r = fetch(api, kernel, ok)
    if not scored(r):
        print("\nNo model got a score.")
    sys.exit(0 if scored(r) else 1)


if __name__ == "__main__":
    main()

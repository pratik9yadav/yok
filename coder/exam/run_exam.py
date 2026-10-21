"""The engineer exam on Kaggle's free GPU (coder/exam/exam.ipynb): start it, wait, print the report. No clicks.

Starts exam.ipynb on Kaggle (2× T4, internet on), waits (each model stops at MINUTES_PER_MODEL, 60: about 4½ hours
for 4 models) and prints, per model: the total (0-100), tasks solved, each part (hidden tests, own tests, lint,
security, explanation), the scores by kind of task, and the 5 weakest tasks.

    python coder/exam/run_exam.py                                       the notebook's 4 models, every task
    python coder/exam/run_exam.py --models qwen3-coder:30b --limit 3    a quick check
    python coder/exam/run_exam.py --wait                                wait for the run already going, then print it
    python coder/exam/run_exam.py --regrade                             as --wait, then grade every saved reply again
                                                                        with the current grader (no GPU, no model)

--regrade writes kaggle_output_exam/exam_regraded.json and prints the same report, each model's scores before and
after, and the start of every reply that still has no files (to see what the model did). It uses the tasks in
coder/exam/tasks and the checkers installed here (pip install bandit, so security is graded as on Kaggle).

If the run is still going when the wait ends (--wait-minutes, default 330), it says so and exits 0: run again with
--wait. Exit 1 when no model got a score. Needs what run_on_kaggle.py needs (KAGGLE_API_TOKEN, pip install kaggle).
"""

import argparse
import copy
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

import run_on_kaggle
from coder.exam import build_exam, grade
from coder.exam.exam import score_task, summarise, table, weakest
from coder.run_eval import models_from
from coder.run_train import still_running
from run_on_kaggle import check_gpu, connect, print_log, wait

SLUG = "wimb-coder-exam"
NOTEBOOK = HERE / build_exam.NOTEBOOK
OUT = HERE.parent.parent / "kaggle_output_exam"
REGRADED_FILE = "exam_regraded.json"
TASKS_DIR = build_exam.TASKS_DIR
REPLY_CHARS = 300  # how much of a reply with no files the regrade report shows
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


def fetch(api, kernel: str, ok: bool = True, show: bool = True) -> dict | None:
    """Download exam.json (and Kaggle's log and the run notebook) and print it (show). None when the run left
    none."""
    OUT.mkdir(exist_ok=True)
    path = OUT / build_exam.RESULTS_FILE
    path.unlink(missing_ok=True)  # never report an older run's results
    api.kernels_output(kernel, str(OUT), file_pattern=r"(exam\.json|.*\.log|.*\.ipynb)$", force=True)
    r = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if r is not None and show:
        report(r)
    if r is None or not ok:
        print(f"\nNo {build_exam.RESULTS_FILE}: the run stopped early." if r is None else "\nThe run did not finish.",
              "Kaggle's log:")
        for log in OUT.glob("*.log"):
            print_log(log)
    print(f"\nDownloaded to {OUT}: {sorted(p.name for p in OUT.iterdir())}")
    print(f"The run on Kaggle: https://www.kaggle.com/code/{kernel}")
    return r


def regrade(results: dict, tasks: list[dict], workdir=None, workers: int = 4) -> dict:
    """A copy of exam.json with every saved reply graded again by the current grader (no model is asked again), and
    each model's totals worked out again. "regraded" says when, with which checkers, and the scores before."""
    r = copy.deepcopy(results)
    by_id = {t["id"]: t for t in tasks}
    models = r.get("models") or []
    before = {m.get("model"): {k: m.get(k) for k in ("score", "score_done", "solved", "done", "no_files", "hidden")}
              for m in models}
    jobs = [(record, attempt) for m in models for record in m.get("tasks") or [] for attempt in record["attempts"]
            if "reply" in attempt]

    def one(job: tuple[dict, dict]) -> None:
        record, attempt = job
        attempt.pop("grade", None)
        if attempt.get("error", "").startswith(("grading failed", "regrade:")):
            del attempt["error"]
        task = by_id.get(record["task"])
        if task is None:
            attempt["error"] = f"regrade: no task {record['task']} in {TASKS_DIR.name}/"
            return
        try:
            attempt["grade"] = grade.grade_reply(task, attempt["reply"], workdir)
        except Exception as e:  # noqa: BLE001 - count it as 0 and go on, as on Kaggle
            attempt["error"] = f"grading failed: {e!r}"[:300]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, jobs))
    for m in models:
        if not m.get("tasks"):
            continue
        for record in m["tasks"]:
            score_task(record)
        m["errors"] = [e for e in m.get("errors") or [] if "grading failed" not in e and "regrade:" not in e] + [
            f"{t['task']}: {a['error']}" for t in m["tasks"] for a in t["attempts"]
            if a.get("error", "").startswith(("grading failed", "regrade:"))]
        summarise(m)
    r["regraded"] = {"at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "checkers": grade.checker_versions(),
                     "tasks_dir": str(TASKS_DIR), "reply_format_asked": results.get("reply_format"), "before": before}
    return r


def no_file_replies(r: dict, chars: int = REPLY_CHARS) -> list[str]:
    """Every graded answer that still has no files: the model, the task, why, and the start of its reply."""
    lines = []
    for m in r.get("models") or []:
        for record in m.get("tasks") or []:
            for attempt in record["attempts"]:
                g = attempt.get("grade")
                if g is None or g.get("files"):
                    continue
                notes = [n for n in ("cut off" if attempt.get("done_reason") == "length" else None, g.get("fallback"))
                         if n]
                lines.append(f"   {m.get('model')} {record['task']}" + (f" ({'; '.join(notes)})" if notes else "")
                             + ": " + json.dumps(attempt.get("reply", "")[:chars], ensure_ascii=False))
    return lines


def report_regrade(r: dict) -> None:
    """Print a regraded exam.json: the report, each model before → after, and the replies that still have no files."""
    print(f"\nRegraded with the current grader on {r['regraded']['at']} (no model asked again; tasks from "
          f"{TASKS_DIR.name}/; checkers here: {r['regraded']['checkers']})")
    report(r)
    print("\nBefore → after the regrade:")
    for m in r.get("models") or []:
        b = r["regraded"]["before"].get(m.get("model")) or {}
        if not m.get("done"):
            continue
        old = b.get("score") if b.get("score") is not None else b.get("score_done")
        new = m.get("score") if m.get("score") is not None else m.get("score_done")
        print(f"   {m['model']}: total {old} → {new}; solved {b.get('solved')} → {m.get('solved')} of {m['done']}; "
              f"no files {b.get('no_files')} → {m.get('no_files')}; read by the fallback {m.get('fallback')}")
    lines = no_file_replies(r)
    print(f"\nStill no files ({len(lines)}), the first {REPLY_CHARS} characters of each reply:" if lines
          else "\nEvery answered task has files now.")
    for line in lines:
        print(line)


def run_regrade(r: dict) -> dict:
    """Regrade exam.json with the tasks in TASKS_DIR, save exam_regraded.json and print it."""
    regraded = regrade(r, grade.load_tasks(TASKS_DIR))
    OUT.mkdir(exist_ok=True)
    (OUT / REGRADED_FILE).write_text(json.dumps(regraded, indent=1, ensure_ascii=False), encoding="utf-8")
    report_regrade(regraded)
    print(f"\nSaved: {OUT / REGRADED_FILE}")
    return regraded


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--models", metavar="A,B", help="Ollama model names, comma-separated (default: the notebook's 4)")
    args.add_argument("--limit", type=int, metavar="N", help="use only the first N tasks (default: all)")
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    args.add_argument("--regrade", action="store_true",
                      help="as --wait, then grade every saved reply again with the current grader (no GPU)")
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
    opts.wait = opts.wait or opts.regrade
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
    r = fetch(api, kernel, ok, show=not opts.regrade)
    if opts.regrade and r is not None:
        r = run_regrade(r)
    if not scored(r):
        print("\nNo model got a score.")
    sys.exit(0 if scored(r) else 1)


if __name__ == "__main__":
    main()

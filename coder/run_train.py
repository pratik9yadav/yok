"""Phase 2 on Kaggle's free T4 (coder_train.ipynb): train Qwen2.5-Coder-14B on open bug-fix commits, score it on
HumanEvalFix before and after, and print before → after. No clicks.

    python coder/run_train.py                    a full run (about 5–8 hours; waits up to 11½ hours)
    python coder/run_train.py --smoke            a 20–30 minute check of the whole pipeline (40 examples, 8 tasks)
    python coder/run_train.py --max-train 3000   train on up to 3,000 commits (default: the notebook's MAX_TRAIN)
    python coder/run_train.py --wait             don't start a run: wait for the one going, then print it

If the run is still going when the wait ends (--wait-minutes), it says so and exits 0: run again with --wait.
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
from coder import build_train
from coder.fixtrain import counts_lines, report_lines
from run_on_kaggle import DONE, check_gpu, connect, print_log, wait

SLUG = "wimb-coder-train"
NOTEBOOK = HERE / build_train.NOTEBOOK
OUT = HERE.parent / "kaggle_output_coder_train"
NEED_HOURS = {True: 0.5, False: 7}  # Kaggle GPU hours a smoke / full run needs (stop before starting otherwise)
WAIT_MINUTES = {True: 90, False: 690}  # a smoke run, a full one (Kaggle stops any run at 12 hours)
LINES = {name: re.compile(rf"^{name} = .*$", re.MULTILINE) for name in ("MAX_TRAIN", "EPOCHS", "SMOKE")}
STILL_RUNNING = ("The Kaggle run is still going (a full run takes about 5–8 hours): still running; run again with "
                 "action fetch-train (or python coder/run_train.py --wait).")


def settings_cell(nb: dict) -> dict:
    for cell in nb["cells"]:
        if "".join(cell["source"]).startswith(f"#@title {build_train.SETTINGS_TITLE}"):
            return cell
    raise RuntimeError(f"{build_train.NOTEBOOK}: the settings cell is missing; run python coder/build.py")


def notebook_with(smoke: bool = False, max_train: int | None = None, epochs: int | None = None) -> dict:
    """coder_train.ipynb with these settings (None: the notebook's own)."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = settings_cell(nb)
    source = "".join(cell["source"])
    for name, value in (("SMOKE", smoke), ("MAX_TRAIN", max_train), ("EPOCHS", epochs)):
        if value is None:
            continue
        source, n = LINES[name].subn(lambda _, name=name, value=value: f"{name} = {value!r}", source)
        if n != 1:
            raise RuntimeError(f"{build_train.NOTEBOOK}: the {name} line is missing; run python coder/build.py")
    cell["source"] = source.splitlines(keepends=True)
    return nb


def settings(nb: dict) -> dict:
    """The settings cell's values (BASE, MAX_TRAIN, SMOKE, ...), as the notebook sees them."""
    scope: dict = {}
    exec("".join(settings_cell(nb)["source"]), scope)  # noqa: S102 - our own generated cell
    return scope


def push(api, kernel: str, nb: dict) -> None:
    s = settings(nb)
    print(f"Base: {s['BASE']}; up to {s['MAX_TRAIN']} training examples, {s['EPOCHS']} epoch(s)"
          + (f", {s['MAX_STEPS']} steps" if s["MAX_STEPS"] else "")
          + f"; tasks: {s['EVAL_LIMIT'] or 'all'}" + ("; SMOKE check" if s["SMOKE"] else ""), flush=True)
    run_on_kaggle.push_notebook(api, kernel, nb, build_train.NOTEBOOK)


def scored_after(r: dict | None) -> bool:
    return bool(((r or {}).get("after") or {}).get("done"))


def report(r: dict) -> None:
    """Print coder_train.json for a person."""
    print(f"\n{r.get('base', '?')}" + (" (SMOKE check: tiny data, few tasks)" if r.get("smoke") else ""))
    print(f"Versions: {r.get('versions')}; GPU: {r.get('gpu') or '?'}")
    lic = r.get("licence") or {}
    print(f"Training data: {r.get('train_dataset', '?')} from {(r.get('data') or {}).get('source') or '?'}; "
          f"licence {lic.get('licence') or 'not checked'}")
    print("\n".join("   " + line for line in counts_lines((r.get("data") or {}).get("counts"))))
    for example in ((r.get("data") or {}).get("counts") or {}).get("contaminated_examples", [])[:5]:
        print(f"   decontamination dropped: {example.get('subject')!r} ({example.get('task_id')}: {example.get('why')})")
    t = r.get("train") or {}
    print(f"\nTraining: {t.get('examples') or '?'} examples, {t.get('steps') or 0} of {t.get('planned_steps') or '?'} "
          f"steps, {t.get('minutes') or '?'} min, loss {t.get('loss') or t.get('last_loss') or '?'}"
          + ("; stopped at the time limit" if t.get("stopped_by_time") else "")
          + (f"; all steps would take about {t['projected_minutes']} min" if t.get("projected_minutes") else ""))
    print(f"Setting: {r.get('eval_setting', '?')}\n")
    for line in report_lines(r):
        print(line)
    for name in ("before", "after"):
        for error in ((r.get(name) or {}).get("errors") or [])[:5]:
            print(f"   {name} error: {error[:300]}")
    print(f"\nLoRA: {r.get('lora_dir') or 'not saved'}. {r.get('export') or ''}")
    if r.get("errors"):
        print("\nErrors:")
        for error in r["errors"]:
            print("   ", error[:500])


def fetch(api, kernel: str, ok: bool = True) -> dict | None:
    """Download coder_train.json (and Kaggle's log and the run notebook) and print it. None when the run left none."""
    OUT.mkdir(exist_ok=True)
    path = OUT / build_train.RESULTS_FILE
    path.unlink(missing_ok=True)  # never report an older run's results
    api.kernels_output(kernel, str(OUT), file_pattern=r"(coder_train\.json|.*\.log|.*\.ipynb)$", force=True)
    r = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if r is not None:
        report(r)
    if r is None or not ok:
        print(f"\nNo {build_train.RESULTS_FILE}: the run stopped early." if r is None else "\nThe run did not finish.",
              "Kaggle's log:")
        for log in OUT.glob("*.log"):
            print_log(log)
    print(f"\nDownloaded to {OUT}: {sorted(p.name for p in OUT.iterdir())}")
    print(f"The run on Kaggle (the LoRA is in its Output tab): https://www.kaggle.com/code/{kernel}")
    return r


def still_running(api, kernel: str) -> bool:
    try:
        status = api.kernels_status(kernel).status
    except Exception as e:  # noqa: BLE001 - then report what is there
        print("Could not read the run's status:", e, flush=True)
        return False
    return getattr(status, "name", str(status)).upper() not in DONE


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--smoke", action="store_true", help="a quick check: 40 examples, 8 steps, 8 tasks")
    args.add_argument("--max-train", type=int, metavar="N", help="train on up to N commits (default: the notebook's)")
    args.add_argument("--epochs", type=int, metavar="N", help="passes over the training examples (default 1)")
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    args.add_argument("--wait-minutes", type=int, metavar="M", help="stop waiting after M minutes (the run goes on)")
    opts = args.parse_args()
    if opts.max_train is not None and not 1 <= opts.max_train <= 100_000:
        args.error("--max-train must be 1 to 100000")
    if opts.epochs is not None and not 1 <= opts.epochs <= 5:
        args.error("--epochs must be 1 to 5")
    if opts.wait_minutes is not None and opts.wait_minutes < 1:
        args.error("--wait-minutes must be 1 or more")
    nb = notebook_with(opts.smoke, opts.max_train, opts.epochs)
    api, kernel = connect(SLUG)
    if not opts.wait:
        check_gpu(api, need_hours=NEED_HOURS[opts.smoke])
        push(api, kernel, nb)
        time.sleep(30)
    ok = wait(api, kernel, limit_min=opts.wait_minutes or WAIT_MINUTES[opts.smoke])
    if not ok and still_running(api, kernel):
        print("\n" + STILL_RUNNING)
        print(f"The run on Kaggle: https://www.kaggle.com/code/{kernel}")
        sys.exit(0)
    r = fetch(api, kernel, ok)
    if not scored_after(r):
        print("\nNo AFTER score.")
    sys.exit(0 if scored_after(r) else 1)


if __name__ == "__main__":
    main()

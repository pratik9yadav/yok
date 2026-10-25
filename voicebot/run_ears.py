"""Teach the ears Indian speech on Kaggle's free GPU (voicebot/ears_training.ipynb) and print before → after.

    python voicebot/run_ears.py            train (about 1–2 hours), then print the word error rates before and after
    python voicebot/run_ears.py --wait     wait for the training already running, then print it

The trained model (wimb-ears/) stays in the Kaggle run's Output tab; the voice bot's eval and live page attach it
from then on (voicebot/run_voicebot.py). Needs KAGGLE_API_TOKEN (and pip install kaggle).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
if sys.path and Path(sys.path[0]).resolve() == HERE.resolve():  # run as a script: don't shadow standard modules
    sys.path.pop(0)
sys.path.insert(0, str(HERE.parent))

import run_on_kaggle
from run_on_kaggle import check_gpu, connect, print_log, wait
from voicebot import build_ears

SLUG = "wimb-ears-training"
OUT = HERE.parent / "kaggle_output_ears"
WAIT_MINUTES = 300


def fetch(api, kernel: str, ok: bool) -> dict | None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    api.kernels_output(kernel, str(OUT), file_pattern=r"(ears_results\.json|\.log)$", force=True)
    found = sorted(OUT.rglob(build_ears.RESULTS_FILE))
    r = json.loads(found[0].read_text(encoding="utf-8")) if found else None
    if r is not None:
        print()
        for line in r.get("summary") or []:
            print(line)
        print("\nData:", json.dumps(r.get("data"), ensure_ascii=False, indent=1))
        for key, v in (r.get("after") or {}).items():
            for ex in v.get("examples", [])[:2]:
                print(f"   {key}: said \"{ex['said']}\" → heard \"{ex['heard']}\"")
        if r.get("errors"):
            print("\nErrors:")
            for e in r["errors"]:
                print("   ", str(e)[:400])
    if r is None or not ok:
        print("\nThe run stopped early." if r is None else "\nThe run did not finish.", "Kaggle's log:")
        for log in OUT.glob("*.log"):
            print_log(log)
    return r


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    opts = args.parse_args()
    api, kernel = connect(SLUG)
    if not opts.wait:
        check_gpu(api, need_hours=2)
        nb = json.loads((HERE / build_ears.NOTEBOOK).read_text(encoding="utf-8"))
        run_on_kaggle.push_notebook(api, kernel, nb, build_ears.NOTEBOOK)
        time.sleep(30)
    ok = wait(api, kernel, limit_min=WAIT_MINUTES)
    r = fetch(api, kernel, ok)
    sys.exit(0 if ok and r is not None and r.get("after") else 1)


if __name__ == "__main__":
    main()

"""Teach the ears Indian speech on Kaggle's free GPU (voicebot/ears_training.ipynb) and print before → after.

    python voicebot/run_ears.py            train (about 1–2 hours), then print the word error rates before and after
    python voicebot/run_ears.py --wait     wait for the training already running, then print it

Each training run is its own Kaggle notebook (SLUG), with the ears the voice bot uses now attached (the best so far,
run_voicebot.EARS_SLUG), so a worse run never replaces them. Its model (wimb-ears/) is in that run's Output tab
only when it beats the best so far; the lab then sets run_voicebot.EARS_SLUG to SLUG. Needs KAGGLE_API_TOKEN.
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
from voicebot.run_voicebot import EARS_SLUG
from voicebot.speechdata import CV_DATASET

SLUG = "wimb-ears-training-3"  # the next ears run (run 2, wimb-ears-training-2, is the best so far)
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
        best = [] if EARS_SLUG == SLUG else [f"{kernel.split('/', 1)[0]}/{EARS_SLUG}"]
        run_on_kaggle.push_notebook(api, kernel, nb, build_ears.NOTEBOOK, kernel_sources=best,
                                    dataset_sources=[CV_DATASET])
        time.sleep(30)
    ok = wait(api, kernel, limit_min=WAIT_MINUTES)
    r = fetch(api, kernel, ok)
    sys.exit(0 if ok and r is not None and r.get("after") else 1)


if __name__ == "__main__":
    main()

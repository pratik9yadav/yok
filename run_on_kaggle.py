"""Run train_colab.ipynb on Kaggle's free GPU, wait for it, and download the results. No clicks.

Needs, once:
  1. A Kaggle account with a verified phone number (Kaggle allows GPU and internet only then).
  2. KAGGLE_API_TOKEN set (Kaggle → Settings → API → Generate New Token).
  3. pip install kaggle

    python run_on_kaggle.py           start a run, wait (about 40 min), print the results
    python run_on_kaggle.py --wait    wait for the run already going, then print the results
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
SLUG = "wimb-shop-model-training"
OUT = HERE / "kaggle_output"
DONE = {"COMPLETE": True, "ERROR": False, "CANCEL_REQUESTED": False, "CANCEL_ACKNOWLEDGED": False}


def connect():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    if not user:
        sys.exit("Kaggle login failed: set KAGGLE_API_TOKEN (Kaggle → Settings → API → Generate New Token).")
    return api, f"{user}/{SLUG}"


def check_gpu(api) -> None:
    """Stop before starting a run that would get no GPU (Kaggle gives none until the phone is verified)."""
    try:
        quota = api.quota_view().gpu_quota
        allowed, used, reserved = quota.total_time_allowed, quota.time_used, quota.time_reserved
    except Exception as e:  # noqa: BLE001 - only a hint; never block a run on it
        print("Could not read the GPU quota:", e, flush=True)
        return
    hours = lambda t: (t.total_seconds() if t else 0) / 3600
    left = hours(allowed) - hours(used) - hours(reserved)
    print(f"Kaggle GPU time left this week: {left:.1f} h of {hours(allowed):.0f} h", flush=True)
    if hours(allowed) == 0:
        sys.exit("This Kaggle account gets no GPU. Verify your phone number at "
                 "https://www.kaggle.com/settings (Phone verification), then run again.")
    if left < 1:
        sys.exit("Less than 1 hour of Kaggle GPU time left this week. It resets weekly.")


def push(api, kernel: str) -> None:
    folder = Path(tempfile.mkdtemp())
    shutil.copy(HERE / "train_colab.ipynb", folder / "train_colab.ipynb")
    meta = {
        "id": kernel, "title": SLUG, "code_file": "train_colab.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    # T4: the older P100 is too old for the training library.
    resp = api.kernels_push(str(folder), acc="NvidiaTeslaT4")
    error = getattr(resp, "error", None)
    if error:
        sys.exit(f"Kaggle refused the run: {error}")
    print("Started:", getattr(resp, "url", None) or f"https://www.kaggle.com/code/{kernel}", flush=True)
    try:  # what Kaggle saved for this run: shows whether the GPU request arrived
        saved = Path(tempfile.mkdtemp())
        api.kernels_pull(kernel, str(saved), metadata=True)
        server = json.loads((saved / "kernel-metadata.json").read_text())
        print("Kaggle saved:", {k: server.get(k) for k in ("enable_gpu", "machine_shape", "enable_internet")},
              flush=True)
    except Exception as e:  # noqa: BLE001 - diagnostics only
        print("Could not read back the run settings:", e, flush=True)


def wait(api, kernel: str, limit_min: int = 120) -> bool:
    last = None
    for _ in range(limit_min * 2):
        resp = api.kernels_status(kernel)
        status = getattr(resp.status, "name", str(resp.status)).upper()
        if status != last:
            print(time.strftime("%H:%M"), status, flush=True)
            last = status
        if status in DONE:
            if getattr(resp, "failure_message", None):
                print("Failure message:", resp.failure_message, flush=True)
            return DONE[status]
        time.sleep(30)
    print(f"Still not finished after {limit_min} min.", flush=True)
    return False


def print_log(path: Path, lines: int = 120) -> None:
    """The end of a Kaggle run log (a JSON list of {stream_name, time, data})."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        text = "".join(entry.get("data", "") for entry in json.loads(text))
    except (ValueError, AttributeError):
        pass
    for line in text.splitlines()[-lines:]:
        print("   |", line)


def table(name: str, before: dict, after: dict, n) -> None:
    print(f"\n== {name}: {n} calls (before -> after)")
    for field, value in before.items():
        print(f"   {field:12s} {value:5.0%} -> {after[field]:5.0%}")


def list_output(api, kernel: str) -> None:
    """The files the run kept, with sizes (shows whether the .gguf model file was saved)."""
    try:
        files = api.kernels_list_files(kernel, page_size=100).files or []
        for f in files:
            size = int(getattr(f, "size", None) or 0)
            print(f"   {getattr(f, 'name', f)}" + (f"  {size / 1e6:,.0f} MB" if size else ""))
    except Exception as e:  # noqa: BLE001 - diagnostics only
        print("Could not list the output files:", e)


def fetch(api, kernel: str) -> None:
    OUT.mkdir(exist_ok=True)
    # Only the small files: results, the log and the executed notebook (not the 2.5 GB model).
    api.kernels_output(kernel, str(OUT), file_pattern=r"(results.*\.json|.*\.log|.*\.ipynb|.*Modelfile)$", force=True)
    results = OUT / "results.json"
    if results.exists():
        data = json.loads(results.read_text(encoding="utf-8"))
        for name, res in data["sets"].items():
            table(f"{name} set", res["before"], res["after"], res["examples"])
            for m in res["mistakes_after"][:10]:
                print(f"\n   said:  {m['said']}\n   want:  {json.dumps(m['want'], ensure_ascii=False)}"
                      f"\n   reply: {m['reply'][:200]}\n   wrong: {', '.join(m['wrong'])}")
    else:
        print("No results.json: the run stopped early. Kaggle's log:")
        for log in OUT.glob("*.log"):
            print_log(log)
    gguf = OUT / "results_gguf.json"
    if gguf.exists():
        data = json.loads(gguf.read_text(encoding="utf-8"))
        if "error" in data:
            print("\nGGUF test did not run:", data["error"])
        for name, res in data.get("sets", {}).items():
            if res["examples"]:
                print(f"\n== GGUF file {data['gguf']}, {name} set: {res['all_correct']:.0%} of "
                      f"{res['examples']} calls fully right (of {res['of']}), took {data['seconds']} s")
    print("\nFiles the run kept:")
    list_output(api, kernel)
    print(f"\nDownloaded to {OUT}: {sorted(p.name for p in OUT.iterdir())}")
    print(f"The model files are in the run's Output tab: https://www.kaggle.com/code/{kernel}/output")


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    opts = args.parse_args()
    api, kernel = connect()
    if not opts.wait:
        check_gpu(api)
        push(api, kernel)
        time.sleep(30)
    ok = wait(api, kernel)
    fetch(api, kernel)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

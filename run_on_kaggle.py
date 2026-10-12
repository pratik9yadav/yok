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
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
SLUG = "wimb-shop-model-training"
OUT = HERE / "kaggle_output"
DONE = {"COMPLETE": True, "ERROR": False, "CANCEL_REQUESTED": False, "CANCEL_ACKNOWLEDGED": False}


def slug_for(train_set: str) -> str:
    """v2 keeps the main notebook (its Output tab holds the model); other sets get their own."""
    return SLUG if train_set == "v2" else f"{SLUG}-{train_set}"


def connect(slug: str = SLUG):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    if not user:
        sys.exit("Kaggle login failed: set KAGGLE_API_TOKEN (Kaggle → Settings → API → Generate New Token).")
    return api, f"{user}/{slug}"


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


def notebook_for(train_set: str, gguf: bool) -> dict:
    """The notebook with its settings cell set to this training set and export choice."""
    nb = json.loads((HERE / "train_colab.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if source.startswith("#@title 3."):
            source = re.sub(r'^TRAIN_SET = "\w+"', f'TRAIN_SET = "{train_set}"', source, flags=re.MULTILINE)
            source = re.sub(r"^MAKE_OLLAMA_FILE = \w+", f"MAKE_OLLAMA_FILE = {gguf}", source, flags=re.MULTILINE)
            cell["source"] = source.splitlines(keepends=True)
    return nb


def push(api, kernel: str, train_set: str = "v2", gguf: bool = True) -> None:
    folder = Path(tempfile.mkdtemp())
    (folder / "train_colab.ipynb").write_text(json.dumps(notebook_for(train_set, gguf), ensure_ascii=False),
                                              encoding="utf-8")
    meta = {
        "id": kernel, "title": kernel.split("/", 1)[1], "code_file": "train_colab.ipynb",
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
    if results.exists():
        print(f"\nTraining set: {data.get('train_set', '?')} ({data.get('train_rows', '?')} rows)")
    print(f"\nDownloaded to {OUT}: {sorted(p.name for p in OUT.iterdir())}")
    print(f"The model files are in the run's Output tab: https://www.kaggle.com/code/{kernel}/output")


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    args.add_argument("--train-set", choices=["v2", "v1"], default="v2", help="training data to use (default v2)")
    args.add_argument("--no-gguf", action="store_true", help="skip exporting and testing the Ollama file")
    opts = args.parse_args()
    api, kernel = connect(slug_for(opts.train_set))
    if not opts.wait:
        check_gpu(api)
        push(api, kernel, opts.train_set, gguf=not opts.no_gguf)
        time.sleep(30)
    ok = wait(api, kernel)
    fetch(api, kernel)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

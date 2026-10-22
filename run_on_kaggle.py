"""Run train_colab.ipynb on Kaggle's free GPU, wait for it, and download the results. No clicks.

Needs, once:
  1. A Kaggle account with a verified phone number (Kaggle allows GPU and internet only then).
  2. KAGGLE_API_TOKEN set (Kaggle → Settings → API → Generate New Token).
  3. pip install kaggle

    python run_on_kaggle.py           start a run, wait (about 40 min), print the results
    python run_on_kaggle.py --wait    wait for the run already going, then print the results
    python run_on_kaggle.py --train-set v4    train on set v4 (its own notebook, wimb-shop-model-training-v4,
                                              so the v2 model the try/ask/voice notebooks use is not overwritten)
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
TRAIN_SETS = ["v2", "v1", "v4"]
# How long a whole run may take (minutes) and the GPU hours it needs. v4 has 6,151 rows (v2: 1,451): about
# 1 hour of training for 2 epochs, so about 2-2.5 hours in all with the tests and the GGUF export.
WAIT_MIN = {"v4": 300}
NEED_HOURS = {"v4": 3}


def slug_for(train_set: str, slug: str | None = None) -> str:
    """v2 keeps the main notebook (its Output tab holds the model); other sets get their own
    (v4: wimb-shop-model-training-v4). slug: a notebook name of your own instead."""
    if slug:
        return slug
    return SLUG if train_set == "v2" else f"{SLUG}-{train_set}"


def connect(slug: str = SLUG):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    if not user:
        sys.exit("Kaggle login failed: set KAGGLE_API_TOKEN (Kaggle → Settings → API → Generate New Token).")
    return api, f"{user}/{slug}"


def check_gpu(api, need_hours: float = 1) -> None:
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
    if left < need_hours:
        sys.exit(f"Less than {need_hours:g} hour of Kaggle GPU time left this week. It resets weekly.")


def notebook_for(train_set: str, gguf: bool) -> dict:
    """The notebook with its settings cell set to this training set and export choice."""
    nb = json.loads((HERE / "train_colab.ipynb").read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if source.startswith("#@title 3."):
            source = re.sub(r'^TRAIN_SET = "\w+"', f'TRAIN_SET = "{train_set}"', source, flags=re.MULTILINE)
            source = re.sub(r"^MAKE_OLLAMA_FILE = \w+", f"MAKE_OLLAMA_FILE = {gguf}", source, flags=re.MULTILINE)
            cell["source"] = source.splitlines(keepends=True)
        elif source.startswith("#@title 4."):  # only the chosen training set: all of them make the notebook too big
            for name in {"v4": ("TRAIN_V2", "TRAIN_V1"), "v2": ("TRAIN_V4", "TRAIN_V1"), "v1": ("TRAIN_V4", "TRAIN_V2")}[train_set]:
                source, n = re.subn(rf"^{name} = .*?splitlines\(\)\]\n", f"{name} = []  # not this run\n", source,
                                    count=1, flags=re.MULTILINE | re.DOTALL)
                assert n == 1, name
            cell["source"] = source.splitlines(keepends=True)
    return nb


def push(api, kernel: str, train_set: str = "v2", gguf: bool = True) -> None:
    push_notebook(api, kernel, notebook_for(train_set, gguf), "train_colab.ipynb")


def push_notebook(api, kernel: str, nb: dict, code_file: str, kernel_sources: list[str] | None = None,
                  dataset_sources: list[str] | None = None) -> None:
    """Start a run of this notebook on Kaggle: private, T4 GPU, internet on.

    kernel_sources: other notebooks whose output is attached as input (under /kaggle/input).
    dataset_sources: Kaggle datasets ("owner/slug") attached as input (under /kaggle/input too).
    """
    folder = Path(tempfile.mkdtemp())
    (folder / code_file).write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": kernel, "title": kernel.split("/", 1)[1], "code_file": code_file,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": dataset_sources or [], "competition_sources": [], "kernel_sources": kernel_sources or [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    # T4: the older P100 is too old for the training library.
    resp = api.kernels_push(str(folder), acc="NvidiaTeslaT4")
    error = getattr(resp, "error", None)
    if error:
        sys.exit(f"Kaggle refused the run: {error}")
    bad = getattr(resp, "invalid_kernel_sources", None)
    if bad:  # Kaggle still starts the run, without that input
        sys.exit(f"Kaggle could not attach {bad} as input (no such notebook, or no access). "
                 "Run the training first (Actions → Train on Kaggle).")
    bad = getattr(resp, "invalid_dataset_sources", None)
    if bad:
        sys.exit(f"Kaggle could not attach the dataset {bad} as input (no such dataset, or no access). "
                 "Check its name: owner/slug, as in its kaggle.com/datasets/owner/slug link.")
    print("Started:", getattr(resp, "url", None) or f"https://www.kaggle.com/code/{kernel}", flush=True)
    try:  # what Kaggle saved for this run: shows whether the GPU request arrived
        saved = Path(tempfile.mkdtemp())
        api.kernels_pull(kernel, str(saved), metadata=True)
        server = json.loads((saved / "kernel-metadata.json").read_text())
        keys = (("enable_gpu", "machine_shape", "enable_internet") + (("kernel_sources",) if kernel_sources else ())
                + (("dataset_sources",) if dataset_sources else ()))
        print("Kaggle saved:", {k: server.get(k) for k in keys}, flush=True)
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
            size = getattr(f, "size", None)  # as Kaggle reports it; its unit is not documented
            print(f"   {getattr(f, 'name', f)}" + (f"  (size {size:,})" if size else ""))
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
        exported = json.loads(gguf.read_text(encoding="utf-8"))
        if "error" in exported:
            print("\nGGUF test did not run:", exported["error"])
        for name, res in exported.get("sets", {}).items():
            if res["examples"]:
                print(f"\n== GGUF file {exported['gguf']}, {name} set: {res['all_correct']:.0%} of "
                      f"{res['examples']} calls fully right (of {res['of']}), took {exported['seconds']} s")
    print("\nFiles the run kept:")
    list_output(api, kernel)
    if results.exists():
        print(f"\nTraining set: {data.get('train_set', '?')} ({data.get('train_rows', '?')} rows)")
    print(f"\nDownloaded to {OUT}: {sorted(p.name for p in OUT.iterdir())}")
    print(f"The model files are in the run's Output tab: https://www.kaggle.com/code/{kernel}/output")


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    args.add_argument("--train-set", choices=TRAIN_SETS, default="v2", help="training data to use (default v2)")
    args.add_argument("--slug", help="Kaggle notebook name to use instead of the one for the training set")
    args.add_argument("--no-gguf", action="store_true", help="skip exporting and testing the Ollama file")
    opts = args.parse_args()
    api, kernel = connect(slug_for(opts.train_set, opts.slug))
    if not opts.wait:
        check_gpu(api, NEED_HOURS.get(opts.train_set, 1))
        push(api, kernel, opts.train_set, gguf=not opts.no_gguf)
        time.sleep(30)
    ok = wait(api, kernel, WAIT_MIN.get(opts.train_set, 120))
    fetch(api, kernel)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

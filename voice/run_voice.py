"""Run Voice v0 on Kaggle's free GPU (voice/voice_v0.ipynb), wait, download the audio and print the summary. No clicks.

Starts voice_v0.ipynb on Kaggle with the training notebook's output (the shop model) attached, waits (about
20–40 minutes) and prints, per answer clip: what Whisper heard, what the shop model understood, the reply, and the
seconds per step. The .wav files (questions, test answers, replies) are downloaded to kaggle_output_voice/ and are
in the run's Output tab.

    python voice/run_voice.py                                     the 6 synthetic answers only
    python voice/run_voice.py --clips-dataset me/wimb-voice-clips   also your recordings (a Kaggle dataset)
    python voice/run_voice.py --wait                              wait for the run already going, then print it

Needs what run_on_kaggle.py needs (KAGGLE_API_TOKEN, pip install kaggle) and one finished training run.
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import run_on_kaggle
from run_on_kaggle import check_gpu, connect, print_log, wait
from try_on_kaggle import short, training_input
from voice import build_voice
from voice.voicecall import summary_lines

SLUG = "wimb-voice-v0"
NOTEBOOK = HERE / build_voice.NOTEBOOK
OUT = HERE.parent / "kaggle_output_voice"
WAIT_MINUTES = 60
DATASET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")  # owner/slug


def dataset_from(text: str | None) -> str | None:
    """--clips-dataset as "owner/slug" (also from a kaggle.com/datasets/owner/slug link). None when empty."""
    text = (text or "").strip()
    if not text:
        return None
    text = re.sub(r"^(https?://)?(www\.)?kaggle\.com/datasets/", "", text).strip("/")
    if not DATASET.match(text) or len(text) > 120:
        raise ValueError(f"not a Kaggle dataset name: {text!r} (like me/wimb-voice-clips)")
    return text


def push(api, kernel: str, clips_dataset: str | None = None) -> None:
    """Start the notebook with the trained model (and the owner's recordings, when given) attached."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    print("Your recordings:", clips_dataset or "none (only the synthetic answers)", flush=True)
    run_on_kaggle.push_notebook(api, kernel, nb, build_voice.NOTEBOOK, kernel_sources=[training_input(api, kernel)],
                                dataset_sources=[clips_dataset] if clips_dataset else [])


def report(r: dict) -> None:
    """Print voice_results.json for a person."""
    print()
    for line in r.get("summary") or summary_lines(r):
        print(line)
    questions = r.get("questions") or []
    if questions:
        made = sum(bool(q.get("audio")) for q in questions)
        print(f"\nQuestions spoken: {made} of {len(questions)}: " + " | ".join(q.get("text", "") for q in questions))
    if r.get("versions"):
        print("Versions:", r["versions"])
    if r.get("errors"):
        print("\nErrors:")
        for error in r["errors"]:
            print("   ", short(error, 500))


def gaps(r: dict) -> list[str]:
    """What the run did not do (empty when every step worked at least once)."""
    found = []
    if not any(q.get("audio") for q in r.get("questions") or []):
        found.append("no question was spoken (the text to speech did not work)")
    clips = r.get("clips") or []
    if not clips:
        found.append("no answer clip was processed")
        return found
    if not any(c.get("transcript") for c in clips):
        found.append("speech to text heard nothing in any clip")
    if not any(c.get("answer") for c in clips):
        found.append("the shop model understood no clip")
    missing = sum(not c.get("reply_audio") for c in clips)
    if missing:
        found.append(f"{missing} of {len(clips)} clips got no spoken reply")
    return found


def fetch(api, kernel: str, ok: bool = True) -> dict | None:
    """Download voice_results.json, the .wav files and Kaggle's log, and print them. None when the run left none."""
    shutil.rmtree(OUT, ignore_errors=True)  # never report an older run's results or audio
    OUT.mkdir()
    api.kernels_output(kernel, str(OUT), file_pattern=r"(voice_results\.json|\.wav|\.log|\.ipynb)$", force=True)
    found = sorted(OUT.rglob(build_voice.RESULTS_FILE))
    r = json.loads(found[0].read_text(encoding="utf-8")) if found else None
    if r is not None:
        report(r)
    if r is None or not ok:
        print(f"\nNo {build_voice.RESULTS_FILE}: the run stopped early." if r is None else "\nThe run did not finish.",
              "Kaggle's log:")
        for log in OUT.glob("*.log"):
            print_log(log)
    wavs = sorted(p.name for p in OUT.rglob("*.wav"))
    print(f"\nDownloaded to {OUT}: {len(wavs)} .wav files" + (f" ({', '.join(wavs)})" if wavs else ""))
    print(f"Listen on Kaggle: https://www.kaggle.com/code/{kernel}/output (folder voice/)")
    print("Or on GitHub: the run's page → Artifacts → kaggle-output-voice (a zip with the .wav files)")
    return r


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--clips-dataset", metavar="OWNER/SLUG",
                      help="a Kaggle dataset with your recordings (q1_….m4a answers question 1), attached as input")
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    opts = args.parse_args()
    try:
        clips = dataset_from(opts.clips_dataset)
    except ValueError as e:
        args.error(str(e))
    if opts.wait and clips:
        args.error("--wait can't be used with --clips-dataset")
    api, kernel = connect(SLUG)
    if not opts.wait:
        check_gpu(api, need_hours=1)
        push(api, kernel, clips)
        time.sleep(30)
    ok = wait(api, kernel, limit_min=WAIT_MINUTES)
    r = fetch(api, kernel, ok)
    missing = gaps(r) if r is not None else ["the run left no results"]
    for why in missing:
        print("Not done:", why)
    sys.exit(0 if ok and r is not None and not missing else 1)


if __name__ == "__main__":
    main()

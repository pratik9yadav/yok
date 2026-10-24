"""Run the voice bot on Kaggle's free GPU (voicebot/voicebot.ipynb) and print what it found. No clicks.

    python voicebot/run_voicebot.py                  eval: measure every part and pick the best (about 60–90 min)
    python voicebot/run_voicebot.py --live 120       live: the web page for 120 minutes; prints its link
    python voicebot/run_voicebot.py --wait           wait for the eval run already going, then print it
    python voicebot/run_voicebot.py --brains gemma3:4b,qwen3:4b-instruct-2507-q4_K_M    other brains (Ollama names)

Needs KAGGLE_API_TOKEN (and pip install kaggle). The live page also needs VOICEBOT_PASSWORD (the page's password,
user name wimb): a repository secret, never printed. The page's link comes back through ntfy.sh (a free notice
service) under a random name made for each run. Live calls stay in the Kaggle run's Output tab (private): only
their count and outcomes are printed here.
"""

import argparse
import json
import os
import re
import secrets
import shutil
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
if sys.path and Path(sys.path[0]).resolve() == HERE.resolve():  # run as a script: don't shadow standard modules
    sys.path.pop(0)
sys.path.insert(0, str(HERE.parent))

import run_on_kaggle
from run_on_kaggle import check_gpu, connect, print_log, wait
from voicebot import build_voicebot

SLUGS = {"eval": "wimb-voicebot-eval", "live": "wimb-voicebot-live"}
NOTEBOOK = HERE / build_voicebot.NOTEBOOK
OUT = HERE.parent / "kaggle_output_voicebot"
EVAL_MINUTES = 150
BRAIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,79}$")
LINK = re.compile(r"^https://[a-z0-9-]+\.gradio\.live/?$")


def notebook_with(settings: dict) -> dict:
    """The notebook with these settings (name -> value) in its settings cell."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in nb["cells"] if "".join(c["source"]).startswith(f"#@title {build_voicebot.SETTINGS_TITLE}"))
    source = "".join(cell["source"])
    for name, value in settings.items():
        pattern = re.compile(rf"^{name} = .*?(  #.*)?$", re.MULTILINE)
        if not pattern.search(source):
            raise KeyError(f"setting {name} is not in the notebook: run python voicebot/build_voicebot.py")
        line = f"{name} = {json.dumps(value, ensure_ascii=False)}"
        source = pattern.sub(lambda m, line=line: line + (m.group(1) or ""), source, count=1)
    cell["source"] = source.splitlines(keepends=True)
    return nb


def brains_from(text: str | None) -> list[str] | None:
    if not text or not text.strip():
        return None
    names = [b.strip() for b in text.split(",") if b.strip()]
    bad = [b for b in names if not BRAIN.match(b)]
    if bad or not names:
        raise ValueError(f"not Ollama model names: {bad} (like gemma3:4b)")
    return names


def wait_for_link(topic: str, minutes: int = 40) -> str | None:
    """The page's link, posted by the notebook to ntfy.sh/<topic>; None after minutes."""
    url = f"https://ntfy.sh/{topic}/json?poll=1&since=all"
    for _ in range(minutes * 2):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                for line in resp.read().decode().splitlines():
                    message = json.loads(line).get("message", "").strip()
                    if LINK.match(message):
                        return message
        except Exception as e:  # noqa: BLE001 - keep waiting
            print("Waiting for the link:", type(e).__name__, flush=True)
        time.sleep(30)
    return None


def report(r: dict) -> None:
    print()
    for line in r.get("summary") or []:
        print(line)
    for name, t in (r.get("text_test") or {}).items():
        for f in t.get("failed", [])[:6]:
            print(f"\n{name}: FAILED {f['id']}: {'; '.join(f['wrong'])}")
            print("\n".join(f["turns"][:14]))
    for label, t in (r.get("spoken_test") or {}).items():
        if isinstance(t, dict) and t.get("misheard"):
            print(f"\nSpoken ({label}): some lines the ears misheard:")
            for m in t["misheard"][:12]:
                print(f"   said \"{m['said']}\" → heard \"{m['heard']}\"")
    if r.get("turn_seconds"):
        print("\nOne turn, average seconds:", r["turn_seconds"])
    if r.get("errors"):
        print("\nErrors:")
        for e in r["errors"]:
            print("   ", str(e)[:500])


def fetch(api, kernel: str, ok: bool = True, live: bool = False) -> dict | None:
    """Download the results (and, for eval, the audio) and print them. Live call recordings stay on Kaggle."""
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    pattern = r"(voicebot_results\.json|\.log)$" if live else r"(voicebot_results\.json|\.wav|\.log)$"
    api.kernels_output(kernel, str(OUT), file_pattern=pattern, force=True)
    for private in OUT.rglob("live"):  # never keep live calls here (the job's files can be public)
        shutil.rmtree(private, ignore_errors=True)
    found = sorted(OUT.rglob(build_voicebot.RESULTS_FILE))
    r = json.loads(found[0].read_text(encoding="utf-8")) if found else None
    if r is not None:
        report(r)
    if r is None or not ok:
        print("\nNo results: the run stopped early." if r is None else "\nThe run did not finish.", "Kaggle's log:")
        for log in OUT.glob("*.log"):
            print_log(log)
    wavs = sorted(p.name for p in OUT.rglob("*.wav"))
    if wavs:
        print(f"\nAudio ({len(wavs)} files): {', '.join(wavs)}; on GitHub: Artifacts → kaggle-output-voicebot")
    return r


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--live", type=int, metavar="MINUTES", help="open the web page for this many minutes")
    args.add_argument("--brains", help="comma-separated Ollama models to use instead of the notebook's")
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current eval run")
    opts = args.parse_args()
    try:
        brains = brains_from(opts.brains)
    except ValueError as e:
        args.error(str(e))
    live = opts.live is not None
    if live and not 10 <= opts.live <= 300:
        args.error("--live: 10 to 300 minutes")
    password = os.environ.get("VOICEBOT_PASSWORD", "").strip()
    if live and len(password) < 6:
        sys.exit("The live page needs a password: add the repository secret VOICEBOT_PASSWORD (6+ characters, your "
                 "choice) at https://github.com/pratik9yadav/yok/settings/secrets/actions/new")
    api, kernel = connect(SLUGS["live" if live else "eval"])
    if opts.wait:
        ok = wait(api, kernel, limit_min=EVAL_MINUTES)
        sys.exit(0 if fetch(api, kernel, ok) is not None and ok else 1)
    settings = {"MODE": "live" if live else "eval"}
    if brains:
        settings["BRAINS"] = brains
    topic = None
    if live:
        topic = "wimb-" + secrets.token_urlsafe(18).replace("_", "x").replace("-", "y")
        settings.update(LIVE_MINUTES=opts.live, LIVE_PASSWORD=password, NOTIFY_TOPIC=topic)
    check_gpu(api, need_hours=(opts.live / 60 + 0.5) if live else 1.5)
    run_on_kaggle.push_notebook(api, kernel, notebook_with(settings), build_voicebot.NOTEBOOK)
    time.sleep(30)
    if live:
        print("Starting the page: installing and loading the models (about 10–20 minutes)…", flush=True)
        link = wait_for_link(topic)
        if link:
            print("\n" + "=" * 60)
            print(f"Voice bot page: {link}")
            print("User name: wimb   Password: your VOICEBOT_PASSWORD secret")
            print(f"Open for {opts.live} minutes from now. Best on a phone: tap the mic, speak, tap again.")
            print("=" * 60 + "\n", flush=True)
        else:
            print("No link came back (see the log below).", flush=True)
    ok = wait(api, kernel, limit_min=(opts.live + 45) if live else EVAL_MINUTES)
    r = fetch(api, kernel, ok, live=live)
    if live and r:
        print("\nLive calls:", (r.get("live") or {}))
    sys.exit(0 if ok and r is not None else 1)


if __name__ == "__main__":
    main()

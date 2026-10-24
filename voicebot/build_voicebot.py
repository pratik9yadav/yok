"""Build voicebot/voicebot.ipynb: the voice bot on Kaggle's free GPU, in one of two modes.

- eval: measure everything and pick the best free parts: each brain (small models in Ollama) on the 48 test calls;
  speech to text on real Hindi and Marathi speech (FLEURS; Lahaja, many Indian accents, with an HF token), clean and
  as a phone line; the test calls spoken (voice → phone line → speech to text → bot); the voices; the time per turn.
- live: the web page to talk to the bot (voicebot/live.py), with a password, for LIVE_MINUTES.

The voicebot package (voicebot/*.py, tasks, test calls) is written into the notebook, so it runs as tested here.
Run after changing anything in voicebot/:  python voicebot/build_voicebot.py   (never edit the notebook by hand)
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
if sys.path and Path(sys.path[0]).resolve() == HERE.resolve():  # run as a script: don't shadow standard modules
    sys.path.pop(0)
sys.path.insert(0, str(HERE.parent))

from build_notebook import METADATA, OLLAMA_API, START_OLLAMA, code, md, run_helpers, text, titled

NOTEBOOK = "voicebot.ipynb"
RESULTS_FILE = "voicebot_results.json"
SETTINGS_TITLE = "1. Settings"
PACKAGE = ["__init__.py", "numerals.py", "lang.py", "speech_text.py", "tasks.py", "understand.py", "engine.py",
           "evaluate.py", "audio.py", "live.py"]
GRADIO_PIP = "gradio==5.50.0"
PARLER_PIP = "parler-tts==0.2.3"  # pins transformers 4.46.1: installed only when its model will be tried

INTRO = """
# WIMB voice bot on Kaggle's free GPU

A voice bot for Indian calls (Hindi, Marathi, English and mixes): it talks for any **task** written as a small JSON
file (ask a shop about stock, book a clinic visit, confirm an order, or your own).

Each turn: **ears** (speech to text) → **brain** (rules + a small model read what was said, checked by type) →
the conversation decides what to say (from the task, never made up) → **mouth** (text to speech).

**MODE = "eval"** measures every part and picks the best free one. **MODE = "live"** opens a web page to talk to
the bot (the link is sent to the GitHub job; it needs the password).

Built by `voicebot/build_voicebot.py`; run it from GitHub: **Actions → Voice bot → Run workflow**.
"""

SETTINGS = """
import itertools, json, os, re, shutil, subprocess, sys, time, urllib.error, urllib.parse, urllib.request

MODE = "eval"  # "eval": measure everything; "live": the web page
BRAINS = ["qwen3:4b-instruct-2507-q4_K_M", "gemma3:4b", "gemma3:12b"]  # eval: all; live: the first that works
ASR = ["faster-whisper", "transformers"]  # speech to text ("indic-conformer" is tried too with an HF token)
TTS = ["kokoro", "parler", "mms"]  # text to speech, in order of preference (parler needs an HF token)
BENCH_ROWS = 60  # real speech per language for the speech-to-text test
LIVE_MINUTES = 120  # live: how long the page stays open
LIVE_PASSWORD = ""  # live: the page's password (user name: wimb)
NOTIFY_TOPIC = ""  # live: the page's link is posted to ntfy.sh/<this> for the GitHub job to print
WORKING_DIR = "/kaggle/working/voicebot"  # everything is saved here (the Output tab)
"""

HELPERS = """
SRC = "/tmp/voicebot_src"
for name, body in PACKAGE_FILES.items():
    path = os.path.join(SRC, "voicebot", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
sys.path.insert(0, SRC)
os.makedirs(WORKING_DIR, exist_ok=True)

from voicebot import evaluate, tasks
from voicebot.engine import Call
from voicebot.lang import detect

RESULTS = {"mode": MODE, "gpu": None, "ollama_version": None, "brains": {}, "best_brain": None, "text_test": {},
           "asr": {"tried": [], "bench": {}}, "tts": {"tried": [], "samples": []}, "best_asr": None,
           "spoken_test": {}, "turn_seconds": {}, "live": {}, "versions": {}, "summary": [], "errors": []}
"""

INSTALL = f"""
def pip(*packages, timeout=1800):
    done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], capture_output=True, text=True,
                          timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError((done.stdout + done.stderr)[-1500:])


def hf_token():
    \"\"\"The Hugging Face token: Kaggle Secrets "HF_TOKEN" (Add-ons → Secrets), else the env. Never printed.\"\"\"
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = None
    return (token or os.environ.get("HF_TOKEN") or "").strip() or None


HF_TOKEN = hf_token()
print("Hugging Face token:", "found" if HF_TOKEN else "none (gated models are skipped)")
RESULTS["hf_token"] = bool(HF_TOKEN)
missing = [p for p in ("ffmpeg", "espeak-ng", "zstd") if not shutil.which(p)]
if missing:  # ffmpeg reads phone audio; espeak-ng: kokoro's Hindi pronunciation; zstd: Ollama's download
    sh("apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg espeak-ng zstd", timeout=900)
for packages in (["faster-whisper", "soundfile", "librosa", "scipy"], ["kokoro-onnx"], ["datasets", "huggingface_hub"]):
    try:
        pip(*packages)
    except Exception as e:
        failed("pip install " + " ".join(packages), e)
try:  # Kokoro on the GPU (the CPU onnxruntime stays as the fallback)
    pip("onnxruntime-gpu")
except Exception as e:
    print("onnxruntime-gpu did not install (Kokoro runs on the CPU):", f"{{e!r}}"[:300])
if HF_TOKEN and "parler" in TTS:
    try:
        pip("{PARLER_PIP}")
    except Exception as e:
        failed("pip install {PARLER_PIP}", e)
if MODE == "live":
    try:
        pip("{GRADIO_PIP}")
    except Exception as e:
        failed("pip install {GRADIO_PIP}", e)
import importlib.metadata as md

for p in ("torch", "transformers", "faster-whisper", "ctranslate2", "kokoro-onnx", "onnxruntime", "onnxruntime-gpu",
          "parler-tts", "datasets", "gradio"):
    try:
        RESULTS["versions"][p] = md.version(p)
    except md.PackageNotFoundError:
        RESULTS["versions"][p] = None
print(RESULTS["versions"])
save_results()
"""

BRAIN = """
LLM_SECONDS = {}  # brain -> seconds of each model call


def pull(model):
    done = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=1800)
    return done.returncode == 0, re.sub(r"\\x1b\\[[0-9;?]*[A-Za-z]", "", done.stderr + done.stdout)[-400:]


def make_llm(model):
    \"\"\"llm(prompt, schema) -> the model's JSON text, through Ollama (JSON forced by the schema).\"\"\"
    LLM_SECONDS.setdefault(model, [])

    def llm(prompt, schema):
        started = time.perf_counter()
        data = generate({"model": model, "prompt": prompt, "format": schema, "stream": False, "keep_alive": "30m",
                         "options": {"temperature": 0, "num_predict": 300, "num_ctx": 4096}})
        LLM_SECONDS[model].append(round(time.perf_counter() - started, 3))
        return data.get("response", "")

    return llm


READY = []  # the brains that pulled and answered
for model in BRAINS if MODE == "eval" else BRAINS[:2]:
    started = time.time()
    ok, log = pull(model)
    entry = {"pulled": ok, "pull_seconds": round(time.time() - started), "log": None if ok else log}
    if ok:
        try:
            make_llm(model)("Reply with JSON: {\\"ok\\": true}", {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                                                 "required": ["ok"]})
            READY.append(model)
        except Exception as e:
            entry["log"] = f"{e!r}"[:400]
    RESULTS["brains"][model] = entry
    print(model, "ready" if model in READY else f"NOT ready: {entry['log']}")
    if MODE == "live" and READY:
        break
save_results()
"""

TEXT_TEST = """
SCENARIOS = evaluate.scenarios()


def text_test(name, llm):
    started = time.time()
    results = [evaluate.play(s, llm=llm) for s in SCENARIOS]
    lines = evaluate.summary_lines(results, f"{name}: ")
    print("\\n".join(lines))
    calls = LLM_SECONDS.get(name, [])
    RESULTS["text_test"][name] = {
        **evaluate.summary(results), "lines": lines, "minutes": round((time.time() - started) / 60, 1),
        "model_seconds_avg": round(sum(calls) / len(calls), 2) if calls else None,
        "failed": [{"id": r["id"], "wrong": r["wrong"], "turns": evaluate.transcript(r)} for r in results if r["wrong"]]}
    save_results()
    return results


if MODE == "eval":
    text_test("rules only", None)
    for model in READY:
        LLM_SECONDS[model] = []
        text_test(model, make_llm(model))
    ranked = sorted(READY, key=lambda m: (-RESULTS["text_test"][m]["all"]["passed"],
                                          RESULTS["text_test"][m]["model_seconds_avg"] or 99))
    RESULTS["best_brain"] = ranked[0] if ranked else None
    if ranked:  # for a live call, speed matters: the fastest brain within one test call of the best
        top = RESULTS["text_test"][ranked[0]]["all"]["passed"]
        quick = [m for m in ranked if RESULTS["text_test"][m]["all"]["passed"] >= top - 1
                 and (RESULTS["text_test"][m]["model_seconds_avg"] or 99) <= 3]
        RESULTS["live_brain"] = min(quick, key=lambda m: RESULTS["text_test"][m]["model_seconds_avg"]) if quick else None
else:
    RESULTS["best_brain"] = READY[0] if READY else None
print("Brain for the rest:", RESULTS["best_brain"] or "none (rules only)")
LLM = make_llm(RESULTS["best_brain"]) if RESULTS["best_brain"] else None
save_results()
"""

EARS_MOUTH = """
from voicebot.audio import ASR_INFO, TTS_INFO, Ears, Mouth, load_first, phone_line, resample, to_int16

ASR_ORDER = (["indic-conformer"] if HF_TOKEN else []) + [a for a in ASR if a != "indic-conformer"]
EARS = {}  # every speech-to-text model that loads (eval) or the first (live)
for backend in ASR_ORDER:
    ears, tried = load_first("ears", [backend], HF_TOKEN)
    RESULTS["asr"]["tried"] += tried
    if ears is not None:
        EARS[backend] = ears
        if MODE == "live":
            break
MOUTHS = {}
for backend in TTS:
    mouth, tried = load_first("mouth", [backend], HF_TOKEN)
    RESULTS["tts"]["tried"] += tried
    if mouth is not None:
        MOUTHS[backend] = mouth
        if MODE == "live":
            break
print("Ears:", list(EARS) or "none", "| Mouths:", list(MOUTHS) or "none")
save_results()
"""

BENCH = """
import io
import unicodedata

import numpy as np
import soundfile as sf


def norm(t):
    t = unicodedata.normalize("NFC", str(t or "")).lower().replace("\\u093c", "").replace("\\u0901", "\\u0902")
    return " ".join(re.sub(r"[^\\w\\s]", " ", t).split())


def edits(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def error_rates(pairs):
    \"\"\"(word error rate %, character error rate %) over (reference, heard) pairs.\"\"\"
    w = sum(edits(norm(r).split(), norm(h).split()) for r, h in pairs)
    c = sum(edits(norm(r).replace(" ", ""), norm(h).replace(" ", "")) for r, h in pairs)
    nw = sum(len(norm(r).split()) for r, _ in pairs) or 1
    nc = sum(len(norm(r).replace(" ", "")) for r, _ in pairs) or 1
    return round(100 * w / nw, 1), round(100 * c / nc, 1)


def decode(audio):
    if isinstance(audio, dict) and audio.get("bytes"):
        data, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    elif isinstance(audio, dict) and audio.get("array") is not None:
        data, rate = np.asarray(audio["array"], dtype=np.float32)[:, None], audio["sampling_rate"]
    else:
        data, rate = sf.read(audio["path"], dtype="float32", always_2d=True)
    return resample(data.mean(axis=1), rate, 16000)


def rows_from(ds, n, text_keys, group_keys=()):
    from datasets import Audio

    columns = list(getattr(ds, "column_names", None) or next(iter(ds.take(1))).keys())
    audio_key = next(k for k in columns if k in ("audio", "audio_filepath", "file") or "audio" in k)
    text_key = next(k for k in text_keys if k in columns)
    group_key = next((k for k in group_keys if k in columns), None)
    try:
        ds = ds.cast_column(audio_key, Audio(decode=False))
    except Exception:
        pass
    out = []
    for row in ds:
        out.append({"audio": decode(row[audio_key]), "text": row[text_key],
                    "group": str(row[group_key]) if group_key else None})
        if len(out) >= n:
            break
    return out, {"text": text_key, "group": group_key}


def fleurs(lang, n):
    from datasets import load_dataset

    code = {"hi": "hi_in", "mr": "mr_in"}[lang]
    errors = []
    for how in ("parquet", "script"):
        try:
            if how == "parquet":
                ds = load_dataset("parquet", split="train", streaming=True, data_files=(
                    f"hf://datasets/google/fleurs@refs/convert/parquet/{code}/test/*.parquet"))
            else:
                ds = load_dataset("google/fleurs", code, split="test", streaming=True, trust_remote_code=True)
            rows, keys = rows_from(ds, n, ("raw_transcription", "transcription"))
            return rows, f"google/fleurs {code} test ({how})"
        except Exception as e:
            errors.append(f"{how}: {e!r}"[:300])
    raise RuntimeError(" | ".join(errors))


def lahaja(n):
    from datasets import load_dataset

    ds = load_dataset("ai4bharat/Lahaja", split="test", streaming=True, token=HF_TOKEN)
    rows, keys = rows_from(ds, n, ("text", "transcription", "transcript", "sentence", "verbatim"),
                           ("native_state", "state", "native_district", "district", "region"))
    return rows, "ai4bharat/Lahaja test (Hindi from many districts)"


def bench(name, rows, lang):
    out = {}
    for backend, ears in EARS.items():
        for condition in ("clean", "phone"):
            pairs, seconds = [], []
            for i, row in enumerate(rows):
                audio = row["audio"] if condition == "clean" else phone_line(row["audio"], seed=i)
                heard, _, s = ears.hear(audio, hint=lang, allowed=[lang], detect=False)
                pairs.append((row["text"], heard))
                seconds.append(s)
            wer, cer = error_rates(pairs)
            key = f"{backend} / {condition}"
            out[key] = {"wer": wer, "cer": cer, "seconds_avg": round(sum(seconds) / len(seconds), 2),
                        "examples": [{"said": r, "heard": h} for r, h in pairs[:3]]}
            groups = sorted({r["group"] for r in rows if r["group"]})
            if groups and condition == "phone":
                out[key]["by_group"] = {g: error_rates([p for p, r in zip(pairs, rows) if r["group"] == g])[0]
                                        for g in groups}
            print(f"{name} | {key}: WER {wer}%  CER {cer}%  ({out[key]['seconds_avg']} s per clip)")
        if lang == "mr" and ears.backend != "indic-conformer":  # Whisper's Marathi is weak: is "hi" better?
            pairs = [(r["text"], ears.hear(phone_line(r["audio"], seed=i), hint="hi", allowed=["hi"],
                                           detect=False)[0]) for i, r in enumerate(rows)]
            out[f"{backend} / phone, decoded as Hindi"] = dict(zip(("wer", "cer"), error_rates(pairs)))
    RESULTS["asr"]["bench"][name] = out
    save_results()


if MODE == "eval" and EARS:
    sets = [("Hindi (FLEURS)", "hi", lambda: fleurs("hi", BENCH_ROWS)),
            ("Marathi (FLEURS)", "mr", lambda: fleurs("mr", BENCH_ROWS))]
    if HF_TOKEN:
        sets.append(("Hindi accents (Lahaja)", "hi", lambda: lahaja(2 * BENCH_ROWS)))
    for name, lang, load in sets:
        try:
            rows, source = load()
            print(f"{name}: {len(rows)} clips from {source}")
            RESULTS["asr"]["bench"][name + " source"] = source
            bench(name, rows, lang)
        except Exception as e:
            failed(f"speech-to-text test {name}", e)
    scores = {b: [v[f"{b} / phone"]["wer"] for k, v in RESULTS["asr"]["bench"].items()
                  if isinstance(v, dict) and f"{b} / phone" in v] for b in EARS}
    scores = {b: sum(s) / len(s) for b, s in scores.items() if s}
    RESULTS["best_asr"] = min(scores, key=scores.get) if scores else next(iter(EARS))
elif EARS:
    RESULTS["best_asr"] = next(iter(EARS))
print("Speech to text for the rest:", RESULTS["best_asr"])
save_results()
"""

SPOKEN = """
from voicebot.speech_text import for_voice

PERSON_MOUTH = next((MOUTHS[b] for b in ("parler", "kokoro", "mms") if b in MOUTHS), None)  # the most varied voices
BOT_MOUTH = next(iter(MOUTHS.values()), None)
VOICES = itertools.cycle(["person", "person2", "person3"])


def spoken_test(label, use_prompt, demo=False):
    ears = EARS[RESULTS["best_asr"]]
    rows = [s for s in SCENARIOS if "latin" not in s.get("tags", [])]  # speech to text writes Devanagari anyway
    heard_log, demos = [], {}

    def run(scenario):
        voice = next(VOICES)
        audio_parts = []

        def say_and_hear(line, call):
            lang = detect(line, default=call.lang)[0]
            try:
                audio, rate, t_say = PERSON_MOUTH.say(for_voice(line, lang), lang, voice)
            except Exception as e:  # this line can't be spoken: the bot reads it as typed (counted)
                heard_log.append({"id": scenario["id"], "said": line, "heard": line, "seconds": 0, "unspoken": repr(e)[:200]})
                return line, 0.0
            audio = phone_line(resample(audio, rate, 16000), seed=len(heard_log))
            bot = [t.text for t in call.turns if t.who == "bot"]
            heard, _, seconds = ears.hear(audio, hint=call.lang, allowed=call.task.languages,
                                          prompt=(bot[-1][-200:] if use_prompt and bot else None))
            heard_log.append({"id": scenario["id"], "said": line, "heard": heard, "seconds": seconds})
            audio_parts.append(("person", audio))
            return heard, seconds

        def on_bot(reply):
            if demo and BOT_MOUTH is not None and reply.speech:
                audio, rate, _ = BOT_MOUTH.say(reply.speech, reply.lang, "bot")
                audio_parts.append(("bot", resample(audio, rate, 16000)))

        result = evaluate.play(scenario, llm=LLM, say_and_hear=say_and_hear, on_bot=on_bot if demo else None)
        if demo and scenario["task"] not in demos and not result["wrong"]:
            gap = np.zeros(int(0.35 * 16000), dtype=np.float32)
            demos[scenario["task"]] = np.concatenate([x for _, a in audio_parts for x in (a, gap)])
        return result

    started = time.time()
    results = []
    for s in rows:
        try:
            results.append(run(s))
        except Exception as e:  # one broken call must not stop the test: it counts as failed
            failed(f"spoken test {label}: {s['id']}", e)
            results.append({"id": s["id"], "task": s["task"], "tags": s.get("tags", []), "wrong": [f"crashed: {e!r}"[:200]],
                            "lines": [], "model_calls": 0, "model_errors": 0, "turns": []})
    lines = evaluate.summary_lines(results, f"Spoken ({label}): ")
    print("\\n".join(lines))
    misheard = [h for h in heard_log if norm(h["said"]) != norm(h["heard"]) and not h.get("unspoken")]
    unspoken = sum(bool(h.get("unspoken")) for h in heard_log)
    RESULTS["spoken_test"][label] = {**evaluate.summary(results), "lines": lines,
                                     "minutes": round((time.time() - started) / 60, 1),
                                     "lines_heard_exactly": f"{len(heard_log) - len(misheard) - unspoken} of "
                                                            f"{len(heard_log) - unspoken}",
                                     "lines_not_spoken": unspoken,
                                     "misheard": misheard[:40],
                                     "failed": [{"id": r["id"], "wrong": r["wrong"], "turns": evaluate.transcript(r)}
                                                for r in results if r["wrong"]]}
    for task_id, audio in demos.items():
        path = os.path.join(WORKING_DIR, f"demo_call_{task_id}.wav")
        sf.write(path, audio, 16000)
        RESULTS["spoken_test"][label].setdefault("demo_calls", []).append(path)
    save_results()


if MODE == "eval" and EARS and PERSON_MOUTH is not None:
    RESULTS["spoken_test"]["voice"] = f"{PERSON_MOUTH.backend} voices, then a phone line (8 kHz, μ-law, noise)"
    for label, use_prompt, demo in (("no hint", False, True), ("hint: the bot's question", True, False)):
        try:
            spoken_test(label, use_prompt, demo)
        except Exception as e:
            failed(f"spoken test ({label})", e)
"""

VOICES_CELL = """
SAMPLES = {"hi": "ठीक है। तो मैंने नोट किया: 16A MCB है, दाम ₹250, डिलीवरी हो जाएगी। सही है?",
           "mr": "ठीक आहे. तर मी नोंद केलं: 16A MCB आहे, किंमत ₹220, डिलिव्हरी नाही. बरोबर आहे का?",
           "en": "Okay. So I have: 16A MCB is in stock, price ₹250, you deliver. Is that right?"}
if MODE == "eval":
    for backend, mouth in MOUTHS.items():
        for lang, line in SAMPLES.items():
            try:
                audio, rate, seconds = mouth.say(for_voice(line, lang), lang, "bot")
                path = os.path.join(WORKING_DIR, f"voice_{backend}_{lang}.wav")
                sf.write(path, audio, rate)
                RESULTS["tts"]["samples"].append({"backend": backend, "lang": lang, "seconds": seconds,
                                                  "audio_seconds": round(len(audio) / rate, 1), "file": path})
                print(f"{backend} {lang}: {seconds} s for {len(audio) / rate:.1f} s of speech -> {path}")
            except Exception as e:
                failed(f"voice {backend} {lang}", e)
    save_results()
"""

SUMMARY = """
def summary_lines():
    lines = []
    for name, r in RESULTS["text_test"].items():
        a = r["all"]
        speed = f", {r['model_seconds_avg']} s per model call" if r.get("model_seconds_avg") else ""
        lines.append(f"Test calls, typed, {name}: {a['passed']}/{a['total']}{speed}")
    if RESULTS["best_brain"]:
        lines.append(f"Best brain: {RESULTS['best_brain']}; for live calls (fast, within one test call): "
                     f"{RESULTS.get('live_brain') or 'none fast enough'}")
    for name, r in RESULTS["asr"]["bench"].items():
        if isinstance(r, dict):
            lines.append(f"Speech to text, {name}: " + "; ".join(
                f"{k} WER {v['wer']}%" for k, v in r.items() if isinstance(v, dict) and "wer" in v))
    for label, r in RESULTS["spoken_test"].items():
        if isinstance(r, dict) and "all" in r:
            lines.append(f"Test calls, spoken over a phone line ({label}): {r['all']['passed']}/{r['all']['total']}; "
                         f"lines heard exactly: {r['lines_heard_exactly']}")
    for s in RESULTS["tts"]["samples"]:
        lines.append(f"Voice {s['backend']} {s['lang']}: {s['seconds']} s to make {s['audio_seconds']} s of speech")
    ok = lambda kind: [t for t in RESULTS[kind]["tried"] if t["ok"]]
    for kind, info in (("asr", ASR_INFO), ("tts", TTS_INFO)):
        for t in RESULTS[kind]["tried"]:
            i = info.get(t["backend"], {})
            flag = "" if i.get("business", True) else " (NOT for business use)"
            lines.append(f"{kind.upper()} {t['backend']} ({i.get('model')}, {i.get('licence')}{flag}): "
                         + ("ok" if t["ok"] else t["why"]))
    if MODE == "live":
        lines.append(f"Live page: {RESULTS['live'].get('calls', 0)} calls, {RESULTS['live'].get('saved', 0)} saved")
    return lines


RESULTS["summary"] = summary_lines()
print("\\n".join(RESULTS["summary"]))
print("\\nErrors:", RESULTS["errors"] or "none")
save_results()
"""

TURN = """
if MODE == "eval" and EARS and BOT_MOUTH is not None:  # one turn, end to end, timed: hear → understand → speak
    ears = EARS[RESULTS["best_asr"]]
    times = {"hear": [], "understand": [], "speak": []}
    for i, line in enumerate(["हाँ है, ढाई सौ का पड़ेगा", "नहीं, डिलीवरी नहीं करते", "हो, आहे, दोनशे वीस रुपये"]):
        call = Call(tasks.load("shop_stock"), llm=LLM)
        call.start()
        audio, rate, _ = PERSON_MOUTH.say(line, detect(line)[0], "person")
        audio = phone_line(resample(audio, rate, 16000), seed=i)
        heard, _, t_hear = ears.hear(audio, hint="hi", allowed=["hi", "mr", "en"])
        started = time.perf_counter()
        reply = call.hear(heard)
        t_think = time.perf_counter() - started
        _, _, t_speak = BOT_MOUTH.say(reply.speech, reply.lang, "bot")
        for k, v in (("hear", t_hear), ("understand", t_think), ("speak", t_speak)):
            times[k].append(v)
    RESULTS["turn_seconds"] = {k: round(sum(v) / len(v), 2) for k, v in times.items()}
    RESULTS["turn_seconds"]["total"] = round(sum(RESULTS["turn_seconds"].values()), 2)
    print("One turn, average seconds:", RESULTS["turn_seconds"])
    save_results()
"""

LIVE = """
def notify(url):
    \"\"\"Post the page's link to ntfy.sh/NOTIFY_TOPIC, where the GitHub job waits for it.\"\"\"
    if not NOTIFY_TOPIC:
        return
    for _ in range(5):
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{NOTIFY_TOPIC}", data=url.encode(),
                                                          method="POST"), timeout=20).read()
            return
        except Exception as e:
            print("ntfy:", e)
            time.sleep(5)


if MODE == "live":
    from voicebot.live import build_app

    ears = EARS.get(RESULTS["best_asr"]) if EARS else None
    mouth = next(iter(MOUTHS.values()), None)
    note = "" if mouth is None or mouth.info["business"] else "Test voice (non-commercial licence). "
    save_dir = os.path.join(WORKING_DIR, "live")
    app = build_app(ears, mouth, LLM, save_dir=save_dir, brain_name=RESULTS["best_brain"] or "rules only", note=note)
    app.queue(default_concurrency_limit=1)
    _, local_url, share_url = app.launch(share=True, auth=("wimb", LIVE_PASSWORD or "wimb"), prevent_thread_lock=True,
                                         show_error=True)
    RESULTS["live"] = {"opened": time.strftime("%H:%M UTC"), "minutes": LIVE_MINUTES, "link": bool(share_url)}
    print("Page:", "ready" if share_url else "no public link (Gradio share failed)")
    if share_url:
        notify(share_url)
    else:
        failed("live page", "Gradio gave no share link")
    save_results()
    end = time.time() + 60 * LIVE_MINUTES
    while time.time() < end and share_url:
        time.sleep(60)
        calls = [d for d in os.listdir(save_dir)] if os.path.isdir(save_dir) else []
        RESULTS["live"].update(saved=len(calls))
        save_results()
    app.close()
    calls = sorted(os.listdir(save_dir)) if os.path.isdir(save_dir) else []
    RESULTS["live"].update(saved=len(calls), calls=len(calls),
                           outcomes=[json.load(open(os.path.join(save_dir, c, "call.json")))["result"]["outcome"]
                                     for c in calls if os.path.exists(os.path.join(save_dir, c, "call.json"))])
    print("Page closed.", RESULTS["live"])
    save_results()
"""


def package_files() -> dict:
    """Every file of the voicebot package the notebook needs: code, tasks and test calls."""
    files = {name: (HERE / name).read_text(encoding="utf-8") for name in PACKAGE}
    for folder in ("tasks", "scenarios"):
        for path in sorted((HERE / folder).iterdir()):
            if path.suffix in (".json", ".jsonl"):
                files[f"{folder}/{path.name}"] = path.read_text(encoding="utf-8")
    return files


def notebook() -> dict:
    helpers = ("#@title 2. Helpers (no need to read: the voicebot package, written to disk)\n"
               + f"PACKAGE_FILES = {package_files()!r}\n" + HELPERS + run_helpers(RESULTS_FILE, 4)
               + "\n\n" + OLLAMA_API.strip("\n") + "\n")
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": [
            md(INTRO),
            code(titled(f"{SETTINGS_TITLE}: mode, models, live page", SETTINGS)),
            code(helpers, hidden=True, folded=True),
            code(titled("3. Install (about 3–5 minutes)", INSTALL)),
            code(titled("4. Brain: Ollama and the small models", START_OLLAMA + BRAIN)),
            code(titled("5. Test calls, typed: rules only, then each brain", TEXT_TEST)),
            code(titled("6. Ears and mouth: load speech to text and text to speech", EARS_MOUTH)),
            code(titled("7. Speech to text on real speech, clean and as a phone line", BENCH)),
            code(titled("8. Test calls, spoken over a phone line", SPOKEN)),
            code(titled("9. Voices: samples and speed", VOICES_CELL)),
            code(titled("10. One turn, end to end, timed", TURN)),
            code(titled("11. Live: the web page (MODE = \"live\")", LIVE)),
            code(titled("12. Summary (saved to voicebot_results.json)", SUMMARY)),
        ],
    }


def main() -> None:
    if len(sys.argv) > 1:
        print(__doc__)
        return
    out = HERE / NOTEBOOK
    out.write_text(text(notebook()), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

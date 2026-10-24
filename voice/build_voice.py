"""Build voice/voice_v0.ipynb: Voice v0, a spoken shop call on Kaggle's free GPU, as a cascade of free open models:
ears (Whisper, speech to text) → brain (the trained shop model in Ollama) → mouth (text to speech).

Run after changing this file, voice/voicecall.py, score.py or the shared Ollama code in build_notebook.py:
    python voice/build_voice.py
Never edit voice/voice_v0.ipynb by hand.
"""

import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from build_notebook import (
    CREATE_MODEL,
    CREATE_TITLE,
    FIND_MODEL,
    FIND_TITLE,
    METADATA,
    OLLAMA_API,
    OLLAMA_TITLE,
    OPTIONS,
    START_OLLAMA,
    code,
    md,
    modelfile_repo,
    run_helpers,
    scoring,
    text,
    titled,
)

NOTEBOOK = "voice_v0.ipynb"
RESULTS_FILE = "voice_results.json"
SETTINGS_TITLE = "1. Settings"
PARLER_PIP = "parler-tts==0.2.3"  # the PyPI release of github.com/huggingface/parler-tts; it pins transformers 4.46.1
SKIP_MARK = "# in the notebook:"  # voicecall.py's imports of score.py and in_words: the notebook has them above

INTRO = """
# Voice v0: a spoken shop call (ears → brain → mouth), free models on Kaggle's free GPU

**What it does:** the AI asks a shopkeeper 3 questions in Hindi, by voice (in stock? price? delivery?).
Each spoken answer goes through three free models, one after the other:
1. **Ears**: Whisper large-v3 turbo turns the speech into text (Hindi).
2. **Brain**: the trained shop model (Ollama, the repo's `Modelfile`) reads the text and gives the 4 fields
   (in stock, price, delivery, other product offered). A small template writes a short polite reply in Hindi.
3. **Mouth**: a text-to-speech model says the reply.

It times each step and saves everything to the **Output** tab, folder `voice/`: the questions (`q1.wav`–`q3.wav`),
the test answers (`synthetic_1.wav`…), the replies (`reply_1.wav`…) and `voice_results.json`.

**Test answers:** your own recordings if a Kaggle dataset with them is attached (`q2_anything.m4a` answers
question 2), and always 6 made-up answers spoken by the TTS voice. Those check the plumbing, not real accents.

**Settings (right side):** Input `wimb-shop-model-training` (the trained model), Accelerator **GPU T4**,
Internet **on**. For the better Hindi voice (Indic Parler-TTS) add a Hugging Face token as the secret `HF_TOKEN`
(Add-ons → Secrets); without it a fallback voice is used (MMS-TTS, non-commercial licence: test only).

**Or with no clicks:** on GitHub, **Actions → Voice v0 (speech to speech) → Run workflow**, or ask Claude.
"""

SETTINGS = """
import json, os, re, shutil, subprocess, sys, time, urllib.error, urllib.request

PRODUCT = "16A MCB"  # what the AI asks about (and what the shop model is told)
PRODUCT_SPOKEN = "सोलह ए एमसीबी"  # the same, as the voice says it (Devanagari: some voices skip Latin letters)
ASR_MODEL = "openai/whisper-large-v3-turbo"  # speech to text (MIT licence, not gated)
# Text to speech, tried in this order: the first that works is used (its licence is saved with the results).
TTS_CANDIDATES = ["ai4bharat/indic-parler-tts", "facebook/mms-tts-hin"]
CLIPS_DIR = "/kaggle/input"  # your recordings: any .wav/.mp3/.m4a/.ogg/.opus under here (an attached Kaggle dataset)
MAX_CLIPS = 20  # at most this many of your recordings
WORKING_DIR = "/kaggle/working/voice"  # everything is saved here (the notebook's Output tab)
"""

STATE = """
RESULTS = new_results(ASR_MODEL, PRODUCT)
BRAIN_ERROR = None  # why the shop model is not running (None: it is, or not tried yet)
if os.path.isdir(os.path.dirname(WORKING_DIR)):
    os.makedirs(WORKING_DIR, exist_ok=True)
"""

INSTALL = f"""
# parler-tts is installed only when its model will be tried (an HF token is there). It pins transformers 4.46.1,
# so it goes in before anything imports transformers.
PARLER_PIP = "{PARLER_PIP}"
HF_TOKEN = hf_token()  # Kaggle Secrets "HF_TOKEN", else the env HF_TOKEN (never printed)
TTS_PLAN, RESULTS["tts_tried"] = tts_plan(TTS_CANDIDATES, HF_TOKEN)
print("Hugging Face token:", "found" if HF_TOKEN else "none (Indic Parler-TTS is skipped)")


def pip(*packages):
    done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages],
                          capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        raise RuntimeError((done.stdout + done.stderr)[-1500:])


try:
    pip("soundfile", "librosa")  # usually there already
except Exception as e:
    failed("pip install soundfile librosa", e)
if not shutil.which("ffmpeg"):  # ffmpeg reads .m4a, .ogg and .opus (phone recordings)
    sh("apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg", timeout=900)
RESULTS["ffmpeg"] = bool(shutil.which("ffmpeg"))
if any("parler" in m for m in TTS_PLAN):
    try:
        pip(PARLER_PIP)
    except Exception as e:
        failed("pip install " + PARLER_PIP, e)
        RESULTS["tts_tried"] += [{{"model": m, "ok": False, "why": "parler-tts did not install"}}
                                 for m in TTS_PLAN if "parler" in m]
        TTS_PLAN = [m for m in TTS_PLAN if "parler" not in m]
try:
    import transformers  # noqa: F401 - only to see that it is there
except ImportError:
    pip("transformers", "accelerate")
import importlib.metadata as md


def version(package):
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


RESULTS["versions"] = {{p: version(p) for p in ("torch", "transformers", "parler_tts", "soundfile", "librosa")}}
print("ffmpeg:", RESULTS["ffmpeg"], "| versions:", RESULTS["versions"], "| TTS to try:", TTS_PLAN)
save_results()
"""

EARS = """
ASR = None  # the speech-to-text pipeline, once loaded


def load_audio(path):
    \"\"\"16 kHz mono samples. ffmpeg reads any format (.m4a, .ogg, .opus too); without it, librosa.\"\"\"
    import tempfile

    import numpy as np
    import soundfile as sf

    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "audio.wav")
            done = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path, "-ac", "1",
                                   "-ar", "16000", wav], capture_output=True, text=True, timeout=300)
            if done.returncode != 0:
                raise RuntimeError(f"ffmpeg could not read {os.path.basename(path)}: {done.stderr[-300:]}")
            audio, _ = sf.read(wav, dtype="float32")
        return audio
    import librosa

    audio, _ = librosa.load(path, sr=16000, mono=True)
    return audio.astype(np.float32)


def transcribe(path, language):
    \"\"\"What was said in the recording: language "hi" (Hindi), or None to let Whisper guess the language.\"\"\"
    if ASR is None:
        raise RuntimeError("the speech-to-text model did not load (see cell 4)")
    options = {"task": "transcribe", **({"language": language} if language else {})}
    return ASR({"raw": load_audio(path), "sampling_rate": 16000}, generate_kwargs=options)["text"].strip()


try:
    import torch
    from transformers import pipeline

    on_gpu = torch.cuda.is_available()
    started = time.time()
    ASR = pipeline("automatic-speech-recognition", model=ASR_MODEL, device="cuda:0" if on_gpu else "cpu",
                   torch_dtype=torch.float16 if on_gpu else torch.float32, chunk_length_s=30)
    RESULTS["asr_device"] = f"GPU ({torch.cuda.get_device_name(0)})" if on_gpu else "CPU"
    RESULTS["asr_load_seconds"] = round(time.time() - started, 1)
    print(f"{ASR_MODEL} loaded in {RESULTS['asr_load_seconds']} s on the {RESULTS['asr_device']}")
except Exception as e:
    failed("load the speech-to-text model", e)
save_results()
"""

MOUTH = """
TTS = None  # {"model", "say": say(text, voice) -> (samples, sample rate)}, once one loads
# Indic Parler-TTS takes the voice as a description (Divya and Rohit are two of its Hindi speakers).
VOICES = {"ai": "Divya speaks in a clear, calm and polite voice at a moderate pace. The recording is very clear, "
                "with no background noise.",
          "shopkeeper": "Rohit speaks quickly in a casual tone. The recording is close, with a little background noise."}


def device_name():
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_parler(model):
    \"\"\"Indic Parler-TTS (needs the HF token and parler-tts). Two voices: the AI's and the shopkeeper's.\"\"\"
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = device_name()
    net = ParlerTTSForConditionalGeneration.from_pretrained(model, token=HF_TOKEN).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model, token=HF_TOKEN)
    describe = AutoTokenizer.from_pretrained(net.config.text_encoder._name_or_path)

    def say(text, voice):
        d = describe(VOICES[voice], return_tensors="pt").to(device)
        p = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            audio = net.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                 prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
        return audio.cpu().float().numpy().squeeze(), net.config.sampling_rate

    return say, device


def load_mms(model):
    \"\"\"Meta's MMS-TTS (VITS, no token needed). One voice for both sides.\"\"\"
    import torch
    from transformers import AutoTokenizer, VitsModel

    device = device_name()
    net = VitsModel.from_pretrained(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model)
    romanize = None
    if getattr(tokenizer, "is_uroman", False):  # this voice reads Latin letters only: romanize the text first
        pip("uroman")
        import uroman

        romanize = uroman.Uroman().romanize_string

    def say(text, voice):
        inputs = tokenizer(romanize(text) if romanize else text, return_tensors="pt").to(device)
        torch.manual_seed(0)  # the same audio on every run
        with torch.no_grad():
            audio = net(**inputs).waveform
        return audio.cpu().float().numpy().squeeze(), net.config.sampling_rate

    return say, device


def speak(text, path, voice="ai"):
    \"\"\"Say the text into a .wav file. voice: "ai" or "shopkeeper".\"\"\"
    import soundfile as sf

    if TTS is None:
        raise RuntimeError("no text-to-speech model loaded (see cell 5)")
    audio, rate = TTS["say"](text, voice)
    if audio.size == 0:
        raise RuntimeError("the text-to-speech model made no sound")
    sf.write(path, audio, rate)


for model in TTS_PLAN:
    started = time.time()
    try:
        say, device = (load_parler if "parler" in model else load_mms)(model)
        say("नमस्ते", "ai")  # a first try: some problems only show here
    except Exception as e:  # e.g. gated: the token's account has not accepted the model's terms
        RESULTS["tts_tried"].append({"model": model, "ok": False, "why": f"{e!r}"[:500]})
        print(f"{model} did not work:", f"{e!r}"[:600])
        continue
    TTS = {"model": model, "say": say}
    use_tts(RESULTS, model)
    RESULTS["tts_device"] = device
    RESULTS["tts_tried"].append({"model": model, "ok": True, "why": f"loaded in {round(time.time() - started, 1)} s"})
    break
if TTS is None:
    failed("load a text-to-speech model", "none of " + ", ".join(TTS_CANDIDATES) + " worked")
else:
    print(f"Text to speech: {RESULTS['tts_model']} ({RESULTS['tts_licence']}) on {RESULTS['tts_device']}")
    if not RESULTS["tts_business_use"]:
        print("NOT FOR BUSINESS USE:", RESULTS["tts_note"])
save_results()
"""

THINK = """
def think(product, transcript):
    \"\"\"The shop model's reply (the 4-field JSON) to what the shopkeeper said, through Ollama's API.\"\"\"
    if BRAIN_ERROR:
        raise RuntimeError(f"the shop model is not running ({BRAIN_ERROR})")
    data = generate({"model": "wimb-shop", "prompt": user_message(product, transcript), "stream": False,
                     "keep_alive": "60m", "options": {""" + OPTIONS + """}})
    return data.get("response", "").strip()


print("The brain is", "ready" if BRAIN_ERROR is None else f"NOT running: {BRAIN_ERROR} (ears and mouth still run)")
"""

LOAD_BRAIN = """
started = time.time()
generate({"model": "wimb-shop", "keep_alive": "60m"})  # no prompt: only load it
RESULTS["brain_load_seconds"] = round(time.time() - started, 1)
print(f"Shop model loaded in {RESULTS['brain_load_seconds']} s")
"""

QUESTIONS_CELL = """
RESULTS["questions"] = []
for q in questions(PRODUCT, PRODUCT_SPOKEN):
    entry = {**q, "audio": None, "seconds": None}
    path = os.path.join(WORKING_DIR, f"q{q['n']}.wav")
    try:
        _, entry["seconds"] = timed(speak, q["speech"], path, "ai")
        entry["audio"] = path
    except Exception as e:
        failed(f"question {q['n']}", e)
    RESULTS["questions"].append(entry)
    print(f"q{q['n']}: {q['text']}  →  {entry['audio'] or 'no audio'}")
save_results()
"""

ANSWERS_CELL = """
CLIPS = find_clips(CLIPS_DIR, MAX_CLIPS)
print(f"Your recordings: {len(CLIPS)}" + ("" if CLIPS else " (none attached: only the synthetic answers)"))
for c in CLIPS:
    print("   ", c["file"], "answers", f"q{c['question']}" if c["question"] else "? (the name doesn't say)")
made = 0
for i, row in enumerate(SYNTHETIC, 1):  # made-up answers, said by the TTS: a plumbing check, not an accent test
    path = os.path.join(WORKING_DIR, f"synthetic_{i}.wav")
    try:
        speak(row["said"], path, "shopkeeper")
    except Exception as e:
        failed(f"synthetic answer {i}", e)
        continue
    made += 1
    CLIPS.append({"file": path, "synthetic": True, "question": row["question"], "said": row["said"],
                  "expected": row["answer"], "alternative_any": row.get("alternative_any", [])})
print(f"Synthetic answers spoken: {made} of {len(SYNTHETIC)}")
RESULTS["clips_planned"] = len(CLIPS)
save_results()
"""

PIPELINE_CELL = """
RESULTS["clips"] = []
for n, clip in enumerate(CLIPS, 1):
    entry = process_clip(n, clip, PRODUCT, PRODUCT_SPOKEN, transcribe, think, speak, WORKING_DIR, RESULTS,
                         save_results)
    print(clip_line(entry))
save_results()
"""

SUMMARY_CELL = """
if BRAIN_ERROR is None:
    RESULTS["brain_processor"] = processor()  # where Ollama ran the shop model: "100% GPU", ...
RESULTS["average_seconds"] = average_seconds(RESULTS["clips"])
RESULTS["summary"] = summary_lines(RESULTS)
print("\\n".join(RESULTS["summary"]))
print("\\nErrors:", "none" if not RESULTS["errors"] else "")
for error in RESULTS["errors"]:
    print("   ", error[:500])
save_results()
print("\\nListen: the Output tab → voice/ (q1-q3.wav: the questions; synthetic_N.wav: the test answers; "
      "reply_N.wav: the replies)")
"""

NEXT = """
## What next
- **Your own voice:** record answers on your phone (`q1_….m4a`, `q2_….m4a`, `q3_….m4a`), put them in a private
  Kaggle dataset and run again with it attached (README: "Voice v0").
- This cascade (ears → brain → mouth) is the start. The goal is a direct speech-to-speech model, trained on real
  calls recorded with the shopkeepers' consent.
"""


def lower_first(title: str) -> str:
    return title[:1].lower() + title[1:]


def guarded(title: str, body: str) -> str:
    """A brain cell that is skipped once a brain step failed, and records its own failure instead of stopping
    Run all (the ears and the mouth still run without the brain)."""
    if '"""' in body or "'''" in body:
        raise ValueError(f"{title}: a multi-line string can't be indented safely")
    inner = textwrap.indent(body.strip("\n"), " " * 8)
    return titled(title, f"""
if BRAIN_ERROR is None:  # skipped when an earlier brain step failed
    try:
{inner}
    except Exception as e:
        BRAIN_ERROR = f"{{type(e).__name__}}: {{e}}"[:500]
        failed("brain", e)
        print("The brain is NOT running (the ears and the mouth still run):", BRAIN_ERROR)
save_results()
""")


def voicecall_code() -> str:
    """voice/voicecall.py without its imports of score.py and in_words (the notebook has them above)."""
    lines = (HERE / "voicecall.py").read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if not (line.startswith("from ") and SKIP_MARK in line)]
    if len(lines) - len(kept) != 2:
        raise ValueError(f"voicecall.py: expected 2 imports marked '{SKIP_MARK}'")
    return "\n".join(kept) + "\n"


def notebook() -> dict:
    """The Voice v0 notebook."""
    helpers = (
        "#@title 2. Helpers (no need to read; built from score.py, voice/voicecall.py and Modelfile)\n"
        + scoring()
        + voicecall_code()
        + "\n"
        + modelfile_repo()
        + STATE
        + run_helpers(RESULTS_FILE, 7)
        + "\n\n"
        + OLLAMA_API.strip("\n")
        + "\n"
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": [
            md(INTRO),
            code(titled(f"{SETTINGS_TITLE}: the product, the models, your recordings", SETTINGS)),
            code(helpers, hidden=True, folded=True),
            code(titled("3. Install the speech tools (about 1–3 minutes)", INSTALL)),
            code(titled("4. Ears: load Whisper (speech to text)", EARS)),
            code(titled("5. Mouth: load the text to speech (the first that works)", MOUTH)),
            code(guarded(f"6. Brain: {lower_first(FIND_TITLE)}", FIND_MODEL)),
            code(guarded(f"7. Brain: {lower_first(OLLAMA_TITLE)}", START_OLLAMA)),
            code(guarded(f"8. Brain: {lower_first(CREATE_TITLE)}", CREATE_MODEL + LOAD_BRAIN) + "\n\n\n" + THINK.strip("\n")),
            code(titled("9. The AI's 3 questions, spoken (q1.wav, q2.wav, q3.wav)", QUESTIONS_CELL)),
            code(titled("10. The answers: your recordings (if attached) and 6 synthetic ones", ANSWERS_CELL)),
            code(titled("11. Each answer: ears → brain → reply → mouth (timed)", PIPELINE_CELL)),
            code(titled("12. Summary (saved to voice/voice_results.json)", SUMMARY_CELL)),
            md(NEXT),
        ],
    }


def main() -> None:
    out = HERE / NOTEBOOK
    out.write_text(text(notebook()), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

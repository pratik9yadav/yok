"""Voice v0: the plain parts of the spoken shop call, shared by voice_v0.ipynb (built in by voice/build_voice.py)
and the tests. No torch here: the ears (ASR), the mouth (TTS) and the brain (the shop model in Ollama) are passed in
as plain functions, so the tests can replace them.

    transcribe(path, language) -> text       language "hi", or None to let Whisper guess
    think(product, transcript) -> reply       the shop model's raw reply (the 4-field JSON)
    speak(text, path, voice) -> None          writes a .wav; voice "ai" or "shopkeeper"
"""

import os
import re
import time

from score import FIELDS, parse, same  # in the notebook: score.py is above
from try_on_kaggle import in_words  # in the notebook: in_words is above

RESULTS_FILE = "voice_results.json"
AUDIO = (".wav", ".mp3", ".m4a", ".ogg", ".opus")  # .opus: WhatsApp voice notes

# Every model the cascade can use, with its licence. business_use False: allowed for this internal test only.
ASR_LICENCES = {"openai/whisper-large-v3-turbo": "MIT"}
TTS_INFO = {
    "ai4bharat/indic-parler-tts": {"licence": "Apache-2.0", "needs_token": True, "business_use": True,
                                   "note": "needs a Hugging Face token whose account accepted the model's terms"},
    "facebook/mms-tts-hin": {"licence": "CC-BY-NC-4.0", "needs_token": False, "business_use": False,
                             "note": "non-commercial licence: allowed for this internal test only, not for business use"},
}
BRAIN_INFO = "wimb-shop: Qwen3-4B-Instruct-2507 (Apache-2.0) + our LoRA, Q4_K_M in Ollama (MIT)"

# What the AI asks: (what is shown, what is said). {p} is the product as written, {s} as spoken.
QUESTIONS = [
    ("नमस्ते, क्या आपके पास {p} है?", "नमस्ते, क्या आपके पास {s} है?"),
    ("{p} का दाम कितना है?", "{s} का दाम कितना है?"),
    ("क्या आप घर पर डिलीवरी करते हैं?", "क्या आप घर पर डिलीवरी करते हैं?"),
]

# Made-up shopkeeper answers about a 16A MCB, spoken by the TTS so the pipeline runs end to end without real clips.
# They are a check of the plumbing, not of real accents. Devanagari only: digits and Latin letters can't be spoken by
# every TTS. question: which question it answers (None: none). alternative_any: any of these words counts as the
# alternative (the model may write it in Latin or in Devanagari).
SYNTHETIC = [
    {"question": 1, "said": "हाँ जी, सोलह ए वाला एमसीबी है। दो सौ बीस रुपये का पड़ेगा।",
     "answer": {"in_stock": True, "price_inr": 220, "delivery": None, "alternative": None}},
    {"question": 1, "said": "नहीं भाई, वो वाला अभी खत्म है। हैवेल्स का बत्तीस ए वाला रखा है, वो ले लो।",
     "answer": {"in_stock": False, "price_inr": None, "delivery": None, "alternative": "Havells 32A"},
     "alternative_any": ["havells", "हैवेल्स"]},
    {"question": 3, "said": "हाँ है, ढाई सौ का है, और घर पे भी पहुँचा देंगे।",
     "answer": {"in_stock": True, "price_inr": 250, "delivery": True, "alternative": None}},
    {"question": 3, "said": "है, एक सौ नब्बे का। पर डिलीवरी नहीं करते, दुकान से ले जाना पड़ेगा।",
     "answer": {"in_stock": True, "price_inr": 190, "delivery": False, "alternative": None}},
    {"question": 2, "said": "यस, स्टॉक में है। रेट टू थर्टी है, होम डिलीवरी कल कर देंगे।",
     "answer": {"in_stock": True, "price_inr": 230, "delivery": True, "alternative": None}},
    {"question": None, "said": "पता नहीं, बाद में कॉल करो।",
     "answer": {"in_stock": None, "price_inr": None, "delivery": None, "alternative": None}},
]

ONES = [  # 0 to 99 in Hindi words
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", "तेरह", "चौदह",
    "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस",
    "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस",
    "अड़तीस", "उनतालीस", "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चवालीस", "पैंतालीस", "छियालीस", "सैंतालीस",
    "अड़तालीस", "उनचास", "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ",
    "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर", "इकहत्तर",
    "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उनासी", "अस्सी", "इक्यासी", "बयासी",
    "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी", "नब्बे", "इक्यानवे", "बानवे", "तिरानवे",
    "चौरानवे", "पचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे",
]


def hindi_number(n: int) -> str:
    """A whole number in Hindi words, e.g. 220 -> "दो सौ बीस", 1500 -> "एक हज़ार पाँच सौ". Digits from a crore up."""
    if n < 0 or n >= 10_000_000:
        return str(n)
    if n < 100:
        return ONES[n]
    words = []
    for size, name in ((100_000, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= size:
            words.append(f"{ONES[n // size]} {name}")
            n %= size
    if n:
        words.append(ONES[n])
    return " ".join(words)


def rupees(price) -> tuple[str, str]:
    """A price as (shown, spoken): 220 -> ("₹220", "दो सौ बीस रुपये")."""
    value = float(price)
    whole, paise = int(value), round((value - int(value)) * 100)
    shown = f"₹{whole}" if not paise else f"₹{value:.2f}"
    spoken = f"{hindi_number(whole)} रुपये" + (f" {hindi_number(paise)} पैसे" if paise else "")
    return shown, spoken


def questions(product: str, spoken: str | None = None) -> list[dict]:
    """The 3 questions about this product: [{"n", "text" (shown), "speech" (what the TTS says)}]."""
    return [{"n": n, "text": text.format(p=product), "speech": speech.format(s=spoken or product)}
            for n, (text, speech) in enumerate(QUESTIONS, 1)]


def is_price(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def write_reply(answer, product: str, spoken: str | None = None) -> dict:
    """A short polite Hindi reply to what the shop model understood: {"text" (shown), "speech" (for the TTS)}.

    For example: in stock, ₹220, no delivery -> "ठीक है, 16A MCB ₹220 में है, डिलीवरी नहीं। धन्यवाद!".
    Nothing understood (no answer, or all four fields null) -> a polite request to say it again.
    """
    answer = answer if isinstance(answer, dict) else {}
    in_stock, price, delivery, alternative = (answer.get(f) for f in FIELDS)
    price = price if is_price(price) else None
    alternative = alternative.strip() if isinstance(alternative, str) and alternative.strip() else None
    if in_stock is None and price is None and delivery is None and alternative is None:
        sorry = "माफ़ कीजिए, ठीक से समझ नहीं आया। क्या आप एक बार फिर बता सकते हैं? नहीं तो हम बाद में कॉल कर लेंगे। धन्यवाद!"
        return {"text": sorry, "speech": sorry}
    out = {}
    for key, name, money in (("text", product, 0), ("speech", spoken or product, 1)):
        cost = rupees(price)[money] if price is not None else None
        clauses = []
        if in_stock is True:
            clauses.append(f"{name} {cost} में है" if cost else f"{name} है")
        elif in_stock is False:
            clauses.append(f"{name} अभी नहीं है")
        elif cost:
            clauses.append(f"{name} का दाम {cost} है")
        if delivery is True:
            clauses.append("डिलीवरी हो जाएगी")
        elif delivery is False:
            clauses.append("डिलीवरी नहीं")
        text = "ठीक है, " + ", ".join(clauses) + "।" if clauses else "ठीक है।"
        if alternative:
            spoken_digits = re.sub(r"\d+", lambda m: f" {hindi_number(int(m[0]))} ", alternative)  # 32A -> बत्तीस A
            other = alternative if key == "text" else " ".join(spoken_digits.split())
            text += f" {other} के बारे में बताया, हम ग्राहक से पूछ लेंगे।"
        if in_stock is True and price is None:
            text += " इसका दाम कितना है?"  # the call goes on: no thanks yet
        else:
            text += " धन्यवाद!"
        out[key] = text
    return out


def question_of(file_name: str) -> int | None:
    """Which question a recording answers, from its name: "q2_anything.m4a" -> 2. None when the name doesn't say."""
    found = re.match(r"q([1-9])(?![0-9])", os.path.basename(file_name).lower())
    n = int(found.group(1)) if found else None
    return n if n is not None and n <= len(QUESTIONS) else None


def find_clips(folder: str, limit: int = 20) -> list[dict]:
    """The owner's recordings under folder (any depth), sorted by name: [{"file", "synthetic", "question"}]."""
    found = []
    for root, _dirs, files in os.walk(folder or "/nonexistent"):
        found += [os.path.join(root, f) for f in files if f.lower().endswith(AUDIO) and not f.startswith(".")]
    found.sort(key=lambda p: (os.path.basename(p).lower(), p))
    return [{"file": p, "synthetic": False, "question": question_of(p)} for p in found[:limit]]


def wrong_fields(answer, expected: dict, alternative_any=()) -> list[str]:
    """The fields the answer got wrong (all four when there is no answer). The alternative also counts as right when
    it holds any of alternative_any (the model may write a brand in Latin or in Devanagari)."""
    if not isinstance(answer, dict):
        return list(FIELDS)
    wrong = []
    for f in FIELDS:
        got, want = answer.get(f), expected[f]
        named = f == "alternative" and isinstance(got, str) and isinstance(want, str)
        if named and any(w.lower() in got.lower() for w in alternative_any):
            continue
        if not same(f, got, want):
            wrong.append(f)
    return wrong


def new_results(asr_model: str, product: str) -> dict:
    """What voice_results.json starts as (the notebook fills it in, stage by stage)."""
    return {"asr_model": asr_model, "asr_licence": ASR_LICENCES.get(asr_model, "unknown: check before use"),
            "asr_device": None, "tts_model": None, "tts_licence": None, "tts_business_use": None, "tts_note": None,
            "tts_tried": [], "tts_device": None, "brain": BRAIN_INFO, "brain_processor": None, "product": product,
            "questions": [], "clips": [], "summary": [], "average_seconds": None, "errors": []}


def use_tts(results: dict, model: str) -> None:
    """Record which TTS model is used, with its licence (and that it is not for business use, when so)."""
    info = TTS_INFO.get(model, {"licence": "unknown: check before any use", "business_use": False,
                                "note": "not in the list of checked models"})
    results.update(tts_model=model, tts_licence=info["licence"], tts_business_use=info["business_use"],
                   tts_note=info["note"])


def tts_plan(candidates: list[str], token: str | None) -> tuple[list[str], list[dict]]:
    """Which TTS models to try, in order, and which are skipped and why (a model that needs a token, without one)."""
    plan, skipped = [], []
    for model in candidates:
        if TTS_INFO.get(model, {}).get("needs_token") and not token:
            skipped.append({"model": model, "ok": False, "why": "skipped: no HF_TOKEN (Kaggle Secrets or env)"})
        else:
            plan.append(model)
    return plan, skipped


def hf_token() -> str | None:
    """The Hugging Face token: Kaggle Secrets "HF_TOKEN" (Add-ons → Secrets), else the env HF_TOKEN. Never printed."""
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:  # noqa: BLE001 - not on Kaggle, or the secret is not attached to this notebook
        token = None
    return (token or os.environ.get("HF_TOKEN") or "").strip() or None


def timed(fn, *args):
    """fn(*args) and the seconds it took."""
    started = time.perf_counter()
    value = fn(*args)
    return value, round(time.perf_counter() - started, 2)


def process_clip(n: int, clip: dict, product: str, spoken: str | None, transcribe, think, speak, out_dir: str,
                 results: dict, save) -> dict:
    """One answer clip through the cascade: ears (ASR) -> brain (shop model) -> reply text -> mouth (TTS).

    The entry is added to results["clips"] first and save() runs after every stage, so a failure or a stop part-way
    loses nothing. A stage that fails is recorded (in the entry and in results["errors"]) and the next one still runs
    when it can: with no transcript or no answer, the reply is the polite request to say it again.
    """
    entry = {"n": n, "file": clip["file"], "synthetic": clip.get("synthetic", False), "question": clip.get("question"),
             "said": clip.get("said"), "expected": clip.get("expected"), "transcript": None, "transcript_auto": None,
             "reply": None, "answer": None, "wrong": None, "reply_text": None, "reply_speech": None,
             "reply_audio": None, "seconds": {"asr": None, "asr_auto": None, "brain": None, "tts": None, "total": None},
             "errors": []}
    results["clips"].append(entry)

    def stage(name, fn, *args):
        try:
            value, entry["seconds"][name] = timed(fn, *args)
            return value
        except Exception as e:  # noqa: BLE001 - record it, go on
            message = f"{name}: {e!r}"[:500]
            entry["errors"].append(message)
            results["errors"].append(f"clip {n} ({os.path.basename(clip['file'])}) {message}")
            return None
        finally:
            save()

    transcript = stage("asr", transcribe, clip["file"], "hi")
    entry["transcript"] = transcript.strip() if isinstance(transcript, str) else None
    auto = stage("asr_auto", transcribe, clip["file"], None)  # the language Whisper guesses (not in the total)
    entry["transcript_auto"] = auto.strip() if isinstance(auto, str) else None
    if entry["transcript"]:
        reply = stage("brain", think, product, entry["transcript"])
        entry["reply"] = reply.strip() if isinstance(reply, str) else None
        entry["answer"] = parse(entry["reply"]) if entry["reply"] else None
    if entry["expected"] is not None:
        entry["wrong"] = wrong_fields(entry["answer"], entry["expected"], clip.get("alternative_any") or ())
    reply = write_reply(entry["answer"], product, spoken)
    entry["reply_text"], entry["reply_speech"] = reply["text"], reply["speech"]
    path = os.path.join(out_dir, f"reply_{n}.wav")
    stage("tts", speak, reply["speech"], path, "ai")
    if entry["seconds"]["tts"] is not None:
        entry["reply_audio"] = path
    steps = [entry["seconds"][k] for k in ("asr", "brain", "tts")]
    entry["seconds"]["total"] = round(sum(s for s in steps if s is not None), 2)
    save()
    return entry


def average_seconds(clips: list[dict]) -> dict:
    """The average seconds per step over the clips where that step ran (None: it never ran)."""
    out = {}
    for step in ("asr", "brain", "tts", "total"):
        values = [c["seconds"][step] for c in clips if c.get("seconds", {}).get(step) is not None]
        out[step] = round(sum(values) / len(values), 2) if values else None
    return out


def short(text, width: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def clip_line(c: dict) -> str:
    """One clip in one line: where it came from, what was heard, understood and replied, how right, how fast."""
    kind = "synthetic" if c.get("synthetic") else "your clip " + os.path.basename(c.get("file") or "?")
    question = f"q{c['question']}" if c.get("question") else "q?"
    if c.get("wrong") is None:
        mark = ""
    else:
        mark = " 4/4 right" if not c["wrong"] else f" {4 - len(c['wrong'])}/4 right (wrong: {', '.join(c['wrong'])})"
    s = c.get("seconds") or {}
    times = ", ".join(f"{k} {s[k]} s" for k in ("asr", "brain", "tts", "total") if s.get(k) is not None)
    heard = short(c["transcript"], 50) if c.get("transcript") else "(nothing heard)"
    return (f"{c.get('n', '?'):>2}. {kind} {question}{mark}: heard \"{heard}\" → {in_words(c.get('answer'))} → "
            f"reply \"{short(c.get('reply_text') or '(none)', 70)}\"" + (f" ({times})" if times else "")
            + (f" ERRORS: {'; '.join(c['errors'])}" if c.get("errors") else ""))


def summary_lines(results: dict) -> list[str]:
    """The run for a person: the models and licences, one line per clip, the synthetic score, the average times."""
    lines = [f"ASR (ears): {results.get('asr_model')} ({results.get('asr_licence')}) on {results.get('asr_device') or '?'}"]
    if results.get("tts_model"):
        flag = "" if results.get("tts_business_use") else " - NOT FOR BUSINESS USE: allowed for this internal test only"
        lines.append(f"TTS (mouth): {results['tts_model']} ({results.get('tts_licence')}){flag}")
    else:
        lines.append("TTS (mouth): none worked")
    for t in results.get("tts_tried") or []:
        if not t.get("ok"):
            lines.append(f"   TTS not used: {t.get('model')}: {short(t.get('why'), 200)}")
    lines.append(f"Brain: {results.get('brain')}; ran on {results.get('brain_processor') or '?'}")
    clips = results.get("clips") or []
    lines += [clip_line(c) for c in clips]
    synthetic = [c for c in clips if c.get("wrong") is not None]
    if synthetic:
        right = sum(not c["wrong"] for c in synthetic)
        lines.append(f"Synthetic answers: {right} of {len(synthetic)} with all 4 fields right "
                     "(a plumbing check with the TTS voice, not a real accent test)")
    avg = average_seconds(clips)
    if clips:
        lines.append(f"Average seconds over {len(clips)} clips: " + ", ".join(
            f"{k} {'?' if v is None else v}" for k, v in avg.items()))
    return lines


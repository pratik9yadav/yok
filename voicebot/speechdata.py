"""Real speech for testing and training the ears (speech to text), and how to score them.

    rows, source = fleurs("mr", 100, split="test")      [{"audio": 16 kHz samples, "text", "group"}], where from
    rows, source = indicvoices("hi", 2000, token)        many speakers and districts (gated: an HF token)
    error_rates([(said, heard), ...]) -> (word error rate %, character error rate %)

Every loader streams only the rows it needs and decodes the audio itself (no audio library in `datasets`).
"""

import contextlib
import io
import unicodedata

FLEURS = {"hi": "hi_in", "mr": "mr_in"}
TEXT_KEYS = ("raw_transcription", "transcription", "text", "sentence", "transcript", "verbatim", "normalized")
GROUP_KEYS = ("native_state", "state", "native_district", "district", "region")


def norm(t) -> str:
    """Text as compared: same Unicode form, no nukta, chandrabindu as anusvara, no punctuation, lowercase."""
    t = unicodedata.normalize("NFC", str(t or "")).lower().replace("़", "").replace("ँ", "ं")
    # punctuation and symbols out; vowel signs (Unicode marks, which \w does not cover) stay
    t = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in t)
    return " ".join(t.split())


def edits(a, b) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def error_rates(pairs) -> tuple[float, float]:
    """(word error rate %, character error rate %) over (reference, heard) pairs."""
    pairs = list(pairs)
    w = sum(edits(norm(r).split(), norm(h).split()) for r, h in pairs)
    c = sum(edits(norm(r).replace(" ", ""), norm(h).replace(" ", "")) for r, h in pairs)
    nw = sum(len(norm(r).split()) for r, _ in pairs) or 1
    nc = sum(len(norm(r).replace(" ", "")) for r, _ in pairs) or 1
    return round(100 * w / nw, 1), round(100 * c / nc, 1)


def decode(audio):
    """16 kHz mono float32 samples from a `datasets` audio cell (bytes, an array, or a path)."""
    import numpy as np
    import soundfile as sf

    from voicebot.audio import resample

    if isinstance(audio, dict) and audio.get("bytes"):
        data, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    elif isinstance(audio, dict) and audio.get("array") is not None:
        data, rate = np.asarray(audio["array"], dtype=np.float32)[:, None], audio["sampling_rate"]
    else:
        data, rate = sf.read(audio["path"], dtype="float32", always_2d=True)
    return resample(data.mean(axis=1), rate, 16000)


def rows_from(ds, n: int, text_keys=TEXT_KEYS, group_keys=GROUP_KEYS, skip: int = 0, max_seconds: float = 30.0):
    """Up to n rows from a streamed dataset (after skipping `skip`), clips over max_seconds left out."""
    from datasets import Audio

    columns = list(getattr(ds, "column_names", None) or next(iter(ds.take(1))).keys())
    audio_key = next(k for k in columns if k in ("audio", "audio_filepath", "file") or "audio" in k)
    text_key = next(k for k in text_keys if k in columns)
    group_key = next((k for k in group_keys if k in columns), None)
    with contextlib.suppress(Exception):  # already undecoded
        ds = ds.cast_column(audio_key, Audio(decode=False))
    out, seen = [], 0
    for row in ds:
        seen += 1
        if seen <= skip:
            continue
        text = str(row[text_key] or "").strip()
        if not text:
            continue
        audio = decode(row[audio_key])
        if len(audio) > max_seconds * 16000 or len(audio) < 8000:
            continue
        out.append({"audio": audio, "text": text, "group": str(row[group_key]) if group_key else None})
        if len(out) >= n:
            break
    return out


def fleurs(lang: str, n: int, split: str = "test"):
    """Google FLEURS read speech (CC-BY-4.0, not gated): (rows, where from)."""
    from datasets import load_dataset

    code = FLEURS[lang]
    errors = []
    for how in ("parquet", "script"):
        try:
            if how == "parquet":
                ds = load_dataset("parquet", split="train", streaming=True, data_files=(
                    f"hf://datasets/google/fleurs@refs/convert/parquet/{code}/{split}/*.parquet"))
            else:
                ds = load_dataset("google/fleurs", code, split=split, streaming=True, trust_remote_code=True)
            return rows_from(ds, n), f"google/fleurs {code} {split} ({how})"
        except Exception as e:  # noqa: BLE001 - try the other way
            errors.append(f"{how}: {e!r}"[:300])
    raise RuntimeError(" | ".join(errors))


def lahaja(n: int, token: str):
    """AI4Bharat Lahaja: Hindi from 132 speakers in 83 districts (CC-BY-4.0, gated: accept it on its page)."""
    from datasets import load_dataset

    ds = load_dataset("ai4bharat/Lahaja", split="test", streaming=True, token=token)
    return rows_from(ds, n), "ai4bharat/Lahaja test (Hindi from many districts)"


def indicvoices(lang: str, n: int, token: str):
    """AI4Bharat IndicVoices: natural speech from many districts (CC-BY-4.0, gated: accept it on its page)."""
    from datasets import load_dataset

    name = {"hi": "hindi", "mr": "marathi"}[lang]
    errors = []
    for config in (name, name.capitalize()):
        try:
            ds = load_dataset("ai4bharat/IndicVoices", config, split="train", streaming=True, token=token)
            return rows_from(ds, n), f"ai4bharat/IndicVoices {config} train"
        except Exception as e:  # noqa: BLE001
            errors.append(f"{config}: {e!r}"[:300])
    raise RuntimeError(" | ".join(errors))

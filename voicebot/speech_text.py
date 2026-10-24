"""Text for the voice, and clean-up of what speech to text heard.

    for_voice("16A MCB ₹220 में है", "hi") -> "सोलह ए एम सी बी दो सौ बीस रुपये में है"
    clean_heard("Thank you for watching!") -> ""          a known speech-to-text hallucination on silence

Voices read words, not digits or capital-letter short forms, and some read only one script; so digits, ₹, times
and short forms like MCB are spelled out in the reply's language first.
"""

import re
import unicodedata

from voicebot.numbers import digit_words, money_words, time_words, to_words

LETTERS_HI = {
    "A": "ए", "B": "बी", "C": "सी", "D": "डी", "E": "ई", "F": "एफ", "G": "जी", "H": "एच", "I": "आई", "J": "जे",
    "K": "के", "L": "एल", "M": "एम", "N": "एन", "O": "ओ", "P": "पी", "Q": "क्यू", "R": "आर", "S": "एस", "T": "टी",
    "U": "यू", "V": "वी", "W": "डब्ल्यू", "X": "एक्स", "Y": "वाई", "Z": "ज़ेड",
}
PHONE_LIKE = re.compile(r"(?<!\d)(?:\+91[\s-]?)?(\d{5})[\s-]?(\d{5})(?!\d)")
CLOCK = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)")
RUPEES = re.compile(r"₹\s?(\d[\d,]*(?:\.\d{1,2})?)(?:/-)?|(?<![\w.])(?:Rs\.?|INR)\s?(\d[\d,]*(?:\.\d{1,2})?)")
CODE = re.compile(r"(?<![\w])(\d+)([A-Za-z]{1,3})(?![\w])")  # 16A, 5kg, 32GB
ACRONYM = re.compile(r"(?<![A-Za-z])([A-Z]{2,6})(?![A-Za-z])")
NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")
UNIT_WORDS = {"kg": ("किलो", "किलो", "kilo"), "g": ("ग्राम", "ग्राम", "gram"), "gb": ("जीबी", "जीबी", "GB"),
              "ml": ("एमएल", "एमएल", "ml"), "mm": ("एमएम", "एमएम", "mm"), "w": ("वाट", "वॅट", "watt"),
              "l": ("लीटर", "लिटर", "litre"), "m": ("मीटर", "मीटर", "metre")}


def spell(letters: str, lang: str) -> str:
    """Capital letters one by one: "MCB" -> "एम सी बी" (English: "M C B")."""
    if lang == "en":
        return " ".join(letters.upper())
    return " ".join(LETTERS_HI.get(c, c) for c in letters.upper())


def _number(text: str) -> float:
    value = float(text.replace(",", ""))
    return int(value) if value.is_integer() else value


def for_voice(text: str, lang: str = "hi") -> str:
    """The text as the voice should say it, in lang ("hi", "mr" or "en"): phone numbers digit by digit, times, ₹
    prices, sizes like 16A, short forms like MCB and every other number in words. Symbols the voice can't say go."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)
    li = {"hi": 0, "mr": 1, "en": 2}.get(lang, 0)
    t = PHONE_LIKE.sub(lambda m: digit_words(m.group(1) + m.group(2), lang), t)
    t = CLOCK.sub(lambda m: time_words(f"{int(m.group(1)):02d}:{m.group(2)}", lang), t)
    t = RUPEES.sub(lambda m: money_words(_number(m.group(1) or m.group(2)), lang), t)

    def code(m):
        unit = UNIT_WORDS.get(m.group(2).lower())
        tail = unit[li] if unit and m.group(2).islower() else spell(m.group(2), lang)
        return f"{to_words(_number(m.group(1)), lang)} {tail}"

    t = CODE.sub(code, t)
    t = ACRONYM.sub(lambda m: spell(m.group(1), lang), t)
    t = NUMBER.sub(lambda m: to_words(_number(m.group(1)), lang), t)
    t = t.replace("%", " प्रतिशत" if lang != "en" else " percent").replace("&", " और " if lang == "hi" else " ")
    t = re.sub(r"[*_#~`|<>\[\]{}^=]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# What Whisper writes on silence or noise: never something a person said on the call.
HALLUCINATIONS = [
    "thank you for watching", "thanks for watching", "please subscribe", "subscribe to", "like and subscribe",
    "subtitles by", "amara.org", "transcribed by", "सब्सक्राइब", "उपशीर्षक", "चैनल को", "वीडियो देखने",
    "ご視聴", "字幕", "www.", ".com", "♪",
]
OTHER_SCRIPTS = re.compile(r"[؀-ۿ฀-๿぀-ヿ一-鿿가-힯Ѐ-ӿ]")


def clean_heard(text: str) -> str:
    """What speech to text heard, cleaned: "" when it is a known hallucination or mostly in another script (Arabic,
    Chinese, ... from noise); a phrase repeated over and over (another Whisper failure) is kept once."""
    t = " ".join((text or "").split())
    if not t:
        return ""
    low = t.lower()
    if any(h in low for h in HALLUCINATIONS):
        return ""
    letters = [c for c in t if c.isalpha()]
    if letters and len(OTHER_SCRIPTS.findall(t)) > len(letters) / 3:
        return ""
    words = t.split()
    for size in range(1, 6):  # "हाँ हाँ हाँ हाँ हाँ हाँ" -> "हाँ हाँ"; long loops of a phrase -> the phrase once
        out, i = [], 0
        while i < len(words):
            chunk = words[i:i + size]
            repeats = 1
            while words[i + repeats * size:i + (repeats + 1) * size] == chunk:
                repeats += 1
            if repeats >= 3:
                out += chunk * (2 if size == 1 else 1)
                i += repeats * size
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)

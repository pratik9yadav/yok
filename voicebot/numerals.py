"""Spoken numbers, prices, times, days and phone numbers in Hindi, Marathi and English, written the way speech to
text writes them (Devanagari or Latin letters, words or digits), turned into values; and values back into words for
the voice.

    find_numbers("ढाई सौ का है")                    -> [Span(value=250, ...)]
    money("सोलह ए वाला है, दो सौ बीस रुपये")          -> 220      (16 is skipped: "ए" (amp) follows it)
    money("rate two thirty hai")                     -> 230      (shop talk: "two thirty" is 230)
    time_of_day("शाम साढ़े चार बजे")                  -> "16:30"
    day("उद्या येतो")                                 -> "tomorrow"
    phone("नौ आठ सात छह पाँच, चार तीन दो एक शून्य")   -> "9876543210"
    to_words(220, "hi") -> "दो सौ बीस";  money_words(220, "mr") -> "दोनशे वीस रुपये";  time_words("16:30", "hi")

Fractions people say: डेढ़ 1.5, ढाई 2.5, सवा +¼, साढ़े +½, पौने −¼ (Marathi दीड, अडीच, सव्वा, साडे, पावणे), so
"साढ़े तीन सौ" is 350 and "पौने दो हज़ार" is 1,750. No imports beyond the standard library.
"""

import re
import unicodedata
from dataclasses import dataclass

HI_ONES = [  # 0 to 99
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
HI_MORE = {  # other spellings speech to text writes
    "छः": 6, "छै": 6, "छे": 6, "सतरह": 17, "अट्ठारह": 18, "उन्नीस": 19, "उन्निस": 19, "इकीस": 21, "बाइस": 22,
    "तेइस": 23, "चौबिस": 24, "पचीस": 25, "पच्चिस": 25, "छबीस": 26, "सताईस": 27, "अठाईस": 28, "अट्ठाइस": 28,
    "चौतीस": 34, "पैतीस": 35, "सैतीस": 37, "अडतीस": 38, "चालिस": 40, "बियालीस": 42, "चौवालीस": 44, "पैतालीस": 45,
    "पचास": 50, "इक्कावन": 51, "तिरेपन": 53, "चौअन": 54, "छप्पन": 56, "सत्तावन": 57, "अठावन": 58, "चौसठ": 64,
    "पैसठ": 65, "अड़सठ": 68, "उन्हत्तर": 69, "पिचहत्तर": 75, "पचत्तर": 75, "उन्यासी": 79, "अस्सि": 80, "नब्बें": 90,
    "पचानबे": 95, "निन्यानबे": 99, "निनानवे": 99,
}
MR_ONES = [  # 0 to 100
    "शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ", "दहा", "अकरा", "बारा", "तेरा", "चौदा",
    "पंधरा", "सोळा", "सतरा", "अठरा", "एकोणीस", "वीस", "एकवीस", "बावीस", "तेवीस", "चोवीस", "पंचवीस", "सव्वीस",
    "सत्तावीस", "अठ्ठावीस", "एकोणतीस", "तीस", "एकतीस", "बत्तीस", "तेहेतीस", "चौतीस", "पस्तीस", "छत्तीस", "सदतीस",
    "अडतीस", "एकोणचाळीस", "चाळीस", "एक्केचाळीस", "बेचाळीस", "त्रेचाळीस", "चव्वेचाळीस", "पंचेचाळीस", "सेहेचाळीस",
    "सत्तेचाळीस", "अठ्ठेचाळीस", "एकोणपन्नास", "पन्नास", "एक्कावन्न", "बावन्न", "त्रेपन्न", "चोपन्न", "पंचावन्न",
    "छप्पन्न", "सत्तावन्न", "अठ्ठावन्न", "एकोणसाठ", "साठ", "एकसष्ठ", "बासष्ठ", "त्रेसष्ठ", "चौसष्ठ", "पासष्ठ",
    "सहासष्ठ", "सदुसष्ठ", "अडुसष्ठ", "एकोणसत्तर", "सत्तर", "एक्काहत्तर", "बाहत्तर", "त्र्याहत्तर", "चौऱ्याहत्तर",
    "पंचाहत्तर", "शहात्तर", "सत्याहत्तर", "अठ्ठ्याहत्तर", "एकोणऐंशी", "ऐंशी", "एक्क्याऐंशी", "ब्याऐंशी", "त्र्याऐंशी",
    "चौऱ्याऐंशी", "पंच्याऐंशी", "शहाऐंशी", "सत्त्याऐंशी", "अठ्ठ्याऐंशी", "एकोणनव्वद", "नव्वद", "एक्क्याण्णव",
    "ब्याण्णव", "त्र्याण्णव", "चौऱ्याण्णव", "पंच्याण्णव", "शहाण्णव", "सत्त्याण्णव", "अठ्ठ्याण्णव", "नव्व्याण्णव", "शंभर",
]
MR_MORE = {"नउ": 9, "तेहतीस": 33, "पंच्याहत्तर": 75, "छप्पन": 56, "एक्याऐंशी": 81}
EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
           "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
           "eighty": 80, "ninety": 90}
EN_DEVANAGARI = {  # English number words as Hindi speech to text writes them
    "ज़ीरो": 0, "वन": 1, "टू": 2, "थ्री": 3, "फोर": 4, "फ़ोर": 4, "फाइव": 5, "फ़ाइव": 5, "सिक्स": 6, "सेवन": 7, "एट": 8,
    "नाइन": 9, "टेन": 10, "इलेवन": 11, "ट्वेल्व": 12, "थर्टीन": 13, "फोर्टीन": 14, "फिफ्टीन": 15, "सिक्सटीन": 16,
    "सेवेंटीन": 17, "एटीन": 18, "नाइनटीन": 19, "ट्वेंटी": 20, "थर्टी": 30, "फोर्टी": 40, "फ़ोर्टी": 40, "फिफ्टी": 50,
    "फ़िफ़्टी": 50, "सिक्सटी": 60, "सेवेंटी": 70, "एटी": 80, "नाइंटी": 90, "नाइनटी": 90,
}
LATIN_HI = {  # Hindi number words in Latin letters (Hinglish); compared by latin_key(), so spellings may vary
    0: "shunya", 1: "ek", 2: "do", 3: "teen tin", 4: "char chaar", 5: "paanch panch paach", 6: "chhe chhah che chah",
    7: "saat", 8: "aath", 9: "nau", 10: "das dus", 11: "gyarah gyara", 12: "barah bara", 13: "terah", 14: "chaudah",
    15: "pandrah", 16: "solah sola", 17: "satrah", 18: "atharah attharah", 19: "unnis unees", 20: "bees bis",
    21: "ikkis", 22: "bais baees", 23: "teis teees", 24: "chaubis chobis", 25: "pachchis pachis pacchis",
    26: "chhabbis chabbis", 27: "sattais", 28: "atthais athais", 29: "untis", 30: "tees", 31: "ikattis iktis",
    32: "battis", 33: "taintis tetis", 34: "chautis chontis", 35: "paintis pentis", 36: "chhattis chattis",
    37: "saintis sentis", 38: "adtis artis", 39: "untalis", 40: "chalis", 41: "iktalis", 42: "bayalis byalis",
    43: "taintalis tetalis", 44: "chavalis chawalis", 45: "paintalis pentalis", 46: "chhiyalis chiyalis",
    47: "saintalis sentalis", 48: "adtalis artalis", 49: "unchas", 50: "pachas", 51: "ikyavan ekyavan",
    52: "bavan", 53: "tirpan trepan", 54: "chauvan chovan", 55: "pachpan", 56: "chhappan chappan", 57: "sattavan",
    58: "atthavan athavan", 59: "unsath", 60: "saath", 61: "iksath", 62: "basath", 63: "tirsath", 64: "chausath chosath",
    65: "painsath pensath", 66: "chhiyasath", 67: "sadsath", 68: "adsath", 69: "unhattar", 70: "sattar",
    71: "ikhattar", 72: "bahattar", 73: "tihattar", 74: "chauhattar", 75: "pachhattar pachattar", 76: "chhihattar",
    77: "sathattar satattar", 78: "athhattar athattar", 79: "unasi", 80: "assi", 81: "ikyasi", 82: "bayasi",
    83: "tirasi", 84: "chaurasi", 85: "pachasi", 86: "chhiyasi", 87: "sattasi", 88: "atthasi athasi",
    89: "navasi nawasi", 90: "nabbe", 91: "ikyanve", 92: "banve", 93: "tiranve", 94: "chauranve", 95: "pachanve",
    96: "chhiyanve", 97: "sattanve", 98: "atthanve", 99: "ninyanve ninanve",
}
MULTIPLIERS = {  # word -> value
    "सौ": 100, "सैकड़ा": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 100_000, "करोड़": 10_000_000, "करोड": 10_000_000,
    "कोटी": 10_000_000, "हंड्रेड": 100, "थाउजेंड": 1000, "थाउज़ेंड": 1000, "थाउसेंड": 1000, "hundred": 100,
    "thousand": 1000, "lakh": 100_000, "lac": 100_000, "crore": 10_000_000, "sau": 100, "so": None,
    "hazaar": 1000, "hazar": 1000, "hajar": 1000, "hajaar": 1000, "karod": 10_000_000, "k": 1000,
}
MULTIPLIERS = {k: v for k, v in MULTIPLIERS.items() if v}  # "so" is English too: not a multiplier
FRACTIONS = {  # a whole value said in one word
    "डेढ़": 1.5, "डेढ": 1.5, "डेड़": 1.5, "ढाई": 2.5, "अढ़ाई": 2.5, "आधा": 0.5, "आधी": 0.5, "दीड": 1.5, "अडीच": 2.5,
    "अर्धा": 0.5, "dedh": 1.5, "derh": 1.5, "dhed": 1.5, "dhai": 2.5, "adhai": 2.5, "aadha": 0.5, "adha": 0.5,
    "deed": 1.5, "adich": 2.5,
}
PREFIXES = {  # added to the next value: साढ़े तीन = 3.5
    "सवा": 0.25, "साढ़े": 0.5, "साढे": 0.5, "साड़े": 0.5, "साडे": 0.5, "पौने": -0.25, "पौन": -0.25, "सव्वा": 0.25,
    "पावणे": -0.25, "पाउणे": -0.25, "sava": 0.25, "sawa": 0.25, "sadhe": 0.5, "saadhe": 0.5, "sade": 0.5,
    "saade": 0.5, "sarhe": 0.5, "paune": -0.25, "pone": -0.25, "paun": -0.25,
}
MONEY_AFTER = {"रुपये", "रुपए", "रूपये", "रूपए", "रुपया", "रुपय", "रुपयांना", "रुपयाला", "रुपयात", "रु", "रू", "₹",
               "rs", "rupees", "rupee", "rupaye", "rupay", "rupiya", "rupya", "inr", "/-", "का", "की", "के", "में",
               "ला", "ना", "मध्ये", "ka", "ki", "ke", "mein", "me", "padega", "पड़ेगा", "पडेगा", "पड़ेंगे", "होगा",
               "होंगे", "lagega", "लगेगा", "लगेंगे", "लागतील", "पडेल", "पडतील"}
MONEY_BEFORE = {"₹", "rs", "inr", "रु", "दाम", "रेट", "भाव", "कीमत", "क़ीमत", "किंमत", "दर", "price", "rate", "daam",
                "bhav", "keemat", "kimat", "cost", "mrp", "एमआरपी", "प्राइस", "चार्ज", "charge", "फीस", "fees", "fee",
                "बिल", "bill", "total", "टोटल"}
WEAK_AFTER = {"का", "की", "के", "में", "ला", "ना", "मध्ये", "ka", "ki", "ke", "mein", "me"}  # "ढाई सौ का"
UNITS = {  # a number right before one of these is a size or a count, not a price
    "ए", "a", "amp", "amps", "ampere", "एम्प", "एम्पियर", "वाट", "watt", "watts", "w", "किलो", "kg", "kilo", "ग्राम",
    "gram", "grams", "g", "लीटर", "litre", "liter", "ml", "एमएम", "mm", "इंच", "inch", "फुट", "feet", "foot", "मीटर",
    "meter", "metre", "नंबर", "number", "no", "साइज", "साइज़", "size", "साल", "वर्ष", "years", "year", "बजे", "बजकर",
    "baje", "वाजता", "वाजून", "दिन", "दिवस", "din", "days", "day", "घंटे", "घंटा", "तास", "ghante", "hours", "मिनट",
    "मिनिट", "minute", "minutes", "तारीख", "तारखेला", "%", "परसेंट", "percent", "पीस", "piece", "pieces", "pcs",
    "जीबी", "gb", "एमबी", "mb", "हफ्ते", "हफ़्ते", "आठवडे", "महीने", "महिने", "पैकेट", "packet", "डिब्बे", "बॉक्स",
    "box", "बोरी", "बैग", "bag", "बजता", "नग", "नग्स", "लोग", "जण", "लोक", "बार", "वेळा", "times", "टन", "ton",
    "वोल्ट", "volt", "v", "kw", "किलोवाट", "hp", "एचपी", "एमसीबी", "mcb", "सिलेंडर", "kmph", "km", "किमी", "किलोमीटर",
}
NUKTA, CANDRABINDU, ANUSVARA = "़", "ँ", "ं"
TOKEN = re.compile(r"₹|/-|\d[\d,]*(?:\.\d+)?|[^\s\d₹.,!?।|;:\"'()\[\]{}<>/\\-]+")
DIGITS = re.compile(r"^\d[\d,]*(?:\.\d+)?$")
MR_SUFFIXES = ("ाला", "ाना", "ाचा", "ाची", "ाचे", "ाने", "ात", "ला", "ना", "चा", "ची", "चे", "ने", "त", "ं", "ची")


def deva_key(word: str) -> str:
    """A Devanagari word without nukta, with chandrabindu as anusvara: "पाँच" and "पांच" match, "हज़ार" and "हजार"."""
    word = unicodedata.normalize("NFC", word)
    return word.replace(NUKTA, "").replace(CANDRABINDU, ANUSVARA).replace("ऩ", "न").replace("ड़", "ड").replace("ढ़", "ढ")


def latin_key(word: str) -> str:
    """A Latin word in a loose spelling: "paanch" and "panch", "pachchis" and "pachis", "hazaar" and "hajar" match."""
    word = word.lower()
    for a, b in (("aa", "a"), ("ee", "i"), ("oo", "u"), ("w", "v"), ("z", "j"), ("ph", "f"), ("q", "k")):
        word = word.replace(a, b)
    word = re.sub(r"(.)\1+", r"\1", word)  # no doubled letters
    word = re.sub(r"(?<=[aeiou])h$", "", word)  # "pandrah" -> "pandra"
    return word.replace("chh", "ch").replace("th", "t").replace("dh", "d")


def is_latin(word: str) -> bool:
    return bool(re.match(r"^[A-Za-z]+$", word))


def key(word: str) -> str:
    return latin_key(word) if is_latin(word) else deva_key(word)


def _table() -> dict[str, tuple[str, float]]:
    """Every number word (by key) -> (kind, value). kind: "val" (a value), "mul" (×100, ×1000, ...), "frac" (a whole
    fraction like डेढ़), "pre" (+/- added to the next value, like साढ़े)."""
    table: dict[str, tuple[str, float]] = {}

    def add(word, kind, value, keep=False):
        k = key(word)
        if keep and k in table:
            return
        table[k] = (kind, value)

    for n, w in enumerate(HI_ONES):
        add(w, "val", n)
    for w, n in HI_MORE.items():
        add(w, "val", n)
    for n, w in enumerate(MR_ONES):  # Marathi and Hindi share many words with the same value
        add(w, "val", n, keep=True)
        if "ष्ठ" in w:
            add(w.replace("ष्ठ", "ष्ट"), "val", n, keep=True)
    for w, n in MR_MORE.items():
        add(w, "val", n, keep=True)
    for n, w in enumerate(EN_ONES):
        add(w, "val", n)
    for w, n in {**EN_TENS, **EN_DEVANAGARI}.items():
        add(w, "val", n)
    for n, words in LATIN_HI.items():
        for w in words.split():
            add(w, "val", n, keep=True)
    for w, m in MULTIPLIERS.items():
        add(w, "mul", m)
    for w, v in FRACTIONS.items():
        add(w, "frac", v)
    for w, v in PREFIXES.items():
        add(w, "pre", v)
    return table


TABLE = _table()
MR_HUNDREDS = {key(w): n for n, w in enumerate(MR_ONES[1:10], 1)}  # दोनशे = 200: the stem before "शे"
PREFIX_KEYS = sorted((k for k, (kind, _) in TABLE.items() if kind == "pre" and not k.isascii()), key=len, reverse=True)


@dataclass
class Span:
    """One number found in a text: its value, its tokens [start, end), the words, and whether it is clearly a number
    (has digits, a multiplier, a fraction or several words, or is 10 or more) rather than e.g. "दो" in "बता दो"."""
    value: float
    start: int
    end: int
    text: str
    strong: bool


def tokens(text: str) -> list[str]:
    return TOKEN.findall(unicodedata.normalize("NFC", text or ""))


def classify(token: str) -> list[tuple[str, float]]:
    """A token as number parts: [] (not a number), [("val", 220)], [("val", 2), ("mul", 100)] for "दोनशे", ..."""
    if DIGITS.match(token):
        value = float(token.replace(",", ""))
        return [("digits", int(value) if value.is_integer() else value)]
    m = re.match(r"^(\d+(?:\.\d+)?)[kK]$", token)  # 2k
    if m:
        return [("digits", float(m.group(1))), ("mul", 1000)]
    k = key(token)
    if k in TABLE:
        return [TABLE[k]]
    if not is_latin(token):
        found = classify_key(k)
        if found:
            return found
        for pre in PREFIX_KEYS:  # joined: साडेतीनशे = साडे + तीनशे, सव्वादोनशे
            if k.startswith(pre) and len(k) > len(pre) + 1:
                rest = classify_key(k[len(pre):])
                if rest and rest[0][0] == "val":
                    return [TABLE[pre], *rest]
        for suffix in MR_SUFFIXES:  # Marathi joins endings on: शंभराला, दोनशेला, हजाराला
            if k.endswith(suffix) and len(k) > len(suffix) + 1:
                stem = k[: -len(suffix)]
                for candidate in (stem, stem + "र" if suffix.startswith("ा") else stem):
                    found = classify_key(candidate)
                    if found:
                        return found
    return []


def classify_key(k: str) -> list[tuple[str, float]]:
    if k in TABLE:
        return [TABLE[k]]
    if k.endswith("शे"):  # Marathi hundreds in one word: दोनशे 200, एकशे 100, दीडशे 150, अडीचशे 250
        stem = k[:-2]
        if stem in MR_HUNDREDS:
            return [("val", MR_HUNDREDS[stem]), ("mul", 100)]
        if TABLE.get(stem, ("",))[0] == "frac":
            return [("frac", TABLE[stem][1]), ("mul", 100)]
    return []


def whole(value: float) -> float:
    return int(value) if float(value).is_integer() else round(value, 2)


def find_numbers(text: str, colloquial: bool = False) -> list[Span]:
    """Every number in the text. colloquial: two values in a row like "दो बीस" or "two thirty" are one price (220,
    230), as shopkeepers say them; otherwise they are two numbers."""
    words = tokens(text)
    spans: list[Span] = []
    i = 0
    while i < len(words):
        parts = classify(words[i])
        if not parts or (parts == [parts[0]] and parts[0][0] == "pre"
                         and not (i + 1 < len(words) and classify(words[i + 1]))):
            i += 1
            continue
        start, total, current, raw, prefix, top, count, strong = i, 0.0, None, True, None, None, 0, False
        while i < len(words):
            parts = classify(words[i])
            if (not parts and words[i].lower() == "and" and current is not None and not raw and i + 1 < len(words)
                    and classify(words[i + 1])):
                i += 1  # "two hundred and twenty"
                continue
            if not parts:
                break
            stop = False
            for kind, v in parts:
                if kind == "pre":
                    if current is not None:
                        stop = True  # a fraction word starts a new number: "दो सौ, साढ़े तीन सौ"
                        break
                    prefix, strong = v, True
                elif kind in ("val", "digits", "frac"):
                    x = v + (prefix or 0)
                    if prefix is not None or kind != "val":
                        strong = True
                    if current is None:
                        current, raw = x, True
                    elif raw and current in EN_TENS.values() and 1 <= x <= 9:
                        current += x  # "twenty five"
                    elif raw and colloquial and 1 <= current <= 9 and float(current).is_integer() and 10 <= x <= 99:
                        current, raw, strong = current * 100 + x, False, True  # "दो बीस" = 220
                    elif not raw and top is not None and x < top and (current % top == 0 or (
                            current % 10 == 0 and current % 100 in EN_TENS.values() and 1 <= x <= 9)):
                        current += x  # "दो सौ बीस": 200 + 20; "two hundred twenty five"
                    else:
                        stop = True
                        break
                    prefix = None
                    count += 1
                elif kind == "mul":
                    strong = True
                    base = current if current is not None else 1 + (prefix or 0)
                    prefix = None
                    if v == 100:
                        if not raw and current is not None and current >= 100:
                            stop = True  # "दो सौ तीन सौ": two numbers
                            break
                        current, raw, top = base * 100, False, 100
                    else:
                        if top is not None and top >= v and current is None:
                            stop = True
                            break
                        total += base * v
                        current, raw, top = None, False, v
            if stop:
                break
            i += 1
        value = total + (current or 0)
        if i == start:  # nothing taken (e.g. a lone prefix)
            i += 1
            continue
        if total == 0 and current is None:
            continue
        strong = strong or count > 1 or value >= 10
        spans.append(Span(whole(value), start, i, " ".join(words[start:i]), strong))
    return spans


def _near(words: list[str], start: int, end: int, cues: set[str], before: int, after: int) -> bool:
    keys = {key(c) for c in cues}
    window = words[max(0, start - before):start] if before else []
    window += words[end:end + after] if after else []
    return any(key(w) in keys or w.lower() in cues for w in window)


def money(text: str, expected: bool = False) -> float | None:
    """The price said in the text (rupees), or None. Sizes and counts ("16A", "दो पीस", "5 किलो") are skipped. With
    expected (the price was just asked), a clear number counts alone; without, it needs a sign that it is money next
    to it ("रुपये", "₹", "दाम", "का", "पड़ेगा")."""
    words = tokens(text)
    spans = find_numbers(text, colloquial=True)
    units = {key(u) for u in UNITS}
    best, best_score = None, (0, False, 0)
    for s in spans:
        nxt = words[s.end] if s.end < len(words) else ""
        if key(nxt) in units or nxt.lower() in UNITS or not 0 < s.value < 10_000_000:
            continue
        cue = _near(words, s.start, s.end, MONEY_AFTER - WEAK_AFTER, 0, 2) or _near(words, s.start, s.end,
                                                                                   MONEY_BEFORE, 3, 0)
        score = 3 * cue + _near(words, s.start, s.end, WEAK_AFTER, 0, 1) + s.strong
        if expected and score == 0 and s.value >= 1 and len(spans) == 1:
            score = 1
        rank = (score, s.strong, s.value)
        if score and (expected or score >= 2) and (best is None or rank > best_score):
            best, best_score = s.value, rank
    return best


def annotate(text: str) -> str:
    """The text with each number said in words followed by its digits, to help a small model read it:
    "ढाई सौ का है" -> "ढाई सौ (=250) का है". Plain digits and lone weak words ("बता दो") are left as they are."""
    words = tokens(text)
    spans = [s for s in find_numbers(text, colloquial=True)
             if s.strong and not all(DIGITS.match(w) for w in words[s.start:s.end])]
    out = text
    for s in reversed(spans):
        pos = _find_end(out, words, s.end - 1)
        if pos is not None:
            out = out[:pos] + f" (={s.value})" + out[pos:]
    return out


def _find_end(text: str, words: list[str], index: int) -> int | None:
    """Where token number index ends in text."""
    pos = 0
    for n, w in enumerate(words):
        found = text.find(w, pos)
        if found < 0:
            return None
        pos = found + len(w)
        if n == index:
            return pos
    return None


# --- times and days ---------------------------------------------------------------------------------------------
CLOCK = {"बजे", "बजकर", "बज", "बजा", "बजेंगे", "baje", "bajkar", "वाजता", "वाजून", "वाजेपर्यंत", "वाजे", "vajta",
         "o'clock", "oclock", "am", "pm", "a.m", "p.m"}
MORNING = {"सुबह", "सवेरे", "सुबे", "सकाळी", "सकाळ", "subah", "subha", "savere", "sakali", "morning", "am"}
NOON = {"दोपहर", "दुपहर", "दुपारी", "दुपार", "dopahar", "dopehar", "dupari", "afternoon", "noon"}
EVENING = {"शाम", "सायं", "संध्याकाळी", "संध्याकाळ", "सायंकाळी", "shaam", "sham", "sandhyakali", "evening", "pm"}
NIGHT = {"रात", "रात्रि", "रात्री", "रात्र", "raat", "rat", "ratri", "night"}
DAYS = {
    "today": ["आज", "आजच", "aaj", "aj", "today"],
    "tomorrow": ["कल", "उद्या", "उद्याला", "kal", "kaal", "udya", "tomorrow", "tmrw"],
    "day_after": ["परसों", "परसो", "परवा", "परवाला", "parson", "parso", "parva", "parwa"],
    "monday": ["सोमवार", "सोमवारी", "somvar", "somwar", "monday", "मंडे"],
    "tuesday": ["मंगलवार", "मंगळवार", "मंगळवारी", "mangalvar", "mangalwar", "tuesday", "ट्यूज़डे", "ट्यूसडे"],
    "wednesday": ["बुधवार", "बुधवारी", "budhvar", "budhwar", "wednesday", "वेडनेसडे", "वेन्सडे"],
    "thursday": ["गुरुवार", "गुरूवार", "बृहस्पतिवार", "वीरवार", "गुरुवारी", "guruvar", "guruwar", "thursday", "थर्सडे"],
    "friday": ["शुक्रवार", "शुक्रवारी", "shukravar", "shukrawar", "friday", "फ्राइडे"],
    "saturday": ["शनिवार", "शनिवारी", "shanivar", "shaniwar", "saturday", "सैटरडे"],
    "sunday": ["रविवार", "इतवार", "रविवारी", "ravivar", "raviwar", "itvar", "sunday", "संडे"],
}
DAY_OF = {key(w): d for d, words in DAYS.items() for w in words}
DAY_AFTER_PHRASES = ["day after tomorrow", "day after"]


def time_of_day(text: str) -> str | None:
    """The clock time said in the text as "HH:MM" (24 h), or None: "शाम साढ़े चार बजे" -> "16:30", "10:30" -> "10:30",
    "सकाळी दहा वाजता" -> "10:00", "चार बजे" -> "16:00" (1 to 7 with no part of the day said: afternoon)."""
    t = unicodedata.normalize("NFC", text or "")
    words = tokens(t)
    lowered = [w.lower() for w in words]
    period = None
    for w in lowered:
        k = key(w)
        for name, group in (("morning", MORNING), ("noon", NOON), ("evening", EVENING), ("night", NIGHT)):
            if w in group or k in {key(g) for g in group}:
                period = name
    hour = minute = None
    clock = re.search(r"\b(\d{1,2})[:.](\d{2})\b", t)
    if clock and int(clock.group(1)) < 24 and int(clock.group(2)) < 60:
        hour, minute = int(clock.group(1)), int(clock.group(2))
    else:
        clock_keys = {key(c) for c in CLOCK}
        cue = next((i for i, w in enumerate(lowered) if w in CLOCK or key(w) in clock_keys), None)
        if cue is None:  # "पाँच बदे": a misheard बजे right after a number
            numbered = {s.end for s in find_numbers(t)}
            cue = next((i for i, w in enumerate(lowered) if i in numbered and not is_latin(w) and len(key(w)) == 3
                        and any(distance(key(w), c) <= 1 for c in ("बजे", "बजक"))), None)
        if cue is not None:
            spans = [s for s in find_numbers(t) if s.end <= cue]
            if spans and spans[-1].end >= cue - 1:
                value = float(spans[-1].value)
                hour, minute = int(value), round((value - int(value)) * 60)
                after = [s for s in find_numbers(t) if s.start > cue]  # "चार बजकर बीस मिनट"
                if after and minute == 0 and after[0].start <= cue + 2 and 0 < after[0].value < 60:
                    minute = int(after[0].value)
        elif period:  # "शाम पाँच" with no बजे: the number right after the part of the day
            for i, w in enumerate(lowered):
                if any(w in g or key(w) in {key(x) for x in g} for g in (MORNING, NOON, EVENING, NIGHT)):
                    nums = [s for s in find_numbers(t) if s.start == i + 1 or (s.start == i + 2)]
                    if nums and 0 < nums[0].value <= 12:
                        value = float(nums[0].value)
                        hour, minute = int(value), round((value - int(value)) * 60)
    if hour is None or not (0 <= hour <= 24) or not (0 <= (minute or 0) < 60):
        return None
    minute = minute or 0
    if hour <= 12:
        if (period in ("evening", "night") and hour < 12) or (period == "noon" and 1 <= hour <= 6):
            hour += 12
        elif period == "night" and hour == 12:
            hour = 0
        elif period is None and 1 <= hour <= 7:
            hour += 12
    hour %= 24
    return f"{hour:02d}:{minute:02d}"


def distance(a: str, b: str) -> int:
    """Edit distance (letters added, dropped or changed)."""
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def near(word: str, choices: dict[str, str]) -> str | None:
    """The value of the choice a misheard word is closest to ("संवार" -> सोमवार, "जविवार" -> रविवार): at most one
    letter off for 4 letters, two for 5 or more, and only when one value is closest; else None."""
    k = key(word)
    if len(k) < 4:
        return None
    limit = 1 if len(k) < 5 else 2
    scored = [(distance(k, c), v) for c, v in choices.items() if len(c) >= 4 and abs(len(c) - len(k)) <= limit]
    scored = [(d, v) for d, v in scored if d <= limit]
    if not scored:
        return None
    best = min(d for d, _ in scored)
    found = {v for d, v in scored if d == best}
    return found.pop() if len(found) == 1 else None


def day(text: str, fuzzy: bool = False) -> str | None:
    """The day said in the text: "today", "tomorrow", "day_after" or a weekday ("monday" ...), or None. "कल" is
    taken as tomorrow (a booking talks about the future). fuzzy (the day was just asked): a weekday misheard by
    speech to text by a letter or two counts too."""
    low = (text or "").lower()
    if any(p in low for p in DAY_AFTER_PHRASES):
        return "day_after"
    found = None
    weekdays = {k: d for k, d in DAY_OF.items() if d not in ("today", "tomorrow", "day_after")}
    for w in tokens(text):
        d = DAY_OF.get(key(w)) or DAY_OF.get(w.lower()) or (near(w, weekdays) if fuzzy and not is_latin(w) else None)
        if d:
            found = d if found in (None, "today") or d not in ("today",) else found
    return found


# --- phone numbers ----------------------------------------------------------------------------------------------
REPEAT = {"डबल": 2, "double": 2, "दुबल": 2, "ट्रिपल": 3, "triple": 3, "ट्रिप्पल": 3}
DIGIT_WORDS = {"oh": 0, "o": 0, "जीरो": 0, "झिरो": 0, "शुन्य": 0}


def phone(text: str) -> str | None:
    """A 10-digit Indian mobile number said in the text (digits, digit words, or pairs like "अट्ठानवे"), or None.
    "+91" and a leading 0 are dropped; "डबल सात" is 77."""
    digits = ""
    repeat = 1
    for w in tokens(text):
        lw = w.lower()
        if key(lw) in {key(r) for r in REPEAT} or lw in REPEAT:
            repeat = REPEAT.get(lw) or next(v for r, v in REPEAT.items() if key(r) == key(lw))
            continue
        piece = None
        if DIGITS.match(w) and "." not in w:
            piece = w.replace(",", "")
        elif lw in DIGIT_WORDS or key(lw) in {key(d) for d in DIGIT_WORDS}:
            piece = "0"
        else:
            parts = classify(w)
            if len(parts) == 1 and parts[0][0] == "val" and float(parts[0][1]).is_integer() and parts[0][1] < 100:
                piece = str(int(parts[0][1]))
        if piece is None:
            repeat = 1
            continue
        digits += piece * repeat if len(piece) == 1 else piece
        repeat = 1
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    return digits if re.fullmatch(r"[6-9]\d{9}", digits) else None


# --- values back to words, for the voice -----------------------------------------------------------------------
MR_HUNDRED_WORDS = {1: "एकशे", 2: "दोनशे", 3: "तीनशे", 4: "चारशे", 5: "पाचशे", 6: "सहाशे", 7: "सातशे", 8: "आठशे",
                    9: "नऊशे"}

def _en_words(n: int) -> str:
    if n < 20:
        return EN_ONES[n]
    if n < 100:
        tens = next(w for w, v in EN_TENS.items() if v == n // 10 * 10 and w != "fourty")
        return tens + (f" {EN_ONES[n % 10]}" if n % 10 else "")
    for size, name in ((10_000_000, "crore"), (100_000, "lakh"), (1000, "thousand"), (100, "hundred")):
        if n >= size:
            rest = n % size
            return f"{_en_words(n // size)} {name}" + (f" {_en_words(rest)}" if rest else "")
    return str(n)


def to_words(n, lang: str = "hi") -> str:
    """A whole number in words: to_words(220, "hi") -> "दो सौ बीस", (220, "mr") -> "दोनशे वीस", (220, "en") ->
    "two hundred twenty". Halves and quarters of small numbers use साढ़े/सवा/पौने. Digits from 100 crore up."""
    value = float(n)
    if value < 0 or value >= 1_000_000_000:
        return str(n)
    if not value.is_integer():
        base, frac = int(value), round(value - int(value), 2)
        words = {0.5: ("साढ़े", "साडे", "and a half"), 0.25: ("सवा", "सव्वा", "and a quarter")}
        if lang in ("hi", "mr") and frac in words and base >= 3:
            return f"{words[frac][0 if lang == 'hi' else 1]} {to_words(base, lang)}"
        if lang in ("hi", "mr") and value in (1.5, 2.5):
            return {(1.5, "hi"): "डेढ़", (2.5, "hi"): "ढाई", (1.5, "mr"): "दीड", (2.5, "mr"): "अडीच"}[(value, lang)]
        return str(value)
    n = int(value)
    if lang == "en":
        return _en_words(n)
    ones = MR_ONES if lang == "mr" else HI_ONES
    if n < 100:
        return ones[n]
    words = []
    names = (("कोटी", "हजार", "लाख") if lang == "mr" else ("करोड़", "हज़ार", "लाख"))
    for size, name in ((10_000_000, names[0]), (100_000, names[2]), (1000, names[1])):
        if n >= size:
            words.append(f"{to_words(n // size, lang)} {name}")
            n %= size
    if n >= 100:
        words.append(MR_HUNDRED_WORDS[n // 100] if lang == "mr" else f"{ones[n // 100]} सौ")
        n %= 100
    if n:
        words.append(ones[n])
    return " ".join(words)


def money_words(price, lang: str = "hi") -> str:
    """A price as the voice says it: 220 -> "दो सौ बीस रुपये" (mr "दोनशे वीस रुपये", en "two hundred twenty rupees");
    paise too: 12.5 -> "बारह रुपये पचास पैसे"."""
    value = float(price)
    rupees, paise = int(value), round((value - int(value)) * 100)
    unit = {"hi": ("रुपये", "पैसे"), "mr": ("रुपये", "पैसे"), "en": ("rupees", "paise")}.get(lang, ("रुपये", "पैसे"))
    said = f"{to_words(rupees, lang)} {unit[0]}"
    return said + (f" {to_words(paise, lang)} {unit[1]}" if paise else "")


def time_words(hhmm: str, lang: str = "hi") -> str:
    """"16:30" as the voice says it: hi "शाम साढ़े चार बजे", mr "संध्याकाळी साडेचार वाजता", en "4:30 in the evening"."""
    hour, minute = (int(x) for x in hhmm.split(":"))
    h12 = hour % 12 or 12
    part = ("morning" if 4 <= hour < 12 else "noon" if 12 <= hour < 16 else "evening" if 16 <= hour < 20 else "night")
    if lang == "en":
        when = {"morning": "in the morning", "noon": "in the afternoon", "evening": "in the evening",
                "night": "at night"}[part]
        said = (f"{_en_words(h12)} o'clock" if not minute else
                f"{_en_words(h12)} {'oh ' if minute < 10 else ''}{_en_words(minute)}")
        return f"{said} {when}"
    mr = lang == "mr"
    period = {"morning": ("सुबह", "सकाळी"), "noon": ("दोपहर", "दुपारी"), "evening": ("शाम", "संध्याकाळी"),
              "night": ("रात", "रात्री")}[part][mr]
    nxt = h12 % 12 + 1
    o_clock = "वाजता" if mr else "बजे"
    if minute == 0:
        said = f"{to_words(h12, lang)} {o_clock}"
    elif minute == 30:
        said = (("दीड" if mr else "डेढ़") if h12 == 1 else ("अडीच" if mr else "ढाई") if h12 == 2
                else f"{'साडे' if mr else 'साढ़े'} {to_words(h12, lang)}") + f" {o_clock}"
    elif minute == 15:
        said = f"{'सव्वा' if mr else 'सवा'} {to_words(h12, lang)} {o_clock}"
    elif minute == 45:
        said = f"{'पावणे' if mr else 'पौने'} {to_words(nxt, lang)} {o_clock}"
    else:
        said = (f"{to_words(h12, lang)} वाजून {to_words(minute, lang)} मिनिटांनी" if mr
                else f"{to_words(h12, lang)} बजकर {to_words(minute, lang)} मिनट")
    return f"{period} {said}"


DAY_WORDS = {
    "today": ("आज", "आज", "today"), "tomorrow": ("कल", "उद्या", "tomorrow"),
    "day_after": ("परसों", "परवा", "the day after tomorrow"), "monday": ("सोमवार", "सोमवारी", "Monday"),
    "tuesday": ("मंगलवार", "मंगळवारी", "Tuesday"), "wednesday": ("बुधवार", "बुधवारी", "Wednesday"),
    "thursday": ("गुरुवार", "गुरुवारी", "Thursday"), "friday": ("शुक्रवार", "शुक्रवारी", "Friday"),
    "saturday": ("शनिवार", "शनिवारी", "Saturday"), "sunday": ("रविवार", "रविवारी", "Sunday"),
}


def day_words(d: str, lang: str = "hi") -> str:
    return DAY_WORDS.get(d, (d, d, d))[{"hi": 0, "mr": 1, "en": 2}.get(lang, 0)]


def digit_words(digits: str, lang: str = "hi") -> str:
    """A phone number read digit by digit, in two groups: "9876543210" -> "नौ आठ सात छह पाँच, चार तीन दो एक शून्य"."""
    ones = MR_ONES if lang == "mr" else EN_ONES if lang == "en" else HI_ONES
    said = [ones[int(c)] for c in digits if c.isdigit()]
    half = (len(said) + 1) // 2
    return " ".join(said[:half]) + (", " + " ".join(said[half:]) if said[half:] else "")

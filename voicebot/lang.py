"""Which language a line is in: Hindi, Marathi or English (Devanagari or Latin letters). Hinglish counts as Hindi.

    detect("हो, आहे ना, दोनशे रुपये") -> ("mr", 3)        the language and how many marker words were found
    detect("haan bhai hai, do sau ka") -> ("hi", 3)
    script("नमस्ते") -> "deva"

Speech to text writes Marathi and Hindi in the same letters, so the words decide: आहे, नाही, होय, मी are Marathi;
है, नहीं, हाँ, मैं are Hindi. The letter ळ is almost only Marathi.
"""

import re
import unicodedata

MARKERS = {
    "mr": """आहे आहेत आहोत आहात नाही नाहीये नाहीत होय काय मी तुम्ही आम्ही आपण पाहिजे नको कधी कुठे किती देतो देते देऊ
             करतो करते मिळेल मिळणार मिळतील होते होता होती उद्या परवा सकाळी संध्याकाळी दुपारी वाजता बरं बरोबर झालं
             संपलं संपला सांगा बोला येईल जाईल लागेल पडेल माझं माझा माझी तुमचं तुमचा तुमची आमचं आमचा आमची हवं हवा
             हवी नाव फोन करा द्या घ्या ठेवतो ठेवा आता इथे तिथे म्हणजे म्हणून आणि पण दोन पाच सहा नऊ दहा शंभर दोनशे
             पाचशे रुपयांना करू नंतर थांबा कोण बोलतंय बोलताय चुकीचा ऐकू येतंय कळलं समजलं""",
    "hi": """है हैं हूँ हूं नहीं नही हाँ हां क्या कितना कितने कितनी आप मैं हम था थी थे होगा होगी जाएगा मिलेगा
             मिलेंगे चाहिए भाई भैया की के में से को पर वाला वाली रहा रही रहे देंगे करेंगे कर देंगे परसों बजे
             ठीक लेकिन और अभी यहाँ वहाँ कौन बोल रहे सुनिए बताइए बताओ दीजिए कीजिए हमारे आपका आपकी मेरा मेरी नाम
             दो पाँच पांच छह नौ दस सौ दुकान बाद मत गलत समझ""",
    "en": """is are the yes no we have it available price deliver delivery what how much not sorry okay ok please
             thank thanks you your our this that there here sir madam stock rupees tomorrow today call later number
             wrong who speaking hello hi send home yeah nope don't""",
}
LATIN_MARKERS = {
    "mr": "aahe ahe ahet nahiye hoy kay mi tumhi amhi apan pahije nako kadhi kuthe kiti deto dete karto karte milel "
          "milnar hota hoti udya parva sakali sandhyakali vajta bara barobar zala sampla sanga bola yeil lagel padel "
          "majha maza tumcha amcha hava nantar thamba kon boltay",
    "hi": "hai hain hoon nahi nahin haan han kya kitna kitne aap main hum tha thi hoga milega chahiye bhai bhaiya ka "
          "ki ke mein se ko par wala wali raha rahi denge karenge kal parson baje theek thik lekin aur abhi kaun "
          "boliye bataiye bata dijiye hamare aapka mera naam sau rupaye rupay dukaan baad mat galat samajh",
}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[ऀ-ॿ]+|[A-Za-z']+")


def _keys(text: str) -> set[str]:
    return set(text.split())


DEVA_SETS = {lang: _keys(unicodedata.normalize("NFC", words)) for lang, words in MARKERS.items() if lang != "en"}
EN_SET = _keys(MARKERS["en"])
LATIN_SETS = {lang: _keys(words) for lang, words in LATIN_MARKERS.items()}


def script(text: str) -> str:
    """"deva", "latin", "mixed" or "none" (no letters)."""
    deva, latin = len(DEVANAGARI.findall(text or "")), len(LATIN.findall(text or ""))
    if not deva and not latin:
        return "none"
    if deva and latin:
        return "deva" if deva >= 3 * latin else "latin" if latin >= 3 * deva else "mixed"
    return "deva" if deva else "latin"


def detect(text: str, default: str = "hi") -> tuple[str, int]:
    """The language of the line and how many marker words decided it (0: none found, the default is returned).
    Hinglish (Hindi words in Latin letters, or mixed with English) is "hi"."""
    words = [w.lower() for w in WORD.findall(unicodedata.normalize("NFC", text or ""))]
    if not words:
        return default, 0
    score = {"hi": 0, "mr": 0, "en": 0}
    for w in words:
        if DEVANAGARI.match(w):
            for lang, markers in DEVA_SETS.items():
                score[lang] += w in markers
            if "ळ" in w:
                score["mr"] += 1
        else:
            for lang, markers in LATIN_SETS.items():
                score[lang] += w in markers
            score["en"] += w in EN_SET
    # Hindi in Latin letters often mixes in English words: it is Hinglish when any Hindi marker is there
    if score["en"] and score["hi"] and score["hi"] >= score["mr"]:
        score["hi"] += score["en"]
        score["en"] = 0
    best = max(score, key=lambda k: (score[k], k == default))
    if score[best] == 0:
        return default, 0
    if score["hi"] == score["mr"] and best in ("hi", "mr"):
        return (default if default in ("hi", "mr") else "hi"), score[best]
    return best, score[best]

"""Understanding one thing the other person said: what they want (the intent) and which slot values they gave.

Two ways, used together:
- rules: fast and sure for short replies ("हाँ", "नाही", "ढाई सौ", "कल शाम पाँच बजे") and for "don't call me again";
- a small language model (any function prompt, schema -> JSON text, e.g. Ollama) for everything longer.

    u = understand("हाँ है, ढाई सौ का", task, View(pending="in_stock"), llm)
    u.intent == "answer";  u.values == {"in_stock": True, "price_inr": 250}

Every value the model gives is checked against its type (a price must be a number, a phone 10 digits, a day one of
the known days) and prices are checked against what the number reader found in the words, so a model's slip on
"ढाई सौ" (250) can't reach the result. With no model, or when it fails, the rules answer alone.
"""

import json
import re
from dataclasses import dataclass, field

from voicebot import numerals
from voicebot.tasks import DAY_VALUES, Task

INTENTS = {
    "answer": "gives information or answers the assistant's question (a bare yes or no too)",
    "confirm_yes": "agrees that the summary the assistant read back is right",
    "confirm_no": "says the summary the assistant read back is wrong",
    "dont_know": "does not know the answer",
    "repeat": "did not hear or understand, asks to say it again",
    "who_are_you": "asks who is calling, from where, or why",
    "hold": "asks to wait a moment",
    "busy": "is busy now and asks to call later",
    "stop_calling": "asks not to be called again, or is not interested",
    "wrong_number": "says this is the wrong number, shop or person",
    "human": "wants to talk to a real person",
    "question": "asks a question of their own",
    "bye": "ends the call",
    "other": "anything else: a greeting, small talk, unclear",
}

YES = {"हाँ", "हां", "हा", "जी", "हांजी", "हाँजी", "बिल्कुल", "बिलकुल", "ज़रूर", "जरूर", "सही", "yes", "yeah", "yep", "yup",
       "sure", "ok", "okay", "ओके", "ओक", "haan", "han", "haa", "ha", "ji", "bilkul", "zaroor", "jaroor", "sahi", "हो",
       "होय", "बरोबर", "नक्की", "correct", "right", "करेक्ट", "राइट", "है", "हैं", "आहे", "आहेत", "hai", "hain", "aahe",
       "ahe", "hoy", "ho", "available", "अवेलेबल", "मिलेगा", "मिळेल", "milega", "milel", "यस", "येस", "ठीक", "theek",
       "thik", "बराबर", "हम्म", "hmm", "हाँहाँ", "करते", "करता", "देते", "देता", "देतो", "करतो", "देंगे", "denge",
       "karte", "kar", "dete", "कर", "दे", "देऊ", "करू", "चलेगा", "chalega", "चालेल"}
NO = {"नहीं", "नही", "ना", "न", "नाही", "नाहीये", "नाहीत", "नको", "no", "nope", "nahi", "nahin", "nai", "naa", "नई",
      "नहीँ", "गलत", "ग़लत", "चुकीचं", "चुकीचे", "चूक", "wrong", "खत्म", "ख़त्म", "संपलं", "संपला", "संपले", "आउट",
      "galat", "nako", "nahiye", "नॉट", "not", "khatam", "sampla"}
FILLER = {"जी", "भाई", "भैया", "साहब", "सर", "मैडम", "तो", "अरे", "हाँ", "बस", "ji", "bhai", "bhaiya", "sir", "madam", "toh",
          "to", "arre", "are", "please", "प्लीज़", "प्लीज", "बरं", "ना", "न", "है", "हैं", "आहे", "का", "की", "वो", "वाला",
          "बोलिए", "बोलो", "बताइए", "बताइये", "बताओ", "हैलो", "हेलो", "hello", "बोला", "सांगा", "हां", "हाँजी", "bolo",
          "boliye", "batao", "bataiye", "हम्म", "आ", "गया", "गई", "आलो", "आले"}
DONT_KNOW = ["पता नहीं", "पता नही", "नहीं पता", "नही पता", "मालूम नहीं", "मालूम नही", "माहीत नाही", "माहिती नाही",
             "माहित नाही", "don't know", "dont know", "do not know", "not sure", "pata nahi", "pata nahin", "malum nahi",
             "maahit nahi", "mahit nahi", "kya pata", "क्या पता", "आइडिया नहीं", "idea nahi", "no idea"]
INTENT_PHRASES = {  # checked in this order
    "stop_calling": ["कॉल मत", "फोन मत", "फ़ोन मत", "मत करना कॉल", "दोबारा मत", "दुबारा मत", "परेशान मत", "परेशान कर",
                     "don't call", "dont call", "do not call", "stop calling", "not interested", "इंटरेस्टेड नहीं",
                     "इंटरेस्ट नहीं", "interested nahi", "फोन करू नका", "कॉल करू नका", "फोन नका करू", "call mat",
                     "phone mat", "dobara call mat", "मत करो कॉल", "कॉल नहीं करना", "रुचि नहीं"],
    "wrong_number": ["रॉन्ग नंबर", "रॉंग नंबर", "राँग नंबर", "गलत नंबर", "ग़लत नंबर", "wrong number", "चुकीचा नंबर",
                     "galat number", "rong number", "गलत जगह", "ये दुकान नहीं", "यह दुकान नहीं"],
    "human": ["इंसान से", "किसी आदमी से", "मैनेजर से", "असली आदमी", "real person", "human", "किसी व्यक्ति से", "माणसाशी",
              "operator", "ऑपरेटर", "agent se", "एजेंट से", "मालिक से बात", "आदमी से बात", "person se baat"],
    "repeat": ["फिर से बोल", "फिर से बता", "फिर से कह", "दोबारा बोल", "दोबारा बता", "दुबारा बोल", "क्या बोला", "क्या बोले",
               "क्या कहा", "सुनाई नहीं", "समझ नहीं आया", "समझा नहीं", "समझी नहीं", "पुन्हा सांगा", "परत सांगा",
               "ऐकू आलं नाही", "ऐकू नाही", "कळलं नाही", "समजलं नाही", "repeat", "say again", "come again", "pardon",
               "kya bola", "phir se", "dobara bol", "samajh nahi", "sunai nahi", "क्या?"],
    "who_are_you": ["कौन बोल", "कौन है", "कौन हो", "आप कौन", "कोण बोलत", "कोण आहे", "तुम्ही कोण", "who is this",
                    "who are you", "who's calling", "who is calling", "kaun bol", "aap kaun", "कहाँ से बोल", "कहां से बोल",
                    "कुठून बोलत", "calling from where", "कहाँ से", "कौन सी कंपनी"],
    "hold": ["एक मिनट", "एक मिनिट", "रुको", "रुकिए", "रुकिये", "होल्ड", "थांबा", "थांब", "wait", "hold on", "one minute",
             "ruko", "ek minute", "ek min", "एक सेकंड", "एक सेकेंड", "जरा रुक", "ज़रा रुक", "दो मिनट"],
    "busy": ["बिज़ी", "बिजी", "व्यस्त", "बाद में कॉल", "बाद में फोन", "बाद में बात", "नंतर फोन", "नंतर कॉल", "busy",
             "call later", "baad mein call", "baad me call", "अभी टाइम नहीं", "अभी समय नहीं", "आत्ता वेळ नाही",
             "वेळ नाही", "time nahi", "बाद में करना", "later"],
    "bye": ["बाय", "bye", "रखता हूँ", "रखती हूँ", "रखता हूं", "ठेवतो", "ठेवते", "चलो रखो", "फोन रखो", "good bye", "goodbye"],
}
STRONG_INTENTS = ("stop_calling", "wrong_number", "human")  # a clear phrase wins over the model
INTENT_MAX_WORDS = {"busy": 10, "who_are_you": 10, "repeat": 8, "hold": 8, "bye": 6}  # longer: likely more than that
QUESTION_WORDS = {"क्या", "कब", "कहाँ", "कहां", "कितना", "कितनी", "कितने", "कैसे", "क्यों", "कौन", "काय", "कधी", "कुठे",
                  "किती", "कसे", "कसं", "का?", "what", "when", "where", "how", "why", "who", "kab", "kahan", "kaise", "kyun",
                  "kitna"}
NAME_DROP = {"मेरा", "मेरी", "नाम", "है", "हैं", "जी", "माझं", "माझे", "माझा", "नाव", "आहे", "my", "name", "is", "i'm",
             "am", "i", "mera", "naam", "hai", "patient", "मरीज़", "मरीज", "का", "की", "रुग्णाचं", "रुग्णाचे", "पेशेंट",
             "this", "it's", "its", "the", "हाँ", "haan", "ji", "मैं", "main", "बोल", "रहा", "रही", "हूँ", "हूं", "बोलतोय",
             "बोलतेय", "मी", "मिस्टर", "mr", "mrs", "श्री", "श्रीमती"}
WORD = re.compile(r"[ऀ-ॿ]+|[A-Za-z']+|\d+|\?")


@dataclass
class View:
    """What the understanding needs to know about the call so far."""
    pending: str | None = None  # the slot the assistant just asked for, "confirm" (after the read-back), or None
    last_bot: str = ""  # what the assistant just said
    values: dict = field(default_factory=dict)  # the slot values found so far


@dataclass
class Understood:
    intent: str = "other"
    values: dict = field(default_factory=dict)  # only the slots this turn gave
    fact: str | None = None  # the fact asked about (a key of the task's facts)
    said_no: bool = False  # a no in the reply (for a text slot: "nothing else")
    source: str = "rules"  # "rules", "rules (fast)", "model", "model+rules"
    fast: bool = False  # sure enough from the rules alone: the model is not asked
    raw: str | None = None  # the model's reply
    error: str | None = None  # why the model's reply was not used

    def as_dict(self) -> dict:
        return {"intent": self.intent, "values": self.values, "fact": self.fact, "source": self.source,
                "error": self.error}


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text or "")]


def has(text: str, phrases) -> bool:
    low = " ".join((text or "").lower().split())
    return any(p in low for p in phrases)


def yes_no(text: str) -> bool | None:
    """True for a yes, False for a no (a no anywhere wins: "हाँ जी, नहीं है" is a no), None when neither."""
    ws = words(text)
    if has(text, DONT_KNOW):
        return None
    tag = {"ना", "न", "naa", "na"}  # "है ना" is a yes: these are a no only first or alone
    if any(w in NO and (w not in tag or i == 0) for i, w in enumerate(ws)) or has(
            text, ["not in stock", "out of stock", "नहीं है", "नाही आहे"]):
        return False
    if any(w in YES for w in ws):
        return True
    return None


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in FILLER and w not in YES and w not in NO and w != "?"]


# --- checking values ---------------------------------------------------------------------------------------------
def check(slot: dict, value):
    """The value in its proper form for the slot's type, or None when it is not a valid value of that type."""
    kind = slot["type"]
    if value is None or value == "" or (isinstance(value, str) and value.strip().lower() in ("null", "none", "n/a")):
        return None
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "yes", "false", "no"):
            return value.strip().lower() in ("true", "yes")
        return None
    if kind in ("money", "number"):
        if isinstance(value, bool):
            return None
        if isinstance(value, str):
            found = re.sub(r"[₹,\s]|rs\.?|/-", "", value.lower())
            try:
                value = float(found)
            except ValueError:
                return None
        if not isinstance(value, (int, float)) or value != value:  # noqa: PLR0124 - NaN
            return None
        if kind == "money" and not 0 < value < 10_000_000:
            return None
        return int(value) if float(value).is_integer() else round(float(value), 2)
    if kind == "phone":
        digits = re.sub(r"\D", "", str(value))
        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]
        if digits.startswith("0") and len(digits) == 11:
            digits = digits[1:]
        return digits if re.fullmatch(r"[6-9]\d{9}", digits) else None
    if kind == "day":
        v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
        v = {"day_after_tomorrow": "day_after", "dayafter": "day_after"}.get(v, v)
        return v if v in DAY_VALUES else numerals.day(str(value))
    if kind == "time":
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value < 24:
            return f"{int(value):02d}:{round((value - int(value)) * 60):02d}"
        m = re.fullmatch(r"\s*(\d{1,2})(?:[:.](\d{2}))?\s*", str(value))
        if m and int(m.group(1)) < 24 and int(m.group(2) or 0) < 60:
            return f"{int(m.group(1)):02d}:{int(m.group(2) or 0):02d}"
        return numerals.time_of_day(str(value))
    if kind == "enum":
        v = str(value).strip().lower()
        options = slot["options"]
        if v in options:
            return v
        for key, option in options.items():
            names = [str(option.get(lang, "")).lower() for lang in ("hi", "mr", "en")] + [
                str(w).lower() for w in option.get("words", [])]
            if v in names:
                return key
        return None
    # text and name
    v = " ".join(str(value).split()).strip(" .,!?।\"'")
    limit = 40 if kind == "name" else 80
    return v[:limit] if v else None


def enum_from_words(slot: dict, text: str):
    ws = set(words(text))
    low = (text or "").lower()
    found = [key for key, option in slot["options"].items()
             if any(str(w).lower() in ws or (" " in str(w) and str(w).lower() in low)
                    for w in [option.get("hi"), option.get("mr"), option.get("en"), *option.get("words", [])] if w)]
    return found[0] if len(found) == 1 else None


def trimmed(text: str) -> str:
    """A text value without the filler around it: "हाँ, हैवेल्स का बत्तीस ए है" -> "हैवेल्स का बत्तीस ए"."""
    ws = (text or "").strip(" .,!?।").split()
    edge = FILLER | YES | {"है।", "आहे."}
    while ws and ws[-1].strip(" .,!?।").lower() in edge:
        ws.pop()
    while ws and ws[0].strip(" .,!?।").lower() in edge:
        ws.pop(0)
    return " ".join(ws).strip(" .,!?।")


NAME_AFTER = re.compile(r"(?:नाम|नाव|naam|name is|name's)\s+(?:है\s+|आहे\s+|hai\s+|is\s+)?"
                        r"([\u0900-\u097FA-Za-z]+(?:\s+[\u0900-\u097FA-Za-z]+)?)", re.IGNORECASE)


def name_from(text: str) -> str | None:
    if "?" in (text or "") or any(w in QUESTION_WORDS for w in words(text)):
        return None
    found = NAME_AFTER.search(text or "")
    if found:
        named = [w for w in found.group(1).split() if w.lower() not in NAME_DROP]
        if named:
            return " ".join(w.capitalize() if w.isascii() else w for w in named)
    kept = [w for w in re.findall(r"[ऀ-ॿ]+|[A-Za-z]+", text or "") if w.lower() not in NAME_DROP]
    if not kept or len(kept) > 4:
        return None
    return " ".join(w.capitalize() if w.isascii() else w for w in kept)


# --- the rules -------------------------------------------------------------------------------------------------
def by_rules(text: str, task: Task, view: View) -> Understood:
    u = Understood()
    ws = words(text)
    for intent, phrases in INTENT_PHRASES.items():
        if has(text, phrases) and len(ws) <= INTENT_MAX_WORDS.get(intent, 99):
            u.intent = intent
            break
    if u.intent == "other" and has(text, DONT_KNOW):
        u.intent = "dont_know"
    answer = yes_no(text)
    u.said_no = answer is False
    pending = task.slot.get(view.pending) if view.pending not in (None, "confirm") else None
    if view.pending == "confirm" and u.intent == "other" and answer is not None:
        u.intent = "confirm_yes" if answer else "confirm_no"
    for slot in task.slots:
        name, kind = slot["name"], slot["type"]
        is_pending = pending is not None and name == pending["name"]
        value = None
        if kind == "bool" and is_pending and u.intent in ("other", "answer"):
            value = answer
        elif kind == "money" and (is_pending or sum(s["type"] == "money" for s in task.slots) == 1):
            value = numerals.money(text, expected=is_pending)
        elif kind == "number" and is_pending:
            spans = numerals.find_numbers(text)
            value = spans[0].value if spans else None
        elif kind == "day":
            value = numerals.day(text)
        elif kind == "time":
            value = numerals.time_of_day(text)
        elif kind == "phone":
            value = numerals.phone(text)
        elif kind == "enum":
            value = enum_from_words(slot, text)
        elif kind == "name" and (is_pending and u.intent == "other" or NAME_AFTER.search(text or "")):
            value = name_from(text)
        elif kind == "text" and is_pending and u.intent == "other" and not u.said_no:
            value = trimmed(text) if content_words(text) else None
        value = check(slot, value)
        if value is not None:
            u.values[name] = value
    if u.intent == "other" and (u.values or (pending is not None and answer is not None)):
        u.intent = "answer"
    if u.intent == "other" and ("?" in (text or "") or any(w in QUESTION_WORDS for w in ws)) and not u.values:
        u.intent = "question"
    # sure enough without the model: a short reply that answers what was asked, or a clear request
    short = len([w for w in ws if w != "?"]) <= 4
    if short and pending is not None and pending["name"] in u.values and pending["type"] not in ("name", "text"):
        u.fast = len(content_words(text)) <= 3 or pending["type"] in ("money", "time", "day", "phone")
    else:
        u.fast = (
            (short and view.pending == "confirm" and u.intent in ("confirm_yes", "confirm_no")
             and not content_words(text))
            or u.intent in STRONG_INTENTS
            or (short and u.intent in ("repeat", "hold", "who_are_you", "busy", "bye"))
            or (short and pending is not None and pending["type"] == "text" and u.said_no
                and not content_words(text)))
    u.source = "rules (fast)" if u.fast else "rules"
    return u


# --- the model -------------------------------------------------------------------------------------------------
def describe(task: Task, slot: dict) -> str:
    return task.say(slot["describe"], "en")


def field_line(task: Task, slot: dict) -> str:
    kind = slot["type"]
    shape = {
        "bool": "true or false", "money": "a number of rupees", "number": "a number", "phone": "10 digits",
        "day": "one of " + ", ".join(DAY_VALUES), "time": '"HH:MM", 24-hour', "text": "short text, as said",
        "name": "the name, as said",
    }.get(kind)
    if kind == "enum":
        shape = "one of " + ", ".join(f'"{k}"' for k in slot["options"])
    return f"- {slot['name']}: {shape}. {describe(task, slot)}"


def schema(task: Task) -> dict:
    """The JSON schema of the model's reply (Ollama's "format"): the intent, the fact, and every slot (null when
    this turn does not say it)."""
    props = {"intent": {"type": "string", "enum": list(INTENTS)}}
    for slot in task.slots:
        kind = slot["type"]
        if kind == "bool":
            props[slot["name"]] = {"type": ["boolean", "null"]}
        elif kind in ("money", "number"):
            props[slot["name"]] = {"type": ["number", "null"]}
        elif kind == "enum":
            props[slot["name"]] = {"type": ["string", "null"], "enum": [*slot["options"], None]}
        elif kind == "day":
            props[slot["name"]] = {"type": ["string", "null"], "enum": [*DAY_VALUES, None]}
        else:
            props[slot["name"]] = {"type": ["string", "null"]}
    props["fact"] = ({"type": ["string", "null"], "enum": [*task.facts, None]} if task.facts else {"type": "null"})
    return {"type": "object", "properties": props, "required": list(props)}


def prompt(task: Task, view: View, text: str) -> str:
    """The instructions for the model: the call, what was just asked, the fields, and the line to read."""
    if view.pending == "confirm":
        waiting = ("The assistant read back a summary and asked if it is right: intent confirm_yes if they agree, "
                   "confirm_no if not. If they correct a value, give the corrected value too.")
    elif view.pending in task.slot:
        waiting = f"The assistant was asking for: {view.pending} ({describe(task, task.slot[view.pending])})."
    else:
        waiting = "The assistant was not asking for anything in particular."
    known = {k: v for k, v in view.values.items() if v is not None}
    facts = "".join(f"\n   {k}: {task.say(f['about'], 'en')}" for k, f in task.facts.items()) or " (none)"
    intents = "".join(f"\n   {k}: {v}" for k, v in INTENTS.items())
    fields = "\n".join(field_line(task, s) for s in task.slots)
    return f"""You read one turn of a phone call and fill in a form. Reply with JSON only.

The call: {task.purpose()}
The other person is {task.who()}.
The assistant just said: "{view.last_bot}"
{waiting}
Known so far: {json.dumps(known, ensure_ascii=False) if known else "nothing"}

The other person now said: "{numerals.annotate(text)}"

Fields (null when this turn does not say it):
{fields}
- intent: one of:{intents}
- fact: when they ask a question, which of these it is about, else null:{facts}

Rules:
- A bare yes or no (हाँ, जी, हो, होय, है, नहीं, नाही, yes, no) answers the assistant's last question.
- Numbers said in words have their digits added, like "ढाई सौ (=250)": use those digits.
- Fill in only what this turn says. Never guess. Keep text values short, in the words they used."""


def first_json(raw: str) -> dict:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    found = json.loads(raw[start:end + 1])
    if not isinstance(found, dict):
        raise TypeError("the reply is not a JSON object")
    return found


def from_model(raw: str, task: Task) -> Understood:
    data = first_json(raw)
    u = Understood(source="model", raw=raw)
    intent = str(data.get("intent") or "other").strip().lower()
    u.intent = intent if intent in INTENTS else "other"
    fact = data.get("fact")
    u.fact = fact if isinstance(fact, str) and fact in task.facts else None
    for slot in task.slots:
        value = check(slot, data.get(slot["name"]))
        if value is not None:
            u.values[slot["name"]] = value
    return u


def reconcile(model: Understood, rules: Understood, text: str, task: Task, view: View) -> Understood:
    """The model's understanding, corrected by the rules where the rules are surer."""
    u = model
    u.said_no = rules.said_no
    if rules.intent in STRONG_INTENTS:
        u.intent = rules.intent
    spans = numerals.find_numbers(text, colloquial=True) + numerals.find_numbers(text)
    said = {s.value for s in spans}
    for slot in task.slots:
        name, kind = slot["name"], slot["type"]
        mine, theirs = u.values.get(name), rules.values.get(name)
        if kind in ("money", "number") and mine is not None:
            if not said:  # a number the person never said
                del u.values[name]
                mine = None
            elif theirs is not None and mine != theirs and mine not in said:
                u.values[name] = theirs
                u.source = "model+rules"
        free_text = kind in ("text", "name")
        if mine is None and theirs is not None and (not free_text or (name == view.pending and u.intent == "answer")):
            if kind == "bool" and name != view.pending:
                continue
            u.values[name] = theirs
            u.source = "model+rules"
        if (kind == "bool" and mine is True and theirs is False and name == view.pending and rules.said_no
                and len(words(text)) <= 3):  # a short "नहीं" can't be a yes
            u.values[name] = False
            u.source = "model+rules"
    if u.intent == "other" and u.values:
        u.intent = "answer"
    if view.pending == "confirm" and u.intent == "answer" and not u.values and rules.intent in ("confirm_yes",
                                                                                                "confirm_no"):
        u.intent = rules.intent
    return u


def understand(text: str, task: Task, view: View, llm=None) -> Understood:
    """What the line means for this call. llm(prompt, schema) -> the model's JSON text; None: rules only."""
    rules = by_rules(text, task, view)
    if llm is None or rules.fast:
        return rules
    try:
        raw = llm(prompt(task, view, text), schema(task))
        model = from_model(raw, task)
    except Exception as e:  # noqa: BLE001 - a model failure must not stop the call: the rules answer
        rules.error = f"model: {type(e).__name__}: {e}"[:300]
        return rules
    return reconcile(model, rules, text, task, view)


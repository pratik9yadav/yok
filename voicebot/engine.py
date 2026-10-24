"""The conversation: one call, turn by turn, for any task (voicebot/tasks/*.json).

    call = Call(load("shop_stock"), llm=my_model)      llm: None = rules only
    reply = call.start()                                the greeting and the first question
    reply = call.hear("हाँ है, ढाई सौ का")               what the other person said (from speech to text)
    reply.text, reply.speech, reply.done                what to show, what the voice says, whether the call is over
    call.result()                                       the values found, the outcome, every turn

The model only reads (what was said, which values); what the bot says next is decided here, by plain rules, from
the task: ask the next missing slot, say it again when not understood (twice at most, then move on), read
everything back at the end and fix what the person corrects. So the bot never makes up a price, a promise or a
fact: it only says the task's own texts and the values the person gave.

Outcomes: done (every needed slot found), partial (the call ended with some missing), callback (busy, call later),
do_not_call, wrong_number, handoff (wants a person), no_answer (silence), too_long.
"""

import time
from dataclasses import dataclass, field

from voicebot import lang as langs
from voicebot.numerals import day_words, digit_words, money_words, time_words
from voicebot.speech_text import clean_heard, for_voice
from voicebot.tasks import Task, render, text_of
from voicebot.understand import Understood, View, content_words, understand, words

MAX_ASKS = 2  # a slot is asked at most this many times, then left as unknown
MAX_READBACKS = 3
MAX_SILENCES = 3

PHRASES = {
    "ok": {"hi": "ठीक है।", "mr": "ठीक आहे.", "en": "Okay."},
    "repeat": {"hi": "जी, फिर से कहती हूँ।", "mr": "हो, पुन्हा सांगते.", "en": "Sure, let me say that again."},
    "hold": {"hi": "जी, ज़रूर, मैं रुकती हूँ।", "mr": "हो, नक्की, मी थांबते.", "en": "Sure, I'll wait."},
    "not_understood": {"hi": "माफ़ कीजिए, ठीक से समझ नहीं आया।", "mr": "माफ करा, नीट समजलं नाही.",
                       "en": "Sorry, I didn't quite get that."},
    "silence": {"hi": "हैलो? क्या आप मुझे सुन पा रहे हैं?", "mr": "हॅलो? तुम्हाला ऐकू येतंय का?",
                "en": "Hello? Can you hear me?"},
    "back": {"hi": "जी।", "mr": "हो.", "en": "Sure."},
    "dont_know": {"hi": "कोई बात नहीं।", "mr": "काही हरकत नाही.", "en": "No problem."},
    "busy": {"hi": "कोई बात नहीं, हम बाद में कॉल कर लेंगे। धन्यवाद!",
             "mr": "काही हरकत नाही, आम्ही नंतर फोन करू. धन्यवाद!", "en": "No problem, we'll call back later. Thank you!"},
    "stop": {"hi": "माफ़ कीजिए, अब हम आपको कॉल नहीं करेंगे। धन्यवाद!",
             "mr": "माफ करा, आम्ही आता तुम्हाला फोन करणार नाही. धन्यवाद!",
             "en": "Sorry about that, we won't call you again. Thank you!"},
    "wrong_number": {"hi": "माफ़ कीजिए, गलत नंबर लग गया। धन्यवाद!", "mr": "माफ करा, चुकीचा नंबर लागला. धन्यवाद!",
                     "en": "Sorry, wrong number. Thank you!"},
    "human": {"hi": "ठीक है, हमारी टीम का कोई व्यक्ति आपको जल्द ही कॉल करेगा। धन्यवाद!",
              "mr": "ठीक आहे, आमच्या टीममधील कोणीतरी तुम्हाला लवकरच फोन करेल. धन्यवाद!",
              "en": "Okay, a person from our team will call you shortly. Thank you!"},
    "unknown_fact": {"hi": "ये मुझे अभी पता नहीं, मैं टीम से पूछकर बता दूँगी।",
                     "mr": "हे मला आत्ता माहीत नाही, मी टीमला विचारून सांगते.",
                     "en": "I don't know that right now, I'll check with the team."},
    "readback": {"hi": "तो मैंने नोट किया: {items}। सही है?", "mr": "तर मी नोंद केलं: {items}. बरोबर आहे का?",
                 "en": "So I have: {items}. Is that right?"},
    "fix": {"hi": "माफ़ कीजिए, क्या गलत है? सही बता दीजिए।", "mr": "माफ करा, काय चुकलं? बरोबर सांगा.",
            "en": "Sorry, what is wrong? Please tell me the right one."},
    "no_answer": {"hi": "लगता है आवाज़ नहीं आ रही। हम बाद में कॉल करेंगे, धन्यवाद!",
                  "mr": "आवाज येत नाहीये असं वाटतं. आम्ही नंतर फोन करू, धन्यवाद!",
                  "en": "I can't hear you, it seems. We'll call back later, thank you!"},
    "bye": {"hi": "ठीक है, धन्यवाद! नमस्ते।", "mr": "ठीक आहे, धन्यवाद! नमस्कार.", "en": "Okay, thank you! Goodbye."},
    "too_long": {"hi": "आपके समय के लिए धन्यवाद, हम बाद में फिर संपर्क करेंगे।",
                 "mr": "तुमच्या वेळेबद्दल धन्यवाद, आम्ही नंतर पुन्हा संपर्क करू.",
                 "en": "Thanks for your time, we'll get back to you later."},
    "yes": {"hi": "हाँ", "mr": "हो", "en": "yes"},
    "no": {"hi": "नहीं", "mr": "नाही", "en": "no"},
    "and": {"hi": ", ", "mr": ", ", "en": ", "},
}


def phrase(name: str, lang: str) -> str:
    return text_of(PHRASES[name], lang)


@dataclass
class Reply:
    text: str  # what is shown
    speech: str  # what the voice says (numbers and short forms in words)
    lang: str
    done: bool = False
    outcome: str | None = None


@dataclass
class Turn:
    who: str  # "bot" or "person"
    text: str
    lang: str
    seconds: float = 0.0
    understood: dict | None = None
    note: str = ""


@dataclass
class Call:
    task: Task
    lang: str | None = None
    llm: object = None  # llm(prompt, schema) -> JSON text; None: rules only
    max_turns: int = 16
    values: dict = field(default_factory=dict)  # slot -> value
    status: dict = field(default_factory=dict)  # slot -> "filled", "unknown" (asked, no answer) or "none" (no)
    asks: dict = field(default_factory=dict)  # slot -> times asked
    pending: str | None = None  # the slot just asked, "confirm" (after the read-back), "fix", or None
    turns: list = field(default_factory=list)
    done: bool = False
    outcome: str | None = None
    readbacks: int = 0
    silences: int = 0
    heard: int = 0
    last_question: tuple = ("", "")  # (text, speech) of the last question, to say again
    holding: bool = False  # asked to wait: the next "हाँ, बोलिए" is them coming back, not an answer
    _maybe_lang: str | None = None

    def __post_init__(self):
        if self.lang not in self.task.languages:
            self.lang = self.task.default_language

    # --- what the bot says ---------------------------------------------------------------------------------------
    def _say(self, template, extra: dict | None = None) -> tuple[str, str]:
        """A task text in the call's language: (shown, spoken)."""
        text = self.task.say(template, self.lang, extra)
        return text, for_voice(text, self.lang)

    def _phrase(self, name: str, items: tuple[str, str] | None = None) -> tuple[str, str]:
        p = phrase(name, self.lang)
        if items is None:
            return p, for_voice(p, self.lang)
        return render(p, {"items": items[0]}), for_voice(render(p, {"items": items[1]}), self.lang)

    def show(self, slot: dict, value) -> tuple[str, str]:
        """A value as shown and as spoken: ₹220 / "दो सौ बीस रुपये", 16:30 / "शाम साढ़े चार बजे", ..."""
        kind, lang = slot["type"], self.lang
        if kind == "money":
            return f"₹{value}", money_words(value, lang)
        if kind == "time":
            words = time_words(value, lang)
            return (words if lang != "en" else f"{value}"), words
        if kind == "day":
            return day_words(value, lang), day_words(value, lang)
        if kind == "phone":
            return f"{value[:5]} {value[5:]}", digit_words(value, lang)
        if kind == "enum":
            label = text_of(slot["options"][value], lang)
            return label, label
        if kind == "bool":
            return phrase("yes" if value else "no", lang), phrase("yes" if value else "no", lang)
        return str(value), str(value)

    def _ask(self, name: str) -> tuple[str, str]:
        self.asks[name] = self.asks.get(name, 0) + 1
        self.pending = name
        self.last_question = self._say(self.task.slot[name]["ask"])
        return self.last_question

    def _readback(self) -> tuple[str, str]:
        shown, spoken = [], []
        for slot in self.task.slots:
            name = slot["name"]
            if not self.task.applies(slot, self.values) or self.status.get(name) != "filled":
                continue
            value = self.values[name]
            template = slot.get("readback")
            if slot["type"] == "bool" and isinstance(template, dict) and "true" in template:
                template = template["true" if value else "false"]
            elif template is None:
                template = {"hi": f"{name}: {{value}}", "mr": f"{name}: {{value}}", "en": f"{name}: {{value}}"}
            a, b = self.show(slot, value)
            shown.append(self.task.say(template, self.lang, {"value": a}))
            spoken.append(self.task.say(template, self.lang, {"value": b}))
        if not shown:
            return "", ""
        self.readbacks += 1
        self.pending = "confirm"
        joiner = phrase("and", self.lang)
        self.last_question = self._phrase("readback", (joiner.join(shown), joiner.join(spoken)))
        return self.last_question

    # --- the flow --------------------------------------------------------------------------------------------------
    def next_slot(self) -> str | None:
        """The next slot to ask: needed now (its "when" holds), not answered, asked fewer than MAX_ASKS times."""
        for slot in self.task.slots:
            name = slot["name"]
            if not self.task.applies(slot, self.values) or name in self.status:
                continue
            if self.asks.get(name, 0) >= MAX_ASKS:
                self.status[name] = "unknown"
                continue
            return name
        return None

    def missing(self) -> list[str]:
        """The needed slots (required, and their "when" holds) with no value."""
        return [s["name"] for s in self.task.slots if s.get("required", True) and self.task.applies(s, self.values)
                and self.status.get(s["name"]) != "filled"]

    def _reply(self, pieces: list[tuple[str, str]], done: bool = False, outcome: str | None = None,
               started: float | None = None) -> Reply:
        pieces = [p for p in pieces if p and p[0]]
        text = " ".join(p[0] for p in pieces)
        speech = " ".join(p[1] for p in pieces)
        if done:
            self.done, self.outcome, self.pending = True, outcome, None
        self.turns.append(Turn("bot", text, self.lang, round(time.perf_counter() - started, 3) if started else 0.0,
                               note=outcome or ""))
        return Reply(text, speech, self.lang, done, outcome)

    def _end(self, pieces, outcome, started=None) -> Reply:
        return self._reply(pieces, done=True, outcome=outcome, started=started)

    def _finish(self, pieces, started=None) -> Reply:
        """Every slot is done: the closing (and the outcome: done, or partial when a needed one is missing)."""
        closing = self.task.closing(self.values, self.lang)
        return self._end(pieces + [(closing, for_voice(closing, self.lang))], "partial" if self.missing() else "done",
                         started)

    def _continue(self, pieces, started=None) -> Reply:
        """Ask the next slot; with none left, read back (once the task wants it) or finish."""
        name = self.next_slot()
        if name:
            return self._reply(pieces + [self._ask(name)], started=started)
        if self.task.confirm and self.readbacks < MAX_READBACKS and any(
                v == "filled" for v in self.status.values()):
            readback = self._readback()
            if readback[0]:
                return self._reply(pieces + [readback], started=started)
        return self._finish(pieces, started)

    def start(self) -> Reply:
        started = time.perf_counter()
        greeting = self._say(self.task.spec["greeting"])
        return self._continue([greeting], started)

    def _switch_language(self, text: str) -> None:
        """Answer in the language the person speaks: switch after two lines in it, or one clear line."""
        found, score = langs.detect(text, default=self.lang)
        if score == 0 or found not in self.task.languages:
            return
        if found != self.lang and (score >= 2 or self._maybe_lang == found):
            self.lang = found
        self._maybe_lang = found

    def _apply(self, u: Understood) -> tuple[list[str], tuple[str, str] | None]:
        """Put the understood values in; returns the slots that changed, and the task's "invalid" text (spoken)
        when a value breaks the task's rules (a Sunday when the clinic is closed)."""
        changed = []
        for name, value in u.values.items():
            slot = self.task.slot[name]
            bad = self._invalid(slot, value)
            if bad:
                self.status.pop(name, None)
                self.values.pop(name, None)
                self.pending = name
                return changed, self._say(slot.get("invalid") or PHRASES["not_understood"])
            if self.values.get(name) != value or self.status.get(name) != "filled":
                changed.append(name)
            self.values[name] = value
            self.status[name] = "filled"
        return changed, None

    @staticmethod
    def _invalid(slot: dict, value) -> bool:
        rules = slot.get("valid") or {}
        if value in (rules.get("not") or []):
            return True
        low, high = rules.get("min"), rules.get("max")
        try:
            return (low is not None and value < low) or (high is not None and value > high)
        except TypeError:
            return False

    def hear(self, text: str, heard_seconds: float = 0.0) -> Reply:
        """The next thing the other person said; returns what the bot says back."""
        if self.done:
            return Reply("", "", self.lang, True, self.outcome)
        started = time.perf_counter()
        text = clean_heard(text)
        if not text:
            self.silences += 1
            self.turns.append(Turn("person", "", self.lang, heard_seconds, note="silence"))
            if self.silences >= MAX_SILENCES:
                return self._end([self._phrase("no_answer")], "no_answer", started)
            return self._reply([self._phrase("silence")] + ([self.last_question] if self.silences > 1 else []),
                               started=started)
        self.silences = 0
        self.heard += 1
        self._switch_language(text)
        view = View(pending=self.pending, last_bot=next((t.text for t in reversed(self.turns) if t.who == "bot"), ""),
                    values=dict(self.values))
        u = understand(text, self.task, view, self.llm)
        self.turns.append(Turn("person", text, self.lang, heard_seconds, understood=u.as_dict()))
        if self.heard > self.max_turns:
            return self._end([self._phrase("too_long")], "too_long", started)
        if u.intent == "stop_calling":
            return self._end([self._phrase("stop")], "do_not_call", started)
        if u.intent == "wrong_number":
            return self._end([self._phrase("wrong_number")], "wrong_number", started)
        if u.intent == "human":
            return self._end([self._phrase("human")], "handoff", started)
        if u.intent == "busy" and not u.values:
            return self._end([self._phrase("busy")], "callback", started)
        if u.intent == "bye" and not u.values:
            if not self.missing():
                return self._finish([], started)
            return self._end([self._phrase("bye")], "partial", started)
        if u.intent == "repeat" and not u.values:
            return self._reply([self._phrase("repeat"), self.last_question], started=started)
        if u.intent == "hold" and not u.values:
            self.holding = True
            return self._reply([self._phrase("hold")], started=started)
        if self.holding:
            self.holding = False
            if len(words(text)) <= 3 and set(u.values) <= {self.pending} and not content_words(text):
                return self._reply([self._phrase("back"), self.last_question], started=started)
        if u.intent == "who_are_you" and not u.values:
            return self._reply([self._say(self.task.spec["greeting"]), self.last_question], started=started)

        pending = self.pending
        changed, invalid = self._apply(u)
        if invalid:
            return self._reply([invalid, self._ask(self.pending)], started=started)
        pieces = []
        if u.intent == "question" or (u.fact and not changed):
            fact = self.task.facts.get(u.fact) if u.fact else None
            pieces.append(self._say(fact["answer"]) if fact else self._phrase("unknown_fact"))
        if pending in ("confirm", "fix"):
            if changed:
                self.readbacks = min(self.readbacks, MAX_READBACKS - 1)
                return self._continue(pieces + [self._phrase("ok")], started)
            if u.intent == "confirm_yes" or (u.intent == "answer" and not u.said_no and pending == "fix"):
                return self._finish(pieces, started)
            if u.intent == "confirm_no" or u.said_no:
                if self.readbacks >= MAX_READBACKS:
                    return self._finish(pieces, started)
                self.pending = "fix"
                self.last_question = self._phrase("fix")
                return self._reply(pieces + [self.last_question], started=started)
            return self._reply(pieces + ([] if pieces else [self._phrase("not_understood")]) + [self.last_question],
                               started=started)
        slot = self.task.slot.get(pending) if pending else None
        if slot is not None and pending not in changed and self.status.get(pending) is None:
            if u.intent == "dont_know":
                self.status[pending] = "unknown"
                pieces.append(self._phrase("dont_know"))
            elif slot["type"] in ("text", "name") and u.said_no and not u.values:
                self.status[pending] = "none"  # "no other brand"
            elif not pieces and not changed:
                pieces.append(self._phrase("not_understood"))
        elif changed and not pieces:
            pieces.append(self._phrase("ok"))
        return self._continue(pieces, started)

    # --- the result --------------------------------------------------------------------------------------------------
    def result(self) -> dict:
        """What the call found: the values of the slots that apply (None: not found), each slot's status, the
        outcome, the language and every turn."""
        values = {s["name"]: (self.values.get(s["name"]) if self.task.applies(s, self.values) else None)
                  for s in self.task.slots}
        return {"task": self.task.id, "outcome": self.outcome, "values": values,
                "status": {k: v for k, v in self.status.items()}, "missing": self.missing(), "lang": self.lang,
                "turns": [t.__dict__ for t in self.turns]}

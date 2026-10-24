"""Tasks: what the voice bot is for on one call, written as a small JSON file (voicebot/tasks/*.json), so the same bot
can ask a shop about stock, book a clinic visit or confirm an order, with no code change.

A task says who the bot talks to and why, the greeting, the facts it may tell, and the "slots": the things it has to
find out, each with a type, the question that asks for it (in Hindi, Marathi and English) and when to ask it.

    task = load("shop_stock")                     voicebot/tasks/shop_stock.json
    task = from_json(text)                         a task someone wrote (checked the same way)
    problems(spec) -> ["slot 'price': ..."]        what is wrong with a task ([] when it is fine)

Slot types: bool (yes/no), money (rupees), number, text, name, phone (10-digit mobile), day (today, tomorrow,
day_after or a weekday), time ("HH:MM"), enum (one of "options").
"""

import json
import re
from pathlib import Path

TASKS_DIR = Path(__file__).parent / "tasks"
LANGS = ("hi", "mr", "en")
TYPES = ("bool", "money", "number", "text", "name", "phone", "day", "time", "enum")
DAY_VALUES = ("today", "tomorrow", "day_after", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
              "sunday")
RESERVED = {"intent", "faq", "language", "slots"}  # field names the understanding step uses itself
NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


class TaskError(ValueError):
    pass


def text_of(value, lang: str, fallback: str = "hi") -> str:
    """A text in lang from {"hi": ..., "mr": ..., "en": ...} (or a plain string): lang, else fallback, else any."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return value.get(lang) or value.get(fallback) or next((v for v in value.values() if v), "")


def _texts_ok(value, where: str, langs, required: bool = True) -> list[str]:
    if value is None:
        return [f"{where}: missing"] if required else []
    if isinstance(value, str):
        return [] if value.strip() else [f"{where}: empty"]
    if not isinstance(value, dict):
        return [f"{where}: must be a text or {{\"hi\": ..., \"mr\": ..., \"en\": ...}}"]
    missing = [lang for lang in langs if not str(value.get(lang) or "").strip()]
    return [f"{where}: no {', '.join(missing)} text"] if missing else []


def problems(spec) -> list[str]:
    """Everything wrong with a task spec, in plain words ([] when it can be used)."""
    if not isinstance(spec, dict):
        return ["the task must be a JSON object"]
    out = []
    if not NAME.match(str(spec.get("id") or "")):
        out.append("id: lowercase letters, digits and _ (like shop_stock)")
    langs = spec.get("languages") or ["hi"]
    if not isinstance(langs, list) or not langs or any(lang not in LANGS for lang in langs):
        out.append(f"languages: a list from {', '.join(LANGS)}")
        langs = ["hi"]
    if spec.get("default_language", langs[0]) not in langs:
        out.append("default_language: must be one of languages")
    if spec.get("bot_role", "caller") not in ("caller", "receiver"):
        out.append('bot_role: "caller" (the bot calls) or "receiver" (the bot answers)')
    for field in ("purpose", "who"):
        if not str(text_of(spec.get(field), "en")).strip():
            out.append(f"{field}: missing")
    out += _texts_ok(spec.get("greeting"), "greeting", langs)
    closing = spec.get("closing")
    if isinstance(closing, list):
        for i, c in enumerate(closing):
            out += _texts_ok(c.get("text") if isinstance(c, dict) else None, f"closing[{i}]", langs)
        if not any(isinstance(c, dict) and not c.get("when") for c in closing):
            out.append("closing: one entry needs no \"when\" (the default)")
    else:
        out += _texts_ok(closing, "closing", langs)
    context = spec.get("context") or {}
    if not isinstance(context, dict):
        out.append("context: must be an object of names and values")
    slots = spec.get("slots")
    if not isinstance(slots, list) or not slots:
        return out + ["slots: at least one"]
    names = []
    for i, slot in enumerate(slots):
        where = f"slot {i + 1}"
        if not isinstance(slot, dict):
            out.append(f"{where}: must be an object")
            continue
        name = slot.get("name")
        where = f"slot '{name}'"
        if not NAME.match(str(name or "")) or name in RESERVED:
            out.append(f"{where}: name must be lowercase letters, digits and _, and not {sorted(RESERVED)}")
        if name in names:
            out.append(f"{where}: used twice")
        names.append(name)
        if slot.get("type") not in TYPES:
            out.append(f"{where}: type must be one of {', '.join(TYPES)}")
        if not str(slot.get("describe") or "").strip():
            out.append(f"{where}: describe: missing (what it means, in English, for the model)")
        out += _texts_ok(slot.get("ask"), f"{where} ask", langs)
        if slot.get("type") == "enum":
            options = slot.get("options")
            if not isinstance(options, dict) or len(options) < 2:
                out.append(f"{where}: options: at least two")
            else:
                for value, o in options.items():
                    out += _texts_ok(o if isinstance(o, dict) else None, f"{where} option '{value}'", langs)
        for cond in (slot.get("when") or {}):
            if cond not in names[:-1]:
                out.append(f"{where}: when: '{cond}' must be a slot listed before it")
    for key, fact in (spec.get("facts") or {}).items():
        if not isinstance(fact, dict) or not str(fact.get("about") or "").strip():
            out.append(f"fact '{key}': about: missing")
        else:
            out += _texts_ok(fact.get("answer"), f"fact '{key}' answer", langs)
    try:
        render_all(spec)
    except KeyError as e:
        out.append(f"a text uses {{{e.args[0]}}}, which is not in context or slots")
    return out


def render(template: str, values: dict) -> str:
    """The template with {name} filled from values; an unknown name raises KeyError."""
    return re.sub(r"\{([a-z_][a-z0-9_]*)\}", lambda m: str(values[m.group(1)]), template)


def render_all(spec: dict) -> None:
    """Fill every text once with the context and slot names, to find unknown {names}."""
    values = {**(spec.get("context") or {}), **{s.get("name"): "x" for s in spec.get("slots") or []
                                                 if isinstance(s, dict)}, "value": "x"}

    def walk(v):
        if isinstance(v, str):
            render(v, values)
        elif isinstance(v, dict):
            for k, x in v.items():
                if k not in ("describe", "about", "purpose", "who", "words"):
                    walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk({k: v for k, v in spec.items() if k not in ("id", "context", "languages")})


class Task:
    """A checked task spec, with easy access to its parts."""

    def __init__(self, spec: dict, context: dict | None = None):
        spec = json.loads(json.dumps(spec))  # a copy
        if context:
            spec["context"] = {**(spec.get("context") or {}), **context}
        found = problems(spec)
        if found:
            raise TaskError("; ".join(found))
        self.spec = spec
        self.id = spec["id"]
        self.languages = list(spec.get("languages") or ["hi"])
        self.default_language = spec.get("default_language") or self.languages[0]
        self.bot_role = spec.get("bot_role", "caller")
        self.context = dict(spec.get("context") or {})
        self.slots = spec["slots"]
        self.slot = {s["name"]: s for s in self.slots}
        self.facts = spec.get("facts") or {}
        self.confirm = bool(spec.get("confirm", False))

    def name(self, lang: str = "en") -> str:
        return text_of(self.spec.get("name"), lang, "en") or self.id

    def purpose(self) -> str:
        return render(text_of(self.spec["purpose"], "en"), self.context)

    def who(self) -> str:
        return render(text_of(self.spec["who"], "en"), self.context)

    def say(self, value, lang: str, extra: dict | None = None) -> str:
        """A text of the task in lang, filled with the context (and extra values)."""
        return render(text_of(value, lang, self.default_language), {**self.context, **(extra or {})})

    def applies(self, slot: dict, values: dict) -> bool:
        """Whether a slot is asked: every "when" condition holds for the values found so far."""
        return all(values.get(k) == v for k, v in (slot.get("when") or {}).items())

    def closing(self, values: dict, lang: str) -> str:
        closing = self.spec["closing"]
        if isinstance(closing, list):
            chosen = next((c for c in closing if c.get("when") and all(values.get(k) == v for k, v in
                                                                         c["when"].items())), None)
            chosen = chosen or next(c for c in closing if not c.get("when"))
            return self.say(chosen["text"], lang)
        return self.say(closing, lang)


def from_json(text: str, context: dict | None = None) -> Task:
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as e:
        raise TaskError(f"not valid JSON: {e}") from e
    return Task(spec, context)


def load(task_id: str, context: dict | None = None) -> Task:
    path = TASKS_DIR / f"{task_id}.json"
    if not NAME.match(task_id) or not path.exists():
        raise TaskError(f"no task '{task_id}' (there are: {', '.join(available())})")
    return from_json(path.read_text(encoding="utf-8"), context)


def available() -> list[str]:
    return sorted(p.stem for p in TASKS_DIR.glob("*.json"))

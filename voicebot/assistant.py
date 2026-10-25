"""The general assistant: talk about anything, like a voice assistant, in the language the person speaks.

    chat = ollama_chat("gemma3:4b")                       chat(messages, stream) -> text, or an iterator of pieces
    helper = Assistant(chat)
    for sentence in helper.reply("मुंबई में कितने लोग रहते हैं?", "hi"):   one sentence at a time, as the model writes
        speak(sentence)

In a task call (shop, clinic...), a question the task's facts don't cover gets a short answer from aside(): general
knowledge is answered; the business's own details (a price, stock, an address) are never made up.
"""

import json
import re
import urllib.request

from voicebot.tasks import Task

LANG_NAMES = {"hi": "Hindi, in Devanagari letters", "mr": "Marathi, in Devanagari letters", "en": "English"}
SYSTEM = """You are Wimb, a friendly voice assistant from India, talking with someone by voice.
Reply in {language}, the language they speak (if they mix Hindi and English, you may too).
Keep it short and spoken: one to three sentences, no lists, no markdown, no emojis, no headings.
Say numbers the way people say them. If you are not sure, or it needs live information (today's news, weather,
prices, scores), say so briefly instead of guessing. Be warm and polite, and use "आप" in Hindi."""
ASIDE = """You are a voice assistant on a phone call. The call: {purpose}
The other person ({who}) just asked something the call's script does not cover: "{question}"
Answer in {language}, in one short spoken sentence, no markdown.
If it is about this business's own details (its prices, stock, fees, address, timings, offers) say politely that
you will check with the team and tell them. Otherwise answer it briefly from general knowledge."""
SENTENCE_END = re.compile(r"([।.!?]+[\"')\]]*)(\s+)")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")
MAX_HISTORY = 12  # messages kept (6 back-and-forths): enough context, still fast


def clean(text: str) -> str:
    """Model text as it should be spoken: no markdown, emojis or list marks."""
    text = EMOJI.sub("", text or "")
    text = re.sub(r"[*_#`>|]+", "", text)
    text = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    return " ".join(text.split())


def sentences(pieces, min_chars: int = 12):
    """Whole sentences from a stream of text pieces (the model's tokens), each as soon as it is complete, so the
    voice can start on the first while the model writes the rest."""
    buffer = ""
    for piece in pieces:
        buffer += piece
        while True:
            found = None
            for m in SENTENCE_END.finditer(buffer):
                if m.end(1) >= min_chars:
                    found = m
                    break
            if not found:
                break
            out = clean(buffer[:found.end(1)])
            buffer = buffer[found.end():]
            if out:
                yield out
    rest = clean(buffer)
    if rest:
        yield rest


class Assistant:
    """A conversation with the model, remembered for the last MAX_HISTORY messages."""

    def __init__(self, chat):
        self.chat = chat
        self.history: list[dict] = []

    def reply(self, text: str, lang: str = "hi"):
        """The answer to text, sentence by sentence (a generator); remembered once complete."""
        system = SYSTEM.format(language=LANG_NAMES.get(lang, LANG_NAMES["hi"]))
        messages = [{"role": "system", "content": system}, *self.history[-MAX_HISTORY:],
                    {"role": "user", "content": text}]
        said = []
        for sentence in sentences(self.chat(messages, stream=True)):
            said.append(sentence)
            yield sentence
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": " ".join(said)}]


def aside(chat, task: Task, question: str, lang: str) -> str | None:
    """A short answer to an off-script question in a task call (None when the model gives nothing)."""
    prompt = ASIDE.format(purpose=task.purpose(), who=task.who(), question=question,
                          language=LANG_NAMES.get(lang, LANG_NAMES["hi"]))
    text = clean(chat([{"role": "user", "content": prompt}], stream=False))
    return text[:300] or None


def ollama_chat(model: str, url: str = "http://127.0.0.1:11434", max_tokens: int = 200):
    """chat(messages, stream) through Ollama's /api/chat: the whole reply, or (stream=True) its pieces."""

    def chat(messages, stream=False):
        body = {"model": model, "messages": messages, "stream": stream, "keep_alive": "60m",
                "options": {"temperature": 0.4, "num_predict": max_tokens, "num_ctx": 4096}}
        request = urllib.request.Request(url + "/api/chat", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        if not stream:
            with urllib.request.urlopen(request, timeout=120) as resp:
                return json.load(resp).get("message", {}).get("content", "")

        def pieces():
            with urllib.request.urlopen(request, timeout=120) as resp:
                for line in resp:
                    if line.strip():
                        data = json.loads(line)
                        yield data.get("message", {}).get("content", "")
                        if data.get("done"):
                            return

        return pieces()

    return chat

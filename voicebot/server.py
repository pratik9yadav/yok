"""The voice page: talk to Wimb hands-free, like a voice assistant. Open it, tap the circle once, then just talk:
the page hears when you start and stop (a voice detector in the browser), Wimb answers by voice, and you can
interrupt it. Modes at the top: Assistant (ask anything) and the tasks (shop, clinic, order...).

    app = make_app(ears, mouth, llm=understanding_llm, chat=ollama_chat("gemma3:4b"), password="...")
    serve(app, 7860); url = open_tunnel(7860)            on Kaggle: a public https link (the page asks the password)

Each turn streams back as lines of JSON, so the voice starts on the first sentence while the rest is still being
written: {"type": "heard"}, then {"type": "say", "text", "audio"} per sentence, then {"type": "done"}.
Calls are saved (audio and text) only when the person leaves "save my voice" on.
"""

import base64
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from voicebot import tasks
from voicebot.assistant import SENTENCE_END, Assistant, aside
from voicebot.engine import Call
from voicebot.lang import detect
from voicebot.speech_text import clean_heard, for_voice
from voicebot.speechdata import norm

WEB = Path(__file__).parent / "web" / "index.html"
VAD_CDN = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.22/dist/"
ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/"
ASSISTANT = "assistant"
GREETING = {"hi": "नमस्ते! मैं विम्ब हूँ। बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
            "mr": "नमस्कार! मी विम्ब. सांगा, मी तुमची काय मदत करू?",
            "en": "Hi, I'm Wimb. How can I help you?"}
NO_BRAIN = {"hi": "माफ़ कीजिए, अभी मेरा मॉडल चालू नहीं है, थोड़ी देर बाद कोशिश कीजिए।",
            "mr": "माफ करा, आत्ता माझं मॉडेल चालू नाही, थोड्या वेळाने प्रयत्न करा.",
            "en": "Sorry, my model is not running right now. Please try again in a bit."}
NOT_HEARD = {"hi": "माफ़ कीजिए, सुनाई नहीं दिया। फिर से बोलिए?", "mr": "माफ करा, ऐकू आलं नाही. पुन्हा बोला?",
             "en": "Sorry, I didn't catch that. Could you say it again?"}
CLOUDFLARED = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
MODEL_LOCK = threading.Lock()  # one GPU: one model call at a time


def modes() -> list[dict]:
    order = sorted(tasks.available(), key=lambda t: (t != "shop_stock", t))
    return [{"id": ASSISTANT, "name": "Assistant"}] + [{"id": t, "name": tasks.load(t).name("en").split(":")[0]}
                                                      for t in order]


def pairs(text: str, speech: str) -> list[tuple[str, str]]:
    """A reply cut into sentences, shown and spoken side by side (so the voice can start on the first)."""
    def cut(t):
        parts, start = [], 0
        for m in SENTENCE_END.finditer(t + " "):
            parts.append(t[start:m.end(1)].strip())
            start = m.end()
        tail = t[start:].strip()
        return [p for p in parts + ([tail] if tail else []) if p]

    shown, said = cut(text), cut(speech)
    if len(shown) != len(said):
        return [(text, speech)]
    return list(zip(shown, said))


def echo_of(heard: str, bot: str) -> bool:
    """Whether what was heard while Wimb spoke is just Wimb's own voice coming back through the microphone."""
    words = norm(heard).split()
    if not words:
        return True
    own = set(norm(bot).split())
    return sum(w in own for w in words) / len(words) >= 0.6


class Conversation:
    """One conversation on the page: the assistant, or a task call; saved only with consent."""

    def __init__(self, mode: str, chat=None, llm=None, save_dir: str | None = None, save: bool = False,
                 lang: str = "hi"):
        self.id = uuid.uuid4().hex[:12]
        self.mode, self.lang, self.chat = mode, lang, chat
        self.helper = Assistant(chat) if mode == ASSISTANT and chat else None
        self.call = None
        if mode != ASSISTANT:
            task = tasks.load(mode)
            answer = (lambda question, lang: aside(chat, task, question, lang)) if chat else None
            self.call = Call(task, llm=llm, aside=answer)
            self.lang = self.call.lang
        self.folder = os.path.join(save_dir, time.strftime("%Y%m%d-%H%M%S-") + self.id) if save_dir and save else None
        if self.folder:
            os.makedirs(self.folder, exist_ok=True)
        self.log: list[dict] = []
        self.last_bot = ""
        self.n = 0
        self.ended = False

    def opening(self):
        """What Wimb says first: [(shown, spoken)], and whether the call is over already."""
        if self.call is not None:
            r = self.call.start()
            return pairs(r.text, r.speech), r.done
        greeting = GREETING.get(self.lang, GREETING["hi"])
        return [(greeting, for_voice(greeting, self.lang))], False

    def answer(self, heard: str, heard_lang: str | None = None):
        """Wimb's answer to what was heard, sentence by sentence: (shown, spoken). Sets self.ended."""
        if self.call is not None:
            r = self.call.hear(heard)
            self.lang, self.ended = r.lang, r.done
            yield from pairs(r.text, r.speech)
            return
        found, score = detect(heard, default=self.lang)
        self.lang = found if score else (heard_lang if heard_lang in GREETING else self.lang)
        if not heard:
            line = NOT_HEARD[self.lang]
            yield line, for_voice(line, self.lang)
            return
        if self.helper is None:
            line = NO_BRAIN[self.lang]
            yield line, for_voice(line, self.lang)
            return
        for sentence in self.helper.reply(heard, self.lang):
            yield sentence, for_voice(sentence, self.lang)

    def found(self) -> str:
        if self.call is None:
            return ""
        r = self.call.result()
        values = ", ".join(f"{k}: {v}" for k, v in r["values"].items() if v is not None)
        return (values or "nothing yet") + (f"  ({r['outcome']})" if r["outcome"] else "")

    def save(self) -> None:
        if not self.folder:
            return
        data = {"id": self.id, "mode": self.mode, "log": self.log,
                "result": self.call.result() if self.call is not None else None}
        with open(os.path.join(self.folder, "call.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)


def wav_b64(audio, rate: int) -> str:
    import soundfile as sf

    from voicebot.audio import to_int16

    buffer = io.BytesIO()
    sf.write(buffer, to_int16(audio), rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode()


def samples_from(body: bytes, content_type: str):
    """16 kHz samples from what the page sent (WAV from the voice detector, or webm/ogg/mp4 from tap-to-talk)."""
    from voicebot.audio import as_samples

    if "wav" in (content_type or "") or body[:4] == b"RIFF":
        import soundfile as sf

        audio, rate = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
        return as_samples((rate, audio.mean(axis=1)))
    suffix = ".ogg" if "ogg" in content_type else ".mp4" if "mp4" in content_type else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(body)
    try:
        return as_samples(fh.name)
    finally:
        os.remove(fh.name)


def line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def make_app(ears=None, mouth=None, llm=None, chat=None, password: str = "", save_dir: str | None = None,
             vad_base: str = VAD_CDN, ort_base: str = ORT_CDN, vendor_dir: str | None = None):
    """The FastAPI app. ears/mouth: voicebot.audio Ears and Mouth (None: text only); llm: the understanding model
    for tasks; chat: the assistant model (voicebot.assistant.ollama_chat); password: asked once (cookie)."""
    from fastapi import FastAPI, Form, Request
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse

    app = FastAPI(title="Wimb voice")
    secret = secrets.token_bytes(32)
    token = hmac.new(secret, (password or "").encode(), hashlib.sha256).hexdigest()
    conversations: dict[str, Conversation] = {}
    page = WEB.read_text(encoding="utf-8").replace("{{VAD_BASE}}", vad_base).replace("{{ORT_BASE}}", ort_base)
    app.state.conversations = conversations

    if vendor_dir:  # local tests: the voice detector's files served from here instead of the CDN
        from fastapi.staticfiles import StaticFiles

        app.mount("/vendor", StaticFiles(directory=vendor_dir), name="vendor")

    def allowed(request: Request) -> bool:
        return not password or hmac.compare_digest(request.cookies.get("wimb", ""), token)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.url.path in ("/login",) or request.url.path.startswith("/vendor") or allowed(request):
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "log in first"}, status_code=401)
        return HTMLResponse(LOGIN.replace("{{ERROR}}", ""))

    @app.post("/login")
    async def login(password_typed: str = Form("", alias="password")):
        if password and not hmac.compare_digest(password_typed, password):
            return HTMLResponse(LOGIN.replace("{{ERROR}}", "Wrong password"), status_code=401)
        response = RedirectResponse("./", status_code=303)
        response.set_cookie("wimb", token, httponly=True, samesite="lax", secure=False, max_age=12 * 3600)
        return response

    @app.get("/")
    async def index():
        return HTMLResponse(page)

    @app.get("/api/modes")
    async def list_modes():
        return modes()

    def speak(text: str, lang: str):
        if mouth is None or not text:
            return None, 0.0
        with MODEL_LOCK:
            audio, rate, seconds = mouth.say(text, lang, "bot")
        return wav_b64(audio, rate), seconds

    def reply_lines(conv: Conversation, pieces, timing: dict, started: float):
        """The 'say' lines, one per sentence with its audio, then 'done'."""
        said = []
        first = True
        try:
            for shown, spoken in pieces:
                audio, seconds = speak(spoken, conv.lang)
                if first:
                    timing["first_sentence"] = round(time.perf_counter() - started - seconds, 2)
                    timing["first_audio"] = round(time.perf_counter() - started, 2)
                    first = False
                timing["speak"] = round(timing.get("speak", 0) + seconds, 2)
                said.append(shown)
                yield line({"type": "say", "text": shown, "audio": audio, "lang": conv.lang})
        except Exception as e:  # noqa: BLE001 - say what broke, keep the page alive
            yield line({"type": "error", "text": f"Sorry, something broke: {type(e).__name__}"})
            timing["error"] = repr(e)[:300]
        conv.last_bot = " ".join(said)
        conv.log.append({"who": "bot", "text": conv.last_bot, "lang": conv.lang, **timing})
        conv.save()
        yield line({"type": "done", "found": conv.found(), "end": conv.ended, "timing": timing})

    @app.post("/api/start")
    async def start(request: Request):
        body = await request.json()
        mode = body.get("mode") or ASSISTANT
        if mode != ASSISTANT and mode not in tasks.available():
            return JSONResponse({"error": "no such mode"}, status_code=400)
        conv = Conversation(mode, chat=chat, llm=llm, save_dir=save_dir, save=bool(body.get("save")),
                            lang=body.get("lang") if body.get("lang") in GREETING else "hi")
        conversations[conv.id] = conv
        started = time.perf_counter()

        def run():
            yield line({"type": "session", "sid": conv.id})
            pieces, conv.ended = conv.opening()
            yield from reply_lines(conv, pieces, {}, started)

        return StreamingResponse(run(), media_type="application/x-ndjson")

    def conversation(sid: str | None) -> Conversation | None:
        return conversations.get(sid or "")

    @app.post("/api/turn")
    async def turn(request: Request, sid: str = "", during: int = 0):
        conv = conversation(sid)
        if conv is None:
            return JSONResponse({"error": "no such conversation: start again"}, status_code=404)
        body = await request.body()
        content_type = request.headers.get("content-type", "")
        started = time.perf_counter()

        def run():
            timing = {}
            conv.n += 1
            heard, heard_lang = "", None
            try:
                samples = samples_from(body, content_type)
                if conv.folder:
                    import soundfile as sf

                    sf.write(os.path.join(conv.folder, f"{conv.n:02d}_you.wav"), samples, 16000)
                if ears is not None and len(samples) > 3200:
                    with MODEL_LOCK:
                        heard, heard_lang, timing["hear"] = ears.hear(samples, hint=conv.lang,
                                                                       allowed=("hi", "mr", "en"))
                    heard = clean_heard(heard)
            except Exception as e:  # noqa: BLE001
                timing["error"] = f"hear: {e!r}"[:300]
            if during and echo_of(heard, conv.last_bot):
                yield line({"type": "heard", "text": heard, "ignored": True})
                return
            conv.log.append({"who": "person", "text": heard, "during_bot": bool(during), **timing})
            yield line({"type": "heard", "text": heard, "ignored": False})
            if conv.ended:
                return
            think = time.perf_counter()
            pieces = conv.answer(heard, heard_lang)
            timing["think_start"] = round(time.perf_counter() - think, 2)
            yield from reply_lines(conv, pieces, timing, started)

        return StreamingResponse(run(), media_type="application/x-ndjson")

    @app.post("/api/text")
    async def text_turn(request: Request):
        body = await request.json()
        conv = conversation(body.get("sid"))
        if conv is None:
            return JSONResponse({"error": "no such conversation: start again"}, status_code=404)
        typed = str(body.get("text") or "")[:500]
        started = time.perf_counter()

        def run():
            conv.log.append({"who": "person", "text": typed, "typed": True})
            yield line({"type": "heard", "text": typed, "ignored": False})
            if not conv.ended:
                yield from reply_lines(conv, conv.answer(typed), {}, started)

        return StreamingResponse(run(), media_type="application/x-ndjson")

    @app.post("/api/end")
    async def end(request: Request):
        body = await request.json()
        conv = conversations.pop(body.get("sid") or "", None)
        if conv is not None:
            conv.save()
        return {"ok": True}

    return app


LOGIN = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,
initial-scale=1"><title>Wimb Voice</title><style>body{margin:0;min-height:100dvh;display:flex;align-items:center;
justify-content:center;background:#0b1020;color:#eef1ff;font-family:system-ui,sans-serif;padding:16px}
form{display:flex;flex-direction:column;gap:12px;width:100%;max-width:320px}input,button{padding:14px;border-radius:12px;
font-size:16px;border:1px solid #2a3360;background:#121834;color:#eef1ff}button{background:#4f8cff;border:none}
p{color:#ff8a95;min-height:1em;margin:0}</style></head><body><form method="post" action="login">
<h2>Wimb Voice</h2><input type="password" name="password" placeholder="Password" autofocus>
<button>Open</button><p>{{ERROR}}</p></form></body></html>"""


def serve(app, port: int = 7860) -> threading.Thread:
    """Run the app on 127.0.0.1:port in a background thread."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    return thread


def open_tunnel(port: int = 7860, log=print) -> str | None:
    """A public https link to 127.0.0.1:port: Gradio's share tunnel, else a Cloudflare quick tunnel."""
    try:
        from gradio.networking import setup_tunnel

        return setup_tunnel("127.0.0.1", port, secrets.token_urlsafe(32), None, None)
    except Exception as e:  # noqa: BLE001
        log("gradio tunnel failed:", repr(e)[:300])
    binary = shutil.which("cloudflared") or "/tmp/cloudflared"
    try:
        if not os.path.exists(binary):
            subprocess.run(["curl", "-fsSL", "-o", binary, CLOUDFLARED], check=True, timeout=300)
            os.chmod(binary, 0o755)
        proc = subprocess.Popen([binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 90
        while time.time() < deadline:
            out = proc.stdout.readline()
            found = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", out or "")
            if found:
                return found.group(0)
    except Exception as e:  # noqa: BLE001
        log("cloudflared failed:", repr(e)[:300])
    return None

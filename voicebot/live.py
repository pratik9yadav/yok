"""The web page to talk to the voice bot (Gradio), run on Kaggle's free GPU and opened with a password.

Pick a task (shop, clinic, order, or your own JSON), press Start, then speak (or type) your side of the call; the bot
answers in voice. On the right: what the bot has found so far. With "save my voice" on, each call's audio and text
are saved to the run's Output tab on Kaggle (private), to make the bot better: real voices and accents.

    app = build_app(ears, mouth, llm, save_dir="/kaggle/working/live")      Ears, Mouth from voicebot.audio
    app.launch(share=True, auth=("wimb", password))
"""

import json
import os
import time
import uuid

from voicebot import tasks
from voicebot.engine import Call

LANG_CHOICES = {"हिंदी (Hindi)": "hi", "मराठी (Marathi)": "mr", "English": "en"}
CUSTOM = "Your own task (JSON below)"


def asr_prompt(call: Call) -> str:
    """A hint for speech to text: what the bot just asked (answers echo the question's words)."""
    bot = [t.text for t in call.turns if t.who == "bot"]
    return bot[-1][-200:] if bot else ""


class Session:
    """One call on the page: the conversation, its audio and timing, saved only with the person's consent."""

    def __init__(self, task: tasks.Task, lang: str, ears, mouth, llm, save_dir: str | None, consent: bool):
        self.call = Call(task, lang=lang, llm=llm)
        self.ears, self.mouth = ears, mouth
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.folder = os.path.join(save_dir, self.id) if save_dir and consent else None
        self.chat: list[dict] = []
        self.timings: list[dict] = []
        self.n = 0
        if self.folder:
            os.makedirs(self.folder, exist_ok=True)

    def _speak(self, reply) -> tuple | None:
        if self.mouth is None or not reply.speech:
            return None
        from voicebot.audio import to_int16

        audio, rate, seconds = self.mouth.say(reply.speech, reply.lang, "bot")
        self.timings[-1]["speak"] = seconds
        if self.folder:
            import soundfile as sf

            sf.write(os.path.join(self.folder, f"{self.n:02d}_bot.wav"), audio, rate)
        return rate, to_int16(audio)

    def _after(self, reply) -> tuple:
        self.chat.append({"role": "assistant", "content": reply.text})
        voice = None
        try:
            voice = self._speak(reply)
        except Exception as e:  # noqa: BLE001 - the text still shows
            self.timings[-1]["error"] = f"voice: {e!r}"[:200]
        self.save()
        return voice, self.chat, self.found(), self.timing_line()

    def start(self) -> tuple:
        self.timings.append({})
        return self._after(self.call.start())

    def hear_audio(self, audio) -> tuple:
        from voicebot.audio import as_samples

        self.n += 1
        samples = as_samples(audio)
        if self.folder:
            import soundfile as sf

            sf.write(os.path.join(self.folder, f"{self.n:02d}_you.wav"), samples, 16000)
        text, lang, seconds = ("", self.call.lang, 0.0)
        if self.ears is not None and len(samples) > 1600:
            text, lang, seconds = self.ears.hear(samples, hint=self.call.lang, allowed=self.call.task.languages,
                                                 prompt=asr_prompt(self.call))
        return self._hear(text, {"hear": seconds, "heard_in": lang})

    def hear_text(self, text: str) -> tuple:
        self.n += 1
        return self._hear(text or "", {})

    def _hear(self, text: str, timing: dict) -> tuple:
        self.chat.append({"role": "user", "content": text or "(…)"})
        started = time.perf_counter()
        reply = self.call.hear(text, timing.get("hear", 0.0))
        timing["think"] = round(time.perf_counter() - started, 3)
        self.timings.append(timing)
        return self._after(reply)

    def found(self) -> dict:
        r = self.call.result()
        return {"outcome": r["outcome"] or "talking…", "language": r["lang"], "values": r["values"],
                "missing": r["missing"]}

    def timing_line(self) -> str:
        t = self.timings[-1] if self.timings else {}
        parts = [f"{name} {t[key]:.1f} s" for key, name in (("hear", "heard"), ("think", "understood"),
                                                              ("speak", "spoke")) if t.get(key) is not None]
        return " · ".join(parts) + (f" · {t['error']}" if t.get("error") else "")

    def save(self) -> None:
        if not self.folder:
            return
        data = {"id": self.id, "task": self.call.task.spec, "result": self.call.result(), "timings": self.timings}
        with open(os.path.join(self.folder, "call.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)


def task_label(task_id: str) -> str:
    return f"{tasks.load(task_id).name('en')} ({task_id})"


def pick_task(label: str, custom_json: str, context_json: str) -> tasks.Task:
    """The task chosen on the page, with its details (context) as edited. Raises TaskError with the reason."""
    try:
        context = json.loads(context_json) if (context_json or "").strip() else {}
    except json.JSONDecodeError as e:
        raise tasks.TaskError(f"details are not valid JSON: {e}") from e
    if not isinstance(context, dict):
        raise tasks.TaskError("details must be a JSON object, like {\"product\": \"LED bulb\"}")
    if label == CUSTOM:
        return tasks.from_json(custom_json, context)
    task_id = label.rsplit("(", 1)[-1].rstrip(")")
    return tasks.load(task_id, context)


def build_app(ears, mouth, llm, save_dir: str | None = None, brain_name: str = "", note: str = ""):
    """The Gradio page."""
    import gradio as gr

    order = sorted(tasks.available(), key=lambda t: (t != "shop_stock", t))  # the shop first
    labels = [task_label(t) for t in order] + [CUSTOM]
    first = tasks.load(order[0])
    example = json.dumps(json.loads((tasks.TASKS_DIR / "order_confirm.json").read_text(encoding="utf-8")),
                         ensure_ascii=False, indent=1)

    def context_of(label):
        if label == CUSTOM:
            return "{}"
        return json.dumps(tasks.load(label.rsplit("(", 1)[-1].rstrip(")")).context, ensure_ascii=False)

    def start(label, lang_label, custom, context, consent):
        try:
            task = pick_task(label, custom, context)
        except tasks.TaskError as e:
            raise gr.Error(str(e)) from e
        session = Session(task, LANG_CHOICES.get(lang_label, "hi"), ears, mouth, llm, save_dir, consent)
        voice, chat, found, timing = session.start()
        return session, voice, chat, found, timing

    def spoken(session, audio):
        if session is None:
            raise gr.Error("Press Start call first")
        if audio is None:
            return session, None, session.chat, session.found(), session.timing_line(), None
        voice, chat, found, timing = session.hear_audio(audio)
        return session, voice, chat, found, timing, None

    def typed(session, text):
        if session is None:
            raise gr.Error("Press Start call first")
        voice, chat, found, timing = session.hear_text(text)
        return session, voice, chat, found, timing, ""

    with gr.Blocks(title="WIMB voice bot") as app:
        gr.Markdown("## WIMB voice bot: test call\nPick a task, press **Start call**, then speak your side "
                    "(tap the mic, speak, tap stop). Hindi, Marathi, English or a mix. "
                    + (f"Brain: `{brain_name}`. " if brain_name else "") + note)
        session = gr.State(None)
        with gr.Row():
            with gr.Column(scale=3):
                task = gr.Dropdown(labels, value=labels[0], label="Task")
                with gr.Row():
                    lang = gr.Dropdown(list(LANG_CHOICES), value=next(iter(LANG_CHOICES)), label="The bot starts in")
                    consent = gr.Checkbox(value=True, label="Save my voice to improve the bot (private, on Kaggle)")
                context = gr.Textbox(json.dumps(first.context, ensure_ascii=False), label="Details (JSON)", lines=1)
                with gr.Accordion("Your own task (JSON)", open=False):
                    custom = gr.Code(example, language="json", label="Task JSON (pick the last task above to use it)")
                go = gr.Button("Start call", variant="primary")
                chat = gr.Chatbot(type="messages", height=380, label="The call")
                bot_voice = gr.Audio(label="Bot", autoplay=True, interactive=False)
                mic = gr.Audio(sources=["microphone"], type="filepath", label="You: tap to speak, tap again to send")
                with gr.Row():
                    text = gr.Textbox(placeholder="…or type what you would say", show_label=False, scale=4)
                    send = gr.Button("Send", scale=1)
            with gr.Column(scale=2):
                found = gr.JSON(label="What the bot found")
                timing = gr.Markdown()
        task.change(context_of, task, context)
        go.click(start, [task, lang, custom, context, consent], [session, bot_voice, chat, found, timing])
        mic.stop_recording(spoken, [session, mic], [session, bot_voice, chat, found, timing, mic])
        send.click(typed, [session, text], [session, bot_voice, chat, found, timing, text])
        text.submit(typed, [session, text], [session, bot_voice, chat, found, timing, text])
    return app

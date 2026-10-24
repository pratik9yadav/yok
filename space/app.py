"""The shop model as a free web page (a Hugging Face Space on a free CPU).

Type the product and what the shopkeeper said; the page shows the 4 answers in plain words, the model's raw JSON
and the seconds it took. On start it downloads the trained model file (.gguf) from the private model repo
(env MODEL_REPO, with the token in env HF_TOKEN) and loads it with llama-cpp-python.

The prompt is exactly the training text: Qwen's chat format with system = score.SYSTEM and
user = score.user_message(product, said). space/deploy.py copies score.py next to this file and writes
shop_words.py (try_on_kaggle.in_words), so there is one source for both.

The pure functions (prompt, answer formatting) need neither gradio nor llama_cpp: tests import this file offline.
Run it on a PC:  MODEL_REPO=<you>/wimb-shop-model-v4 HF_TOKEN=... python space/app.py
"""

import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not (HERE / "score.py").exists():  # in the repo: score.py is one folder up (deploy.py copies it next to app.py)
    sys.path.append(str(HERE.parent))

from score import SYSTEM, parse, user_message

try:
    from shop_words import in_words  # written by deploy.py from try_on_kaggle.in_words
except ImportError:  # in the repo
    from try_on_kaggle import in_words

MODEL_FILE = os.environ.get("MODEL_FILE", "wimb-shop-model.Q4_K_M.gguf")
N_CTX = 2048  # as the Modelfile's num_ctx; calls are short
MAX_TOKENS = 96  # as in training and the tests
STOP = ["<|im_end|>"]
MAX_CHARS = 500
LOADING = "Loading the model (first start takes 1–2 minutes)…"

INTRO = """\
# WIMB shop model
Type the **product** and what the **shopkeeper said** on the call (Hindi, Hinglish or Marathi; Devanagari or
Latin letters), then tap **Ask**. The model tells you: is it in stock, the price, is there delivery, and any other
product the shopkeeper offered. It runs on a free computer (CPU), so an answer takes several seconds.

दुकानदार ने फ़ोन पर जो कहा वह लिखिए, मॉडल बताएगा: माल है या नहीं, दाम, डिलीवरी, और कोई दूसरा सामान।
"""

# New sentences written for this page (not from any test set or the demo questions).
EXAMPLES = [
    ["Asian Paints Tractor Emulsion 10 litre", "haan madam stock mein hai, 2850 lagega, kal subah ghar bhej denge"],
    ["Havells 1.5 sq mm wire 90 metre", "आहे ना, अठराशे पन्नास ला देतो, पण घरी पोचवणार नाही, दुकानातून घेऊन जा"],
    ["Stanley 10 mm impact drill", "वो वाला तो ख़त्म हो गया, ब्लैक एंड डेकर का 10 mm रखा है, चलेगा तो बताइए"],
    ["Amaron 165Ah tubular battery", "ek piece pada hai, saadhe tera hazaar final, fitting wala ladka aake laga dega"],
]


def prompt(product: str, said: str) -> str:
    """The text the model was trained on, up to where its answer starts (as the repo's Modelfile TEMPLATE)."""
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_message(product, said)}<|im_end|>\n"
            "<|im_start|>assistant\n")


def problem(product: str, said: str) -> str | None:
    """What is wrong with a question, or None."""
    if not (said or "").strip():
        return "Please type what the shopkeeper said."
    if len(product or "") > MAX_CHARS or len(said) > MAX_CHARS:
        return f"Please keep each box under {MAX_CHARS} characters."
    return None


def words(reply: str) -> str:
    """The answer in plain words, one per line (Markdown), from the model's reply."""
    answer = parse(reply)
    if answer is None:
        return ("**No answer:** the model's reply was not the 4-part JSON (see the raw reply below). "
                "Try writing the sentence a little differently.")
    parts = in_words(answer).split("; ", 3)  # "in stock: yes; price: ₹220; delivery: no; other product offered: …"
    return "\n".join(f"- {part[0].upper()}{part[1:]}" for part in parts)


def took(seconds: float) -> str:
    return f"Answered in {seconds:.1f} s on a free CPU."


def cpu_threads(cgroup: Path = Path("/sys/fs/cgroup")) -> int:
    """The CPUs this container may use. os.cpu_count() can say the host's (16 or more) on a 2-CPU Space, and too many
    threads make llama.cpp much slower; the container limit (cgroup cpu.max) is the real number. N_THREADS overrides."""
    if os.environ.get("N_THREADS", "").isdigit() and int(os.environ["N_THREADS"]) > 0:
        return int(os.environ["N_THREADS"])
    count = os.cpu_count() or 2
    try:
        quota, period = (cgroup / "cpu.max").read_text().split()[:2]
        if quota != "max":
            count = min(count, max(1, round(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return count


class Model:
    """The model, loaded once in the background so the page opens at once. ask() waits until it is ready."""

    def __init__(self) -> None:
        self.llm = None
        self.error: str | None = None
        self.stage = "starting"
        self.done = threading.Event()
        self.lock = threading.Lock()  # one question at a time: llama.cpp keeps one state

    def start(self) -> "Model":
        threading.Thread(target=self.load, daemon=True).start()
        return self

    def load(self) -> None:
        try:
            repo = os.environ.get("MODEL_REPO")
            if not repo:
                raise RuntimeError("the Space variable MODEL_REPO is not set (space/deploy.py sets it)")
            self.stage = "downloading the model file (about 2.5 GB)"
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=repo, filename=MODEL_FILE, token=os.environ.get("HF_TOKEN") or None)
            self.stage = "loading the model into memory"
            from llama_cpp import Llama

            threads = cpu_threads()
            started = time.time()
            self.llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=threads, n_threads_batch=threads, verbose=False)
            print(f"Model loaded in {time.time() - started:.0f} s, {threads} threads", flush=True)
            self.stage = "warming up"
            started = time.time()
            self.complete(prompt("test", "haan hai"))  # keeps the system prompt in llama.cpp's cache for later asks
            print(f"Warm-up answer in {time.time() - started:.1f} s", flush=True)
            self.stage = "ready"
        except Exception as e:  # noqa: BLE001 - shown on the page
            self.error = f"{type(e).__name__}: {e}"
            print("The model did not load:", self.error, flush=True)
        finally:
            self.done.set()

    def status(self) -> str:
        if self.error:
            return f"The model did not load: {self.error}"
        if self.done.is_set():
            return "The model is ready."
        return f"{LOADING} Now: {self.stage}."

    def complete(self, text: str) -> str:
        with self.lock:
            out = self.llm.create_completion(text, max_tokens=MAX_TOKENS, temperature=0, stop=STOP)
        return out["choices"][0]["text"].strip()

    def ask(self, product: str, said: str) -> str:
        return self.complete(prompt(product, said))


def answer(model: Model, product: str, said: str):
    """What the page shows for one question: (plain words, raw reply, seconds). A generator, so the page can say
    "loading" while the model is not ready yet."""
    product, said = (product or "").strip(), (said or "").strip()
    bad = problem(product, said)
    if bad:
        yield bad, "", ""
        return
    if not model.done.is_set():
        yield LOADING, "", ""
        model.done.wait()
    if model.error:
        yield f"The model did not load: {model.error}", "", ""
        return
    started = time.time()
    reply = model.ask(product, said)
    seconds = time.time() - started
    print(f"Answer in {seconds:.1f} s", flush=True)  # the Space's log: speed only, not what was asked
    yield words(reply), reply, took(seconds)


def build(model: Model):
    """The page (gradio is imported only here)."""
    import gradio as gr

    with gr.Blocks(title="WIMB shop model") as page:
        gr.Markdown(INTRO)
        status = gr.Markdown(model.status())
        product = gr.Textbox(label="Product", placeholder="e.g. 16A MCB", max_lines=1)
        said = gr.Textbox(label="Shopkeeper said", lines=3,
                          placeholder="e.g. haan hai, 220 ka, delivery nahi  /  हाँ है, 220 का, डिलीवरी नहीं")
        ask = gr.Button("Ask", variant="primary")
        result = gr.Markdown()
        seconds = gr.Markdown()
        raw = gr.Textbox(label="The model's reply (JSON)", interactive=False, lines=2)
        gr.Examples(examples=EXAMPLES, inputs=[product, said], label="Examples (tap one, then Ask)")

        def run(product_text, said_text):
            yield from answer(model, product_text, said_text)

        for event in (ask.click, product.submit):
            event(run, [product, said], [result, raw, seconds], concurrency_limit=1)

        timer = gr.Timer(2)  # updates the loading line until the model is ready, then stops
        timer.tick(lambda: (model.status(), gr.Timer(active=not model.done.is_set())), outputs=[status, timer])
    return page


if __name__ == "__main__":
    build(Model().start()).queue(default_concurrency_limit=1).launch(server_name="0.0.0.0")

"""Try the trained model on Kaggle's free GPU: Ollama with the repo's Modelfile, as on a PC. No clicks.

Starts try_model.ipynb on Kaggle with the training notebook's output (the model file) attached, waits
(about 10 minutes) and prints: right or wrong on the example calls, the speed on GPU and on CPU only,
and the answers to your own questions.

    python try_on_kaggle.py                                       the example calls only
    python try_on_kaggle.py --ask "16A MCB" "haan hai, 220 ka"    also your own question (repeat --ask for more)
    python try_on_kaggle.py --questions-json '[{"product": "16A MCB", "said": "haan hai, 220 ka"}]'
    python try_on_kaggle.py --wait                                wait for the run already going, then print it
    python try_on_kaggle.py --make-ask-notebook                   make or refresh your "Ask the shop model" notebook

--make-ask-notebook puts ask_model.ipynb on Kaggle as your notebook wimb-shop-model-ask (run the model once, then
ask it many questions from a phone), runs it once as a check (about 5 minutes) and prints its link.

Needs what run_on_kaggle.py needs (KAGGLE_API_TOKEN, pip install kaggle) and one finished training run.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import run_on_kaggle
from run_on_kaggle import check_gpu, connect, print_log, wait
from score import parse

HERE = Path(__file__).parent
SLUG = "wimb-shop-model-try"
NOTEBOOK = "try_model.ipynb"
ASK_CELL = "#@title 1. Ask your own"
OUT = HERE / "kaggle_output_try"
ASK_SLUG = "wimb-shop-model-ask"  # the "Ask the shop model" notebook: run once, then ask many questions
ASK_NOTEBOOK = "ask_model.ipynb"
ASK_CHECK_FILE = "ask_check.json"  # what its self-check cell saves
ASK_NO_BOX = "No box, only text like VBox(...)? Use cell 7, under it."  # the editor did not draw the ask box
ASK_OUT = HERE / "kaggle_output_ask"
MAX_QUESTIONS = 30
MAX_CHARS = 500
QUESTIONS_LINE = re.compile(r"^QUESTIONS = json\.loads\(.*\)$", re.MULTILINE)


def in_words(answer):
    """An answer in plain words, e.g. "in stock: yes; price: ₹220; delivery: no; other product offered: none"."""
    if not isinstance(answer, dict):
        return "no answer (the reply was not the 4-field JSON)"

    def yes_no(value):
        if value is None:
            return "not said"
        return ("yes" if value else "no") if isinstance(value, bool) else repr(value)

    price = answer.get("price_inr")
    if price is None:
        price = "not said"
    elif isinstance(price, (int, float)) and not isinstance(price, bool):
        price = f"₹{int(price)}" if float(price).is_integer() else f"₹{price}"
    else:
        price = repr(price)
    alternative = answer.get("alternative")
    return (f"in stock: {yes_no(answer.get('in_stock'))}; price: {price}; delivery: {yes_no(answer.get('delivery'))}; "
            f"other product offered: {'none' if alternative is None else alternative}")


def problem(q) -> str | None:
    """What is wrong with one question, or None."""
    if not isinstance(q, dict):
        return 'it is not an object like {"product": ..., "said": ...}'
    product, said = q.get("product", ""), q.get("said")
    if not isinstance(product, str):
        return "product must be text"
    if not isinstance(said, str) or not said.strip():
        return "said (what the shopkeeper said) must be non-empty text"
    if len(said) > MAX_CHARS or len(product) > MAX_CHARS:
        return f"at most {MAX_CHARS} characters each"
    try:
        (product + said).encode("utf-8")
    except UnicodeEncodeError:  # e.g. a lone "\ud800": the notebook could neither print nor save it
        return "product and said must be normal text (a broken character was found)"
    return None


def questions_from(asks: list[list[str]], questions_json: str | None) -> list[dict]:
    """The --ask pairs and the --questions-json list as [{"product", "said"}] (ValueError or TypeError if bad)."""
    try:
        extra = json.loads(questions_json) if questions_json else []
    except json.JSONDecodeError as e:
        raise ValueError(f"--questions-json is not valid JSON: {e}") from None
    if not isinstance(extra, list):
        raise TypeError('--questions-json must be a JSON list like [{"product": "16A MCB", "said": "haan hai, 220 ka"}]')
    items = [{"product": product, "said": said} for product, said in asks] + extra
    if len(items) > MAX_QUESTIONS:
        raise ValueError(f"at most {MAX_QUESTIONS} questions in one run (got {len(items)})")
    for i, q in enumerate(items, 1):
        if problem(q):
            raise ValueError(f"question {i}: {problem(q)}")
    return [{"product": q.get("product", "").strip(), "said": q["said"].strip()} for q in items]


def notebook_with(questions: list[dict]) -> dict:
    """try_model.ipynb with these questions in its QUESTIONS line (one line, any characters)."""
    nb = json.loads((HERE / NOTEBOOK).read_text(encoding="utf-8"))
    line = f"QUESTIONS = json.loads({json.dumps(questions, ensure_ascii=False)!r})"
    for cell in nb["cells"]:
        source = "".join(cell["source"])
        if source.startswith(ASK_CELL):
            source, n = QUESTIONS_LINE.subn(lambda _: line, source)
            if n != 1:
                raise RuntimeError(f"{NOTEBOOK}: the QUESTIONS line is missing; run build_notebook.py")
            cell["source"] = source.splitlines(keepends=True)
            return nb
    raise RuntimeError(f"{NOTEBOOK}: the '{ASK_CELL}' cell is missing; run build_notebook.py")


MODEL = "v4"  # which trained model to attach: v4 (newest, speech-to-text style; run 11) or v2


def training_input(api, kernel: str, model: str = MODEL) -> str:
    """The training notebook (same owner) whose output, the model file, is attached. Prints how its last run went."""
    owner = kernel.split("/", 1)[0]
    training = f"{owner}/{run_on_kaggle.slug_for(model)}"
    print(f"Model: shop model {model}", flush=True)
    try:  # only a hint: the model comes from the training notebook's latest run
        status = api.kernels_status(training).status
        print(f"Training notebook {training}: last run {getattr(status, 'name', status)}", flush=True)
    except Exception as e:  # noqa: BLE001 - diagnostics only
        print(f"Could not read the training notebook {training}:", e, flush=True)
    return training


def push(api, kernel: str, questions: list[dict], model: str = MODEL) -> None:
    """Start the notebook with the training notebook's output (the model file) attached."""
    training = training_input(api, kernel, model)
    print(f"Your questions: {len(questions)}", flush=True)
    run_on_kaggle.push_notebook(api, kernel, notebook_with(questions), NOTEBOOK, kernel_sources=[training])


def push_ask(api, kernel: str, model: str = MODEL) -> None:
    """Make or refresh the "Ask the shop model" notebook on Kaggle, with the model attached. Kaggle runs it once."""
    nb = json.loads((HERE / ASK_NOTEBOOK).read_text(encoding="utf-8"))
    run_on_kaggle.push_notebook(api, kernel, nb, ASK_NOTEBOOK, kernel_sources=[training_input(api, kernel, model)])


def short(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def seconds(value) -> str:
    return "?" if value is None else f"{value:g} s"


TEMPLATES = {"go_template": "the Modelfile's TEMPLATE",
             "gguf_chat_template": "the chat template inside the .gguf (Ollama chose it over the Modelfile's TEMPLATE)"}


def ran_on(processor: str | None) -> str:
    """Where Ollama ran the model, from `ollama ps` ("GPU: Tesla T4" only says the card is there)."""
    if processor == "100% GPU":
        return "the GPU"
    if processor == "100% CPU":
        return "the CPU only (Ollama did not use the GPU)"
    return f"GPU and CPU ({processor})" if processor else "GPU or CPU (not known)"


def report(r: dict) -> None:
    """Print try_results.json for a person: the setup, the example calls, the speed, the questions."""
    print(f"\nOllama {r.get('ollama_version') or '?'}; GPU: {r.get('gpu') or 'none found'}")
    if r.get("model_file"):
        print(f"Model file: {Path(r['model_file']).name} ({r.get('model_mb') or 0:,} MB)")
    print(f"ollama create took {seconds(r.get('create_seconds'))}; loading the model took {seconds(r.get('load_seconds'))}")
    template = r.get("template")
    print("Prompt template:", TEMPLATES.get(template, template) if template else "not in Ollama's log")

    demo = r.get("demo") or []
    where = ran_on(r.get("processor"))
    if demo:
        print(f"\n== Example calls ({len(demo)}, on {where})")
        for i, d in enumerate(demo, 1):
            mark = "✓" if not d["wrong"] else "wrong: " + ", ".join(d["wrong"])
            print(f"{i:3d}. {short(d['said'], 42):42s}  {short(d['reply'], 90)}  {mark}  ({seconds(d.get('seconds'))})")
        print(f"\n{r.get('demo_right', 0)} of {r.get('demo_total', len(demo))} fully right")
    print(f"Speed on {where}: {seconds(r.get('gpu_avg_seconds'))} per answer on average, "
          f"{r.get('gpu_tokens_per_s') or '?'} tokens/s")
    if r.get("cpu"):
        print(f"CPU only: {seconds(r.get('cpu_avg_seconds'))} per answer on average "
              "(not counting loading the model; about what a laptop CPU would take)")
    if r.get("gpu_memory"):
        print("GPU memory used, total:", r["gpu_memory"])
    for key, label in (("ollama_ps", "ollama ps (example calls)"), ("cpu_ollama_ps", "ollama ps (CPU only check)")):
        if r.get(key):
            print(label + ":")
            for line in r[key].splitlines():
                print("   ", line)

    questions = r.get("questions") or []
    if questions:
        print(f"\n== Your questions ({len(questions)}, on {ran_on(r.get('questions_processor'))})")
        for i, q in enumerate(questions, 1):
            print(f"{i}. Product: {q['product']}\n   Shopkeeper: {q['said']}\n   Answer: {in_words(q.get('parsed'))}"
                  f"\n   (reply: {short(q['reply'], 200)})")
    if r.get("errors"):
        print("\nErrors:")
        for error in r["errors"]:
            print("   ", short(error, 500))


def gaps(r: dict, pushed: list[dict] | None = None) -> list[str]:
    """What the run did not answer (empty when it answered everything it was asked).

    pushed: the questions this run was started with (None: not known, e.g. with --wait).
    """
    found = []
    asked = r.get("questions_asked")
    if pushed is not None and asked is not None and asked != pushed:
        found.append("these answers are for other questions (was another run of this notebook started?)")
    if not r.get("demo_total") and not r.get("questions"):
        found.append("the model answered nothing")
    elif r.get("demo_total", 0) < r.get("demo_asked", 0):
        found.append(f"{r['demo_asked'] - r['demo_total']} of {r['demo_asked']} example calls got no answer")
    if asked is not None and len(r.get("questions") or []) < len(asked):
        found.append(f"{len(asked) - len(r.get('questions') or [])} of your {len(asked)} questions got no answer")
    return found


def fetch(api, kernel: str, ok: bool = True) -> dict | None:
    """Download the results and print them, and the end of Kaggle's log when the run did not finish (ok False).

    The results, or None when the run left no try_results.json.
    """
    OUT.mkdir(exist_ok=True)
    results = OUT / "try_results.json"
    results.unlink(missing_ok=True)  # never report an older run's answers
    api.kernels_output(kernel, str(OUT), file_pattern=r"(try_results\.json|.*\.log|.*\.ipynb)$", force=True)
    r = json.loads(results.read_text(encoding="utf-8")) if results.exists() else None
    if r is not None:
        report(r)
    if r is None or not ok:
        print("\nNo try_results.json: the run stopped early." if r is None else "\nThe run did not finish.",
              "Kaggle's log:")
        for log in OUT.glob("*.log"):
            print_log(log)
    if r is not None:
        print(f"\nDownloaded to {OUT}: {sorted(p.name for p in OUT.iterdir())}")
        print(f"The run on Kaggle: https://www.kaggle.com/code/{kernel}")
    return r


ASK_STEPS = f"""
1. Open it → Edit. (The page that opens first is the saved check: no one can type in it.)
2. Right side: Input shows wimb-shop-model-training, Accelerator GPU T4, Internet on.
3. Run all (about 3–4 minutes each time you start it).
4. At the bottom, type the product and what the shopkeeper said in the Ask box, tap Ask. Ask as many as you like.
   {ASK_NO_BOX}
When done: Run → Stop session, so your free GPU hours are not used up."""


def answered(check: dict | None) -> bool:
    """True when the ask notebook's self-check got the 4-field JSON from the model."""
    reply = ((check or {}).get("self_check") or {}).get("reply")
    return isinstance(reply, str) and parse(reply) is not None


def report_ask(check: dict) -> None:
    """Print ask_check.json for a person: did the model answer, how fast, on what."""
    print(f"\nOllama {check.get('ollama_version') or '?'}; GPU: {check.get('gpu') or 'none found'}")
    print(f"ollama create took {seconds(check.get('create_seconds'))}; "
          f"loading the model took {seconds(check.get('load_seconds'))}")
    sc = check.get("self_check") or {}
    if sc:
        print(f"\nSelf-check: {sc.get('product')} / {sc.get('said')}")
        print(f"   Answer: {in_words(parse(sc.get('reply') or ''))}")
        print(f"   (reply: {short(sc.get('reply', ''), 200)}; {seconds(sc.get('seconds'))} on "
              f"{ran_on(check.get('processor'))})")
    print("Self-check: " + ("the model answered." if answered(check) else "NO ANSWER from the model."))
    print("Ask box:", check.get("ask_box") or "not shown (the run stopped before it)")
    if (check.get("ask_box") or "").startswith(("built", "shown")):
        print("   (this check can only build the box; whether the editor draws it is seen only there, after Run all)")
    if check.get("errors"):
        print("Errors:")
        for error in check["errors"]:
            print("   ", short(error, 500))


def fetch_ask(api, kernel: str, ok: bool = True) -> dict | None:
    """Download the ask notebook's ask_check.json (and Kaggle's log) and print it. None when the run left none."""
    ASK_OUT.mkdir(exist_ok=True)
    path = ASK_OUT / ASK_CHECK_FILE
    path.unlink(missing_ok=True)  # never report an older run's check
    api.kernels_output(kernel, str(ASK_OUT), file_pattern=r"(ask_check\.json|.*\.log)$", force=True)
    check = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if check is not None:
        report_ask(check)
    if check is None or not ok:
        print(f"\nNo {ASK_CHECK_FILE}: the run stopped early." if check is None else "\nThe run did not finish.",
              "Kaggle's log:")
        for log in ASK_OUT.glob("*.log"):
            print_log(log)
    return check


def make_ask_notebook(api, kernel: str, model: str = MODEL) -> bool:
    """Push the ask notebook, wait for its check run, print the result and the link. True when it works."""
    push_ask(api, kernel, model)
    time.sleep(30)
    ok = wait(api, kernel, limit_min=30)
    check = fetch_ask(api, kernel, ok)
    works = ok and answered(check) and check.get("ask_box") != "failed"
    print(f"\nYour notebook: https://www.kaggle.com/code/{kernel}")
    print(ASK_STEPS if works else "It is not ready yet: the check above failed (send Claude this log).")
    return works


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--ask", nargs=2, action="append", default=[], metavar=("PRODUCT", "SAID"),
                      help="a question of your own: the product, and what the shopkeeper said (repeatable)")
    args.add_argument("--questions-json", metavar="TEXT",
                      help='questions as a JSON list: [{"product": ..., "said": ...}] (up to 30)')
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    args.add_argument("--make-ask-notebook", action="store_true",
                      help='make or refresh your "Ask the shop model" notebook on Kaggle, and check it once')
    args.add_argument("--model", choices=["v4", "v2"], default=MODEL,
                      help=f"which trained shop model to use (default {MODEL})")
    opts = args.parse_args()
    if opts.make_ask_notebook:
        if opts.ask or opts.questions_json is not None or opts.wait:
            args.error("--make-ask-notebook can't be used with --ask, --questions-json or --wait")
        api, kernel = connect(ASK_SLUG)
        check_gpu(api, need_hours=0.5)
        sys.exit(0 if make_ask_notebook(api, kernel, opts.model) else 1)
    try:
        questions = questions_from(opts.ask, opts.questions_json)
    except (TypeError, ValueError) as e:
        args.error(str(e))
    api, kernel = connect(SLUG)
    if not opts.wait:
        check_gpu(api, need_hours=0.5)
        push(api, kernel, questions, opts.model)
        time.sleep(30)
    ok = wait(api, kernel, limit_min=60)
    r = fetch(api, kernel, ok)
    missing = gaps(r, None if opts.wait else questions) if r is not None else []
    for why in missing:
        print("Not answered in full:", why)
    sys.exit(0 if ok and r is not None and not missing else 1)


if __name__ == "__main__":
    main()

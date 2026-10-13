"""Build train_colab.ipynb, try_model.ipynb and ask_model.ipynb from score.py, the data files and Modelfile.

Run after changing the data, score.py or Modelfile:  python build_notebook.py
"""

import inspect
import json
from pathlib import Path

import try_on_kaggle

HERE = Path(__file__).parent
REPO = "pratik9yadav/yok"
BRANCH = "main"
NOTEBOOK = "train_colab.ipynb"
TRY_NOTEBOOK = try_on_kaggle.NOTEBOOK
ASK_NOTEBOOK = try_on_kaggle.ASK_NOTEBOOK
DEMO_FILE = "demo_questions.jsonl"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


# Folds a cell's code on Kaggle (its own key) and in Jupyter too; Colab's "cellView: form" only works in Colab.
FOLDED = {"_kg_hide-input": True, "jupyter": {"source_hidden": True}}


def code(text: str, hidden: bool = False, folded: bool = False) -> dict:
    return {
        "cell_type": "code",
        "metadata": {**({"cellView": "form"} if hidden else {}), **(FOLDED if folded else {})},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def titled(title: str, body: str) -> str:
    """A code cell's text: its #@title line, then the body."""
    return f"#@title {title}\n{body.strip(chr(10))}"


def embedded(name: str) -> str:
    text = (HERE / "data" / name).read_text(encoding="utf-8").strip()
    if "'''" in text or text.endswith("\\"):
        raise ValueError(f"{name} can't be embedded in a raw string")
    return text


INTRO = f"""
# WIMB: train your own model on a free Colab GPU

**How to run it**
1. Check the top right says **T4**. If it doesn't: **Runtime → Change runtime type → T4 GPU → Save**.
2. Click **Runtime → Run all**.
3. In the first minute Google asks to connect **Google Drive**. Click **Connect** and **Allow**
   (the trained model is saved there).
4. Keep this tab open. It takes about 30–40 minutes.
5. When it's done: **File → Save a copy in GitHub** → repository `{REPO}`,
   branch `{BRANCH}`, file path `{NOTEBOOK}` → **OK**.
   Then tell Claude it's done. Claude reads the results from GitHub.

**What it does:** takes a free model (Qwen3 4B), tests it on shop calls, trains it on
400 examples, tests it again, then tests the saved Ollama file too. Two test sets: 30 calls
and 56 harder ones written in styles the training data does not have. The model goes to Google Drive.

It also runs on Kaggle's free GPU: `run_on_kaggle.py` starts it there with no clicks.
"""

CHECK_GPU = """
#@title 1. Check the GPU
import os, shutil, subprocess, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Kaggle gives two GPUs; training uses one
# nvidia-smi is not always installed, so also ask torch (in its own process: nothing is imported here yet).
cuda = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                      capture_output=True, text=True).stdout.strip() == "True"
if not (cuda or shutil.which("nvidia-smi")):
    # What the machine has, to see why there is no GPU (no secret values printed).
    print("nvidia devices:", [d for d in os.listdir("/dev") if d.startswith("nvidia")])
    print("Kaggle settings:", {k: v for k, v in os.environ.items()
                               if k.startswith("KAGGLE") and any(w in k for w in ("ACCELERATOR", "GPU", "MACHINE", "RUN_TYPE"))})
    raise RuntimeError("No GPU. Colab: Runtime > Change runtime type > T4 GPU > Save, then Run all. "
                       "Kaggle: verify your phone, then turn on GPU T4 once in the notebook editor (Settings > Accelerator).")
print(subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.get_device_name(0))"],
                     capture_output=True, text=True).stdout or "GPU found")
"""

INSTALL = """
#@title 2. Install the training tools (about 3–5 minutes)
!pip install -q unsloth
"""

SETTINGS = """
#@title 3. Settings (and connect Google Drive now, so nothing waits for you later)
MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"  #@param {type:"string"}
EPOCHS = 2  #@param {type:"integer"}
# v2: training data built from real products, prices and spellings, plus 151 real Hindi answers.
# v1: the first 400 made-up examples (kept to compare against).
TRAIN_SET = "v2"  #@param ["v2", "v1"]
OUTPUT_NAME = "wimb-shop-model"  #@param {type:"string"}
SAVE_TO_DRIVE = True  #@param {type:"boolean"}
MAKE_OLLAMA_FILE = True  #@param {type:"boolean"}
# This notebook is set up for Qwen3 instruct models (their chat format is used below).

ON_KAGGLE = os.path.exists("/kaggle/working")
WORK = "/tmp" if ON_KAGGLE else "/content"  # big temporary files (Kaggle saves all of /kaggle/working)
if ON_KAGGLE:
    SAVE_DIR = f"/kaggle/working/{OUTPUT_NAME}"  # shown in the notebook's Output tab
elif SAVE_TO_DRIVE:
    SAVE_DIR = f"/content/drive/MyDrive/{OUTPUT_NAME}"
    from google.colab import drive
    drive.mount("/content/drive")
else:
    SAVE_DIR = None
"""

LOAD = """
#@title 5. Load the base model (about 2–3 minutes)
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import importlib.metadata as md
import torch

print({p: md.version(p) for p in ("unsloth", "unsloth_zoo", "torch", "transformers", "trl", "peft")})

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=1024, load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
# LoRA: train a small add-on (about 1% of the weights) instead of the whole model.
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=3407,
)
"""

BEFORE = """
#@title 6. Test the model BEFORE training
def ask(product, said):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message(product, said)}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_tests(label):
    FastLanguageModel.for_inference(model)
    try:
        results = evaluate(ask, SETS)
    finally:
        FastLanguageModel.for_training(model)
    show(label, results)
    return results


before = run_tests("BEFORE training")
"""

TRAIN = """
#@title 7. Train (about 10–15 minutes)
from datasets import Dataset
from trl import SFTConfig, SFTTrainer


def to_text(ex):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message(ex["product"], ex["said"])},
                {"role": "assistant", "content": answer_text(ex["answer"])}]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


dataset = Dataset.from_list([to_text(ex) for ex in TRAIN_DATA])
print(dataset[0]["text"])

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text", max_length=1024,
        per_device_train_batch_size=4, gradient_accumulation_steps=2,
        num_train_epochs=EPOCHS, learning_rate=2e-4, warmup_steps=5, lr_scheduler_type="linear",
        optim="adamw_8bit", weight_decay=0.01, logging_steps=10, seed=3407,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        output_dir="outputs", report_to="none",
    ),
)
# Learn only the answers, not the questions.
trainer = train_on_responses_only(
    trainer, instruction_part="<|im_start|>user\\n", response_part="<|im_start|>assistant\\n",
)
stats = trainer.train()
print(f"Training took {stats.metrics['train_runtime'] / 60:.1f} min, loss {stats.metrics['train_loss']:.3f}")
"""

AFTER = """
#@title 8. Test the model AFTER training
after = run_tests("AFTER training")
payload = results_payload(MODEL_NAME, EPOCHS, before, after)
payload.update(train_set=TRAIN_SET, train_rows=len(TRAIN_DATA))
print("\\n== Before vs after (all 4 fields right)")
for name, res in payload["sets"].items():
    print(f"   {name:5s} {res['before']['all_correct']:4.0%} -> {res['after']['all_correct']:4.0%}  ({res['examples']} calls)")
print("\\nRESULTS_JSON", json.dumps({k: {n: r[k] for n, r in payload["sets"].items()} for k in ("before", "after")}))
if ON_KAGGLE:  # run_on_kaggle.py downloads this
    with open("/kaggle/working/results.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
"""

TRY = """
#@title 9. Try your own sentence
product = "Activa clutch plate"  #@param {type:"string"}
said = "haan hai, 650 ka padega, ghar pe bhej denge"  #@param {type:"string"}
FastLanguageModel.for_inference(model)
print(ask(product, said))
"""

SAVE = '''
#@title 10. Save your model: Google Drive on Colab, the Output tab on Kaggle (the Ollama file takes 10–15 minutes)
import glob, re

lora_dir = f"{WORK}/{OUTPUT_NAME}-lora"
model.save_pretrained(lora_dir)
tokenizer.save_pretrained(lora_dir)
if SAVE_DIR:
    os.makedirs(SAVE_DIR, exist_ok=True)
    shutil.copytree(lora_dir, f"{SAVE_DIR}/lora", dirs_exist_ok=True)
    print("Saved the LoRA add-on to", f"{SAVE_DIR}/lora")

if MAKE_OLLAMA_FILE:
    gguf_dir = f"{WORK}/{OUTPUT_NAME}"
    result = model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
    files = (result or {}).get("gguf_files") if isinstance(result, dict) else None
    files = files or glob.glob(f"{gguf_dir}*/**/*.gguf", recursive=True)
    if not files:
        raise RuntimeError("No .gguf file was made. The LoRA add-on is still saved; send Claude the error above.")
    q4 = [f for f in files if "q4_k_m" in os.path.basename(f).lower()] or sorted(files, key=os.path.getsize)
    gguf = q4[0]
    name = f"{OUTPUT_NAME}.Q4_K_M.gguf"

    modelfile = make_modelfile(name)
    modelfile_path = os.path.join(os.path.dirname(gguf), "Modelfile.wimb")  # gguf_dir itself may be cleaned up
    with open(modelfile_path, "w", encoding="utf-8") as fh:
        fh.write(modelfile)
    print(modelfile)

    if SAVE_DIR:
        shutil.copy(gguf, f"{SAVE_DIR}/{name}")
        shutil.copy(modelfile_path, f"{SAVE_DIR}/Modelfile")

if SAVE_DIR:
    for path in sorted(glob.glob(f"{SAVE_DIR}/*")):
        size = os.path.getsize(path) / 1e6 if os.path.isfile(path) else 0
        print(f"   {path}  {size:,.0f} MB" if size else f"   {path}/")
'''

GGUF_TEST = '''
#@title 11. Test the exported GGUF file itself (CPU, up to 25 minutes; the answers Ollama would give)
import socket, subprocess, time, urllib.request

gguf_report = {"error": "not run"}
server = None
try:
    if not MAKE_OLLAMA_FILE:
        raise RuntimeError("MAKE_OLLAMA_FILE is off, so there is no GGUF file to test")
    binary = subprocess.run(
        ["bash", "-c", "find / -xdev -name llama-server -type f -perm -u+x 2>/dev/null | head -1"],
        capture_output=True, text=True).stdout.strip()
    if not binary:
        raise RuntimeError("llama-server was not found (Unsloth builds it while making the GGUF)")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen([binary, "-m", gguf, "--port", str(port), "--host", "127.0.0.1", "-c", "2048",
                               "-t", str(os.cpu_count() or 2), "--jinja"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):  # up to 6 minutes to load the model
        if server.poll() is not None:
            raise RuntimeError(f"llama-server stopped at start-up (exit code {server.returncode})")
        try:
            urllib.request.urlopen(base + "/health", timeout=5).read()
            break
        except Exception:
            time.sleep(3)
    else:
        raise RuntimeError("llama-server did not become ready")

    def ask_gguf(product, said):
        body = json.dumps({"messages": [{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": user_message(product, said)}],
                           "temperature": 0, "max_tokens": 96}).encode()
        req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)["choices"][0]["message"]["content"].strip()

    started = time.time()
    gguf_results = evaluate(ask_gguf, SETS, deadline=started + 25 * 60)
    show("GGUF Q4_K_M", gguf_results)
    gguf_report = {"gguf": os.path.basename(gguf), "seconds": round(time.time() - started),
                   "sets": {n: {"examples": r["examples"], "of": r["of"],
                                **({k: round(r[k], 3) for k in KEYS} if r["examples"] else {}),
                                "mistakes": r["mistakes"], "replies": r["replies"]}
                            for n, r in gguf_results.items()}}
except Exception as e:  # the main results are already saved; this test must never fail the run
    print("GGUF test failed:", repr(e))
    gguf_report = {"error": repr(e)}
finally:
    if server is not None:
        server.terminate()
if ON_KAGGLE:
    with open("/kaggle/working/results_gguf.json", "w", encoding="utf-8") as fh:
        json.dump(gguf_report, fh, ensure_ascii=False, indent=1)
'''

NEXT = f"""
## Done. Next steps

**1. Save this notebook to GitHub** so Claude can read the results:
**File → Save a copy in GitHub** → `{REPO}`, branch `{BRANCH}`,
path `{NOTEBOOK}`.

**2. Run your model on your PC**
1. Install Ollama from ollama.com.
2. From Google Drive, download the `wimb-shop-model` folder (the `.gguf` file and `Modelfile`).
3. Open a terminal in that folder and run:
```
ollama create wimb-shop -f Modelfile
ollama run wimb-shop "Product: 16A MCB
Shopkeeper: haan hai, 220 ka, delivery nahi"
```
"""


TRY_INTRO = """
# WIMB: try the trained shop model (Ollama on Kaggle's free GPU)

**What it does:** takes the model made by the notebook `wimb-shop-model-training`, loads it into
Ollama exactly as you would on a PC (with the repo's `Modelfile`), answers some example shop calls,
shows how fast it is on the GPU and on CPU only, and then answers your own questions.
It takes about 5–10 minutes.

**Ask your own question here, by hand**
1. Open this notebook on kaggle.com → **Edit**.
2. The trained model must be attached: right side → **Add Input** → **Notebooks** → **Your Work** →
   `wimb-shop-model-training` → **+**. (If it is already under **Input**, skip this.)
3. Settings (right side): Accelerator **GPU T4**, Internet **on**.
4. In cell **1. Ask your own**, type the product in `PRODUCT` and what the shopkeeper said in `SAID`.
5. **Run all**. Your answer is in cell **8. Your questions**.

**Or with no clicks:** ask Claude ("ask the model: 16A MCB / haan hai, 220 ka, delivery nahi"),
or on GitHub: **Actions → Ask the model on Kaggle → Run workflow**.
"""

TRY_ASK = """
#@title 1. Ask your own: type the product and what the shopkeeper said, then Run all
import json, os, re, shutil, subprocess, time, urllib.request

# For example:  PRODUCT = "16A MCB"  and  SAID = "haan hai, 220 ka, delivery nahi"
PRODUCT = ""
SAID = ""
# Many questions at once: a JSON list of {"product": ..., "said": ...} (Claude and GitHub Actions fill this line).
QUESTIONS = json.loads('[]')
CPU_CHECK = 3  # how many example calls to also time on CPU only (no GPU, like a laptop)
WORKING_DIR = "/kaggle/working"  # try_results.json is saved here (the notebook's Output tab)
"""

def run_helpers(results_file: str, ollama_cell: int) -> str:
    """save_results(), failed() and sh(): shared by the try and ask notebooks (each fills its own RESULTS)."""
    return f'''OLLAMA_LOG = "/tmp/ollama.log"  # what `ollama serve` prints (cell {ollama_cell})


def save_results():
    """Write RESULTS to {results_file}. Done after every stage, so a stop part-way loses nothing."""
    if os.path.isdir(WORKING_DIR):
        with open(os.path.join(WORKING_DIR, "{results_file}"), "w", encoding="utf-8") as fh:
            json.dump(RESULTS, fh, ensure_ascii=False, indent=1)


def failed(stage, error):
    """Record a failed stage and go on (Kaggle's Run all stops at the first error that is not caught)."""
    message = f"{{stage}}: {{error!r}}" if isinstance(error, Exception) else f"{{stage}}: {{error}}"
    print("FAILED", message)
    RESULTS["errors"].append(message)
    save_results()


def sh(command, timeout=120):
    """What a shell command printed, or "" when the command is missing or fails."""
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception:
        return ""
'''


TRY_HELPERS = '''
RESULTS = {"ollama_version": None, "gpu": None, "model_file": None, "model_mb": None, "create_seconds": None,
           "load_seconds": None, "demo": [], "demo_asked": 0, "demo_right": 0, "demo_total": 0, "processor": None,
           "template": None, "gpu_avg_seconds": None, "gpu_tokens_per_s": None, "gpu_memory": None, "cpu": [],
           "cpu_avg_seconds": None, "ollama_ps": None, "cpu_ollama_ps": None, "questions_asked": None,
           "questions": [], "questions_processor": None, "errors": []}
''' + run_helpers("try_results.json", 4) + '''

def average(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


print(len(DEMO), "example calls")
'''

# Shared by try_model.ipynb and ask_model.ipynb: find the model file, start Ollama, make the model (ollama create).
FIND_TITLE = "Find the trained model (the training notebook's output, attached as input)"
FIND_MODEL = """
import glob


def rank(path):
    name = os.path.basename(path).lower()
    return ("wimb-shop-model" not in name or "q4_k_m" not in name, "q4_k_m" not in name, path)


found = sorted(glob.glob("/kaggle/input/**/*.gguf", recursive=True), key=rank)
if not found:
    print("No .gguf file under /kaggle/input. What is attached:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root[len("/kaggle/input"):].count(os.sep)
        print("   " * depth + (os.path.basename(root) or root) + "/")
        for name in sorted(files)[:20]:
            print("   " * (depth + 1) + name)
        if depth >= 3:
            dirs[:] = []
    failed("find the model", "no .gguf file under /kaggle/input")
    raise RuntimeError("No model file. Attach the training notebook's output: Add Input → Notebooks → Your Work → "
                       "wimb-shop-model-training, then Run all again.")
MODEL_FILE = found[0]
RESULTS["model_file"] = MODEL_FILE
RESULTS["model_mb"] = round(os.path.getsize(MODEL_FILE) / 1e6)
print(f"Model file: {MODEL_FILE}  ({RESULTS['model_mb']:,} MB)")
save_results()
"""

OLLAMA_TITLE = "Install Ollama and start it (about 1–2 minutes)"
START_OLLAMA = """
OLLAMA_URL = "http://127.0.0.1:11434"


def ollama_version():
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/version", timeout=5) as resp:
            return json.load(resp).get("version")
    except Exception:
        return None


if not ollama_version():
    if not shutil.which("ollama"):
        # zstd unpacks Ollama's download (.tar.zst). The download already holds the CUDA libraries; lspci is
        # left out on purpose: with it (and no nvidia-smi) install.sh would try to install NVIDIA drivers.
        if not shutil.which("zstd"):
            sh("apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd", timeout=600)
        install = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True,
                                 capture_output=True, text=True, timeout=1200)
        print("\\n".join((install.stdout + install.stderr).replace("\\r", "\\n").splitlines()[-15:]))
        if not shutil.which("ollama"):
            failed("install Ollama", (install.stdout + install.stderr)[-1500:])
            raise RuntimeError("Ollama did not install (see above)")
    print(sh("ollama --version 2>&1"))
    log = open(OLLAMA_LOG, "w")  # stays open: the server writes to it
    server = subprocess.Popen(["ollama", "serve"], env={**os.environ, "OLLAMA_HOST": "127.0.0.1:11434"},
                              stdout=log, stderr=subprocess.STDOUT)
    for _ in range(60):
        if ollama_version() or server.poll() is not None:
            break
        time.sleep(1)
if not ollama_version():
    tail = sh(f"tail -n 40 {OLLAMA_LOG}")
    print(tail)
    failed("start Ollama", tail[-1500:])
    raise RuntimeError("Ollama did not start (its log is above)")
RESULTS["ollama_version"] = ollama_version()
RESULTS["gpu"] = (sh("nvidia-smi --query-gpu=name --format=csv,noheader").splitlines() or [None])[0]
print("Ollama", RESULTS["ollama_version"], "is running. GPU:", RESULTS["gpu"])
save_results()
"""

CREATE_TITLE = "Make the Ollama model from the file, exactly as on a PC (ollama create, about 1 minute)"
CREATE_MODEL = """
os.makedirs("/tmp/wimb", exist_ok=True)
link = "/tmp/wimb/wimb-shop-model.Q4_K_M.gguf"
if os.path.lexists(link):
    os.remove(link)
os.symlink(MODEL_FILE, link)
modelfile = make_modelfile("wimb-shop-model.Q4_K_M.gguf")
if modelfile != MODELFILE_REPO:
    failed("Modelfile", "the Modelfile made here differs from the repo's Modelfile")
    raise RuntimeError("The Modelfile made here differs from the repo's Modelfile: run build_notebook.py again")
with open("/tmp/wimb/Modelfile", "w", encoding="utf-8") as fh:
    fh.write(modelfile)
print(modelfile)


def create():
    started = time.time()
    done = subprocess.run(["ollama", "create", "wimb-shop", "-f", "/tmp/wimb/Modelfile"],
                          capture_output=True, text=True, timeout=1800)
    output = re.sub(r"\\x1b\\[[0-9;?]*[A-Za-z]", "", done.stdout + done.stderr).replace("\\r", "\\n")  # no progress codes
    print("\\n".join(line for line in output.splitlines()[-8:] if line.strip()))
    return done.returncode, output, round(time.time() - started, 1)


exit_code, output, RESULTS["create_seconds"] = create()
if exit_code != 0:  # some versions do not follow the link: try once with a real copy
    print("ollama create failed with the linked file; trying again with a copy")
    os.remove(link)
    shutil.copy(MODEL_FILE, link)
    exit_code, output, RESULTS["create_seconds"] = create()
if exit_code != 0:
    failed("ollama create", output[-1500:])
    raise RuntimeError("ollama create failed (see above)")
print(f"ollama create took {RESULTS['create_seconds']} s")
save_results()
"""
TRY_FIND = titled(f"3. {FIND_TITLE}", FIND_MODEL)
TRY_OLLAMA = titled(f"4. {OLLAMA_TITLE}", START_OLLAMA)
TRY_CREATE = titled(f"5. {CREATE_TITLE}", CREATE_MODEL)

OLLAMA_API = """
def generate(body):
    request = urllib.request.Request(OLLAMA_URL + "/api/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as resp:
        return json.load(resp)


def in_memory():
    \"\"\"The model as Ollama holds it now (from /api/ps: size, size_vram, ...), or None when it is not loaded.\"\"\"
    with urllib.request.urlopen(OLLAMA_URL + "/api/ps", timeout=10) as resp:
        models = json.load(resp).get("models") or []
    return next((m for m in models if m.get("name", "").split(":")[0] == "wimb-shop"), None)


def processor():
    \"\"\"Where the model runs now, as `ollama ps` says it: "100% GPU", "100% CPU", "25%/75% CPU/GPU", or None.\"\"\"
    try:
        m = in_memory()
    except Exception:
        return None
    if not m or not m.get("size"):
        return None
    cpu = round(100 * (m["size"] - (m.get("size_vram") or 0)) / m["size"])
    return "100% GPU" if cpu <= 0 else "100% CPU" if cpu >= 100 else f"{cpu}%/{100 - cpu}% CPU/GPU"
"""

# Ollama's generate options, sent with every question by both notebooks (so their answers match).
OPTIONS = '"temperature": 0, "num_predict": 96'

TRY_API = """
#@title 6. ask(): one question to the model through Ollama's API, as an app would
def ask(product, said, cpu=False):
    \"\"\"The model's reply and how long it took. No system message is sent: the Modelfile's SYSTEM is used,
    exactly as with `ollama run`. cpu=True runs it on the CPU only (num_gpu 0).\"\"\"
    options = {""" + OPTIONS + """, **({"num_gpu": 0} if cpu else {})}
    started = time.time()
    data = generate({"model": "wimb-shop", "prompt": user_message(product, said), "stream": False, "options": options})
    seconds = round(time.time() - started, 2)
    speed = load = None
    if data.get("eval_count") and data.get("eval_duration"):
        speed = round(data["eval_count"] / (data["eval_duration"] / 1e9), 1)
    if data.get("load_duration") is not None:
        load = round(data["load_duration"] / 1e9, 2)
    return {"reply": data.get("response", "").strip(), "seconds": seconds, "tokens_per_s": speed, "load_seconds": load}

""" + OLLAMA_API + """

def back_to_gpu():
    \"\"\"Unload the model and load it again with the normal settings (the GPU). Needed after the CPU check:
    a request without num_gpu never makes Ollama reload, so it would go on using the CPU copy.\"\"\"
    generate({"model": "wimb-shop", "keep_alive": 0})  # no prompt and keep_alive 0: unload
    for _ in range(60):
        if in_memory() is None:
            break
        time.sleep(1)
    generate({"model": "wimb-shop"})  # no prompt: only load


def template_used():
    \"\"\"Which prompt template Ollama chose when it loaded the model, from its log: "go_template" (the Modelfile's
    TEMPLATE) or "gguf_chat_template" (the one inside the .gguf), or None (not logged).\"\"\"
    try:
        with open(OLLAMA_LOG, encoding="utf-8", errors="replace") as fh:
            found = re.findall(r"template selection.*?\\bselected=(\\w+)", fh.read())
    except OSError:
        return None
    return found[-1] if found else None


def wrong_fields(reply, want):
    got = parse(reply)
    return list(FIELDS) if got is None else [f for f in FIELDS if not same(f, got[f], want[f])]


print('Ready. For example: ask("16A MCB", "haan hai, 220 ka, delivery nahi")')
"""

TRY_DEMO = """
#@title 7. The example calls: right or wrong, and how fast (GPU, then CPU only)
RESULTS.update(demo=[], cpu=[], demo_asked=len(DEMO))
try:  # the first answer also loads the model into the GPU
    warm = ask(DEMO[0]["product"], DEMO[0]["said"]) if DEMO else ask("16A MCB", "haan hai")
    RESULTS["load_seconds"] = warm["load_seconds"]
    print(f"Model loaded in {warm['load_seconds']} s (the first answer took {warm['seconds']} s in all)")
except Exception as e:
    failed("warm-up", e)

for i, ex in enumerate(DEMO, 1):
    try:
        res = ask(ex["product"], ex["said"])
    except Exception as e:
        failed(f"example {i}", e)
        continue
    wrong = wrong_fields(res["reply"], ex["answer"])
    RESULTS["demo"].append({"product": ex["product"], "said": ex["said"], "want": ex["answer"], "reply": res["reply"],
                            "wrong": wrong, "seconds": res["seconds"], "tokens_per_s": res["tokens_per_s"]})
    print(f"\\n{i}. Product:    {ex['product']}\\n   Shopkeeper: {ex['said']}\\n   Model:      {res['reply']}")
    print("   " + ("all 4 right" if not wrong else "wrong: " + ", ".join(wrong)) + f"  ({res['seconds']} s)")

try:
    demo = RESULTS["demo"]
    RESULTS["demo_right"] = sum(not d["wrong"] for d in demo)
    RESULTS["demo_total"] = len(demo)
    RESULTS["gpu_avg_seconds"] = average([d["seconds"] for d in demo])
    RESULTS["gpu_tokens_per_s"] = average([d["tokens_per_s"] for d in demo])
    RESULTS["ollama_ps"] = sh("ollama ps") or None
    RESULTS["processor"] = processor()  # the PROCESSOR column of `ollama ps`
    RESULTS["template"] = template_used()
    RESULTS["gpu_memory"] = sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader") or None
    print(f"\\n== {RESULTS['demo_right']} of {RESULTS['demo_total']} example calls fully right"
          + (f" ({len(DEMO)} asked)" if len(demo) < len(DEMO) else ""))
    print(f"Ran on {RESULTS['processor'] or '? (unknown)'}: {RESULTS['gpu_avg_seconds']} s per answer on average, "
          f"{RESULTS['gpu_tokens_per_s']} tokens/s")
    print("ollama ps (PROCESSOR shows the GPU/CPU split):\\n" + (RESULTS["ollama_ps"] or "   (not available)"))
    print("GPU memory used, total:", RESULTS["gpu_memory"] or "(not available)")
    print("Prompt template Ollama chose:", RESULTS["template"] or "(not in its log)")
    if demo and RESULTS["processor"] != "100% GPU":
        failed("GPU", f"Ollama did not run the model fully on the GPU (ollama ps: {RESULTS['processor']})")
except Exception as e:
    failed("summary", e)
save_results()

for ex in DEMO[:CPU_CHECK]:  # the first of these also loads the model again, into normal memory
    try:
        res = ask(ex["product"], ex["said"], cpu=True)
    except Exception as e:
        failed("CPU check", e)
        continue
    RESULTS["cpu"].append({"said": ex["said"], "reply": res["reply"], "wrong": wrong_fields(res["reply"], ex["answer"]),
                           "seconds": res["seconds"], "load_seconds": res["load_seconds"],
                           "tokens_per_s": res["tokens_per_s"]})
RESULTS["cpu_ollama_ps"] = sh("ollama ps") or None  # should say 100% CPU
# Not counting the time to load the model into memory (that happens once).
RESULTS["cpu_avg_seconds"] = average([max(c["seconds"] - (c["load_seconds"] or 0), 0) for c in RESULTS["cpu"]])
if RESULTS["cpu"]:
    print(f"\\nCPU only: about {RESULTS['cpu_avg_seconds']} s per answer "
          f"({average([c['tokens_per_s'] for c in RESULTS['cpu']])} tokens/s), about what a laptop CPU would take")
if DEMO[:CPU_CHECK]:
    try:  # else your questions would run on the CPU copy too
        back_to_gpu()
        print("Model loaded again for your questions:", processor() or "?")
    except Exception as e:
        failed("back to the GPU", e)
save_results()
"""

TRY_YOURS = """
#@title 8. Your questions: the model's answers
RESULTS["questions"] = []
asked = QUESTIONS + ([{"product": PRODUCT, "said": SAID}] if SAID.strip() else [])
RESULTS["questions_asked"] = asked
for q in asked:
    product, said = q.get("product", ""), q["said"]
    try:
        res = ask(product, said)
    except Exception as e:
        failed("your question", e)
        continue
    parsed = parse(res["reply"])
    RESULTS["questions"].append({"product": product, "said": said, "reply": res["reply"], "parsed": parsed,
                                 "seconds": res["seconds"]})
    print(f"\\nProduct:    {product}\\nShopkeeper: {said}\\nModel:      {res['reply']}\\nIn words:   {in_words(parsed)}")
if RESULTS["questions"]:
    RESULTS["questions_processor"] = processor()
    print("\\nYour questions ran on:", RESULTS["questions_processor"] or "?")
if not asked:
    print('No question of your own. Type it in cell 1, for example PRODUCT = "16A MCB" and '
          'SAID = "haan hai, 220 ka, delivery nahi", then Run all (or run cells 1 and 8 again).')
save_results()
"""

TRY_SAVE = """
#@title 9. Save the results (try_results.json, in the Output tab)
save_results()
print("Errors:", RESULTS["errors"] or "none")
if os.path.isdir(WORKING_DIR):
    print("Saved", os.path.join(WORKING_DIR, "try_results.json"))
"""

TRY_NEXT = """
## Run it on your PC

1. Install Ollama from ollama.com.
2. Download `wimb-shop-model/wimb-shop-model.Q4_K_M.gguf` from the Output tab of `wimb-shop-model-training`,
   and the `Modelfile` from the GitHub repo (the same one used above). Put both in one folder.
3. Open a terminal in that folder and run:
```
ollama create wimb-shop -f Modelfile
ollama run wimb-shop "Product: 16A MCB
Shopkeeper: haan hai, 220 ka, delivery nahi"
```
From an app: `POST http://localhost:11434/api/generate` with
`{"model": "wimb-shop", "prompt": "Product: ...\\nShopkeeper: ...", "stream": false}`.
"""


NO_BOX = try_on_kaggle.ASK_NO_BOX  # what to do when the editor does not draw the ask box
# Cells 6 and 7 say this when Kaggle stopped or restarted the session (nothing above has run in the new one).
RESTARTED = "The session restarted: tap Run all again (about 3–4 minutes)"

ASK_INTRO = f"""
# Ask the shop model

Type what a shopkeeper said on a call. The model tells you: **in stock**, **price**, **delivery** and
**other product offered**. The text can be Hindi, Marathi or Hinglish. It is not a chatbot.

**How to use it**
1. Open this notebook on kaggle.com → **Edit**.
2. Right panel:
   - **Input** shows `wimb-shop-model-training`. If not: **Add Input → Notebooks → Your Work →
     wimb-shop-model-training**.
   - **Accelerator: GPU T4**, **Internet: on**.
3. **Run all**. About 3–4 minutes each time you start it.
4. At the bottom: type the product and what the shopkeeper said, tap **Ask**. The answer comes in about
   half a second. Ask as many as you like. {NO_BOX}
5. When done, stop the session (**Run → Stop session**, or the power icon), so your free GPU hours are not
   used up.

Tip: Accelerator **None** also works (about 10 s per answer) and uses no GPU hours.
"""

ASK_STATE = """
WORKING_DIR = "/kaggle/working"  # ask_check.json is saved here (the notebook's Output tab)
RESULTS = {"ollama_version": None, "gpu": None, "model_file": None, "model_mb": None, "create_seconds": None,
           "load_seconds": None, "processor": None, "self_check": None, "ask_box": None, "errors": []}
"""

ASK_HELPERS = """
NOT_RUNNING = "The model is not running (the session may have restarted): tap Run all again"
SAVED_RUN = "This is the saved check, where nobody can type. To ask questions: tap Edit, then Run all."
NO_BOX = """ + json.dumps(NO_BOX) + """


def saved_run():
    \"\"\"True in Kaggle's saved run (Save Version, or try_on_kaggle.py --make-ask-notebook): no one can type there.\"\"\"
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch"  # "Interactive" in the editor


def ask(product, said):
    \"\"\"The model's reply and the seconds it took, through Ollama's API as an app would. No system message is
    sent (the Modelfile has it); keep_alive keeps the model loaded for 60 minutes after each question.\"\"\"
    body = {"model": "wimb-shop", "prompt": user_message(product, said), "stream": False, "keep_alive": "60m",
            "options": {""" + OPTIONS + """}}
    started = time.time()
    try:
        data = generate(body)
    except urllib.error.HTTPError as e:  # Ollama runs but refused, e.g. the model was not made
        raise RuntimeError(f"Ollama said: {e.read().decode(errors='replace')[:300]} (tap Run all again)") from e
    except (urllib.error.URLError, ConnectionError) as e:  # nothing answers: Ollama is not running
        raise RuntimeError(NOT_RUNNING) from e
    return data.get("response", "").strip(), round(time.time() - started, 2)


def where_it_runs(ps, gpu):
    \"\"\"Where the model runs, in words. ps: what processor() gave; gpu: the GPU's name (None: no GPU).\"\"\"
    if ps == "100% GPU":
        return "running on the GPU: about half a second per answer"
    if ps == "100% CPU":
        return "running on CPU only: about 10 s per answer" + (" (Ollama did not use the GPU)" if gpu else "")
    return f"running on GPU and CPU ({ps})" if ps else "running on GPU or CPU (not known)"


def answer_lines(product, said, reply, seconds):
    \"\"\"One question and its answer as lines for a person: the answer in plain words, the model's JSON, the time.\"\"\"
    return [f"Product: {product}", f"Shopkeeper: {said}", f"Answer: {in_words(parse(reply))}",
            f"Model's reply: {reply}", f"Took {seconds} s"]


def show_answer(product, said):
    \"\"\"Ask, then print the answer (or why it could not ask).\"\"\"
    try:
        reply, seconds = ask(product, said)
    except Exception as e:
        print("Could not ask:", str(e) or repr(e))
        return
    print("\\n".join(answer_lines(product, said, reply, seconds)))


def make_ask_box(ask_fn):
    \"\"\"The ask box: Product, Shopkeeper said, an Ask button, and the answers so far (newest first).

    ask_fn(product, said) gives (reply, seconds). Its errors are shown in the box (ipywidgets would hide them).
    \"\"\"
    import html
    import ipywidgets as widgets

    product = widgets.Text(placeholder="e.g. 16A MCB", layout=widgets.Layout(width="100%"))
    said = widgets.Textarea(placeholder="e.g. haan hai, 220 ka, delivery nahi\\nor: nahi hai, 32A wala le lo",
                            rows=3, layout=widgets.Layout(width="100%"))
    button = widgets.Button(description="Ask", button_style="primary")
    status = widgets.Label()
    history = widgets.HTML()
    shown = []  # the answers so far, newest first

    def on_click(_):
        question = (product.value.strip(), said.value.strip())
        if not question[1]:
            status.value = "Type what the shopkeeper said, then tap Ask."
            return
        button.disabled, status.value = True, "Asking…"
        try:
            reply, seconds = ask_fn(*question)
            lines = answer_lines(*question, reply, seconds)
            said.value = ""  # ready for the next question; the product stays
        except Exception as e:
            lines = [f"Product: {question[0]}", f"Shopkeeper: {question[1]}", f"Could not ask: {str(e) or repr(e)}"]
        finally:
            button.disabled, status.value = False, ""
        shown.insert(0, "\\n".join(lines))
        history.value = "".join(f"<pre style='white-space: pre-wrap'>{html.escape(t)}</pre>" for t in shown)

    button.on_click(on_click)
    return widgets.VBox([widgets.Label("Product"), product, widgets.Label("Shopkeeper said"), said, button, status,
                         history])


def show_ask_box():
    \"\"\"Show the ask box (under the cell that calls this). Without ipywidgets, say to use the next cell.

    In Kaggle's saved run the box is only built (to check that it can be), not shown: no one can type there.
    \"\"\"
    try:
        import ipywidgets
        from IPython.display import display
    except ImportError:
        RESULTS["ask_box"] = "no ipywidgets"
        print("The ask box needs ipywidgets, which is missing here: use the next cell instead.")
    else:
        try:
            box = make_ask_box(ask)
            if saved_run():
                RESULTS["ask_box"] = f"built, not shown (Kaggle's saved run; ipywidgets {ipywidgets.__version__})"
                print(SAVED_RUN)
            else:
                print(NO_BOX)  # the box itself is drawn by the editor: this check can't see it
                display(box)
                RESULTS["ask_box"] = f"shown (ipywidgets {ipywidgets.__version__})"
        except Exception as e:
            RESULTS["ask_box"] = "failed"
            failed("ask box", e)
            print("Use the next cell instead.")
    save_results()
"""

# The owner runs this notebook by hand, where Internet may be off: a failed install says what to check.
INSTALL_FAILED = 'raise RuntimeError("Ollama did not install (see above)")'
NO_INTERNET = ('raise RuntimeError("Ollama did not install: check Internet is on (right panel → Session options), '
               'then Run all again")')
if INSTALL_FAILED not in START_OLLAMA:
    raise ValueError("START_OLLAMA changed: update INSTALL_FAILED")
# ollama serve reads OLLAMA_KEEP_ALIVE: how long the model stays loaded after a question (default 5 minutes).
ASK_OLLAMA = titled(f"3. {OLLAMA_TITLE}", 'os.environ["OLLAMA_KEEP_ALIVE"] = "60m"  # keep the model loaded between '
                    "questions\n" + START_OLLAMA.replace(INSTALL_FAILED, NO_INTERNET))

ASK_SELF_CHECK = """
#@title 5. Load the model and check that it answers (about 1 minute the first time)
CHECK = {"product": "16A MCB", "said": "haan hai, 220 ka, delivery nahi"}
try:
    started = time.time()
    generate({"model": "wimb-shop", "keep_alive": "60m"})  # no prompt: only load it
    RESULTS["load_seconds"] = round(time.time() - started, 1)
    print(f"Model loaded in {RESULTS['load_seconds']} s")
except Exception as e:
    failed("load the model", e)
RESULTS["self_check"] = None
try:
    reply, seconds = ask(CHECK["product"], CHECK["said"])
    RESULTS["self_check"] = {**CHECK, "reply": reply, "parsed": parse(reply), "seconds": seconds}
    print("\\n".join(answer_lines(CHECK["product"], CHECK["said"], reply, seconds)))
except Exception as e:
    failed("self-check", e)
RESULTS["processor"] = processor()
if RESULTS["self_check"] is None:
    print("\\nThe model is NOT running (see FAILED above): check the right panel (Input, Accelerator, Internet) "
          "and tap Run all again")
else:
    print("\\nThe model is", where_it_runs(RESULTS["processor"], RESULTS["gpu"]))
if RESULTS["processor"] == "100% CPU" and RESULTS["gpu"]:
    failed("GPU", f"there is a GPU ({RESULTS['gpu']}) but Ollama runs the model on the CPU")
if RESULTS["self_check"] and RESULTS["self_check"]["parsed"]:
    print(SAVED_RUN if saved_run() else "Ready: ask your questions in the box below.")
save_results()
"""

# Cells 6 and 7 first check that the cells above have run in this session (else a NameError, not the hint).
ASK_BOX = f"""
#@title 6. Ask here: type the product and what the shopkeeper said, then tap Ask
if "OLLAMA_URL" in globals() and "show_ask_box" in globals():
    show_ask_box()
else:  # a new session (Kaggle stopped or restarted it): nothing above has run in it
    print("{RESTARTED}")
"""

ASK_TYPE = f"""
#@title 7. Or type here: change the two lines, then run this cell again (Shift+Enter, or the ▶ button)
PRODUCT = "16A MCB"
SAID = "haan hai, 220 ka, delivery nahi"
if "OLLAMA_URL" in globals() and "show_answer" in globals():
    show_answer(PRODUCT, SAID)
else:  # a new session (Kaggle stopped or restarted it): nothing above has run in it
    print("{RESTARTED}")
"""

ASK_TIPS = f"""
## Tips
- {NO_BOX}
- One product per question.
- Write what the shopkeeper said as text, the way they said it (Hindi, Marathi or Hinglish).
- It answers four things only: in stock, price, delivery, other product offered. It is not a chatbot.
- If it says the model is not running, or the session has stopped: **Run all** again.
- When done, stop the session (**Run → Stop session**), so your free GPU hours are not used up.
"""


def real_rows(path: Path) -> str:
    """A data/real/*.jsonl file in the training-row format (only product, said, answer)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    text = "\n".join(json.dumps({k: r[k] for k in ("product", "said", "answer")}, ensure_ascii=False) for r in rows)
    if "'''" in text or text.endswith("\\"):
        raise ValueError(f"{path} can't be embedded in a raw string")
    return text


def block(name: str, text: str) -> str:
    return f"{name} = [json.loads(line) for line in r'''\n{text}\n'''.strip().splitlines()]\n"


METADATA = {
    "colab": {"provenance": [], "gpuType": "T4"},
    "accelerator": "GPU",
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}


def train_notebook() -> dict:
    score_src = (HERE / "score.py").read_text(encoding="utf-8")
    fresh = (HERE / "data" / "fresh_test.jsonl").exists()
    data = (
        "#@title 4. Training data and scoring (hidden; built from data/ and score.py)\n"
        + score_src
        + "\n\n"
        + block("TRAIN_V2", embedded("train.jsonl"))
        + "# Real Hindi answers from Amazon xPQA (CDLA-Sharing-1.0; see data/real/NOTICE-xpqa.md)\n"
        + block("REAL_XPQA", real_rows(HERE / "data" / "real" / "xpqa_hi_examples.jsonl"))
        + block("TRAIN_V1", embedded("train_v1.jsonl"))
        + 'TRAIN_SET = globals().get("TRAIN_SET", "v2")\n'
        + 'TRAIN_DATA = TRAIN_V2 + REAL_XPQA if TRAIN_SET == "v2" else TRAIN_V1\n'
        + block("TEST", embedded("test.jsonl"))
        + block("HARD", embedded("hard_test.jsonl"))
        + (block("FRESH", embedded("fresh_test.jsonl")) if fresh else "FRESH = []\n")
        + "SETS = {'test': TEST, 'hard': HARD, **({'fresh': FRESH} if FRESH else {})}\n"
        + 'print(TRAIN_SET, len(TRAIN_DATA), "training rows;", {k: len(v) for k, v in SETS.items()}, "test calls")\n'
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": [
            md(INTRO), code(CHECK_GPU), code(INSTALL), code(SETTINGS), code(data, hidden=True),
            code(LOAD), code(BEFORE), code(TRAIN), code(AFTER), code(TRY), code(SAVE), code(GGUF_TEST), md(NEXT),
        ],
    }


def scoring() -> str:
    """score.py and in_words(), as notebook code (the same in the try and ask notebooks)."""
    return (HERE / "score.py").read_text(encoding="utf-8") + "\n\n" + inspect.getsource(try_on_kaggle.in_words) + "\n\n"


def modelfile_repo() -> str:
    """MODELFILE_REPO: the repo's Modelfile, which the create cell checks its own Modelfile against."""
    return ("# The repo's Modelfile, to check the one made here is the same\n"
            f"MODELFILE_REPO = {(HERE / 'Modelfile').read_text(encoding='utf-8')!r}\n")


def try_notebook() -> dict:
    """The notebook that runs the trained model in Ollama on Kaggle and answers questions."""
    demo = embedded(DEMO_FILE) if (HERE / "data" / DEMO_FILE).exists() else ""
    data = (
        "#@title 2. Example calls, scoring and helpers (hidden; built from data/, score.py and Modelfile)\n"
        + scoring()
        + f"# The example calls: {DEMO_FILE} (a demo, not a test set)\n"
        + block("DEMO", demo)
        + modelfile_repo()
        + TRY_HELPERS
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": [
            md(TRY_INTRO), code(TRY_ASK), code(data, hidden=True), code(TRY_FIND), code(TRY_OLLAMA),
            code(TRY_CREATE), code(TRY_API), code(TRY_DEMO), code(TRY_YOURS), code(TRY_SAVE), md(TRY_NEXT),
        ],
    }


def ask_notebook() -> dict:
    """The notebook where the owner runs the model once, then asks it many questions (an ask box, from a phone)."""
    helpers = (
        "#@title 1. Helpers (no need to read; built from score.py, try_on_kaggle.py and Modelfile)\n"
        "import os, shutil, subprocess, urllib.error, urllib.request  # json, re and time: score.py below\n\n"
        + scoring()
        + modelfile_repo()
        + ASK_STATE
        + run_helpers(try_on_kaggle.ASK_CHECK_FILE, 3)
        + "\n\n"
        + OLLAMA_API.strip("\n")
        + "\n\n"
        + ASK_HELPERS
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": [
            md(ASK_INTRO), code(helpers, hidden=True, folded=True), code(titled(f"2. {FIND_TITLE}", FIND_MODEL)), code(ASK_OLLAMA),
            code(titled(f"4. {CREATE_TITLE}", CREATE_MODEL)), code(ASK_SELF_CHECK), code(ASK_BOX), code(ASK_TYPE),
            md(ASK_TIPS),
        ],
    }


def notebooks() -> dict[str, dict]:
    """Every generated notebook, by file name."""
    return {NOTEBOOK: train_notebook(), TRY_NOTEBOOK: try_notebook(), ASK_NOTEBOOK: ask_notebook()}


def text(nb: dict) -> str:
    """The notebook file's exact text."""
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def main() -> None:
    for name, nb in notebooks().items():
        out = HERE / name
        out.write_text(text(nb), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

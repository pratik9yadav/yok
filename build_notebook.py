"""Build train_colab.ipynb from score.py and the data files, so the notebook needs no repo access.

Run after changing the data or score.py:  python build_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = "pratik9yadav/yok"
BRANCH = "main"
NOTEBOOK = "train_colab.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str, hidden: bool = False) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"cellView": "form"} if hidden else {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


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

**What it does:** takes a free model (Qwen3 4B), tests it on 30 shop calls, trains it on
400 examples, tests it again, and saves your model to Google Drive for Ollama.

It also runs on Kaggle's free GPU: `run_on_kaggle.py` starts it there with no clicks.
"""

CHECK_GPU = """
#@title 1. Check the GPU
import os, shutil, subprocess, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Kaggle gives two GPUs; training uses one
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


def run_test(label):
    FastLanguageModel.for_inference(model)
    replies = [ask(ex["product"], ex["said"]) for ex in TEST]
    FastLanguageModel.for_training(model)
    result = score(replies, TEST)
    print(f"== {label}: all 4 fields right in {result['all_correct']:.0%} of {result['examples']} calls")
    for f in ("json_ok",) + FIELDS:
        print(f"   {f:12s} {result[f]:.0%}")
    for m in result["mistakes"][:15]:
        print("\\n   said: ", m["said"])
        print("   want: ", answer_text(m["want"]))
        print("   reply:", m["reply"][:200])
        print("   wrong:", ", ".join(m["wrong"]))
    return result


before = run_test("BEFORE training")
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
after = run_test("AFTER training")
print("\\n== Before vs after")
for f in ("all_correct", "json_ok") + FIELDS:
    print(f"   {f:12s} {before[f]:5.0%} -> {after[f]:5.0%}")
summary = {k: {f: round(r[f], 3) for f in ("all_correct", "json_ok") + FIELDS}
           for k, r in (("before", before), ("after", after))}
print("\\nRESULTS_JSON", json.dumps({"model": MODEL_NAME, "epochs": EPOCHS, **summary}))
if ON_KAGGLE:  # run_on_kaggle.py downloads this
    with open("/kaggle/working/results.json", "w", encoding="utf-8") as fh:
        json.dump({"model": MODEL_NAME, "epochs": EPOCHS, **summary,
                   "mistakes_after": after["mistakes"], "mistakes_before": before["mistakes"]},
                  fh, ensure_ascii=False, indent=1)
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

    found = (result or {}).get("modelfile_location") if isinstance(result, dict) else None
    modelfile = open(found, encoding="utf-8").read() if found and os.path.exists(found) else (
        'TEMPLATE """{{ if .System }}<|im_start|>system\\n{{ .System }}<|im_end|>\\n{{ end }}'
        '{{ if .Prompt }}<|im_start|>user\\n{{ .Prompt }}<|im_end|>\\n{{ end }}'
        '<|im_start|>assistant\\n{{ .Response }}<|im_end|>\\n"""\\n'
        'PARAMETER stop "<|im_end|>"\\n'
    )
    modelfile = re.sub(r"^FROM .*\\n?", "", modelfile, flags=re.M)
    modelfile = re.sub(r"^PARAMETER temperature .*\\n?", "", modelfile, flags=re.M)
    modelfile = f"FROM ./{name}\\n{modelfile.strip()}\\nPARAMETER temperature 0\\nSYSTEM \\"\\"\\"{SYSTEM}\\"\\"\\"\\n"
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


def main() -> None:
    score_src = (HERE / "score.py").read_text(encoding="utf-8")
    data = (
        "#@title 4. Training data and scoring (hidden; built from data/ and score.py)\n"
        + score_src
        + "\n\nTRAIN_DATA = [json.loads(line) for line in r'''\n"
        + embedded("train.jsonl")
        + "\n'''.strip().splitlines()]\nTEST = [json.loads(line) for line in r'''\n"
        + embedded("test.jsonl")
        + "\n'''.strip().splitlines()]\n"
        + 'print(len(TRAIN_DATA), "training examples,", len(TEST), "test examples")\n'
    )
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            md(INTRO), code(CHECK_GPU), code(INSTALL), code(SETTINGS), code(data, hidden=True),
            code(LOAD), code(BEFORE), code(TRAIN), code(AFTER), code(TRY), code(SAVE), md(NEXT),
        ],
    }
    out = HERE / NOTEBOOK
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

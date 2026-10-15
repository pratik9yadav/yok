"""Build coder/coder_train.ipynb: train Qwen2.5-Coder-14B (QLoRA, Unsloth) on open bug-fix commits on Kaggle's free
T4, and score it on HumanEvalFix before and after.

`python coder/build.py` writes it (with coder_eval.ipynb). The notebook reuses the GPU check and the install cell
of the shop-model training notebook (build_notebook.py), and coder/fixeval.py + coder/fixtrain.py as its helpers.
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from build_notebook import CHECK_GPU, INSTALL, METADATA, code, md, run_helpers, text, titled

NOTEBOOK = "coder_train.ipynb"
RESULTS_FILE = "coder_train.json"
SETTINGS_TITLE = "3. Settings"

INTRO = """
# Coder, phase 2: train Qwen2.5-Coder-14B to fix bugs (QLoRA on Kaggle's free T4)

**What it does:**
1. Downloads real bug-fix commits from GitHub: **CommitPackFT (Python)** (`bigcode/commitpackft`). It stops if the
   dataset's licence is not an open one, and keeps only commits from permissively licensed repositories.
2. Keeps small bug fixes (the message says fix/bug/error/...; at most 20 lines changed) and **drops every commit
   that shares text with the test tasks** (any 10 words in a row, or the same function signature), so the score
   is honest.
3. Scores the model as it is on **HumanEvalFix (Python)** (164 buggy functions with hidden tests): the BEFORE score.
4. Trains a small LoRA add-on (QLoRA, 4-bit) on the commits: the model sees the file before the fix and learns to
   write the file after it.
5. Scores it again: the AFTER score. Phase 1's best was qwen3-coder:30b with 73%.

Everything goes to `coder_train.json` (Output tab) after each step, so nothing is lost if it stops.
Time: about 5–8 hours (each score 20–60 minutes, at most 2 hours; training at most 4 hours). `SMOKE = True`: about 20–30 minutes.

**Settings (right side):** Accelerator **GPU T4 x2**, Internet **on**. Then **Run all**.

**Or with no clicks:** on GitHub, **Actions → Coder model on Kaggle → Run workflow → action train**, or ask Claude.
"""

SETTINGS = """
import json, os, random, re, time

# The model to train: 4-bit Qwen2.5-Coder-14B-Instruct (Apache 2.0), ready for Unsloth.
BASE = "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
# How many bug-fix commits to train on at most. Claude and GitHub Actions fill this line.
MAX_TRAIN = 2000
MAX_LEN = 2048  # tokens; a longer example (question + answer) is left out
# Passes over the training examples. Claude and GitHub Actions fill this line.
EPOCHS = 1
EVAL_LIMIT = None  # how many HumanEvalFix tasks to score (None = all 164)
# True = a quick check of the whole run (40 examples, 8 steps, 8 tasks; about 20-30 minutes). Claude and GitHub
# Actions fill this line.
SMOKE = False
TRAIN_MINUTES = 240  # training stops after this, so the whole run ends inside Kaggle's 12 hours
EVAL_MINUTES = 120  # each score (before, after) stops after this
EVAL_BATCH = 4  # tasks answered together (halved by itself when the GPU runs out of memory)
MAX_NEW_TOKENS = 1024  # longest reply
MAX_STEPS = None  # None: as many steps as the examples and EPOCHS need
LOGGING_STEPS = 10
SEED = 3407
WORKING_DIR = "/kaggle/working"  # coder_train.json and the LoRA are saved here (the notebook's Output tab)
LORA_DIR = os.path.join(WORKING_DIR, "coder-lora")
if SMOKE:
    MAX_TRAIN, MAX_STEPS, EVAL_LIMIT, LOGGING_STEPS = 40, 8, 8, 1
"""

STATE = """
SETTINGS = {k: globals()[k] for k in ("BASE", "MAX_TRAIN", "MAX_LEN", "EPOCHS", "EVAL_LIMIT", "SMOKE", "TRAIN_MINUTES",
                                      "EVAL_MINUTES", "EVAL_BATCH", "MAX_NEW_TOKENS", "MAX_STEPS", "SEED")}
RESULTS = {"phase": 2, "base": BASE, "smoke": SMOKE, "settings": SETTINGS, "benchmark": BENCHMARK,
           "eval_dataset": DATASET, "eval_licence": LICENCE, "eval_setting": EVAL_SETTING, "eval_source": None,
           "instruction": INSTRUCTION, "train_instruction": TRAIN_INSTRUCTION, "train_dataset": TRAIN_DATASET,
           "licence": None, "data": {"source": None, "fields": None, "example": None, "counts": None, "used": None},
           "train": {"examples": None, "steps": None, "planned_steps": None, "loss": None, "minutes": None,
                     "stopped_by_time": False, "log": []},
           "before": {}, "after": {}, "flips": None, "lora_dir": None, "export": NO_EXPORT, "phase1": PHASE1,
           "versions": None, "gpu": None, "errors": []}
"""

LOAD = """
from unsloth import FastLanguageModel, is_bfloat16_supported  # first: Unsloth speeds up what it imports after it
from unsloth.chat_templates import train_on_responses_only
import importlib.metadata as md
import torch


def version(package):
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


try:
    RESULTS["versions"] = {p: version(p) for p in ("unsloth", "torch", "transformers", "trl", "peft", "bitsandbytes")}
    RESULTS["gpu"] = torch.cuda.get_device_name(0)
    print(RESULTS["versions"], RESULTS["gpu"])
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=BASE, max_seq_length=MAX_LEN, load_in_4bit=True)
    # LoRA: train a small add-on (about 0.5% of the weights). Its new part starts at zero, so until training the
    # model answers exactly as the base model does: the BEFORE score is the base model's.
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED,
    )
except Exception as e:
    failed("load the model", e)
    raise
save_results()

eos = model.generation_config.eos_token_id
STOP_IDS = {i for i in [tokenizer.eos_token_id, tokenizer.pad_token_id, *(eos if isinstance(eos, list) else [eos])]
            if i is not None}


def generate_replies(prompts):
    \"\"\"Greedy answers to a batch of prompts (each the only user message, in the base model's chat format).\"\"\"
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True) for p in prompts]
    side, tokenizer.padding_side = tokenizer.padding_side, "left"  # batched answers need padding on the left
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, repetition_penalty=1.0, use_cache=True,
                                 pad_token_id=tokenizer.pad_token_id)
    except Exception:
        torch.cuda.empty_cache()
        raise
    finally:
        tokenizer.padding_side = side
    answers = []
    for row in out[:, inputs["input_ids"].shape[1]:].tolist():
        n = next((k + 1 for k, t in enumerate(row) if t in STOP_IDS), len(row))
        answers.append({"reply": tokenizer.decode(row[:n], skip_special_tokens=True), "tokens": n,
                        "cut_off": n >= MAX_NEW_TOKENS and row[n - 1] not in STOP_IDS})
    return answers
"""

DATA = """
try:
    HUMANEVAL, RESULTS["eval_source"] = load_tasks()  # the test tasks: needed here to keep them out of training
    EVAL_TASKS = HUMANEVAL[:EVAL_LIMIT] if EVAL_LIMIT else HUMANEVAL
    RESULTS["licence"] = dataset_licence()
    print("Licence of", TRAIN_DATASET, RESULTS["licence"])
    commits, RESULTS["data"]["source"] = load_commits()
    RESULTS["data"]["fields"] = sorted(commits[0])
    RESULTS["data"]["example"] = {k: str(v)[:300] for k, v in commits[0].items()}
    print(f"{len(commits):,} commits from {RESULTS['data']['source']}")
    print("Fields:", ", ".join(RESULTS["data"]["fields"]))
    for field, value in RESULTS["data"]["example"].items():  # the first commit, field by field
        print(f"--- {field}:\\n{value}")
    save_results()

    def count_tokens(row):
        chat = tokenizer.apply_chat_template(messages_for(row), tokenize=False)
        return len(tokenizer(chat, add_special_tokens=False)["input_ids"])

    kept, RESULTS["data"]["counts"] = select_rows(commits, count_tokens, MAX_LEN, HUMANEVAL)
    del commits
    TRAIN_ROWS = pick(kept, MAX_TRAIN, SEED)
    RESULTS["data"]["used"] = len(TRAIN_ROWS)
except Exception as e:
    failed("training data", e)
    raise
print("\\n".join(counts_lines(RESULTS["data"]["counts"])))
print(f"Training on {len(TRAIN_ROWS)} of them; decontamination dropped {RESULTS['data']['counts']['contaminated']}")
for example in RESULTS["data"]["counts"]["contaminated_examples"][:5]:
    print("   dropped:", example)
print("--- One training example, as the model sees it:")
for message in messages_for(TRAIN_ROWS[0]):
    print(f"[{message['role']}]\\n{message['content'][:1500]}")
save_results()
"""

BEFORE = """
print(f"BEFORE: {len(EVAL_TASKS)} tasks, {EVAL_BATCH} at a time")
FastLanguageModel.for_inference(model)
try:
    score(generate_replies, EVAL_TASKS, RESULTS["before"], EVAL_MINUTES, EVAL_BATCH, save_results)
except Exception as e:  # record it; the training still runs
    failed("BEFORE score", e)
finally:
    FastLanguageModel.for_training(model)
save_results()
print("\\n".join(report_lines(RESULTS)))
"""

TRAIN = """
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer


class Progress(TrainerCallback):
    \"\"\"Writes the loss to coder_train.json at each log, and stops the training after TRAIN_MINUTES.\"\"\"

    def __init__(self):
        self.started = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        note_progress(RESULTS["train"], state.global_step, state.max_steps, logs, time.time() - self.started)
        save_results()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.started > TRAIN_MINUTES * 60 and not control.should_training_stop:
            RESULTS["train"]["stopped_by_time"] = True
            print(f"Stopping: {TRAIN_MINUTES} minutes have passed (step {state.global_step} of {state.max_steps})")
            control.should_training_stop = True
        return control


dataset = Dataset.from_list([{"text": tokenizer.apply_chat_template(messages_for(r), tokenize=False)}
                             for r in TRAIN_ROWS])
RESULTS["train"]["examples"] = len(dataset)
progress = Progress()
try:
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        callbacks=[progress],
        args=SFTConfig(
            dataset_text_field="text", max_length=MAX_LEN,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            num_train_epochs=EPOCHS, max_steps=MAX_STEPS or -1,
            learning_rate=2e-4, warmup_steps=10, lr_scheduler_type="linear",
            optim="adamw_8bit", weight_decay=0.01, logging_steps=LOGGING_STEPS, seed=SEED,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
            output_dir="/tmp/coder-outputs", save_strategy="no", report_to="none",
        ),
    )
    # Learn only the answers (the fixed file), not the questions.
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\\n", response_part="<|im_start|>assistant\\n",
    )
    stats = trainer.train()
    RESULTS["train"].update(steps=trainer.state.global_step, planned_steps=trainer.state.max_steps,
                            loss=round(stats.metrics["train_loss"], 4),
                            minutes=round((time.time() - progress.started) / 60, 1))
except Exception as e:
    RESULTS["train"]["minutes"] = round((time.time() - progress.started) / 60, 1)
    failed("training", e)
    raise
print(f"Trained {RESULTS['train']['steps']} of {RESULTS['train']['planned_steps']} steps "
      f"({len(dataset)} examples) in {RESULTS['train']['minutes']} min, loss {RESULTS['train']['loss']}")

try:  # the LoRA add-on (a few hundred MB), in the Output tab
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)
    RESULTS["lora_dir"] = LORA_DIR
    print("Saved the LoRA add-on to", LORA_DIR)
except Exception as e:
    failed("save the LoRA", e)
save_results()
"""

AFTER = """
torch.cuda.empty_cache()
print(f"AFTER: {len(EVAL_TASKS)} tasks, {EVAL_BATCH} at a time")
FastLanguageModel.for_inference(model)
try:
    score(generate_replies, EVAL_TASKS, RESULTS["after"], EVAL_MINUTES, EVAL_BATCH, save_results)
except Exception as e:
    failed("AFTER score", e)
RESULTS["flips"] = flips(RESULTS["before"], RESULTS["after"])
save_results()
"""

REPORT = """
print("\\n".join(report_lines(RESULTS)))
print("Training:", {k: v for k, v in RESULTS["train"].items() if k != "log"})
print("Data:", RESULTS["data"]["used"], "examples;", RESULTS["licence"])
print(RESULTS["export"])
for error in RESULTS["errors"]:
    print("Error:", error)
save_results()
print("Saved:", os.path.join(WORKING_DIR, RESULTS_FILE))
"""

NEXT = """
## What next
If AFTER beats BEFORE clearly (more tasks now fixed than now broken), train on more commits (`MAX_TRAIN`) or for
longer. To use the model in Ollama, the LoRA (`coder-lora`, Output tab) must be merged into the base model and
exported as a GGUF file: that needs more memory than Kaggle gives safely for a 14B model, so it is not done here.
`coder_train.json` keeps every reply, so the tasks that changed can be read.
"""


def notebook() -> dict:
    """The training notebook."""
    helpers = (
        "#@title 4. Helpers (no need to read; built from coder/fixeval.py and coder/fixtrain.py)\n"
        + (HERE / "fixeval.py").read_text(encoding="utf-8")
        + "\n\n"
        + (HERE / "fixtrain.py").read_text(encoding="utf-8")
        + f"\n\nRESULTS_FILE = {RESULTS_FILE!r}\n"
        + STATE
        + run_helpers(RESULTS_FILE, 0).split("\n", 1)[1]  # save_results(), failed(), sh(); no Ollama here
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": [
            md(INTRO),
            code(CHECK_GPU),
            code(INSTALL),
            code(titled(f"{SETTINGS_TITLE}: base model, how many examples, time limits", SETTINGS)),
            code(helpers, hidden=True, folded=True),
            code(titled("5. Load the base model in 4 bits and attach LoRA (about 5–10 minutes)", LOAD)),
            code(titled("6. Training data: open bug-fix commits, filtered and decontaminated", DATA)),
            code(titled("7. Score the model BEFORE training (HumanEvalFix, up to EVAL_MINUTES)", BEFORE)),
            code(titled("8. Train (QLoRA, up to TRAIN_MINUTES) and save the LoRA", TRAIN)),
            code(titled("9. Score the model AFTER training", AFTER)),
            code(titled("10. Results: before → after", REPORT)),
            md(NEXT),
        ],
    }


def main() -> None:
    out = HERE / NOTEBOOK
    out.write_text(text(notebook()), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Build voicebot/ears_training.ipynb: teach the ears (Whisper large-v3 turbo) Indian speech, on Kaggle's free GPU.

LoRA fine-tuning on real Hindi and Marathi speech (FLEURS; with an HF token also IndicVoices: many speakers and
districts), half of it heard through a pretend phone line (8 kHz, μ-law, noise) and at slightly changed speeds, so
the ears learn accents, phone calls and Marathi (Whisper's weakest). Measured before and after on held-out speech
(FLEURS test, never trained on), clean and as a phone line. The merged model is saved as wimb-ears/ in the Output
tab only when it is better; the voice bot uses it from then on.

Run after changing this file or voicebot/:  python voicebot/build_ears.py   (never edit the notebook by hand)
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
if sys.path and Path(sys.path[0]).resolve() == HERE.resolve():  # run as a script: don't shadow standard modules
    sys.path.pop(0)
sys.path.insert(0, str(HERE.parent))

from build_notebook import METADATA, code, md, run_helpers, text, titled
from voicebot import build_voicebot

NOTEBOOK = "ears_training.ipynb"
RESULTS_FILE = "ears_results.json"
SETTINGS_TITLE = "1. Settings"

INTRO = """
# Teach the ears Indian speech (Whisper large-v3 turbo, LoRA), on Kaggle's free GPU

Real Hindi and Marathi speech, half of it through a pretend phone line, measured before and after on speech the
training never saw. The better model is saved as `wimb-ears/` (Output tab) and the voice bot uses it next time.
Built by `voicebot/build_ears.py`; run it from GitHub: **Actions → Voice bot → Run workflow**, mode `train-ears`.
"""

SETTINGS = """
import json, os, re, shutil, subprocess, sys, time

BASE = "openai/whisper-large-v3-turbo"
LANGS = ["hi", "mr"]
TRAIN_ROWS = 3000  # per language and source, at most
EVAL_ROWS = 100  # per language, held out (FLEURS test)
EPOCHS = 3
BATCH = 8
LEARNING_RATE = 1e-4
LORA_RANK = 32
PHONE_SHARE = 0.5  # this share of the training clips is heard through a pretend phone line
WORKING_DIR = "/kaggle/working/ears"
SAVE_DIR = "/kaggle/working/wimb-ears"  # the trained model (only when it is better)
"""

HELPERS = """
SRC = "/tmp/voicebot_src"
for name, body in PACKAGE_FILES.items():
    path = os.path.join(SRC, "voicebot", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
sys.path.insert(0, SRC)
os.makedirs(WORKING_DIR, exist_ok=True)
RESULTS = {"base": BASE, "data": {}, "before": {}, "after": {}, "train": {}, "saved": False, "summary": [],
           "versions": {}, "errors": []}
"""

INSTALL = """
def pip(*packages):
    done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], capture_output=True, text=True,
                          timeout=1800)
    if done.returncode != 0:
        raise RuntimeError((done.stdout + done.stderr)[-1500:])


try:
    pip("peft", "datasets", "soundfile", "librosa", "scipy", "accelerate")
except Exception as e:
    failed("pip install", e)
if not shutil.which("ffmpeg"):
    sh("apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg", timeout=900)


def hf_token():
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = None
    return (token or os.environ.get("HF_TOKEN") or "").strip() or None


HF_TOKEN = hf_token()
print("Hugging Face token:", "found (IndicVoices too)" if HF_TOKEN else "none (FLEURS only)")
import importlib.metadata as md

for p in ("torch", "transformers", "peft", "datasets"):
    try:
        RESULTS["versions"][p] = md.version(p)
    except md.PackageNotFoundError:
        RESULTS["versions"][p] = None
print(RESULTS["versions"])
save_results()
"""

DATA = """
from voicebot.speechdata import error_rates, fleurs, indicvoices

TRAIN, TEST = [], {}
for lang in LANGS:
    try:
        rows, source = fleurs(lang, EVAL_ROWS, "test")
        TEST[lang] = rows
        RESULTS["data"][f"test {lang}"] = f"{len(rows)} clips, {source}"
    except Exception as e:
        failed(f"test data {lang}", e)
    sources = [("FLEURS train", lambda: fleurs(lang, TRAIN_ROWS, "train")),
               ("FLEURS dev", lambda: fleurs(lang, TRAIN_ROWS // 3, "validation"))]
    if HF_TOKEN:
        sources.append(("IndicVoices", lambda: indicvoices(lang, TRAIN_ROWS, HF_TOKEN)))
    for name, load in sources:
        started = time.time()
        try:
            rows, source = load()
        except Exception as e:
            failed(f"train data {lang} {name}", e)
            continue
        TRAIN += [{**r, "lang": lang} for r in rows]
        hours = sum(len(r["audio"]) for r in rows) / 16000 / 3600
        RESULTS["data"][f"train {lang} {name}"] = f"{len(rows)} clips, {hours:.1f} h, {source}"
        print(f"{lang} {name}: {len(rows)} clips, {hours:.1f} h ({time.time() - started:.0f} s)")
print("Training clips:", len(TRAIN), "| test:", {k: len(v) for k, v in TEST.items()})
save_results()
"""

MEASURE = """
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

from voicebot.audio import NO_REPEAT, phone_line, token_limit

PROCESSOR = WhisperProcessor.from_pretrained(BASE)


def measure(model, label):
    \"\"\"Word and character error rates per language, clean and as a phone line, on the held-out clips.\"\"\"
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=PROCESSOR.tokenizer,
                   feature_extractor=PROCESSOR.feature_extractor, device="cuda:0", torch_dtype=torch.float16)
    out = {}
    for lang, rows in TEST.items():
        for condition in ("clean", "phone"):
            pairs = []
            for i, r in enumerate(rows):
                audio = r["audio"] if condition == "clean" else phone_line(r["audio"], seed=i)
                options = {"task": "transcribe", "language": lang, "max_new_tokens": token_limit(audio),
                           "no_repeat_ngram_size": NO_REPEAT}
                heard = asr({"raw": audio, "sampling_rate": 16000}, generate_kwargs=options)["text"]
                pairs.append((r["text"], heard))
            wer, cer = error_rates(pairs)
            out[f"{lang} {condition}"] = {"wer": wer, "cer": cer, "examples": [
                {"said": s, "heard": h} for s, h in pairs[:3]]}
            print(f"{label} | {lang} {condition}: WER {wer}%  CER {cer}%")
    RESULTS[label] = out
    save_results()
    return out


BASE_MODEL = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda:0")
BEFORE = measure(BASE_MODEL, "before")
"""

TRAIN_CELL = """
import random

from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset

from voicebot.audio import resample

TOKENIZER = PROCESSOR.tokenizer
START = TOKENIZER.convert_tokens_to_ids("<|startoftranscript|>")
PREFIX = {lang: [TOKENIZER.convert_tokens_to_ids(t) for t in (f"<|{lang}|>", "<|transcribe|>", "<|notimestamps|>")]
          for lang in LANGS}


class Speech(Dataset):
    \"\"\"Training clips, each time a little different: half through a phone line, speed 0.9–1.1, volume.\"\"\"

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio = r["audio"]
        speed = random.uniform(0.9, 1.1)
        if abs(speed - 1) > 0.02 and len(audio) < 27 * 16000:  # slowed down, a clip must still fit in 30 s
            audio = resample(audio, 16000, int(16000 / speed))
        if random.random() < PHONE_SHARE:
            audio = phone_line(audio, snr_db=random.uniform(10, 30), seed=random.randrange(10**6))
        audio = audio * random.uniform(0.5, 1.0) / (np.abs(audio).max() + 1e-6)
        features = PROCESSOR.feature_extractor(audio, sampling_rate=16000, return_tensors="np").input_features[0]
        labels = PREFIX[r["lang"]] + TOKENIZER(r["text"], add_special_tokens=False).input_ids[:440] + [TOKENIZER.eos_token_id]
        return {"features": features, "labels": labels}


def collate(batch):
    features = torch.tensor(np.stack([b["features"] for b in batch]))
    width = max(len(b["labels"]) for b in batch)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
    return features, labels


del BASE_MODEL
torch.cuda.empty_cache()
model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float16)
model.config.forced_decoder_ids = None
model.config.use_cache = False
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
lora = LoraConfig(r=LORA_RANK, lora_alpha=2 * LORA_RANK, lora_dropout=0.05,
                  target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])
model = get_peft_model(model, lora)
for p in model.parameters():  # the small trained part in float32 (stable), the frozen rest in float16
    if p.requires_grad:
        p.data = p.data.float()
model.to("cuda:0")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable / 1e6:.1f} M parameters")
loader = DataLoader(Speech(TRAIN), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2, drop_last=True)
steps = EPOCHS * len(loader)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=max(steps, 1),
                                                pct_start=0.06, anneal_strategy="linear")
scaler = torch.amp.GradScaler("cuda")
model.train()
started, step, losses = time.time(), 0, []
RESULTS["train"] = {"clips": len(TRAIN), "steps": steps, "epochs": EPOCHS, "trainable_m": round(trainable / 1e6, 1),
                    "loss": []}
for epoch in range(EPOCHS):
    for features, labels in loader:
        features, labels = features.to("cuda:0", torch.float16), labels.to("cuda:0")
        decoder_input = torch.cat([torch.full((labels.shape[0], 1), START, device="cuda:0"),
                                   labels[:, :-1].clamp(min=0)], dim=1)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=features, decoder_input_ids=decoder_input, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        losses.append(loss.item())
        if step % 50 == 0 or step == steps:
            avg = sum(losses[-50:]) / len(losses[-50:])
            RESULTS["train"]["loss"].append([step, round(avg, 4)])
            print(f"step {step}/{steps}  loss {avg:.3f}  ({(time.time() - started) / 60:.0f} min)", flush=True)
            save_results()
RESULTS["train"]["minutes"] = round((time.time() - started) / 60, 1)
model = model.merge_and_unload()
model.half().eval()
model.config.use_cache = True
save_results()
"""

AFTER = """
AFTER = measure(model, "after")


def average(scores, condition):
    values = [v["wer"] for k, v in scores.items() if k.endswith(condition)]
    return sum(values) / len(values) if values else 999


better = average(AFTER, "phone") < average(BEFORE, "phone") and average(AFTER, "clean") <= average(BEFORE, "clean") + 1
if better:
    model.save_pretrained(SAVE_DIR, safe_serialization=True)
    PROCESSOR.save_pretrained(SAVE_DIR)
    RESULTS["saved"] = True
    print("Better: saved to", SAVE_DIR)
else:
    print("Not better: the base model stays")
lines = [f"Training: {RESULTS['train'].get('clips')} clips, {RESULTS['train'].get('epochs')} epochs, "
         f"{RESULTS['train'].get('minutes')} min"]
for key in BEFORE:
    lines.append(f"{key}: WER {BEFORE[key]['wer']}% -> {AFTER[key]['wer']}%  (CER {BEFORE[key]['cer']}% -> "
                 f"{AFTER[key]['cer']}%)")
lines.append("Saved as wimb-ears/: " + ("yes, the voice bot uses it from now on" if better else "no (not better)"))
RESULTS["summary"] = lines
print("\\n".join(lines))
save_results()
"""


def notebook() -> dict:
    files = {k: v for k, v in build_voicebot.package_files().items() if k.endswith(".py")}
    helpers = ("#@title 2. Helpers (no need to read: the voicebot package, written to disk)\n"
               + f"PACKAGE_FILES = {files!r}\n" + HELPERS + run_helpers(RESULTS_FILE, 0))
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": METADATA,
        "cells": [
            md(INTRO),
            code(titled(f"{SETTINGS_TITLE}: model, data, training", SETTINGS)),
            code(helpers, hidden=True, folded=True),
            code(titled("3. Install (about 1–2 minutes)", INSTALL)),
            code(titled("4. Real speech: training clips and held-out test clips", DATA)),
            code(titled("5. Before: how well the ears hear the test clips now", MEASURE)),
            code(titled("6. Train (LoRA; about 40–70 minutes)", TRAIN_CELL)),
            code(titled("7. After: the same test clips; save when better", AFTER)),
        ],
    }


def main() -> None:
    if len(sys.argv) > 1:
        print(__doc__)
        return
    out = HERE / NOTEBOOK
    out.write_text(text(notebook()), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

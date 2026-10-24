---
title: WIMB shop model
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: other
short_description: Shopkeeper's words to stock, price, delivery
---

# WIMB shop model

Type the product and what the shopkeeper said on a phone call (Hindi, Hinglish or Marathi; Devanagari or Latin
letters). The model answers four things: in stock?, the price in ₹, delivery?, and any other product offered.

- Model: Qwen3-4B-Instruct-2507 (Apache-2.0) with a LoRA trained for this task (project WIMB / yok), as a
  4-bit GGUF file run by llama.cpp on the free CPU. An answer takes several seconds.
- The model file is in a private model repo; this Space reads it with the `HF_TOKEN` secret (`MODEL_REPO` variable).
- Licence `other`: the code is Apache-2.0 like the base model, but the training data includes CC BY-SA 4.0,
  ODC-By and CDLA-Sharing-1.0 material (product names, number words, spellings), so the trained model is kept
  private. Details are in the model repo's card.

Made by `space/deploy.py` in the yok repository (Actions → Put the model on Hugging Face). Don't edit here: the
next deploy overwrites these files.

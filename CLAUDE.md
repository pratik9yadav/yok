# yok: train a small model for free

The owner wants Claude to do the work when asked, and answers in short, simple English
(they often write from a phone). Lead with the result, then the one next step.

## What is here
A notebook that fine-tunes Qwen3 4B (QLoRA, Unsloth) to read what a shopkeeper said on a
call and return JSON: `in_stock`, `price_inr`, `delivery`, `alternative`. It is part of the
WIMB project (pratik9yadav/WIMB). The 400 training examples in `data/train.jsonl` are made
up; `data/test.jsonl` holds 30 hand-written test calls that are never trained on.

`train_colab.ipynb` is generated: edit `build_notebook.py`, `score.py` or `data/`, then run
`python build_notebook.py` and `python -m pytest`. Never edit the notebook by hand.

## "Train the model" / "run it"
1. Check `KAGGLE_API_TOKEN` is set and kaggle.com is reachable. If not, tell the owner the
   one-time setup in README.md ("Or let Claude run it for you"); don't ask for the token in chat.
2. `pip install -r requirements-dev.txt`, then run `python run_on_kaggle.py` in the
   background (about 40 minutes; it exits when the run ends).
3. Report the before → after scores and the remaining mistakes from
   `kaggle_output/results.json`. If it failed, read the log in `kaggle_output/`, fix the
   cause, run the tests, push, and run again.
4. The model files (`*.Q4_K_M.gguf`, `Modelfile`, `lora/`) are in the Kaggle run's Output tab.

Colab instead: the owner opens
https://colab.research.google.com/github/pratik9yadav/yok/blob/main/train_colab.ipynb,
runs it, and saves it back with File → Save a copy in GitHub; then read its outputs from git.

## Using real data
Real call transcripts go in `data/train.jsonl`, one JSON object per line:
`{"product": ..., "said": ..., "answer": {"in_stock": ..., "price_inr": ..., "delivery": ..., "alternative": ...}}`.
Keep `data/test.jsonl` separate from training data (a test checks this).

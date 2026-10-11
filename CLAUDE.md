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
Preferred: GitHub Actions. The workflow `.github/workflows/train.yml` ("Train on Kaggle") reads
the repository secret `KAGGLE_API_TOKEN` and runs `run_on_kaggle.py` on a GitHub runner. Start
it with the GitHub tools (workflow_dispatch on `main`), wait for it to finish, then read the job
log: it prints the before → after scores and the mistakes. If the secret is missing, the
owner adds it at https://github.com/pratik9yadav/yok/settings/secrets/actions/new.

From this container instead (needs kaggle.com allowed and the token in the environment):
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

## Results so far and what is next
See "Results" in README.md (run 8: test 83% / hard 43% after training; exported file 87% / 50%).
`results.json` in a run's output holds, per set, scores, mistakes and every raw reply. The
notebook also tests the exported GGUF through llama-server on CPU (`results_gguf.json`).
Next step that helps most: more varied training data (real transcripts if the owner has any;
otherwise realistic examples that are NOT copied from `data/hard_test.jsonl`, which stays held out).
Then run again (about 30 minutes, free). Use the repo's `Modelfile`, not the one Unsloth writes.

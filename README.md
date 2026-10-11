# Train your own model (free, on Google Colab)

This repo trains a small free model (Qwen3 4B) to read what a shopkeeper said on a call and
return JSON:

```
Product: 16A MCB
Shopkeeper: haan hai, 220 ka, delivery nahi
→ {"in_stock": true, "price_inr": 220, "delivery": false, "alternative": null}
```

## Run it
1. Open [`train_colab.ipynb` in Colab](https://colab.research.google.com/github/pratik9yadav/yok/blob/main/train_colab.ipynb)
   (on a phone: keep Colab open and the screen on while it runs).
2. **Runtime → Run all**. Allow Google Drive when asked. Wait 30–40 minutes.
3. **File → Save a copy in GitHub** → same repo, branch, and path. The results are then in git.
4. On your PC: download the `wimb-shop-model` folder from Google Drive, then
   `ollama create wimb-shop -f Modelfile` and `ollama run wimb-shop`.

## Or let Claude run it for you (Kaggle, no clicks)
Kaggle also gives a free GPU, and unlike Colab it has an API, so Claude can start the run,
wait for it, read the results and fix errors on its own. One-time setup:
1. Make a free account at kaggle.com and **verify your phone number** (Settings → Phone
   verification). Kaggle allows GPU and internet only after that.
2. Kaggle → **Settings → API → Generate New Token**. Copy it (don't paste it in the chat).
3. In the Claude Code environment settings (environment menu in the session's title bar → Edit):
   - add the token as `KAGGLE_API_TOKEN` (network secret / environment variable);
   - under Network access, allow `kaggle.com`, `www.kaggle.com`, `api.kaggle.com`.
4. Start a new session on this repo and say "run the training on Kaggle". Claude runs
   `pip install kaggle && python run_on_kaggle.py` (about 40 minutes).

The model files end up in the run's **Output** tab on kaggle.com.

### Or from GitHub (easiest on a phone)
1. Save the token once as a repository secret named `KAGGLE_API_TOKEN`:
   https://github.com/pratik9yadav/yok/settings/secrets/actions/new
2. Ask Claude to "train the model", or tap **Actions → Train on Kaggle → Run workflow**.

## Results (Kaggle T4, Qwen3-4B-Instruct-2507, 2 epochs, run 8)
Share of calls where all four fields are exactly right. `test` = 30 hand-written calls, `hard` =
56 calls in styles the training data does not have. Neither is ever trained on.

| | simple rules, no AI* | model before training | after training | exported Q4_K_M file |
|---|---|---|---|---|
| test (30) | 77% | 63% | 83% | 87% |
| hard (56) | 30% | 27% | 43% | 50% |

*A keyword/regex program written by an independent reviewer (not in this repo).

How to read it: training clearly helps (+20 and +16 points) and the exported file is as good as the
trained model. But 30 and 56 calls are few (one call is 3 and 2 points), so 83% vs 87% is noise, and on
the easy set the model is only about level with the rules. The hard set shows the real gap:
the model still picks the wrong number when a reply has several (bulk price, old price, the price of
the alternative) and answers `false` for delivery when nothing was said. The 400 training examples are
made up from templates; real call transcripts are the next big improvement.

## Use the model
1. Kaggle → your notebook `wimb-shop-model-training` → **Output** tab → download
   `wimb-shop-model/wimb-shop-model.Q4_K_M.gguf` (about 2.5 GB).
2. Put it in a folder together with this repo's **`Modelfile`** (use this one, not the Modelfile in
   the Kaggle output: this one matches the training text exactly).
3. `ollama create wimb-shop -f Modelfile`, then
   `ollama run wimb-shop "Product: 16A MCB` + newline + `Shopkeeper: haan hai, 220 ka, delivery nahi"`.
   Ollama itself has not been run here (this environment cannot reach it); the same file was loaded and
   answered all 86 test calls through llama.cpp.

## Files
| File | What it is |
|---|---|
| `train_colab.ipynb` | The notebook (Colab or Kaggle). Built by `build_notebook.py`, don't edit by hand. |
| `run_on_kaggle.py` | Starts the notebook on Kaggle, waits, downloads the results. |
| `data/train.jsonl` | 400 **made-up** training examples (Hindi, Marathi, English). Made by `make_data.py`. |
| `data/test.jsonl` | 30 hand-written test calls, never used for training. |
| `score.py` | The prompt, and how answers are scored. |
| `test_training.py` | Checks the data and scoring: `python -m pytest` |

## Use real data
The sample data is made up, to prove training works. For a real model, put real call
transcripts in `data/train.jsonl` (same format, one per line), then run
`python build_notebook.py` and train again.

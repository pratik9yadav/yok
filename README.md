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

## Results (Kaggle T4, Qwen3-4B-Instruct-2507, 2 epochs, runs 9 and 10)
Share of calls where all four fields are exactly right. Three held-out sets, never trained on:
`test` = 30 hand-written calls, `hard` = 56 calls in harder styles, `fresh` = 60 calls written blind
(the writer never saw any training data).

| | simple rules, no AI* | model before training | trained on v1 (400 made-up) | trained on v2 (real data) | v2 exported Q4_K_M file |
|---|---|---|---|---|---|
| test (30) | 77% | 63% | 83% | **97%** | 97% |
| hard (56) | 30% | 27% | 43% | **84%** | 79% |
| fresh (60) | 37% | 45% | 60% | **77%** | 70% |

*A keyword/regex program written by an independent reviewer (not in this repo).

How to read it: same model, same settings, only the training data differs. v2 (1,300 rows built from
real products, real prices and real spoken number words, plus 151 real Hindi answers) beats v1 on
every set. `fresh` is the fairest number (+17 points over v1): v2 was designed after looking at
v1's mistakes on `test` and `hard` (kinds of mistakes, not their wording: only 1 of the 116 hard and
fresh replies shares 6 words in a row with the training data). The exported file loses a few calls on `hard` and `fresh`
(3 and 4 calls) to 4-bit rounding.

What is still wrong (after v2):
- `alternative` written shorter than the answer key: "Dewalt" for "Dewalt 13 mm drill", "12 inch" for
  "12 inch wala". The prompt never says how full the name should be, and the `fresh` key uses full
  names (sometimes with words the shopkeeper did not say), so some of these are a wording rule
  still to decide, not a misread.
- Spoken prices with "paune" (quarter less): पौने सत्रह सौ = 1675 read as 1700, पावणे दोनशे = 175 as 195;
  "paintalis" (45) read as 125, "doshe pannas" (250) as 150. And "ladka chhod dega ghar pe" not read
  as delivery.

## Try the model
`try_model.ipynb` runs the trained model the way you would on a PC: on Kaggle's free T4 GPU it installs
Ollama, makes the model with this repo's `Modelfile` (`ollama create`), answers the example calls in
`data/demo_questions.jsonl` (a demo, not a test set), times them on the GPU and on CPU only, and then
answers your own questions. It reads the model file straight from the training run's output.
First run (Ollama 0.40.3, Kaggle T4): 9 of the 12 example calls fully right; **0.55 s per answer on the
GPU, about 10 s on CPU only** (about what a laptop takes). Making the model (`ollama create`) took 44 s and
loading it 46 s. The 3 misses: two Marathi spoken prices (साडेचारशे = 450 read as 350, पावणेतीनशे = 275 as
350) and "ladka ghar pe de jayega" not read as delivery. That run used Ollama's default context
(32,768 tokens), which held 7.5 GB of memory; the `Modelfile` now sets `num_ctx 2048`: checked on the next
run, 2.9 GB, same answers, same speed (0.57 s GPU, 10 s CPU). The very first answer after loading can take
about 30 s (start-up); the ones after it are fast.

Five ways to ask it:
1. **Ask Claude** in the chat: "ask the model: 16A MCB / haan hai, 220 ka, delivery nahi". Claude starts the
   run below and tells you the answer in plain words.
2. **GitHub** (easiest on a phone): **Actions → Ask the model on Kaggle → Run workflow**, type the product
   and what the shopkeeper said. About 10 minutes; the answers are at the end of the job log. It uses the
   same `KAGGLE_API_TOKEN` secret as the training.
3. **Kaggle, ask many questions (best for testing)**: your own notebook "Ask the shop model". Make it once:
   **Actions → Ask the model on Kaggle → Run workflow → run: notebook** (or ask Claude "make the ask notebook";
   about 5 minutes, it checks the notebook once: that the model answers and the ask box can be built, not that
   the editor draws the box). After each new training run, run it again with run = notebook, so the notebook
   uses the new model. Then open
   `https://www.kaggle.com/code/<your Kaggle name>/wimb-shop-model-ask` → **Edit** → **Run all** (about 3–4
   minutes each time you start it) → at the bottom, type the product and what the shopkeeper said in the ask
   box, tap **Ask**: about half a second per answer, as many as you like. No box, only text like `VBox(...)`?
   Use cell 7, under it. When done: **Run → Stop session** (saves your free GPU hours).
4. **Kaggle, by hand**: open your notebook `wimb-shop-model-try` on kaggle.com (made by the first run of 1 or 2;
   or upload `try_model.ipynb` as a new notebook) → **Edit**. The training output must be attached:
   **Add Input → Notebooks → Your Work → `wimb-shop-model-training`**. Type `PRODUCT` and `SAID` in cell 1,
   then **Run all**.
5. **On your PC with Ollama** (ollama.com):
   1. Kaggle → your notebook `wimb-shop-model-training` → **Output** tab → download
      `wimb-shop-model/wimb-shop-model.Q4_K_M.gguf` (about 2.5 GB).
   2. Put it in a folder together with this repo's **`Modelfile`** (use this one, not the Modelfile in
      the Kaggle output: this one matches the training text exactly).
   3. `ollama create wimb-shop -f Modelfile`, then
      `ollama run wimb-shop "Product: 16A MCB` + newline + `Shopkeeper: haan hai, 220 ka, delivery nahi"`.
      Newer Ollama versions may build the prompt from the chat template inside the `.gguf` (the one used in
      training) instead of the Modelfile's `TEMPLATE`; for these questions both give the same prompt. The
      Modelfile's `SYSTEM` and settings are used either way. The try run's report says which one Ollama chose.
   4. From an app, call Ollama's API. The JSON answer is in `response`. Don't send a system message: the
      Modelfile has it.
      ```
      curl http://localhost:11434/api/generate -d '{"model": "wimb-shop", "stream": false,
        "prompt": "Product: 16A MCB\nShopkeeper: haan hai, 220 ka, delivery nahi"}'
      ```

## Real data
Training set v2 (`data/train.jsonl`, 1,300 rows) is generated by `make_data.py` from real material,
and the 151 real Hindi answers in `data/real/xpqa_hi_examples.jsonl` are added at training time.
`fetch_real_data.py` downloads real material from open
datasets (licences checked; details and attribution in `data/real/NOTICE.md`):
- real Indian product names with MRP and selling price (Flipkart, Amazon India; Kaggle),
- real Hindi and Marathi role-play phone conversations (AI4Bharat IndicVoices, CC BY 4.0).

Run it from **Actions → Fetch real data → Run workflow**. Results land on a `real-data/run-…` branch
for review, never straight on `main`.

IndicVoices needs a one-time click: log in to Hugging Face, open
https://huggingface.co/datasets/ai4bharat/IndicVoices and accept the terms, then create a **Read**
token (Settings → Access Tokens) and save it as the repository secret `HF_TOKEN`:
https://github.com/pratik9yadav/yok/settings/secrets/actions/new

## Files
| File | What it is |
|---|---|
| `train_colab.ipynb` | The notebook (Colab or Kaggle). Built by `build_notebook.py`, don't edit by hand. |
| `run_on_kaggle.py` | Starts the notebook on Kaggle, waits, downloads the results. |
| `try_model.ipynb` | Tries the trained model in Ollama on Kaggle and answers your questions. Built by `build_notebook.py`. |
| `try_on_kaggle.py` | Starts `try_model.ipynb` on Kaggle with the training output attached, waits, prints the answers. `--make-ask-notebook` puts `ask_model.ipynb` on Kaggle instead. |
| `ask_model.ipynb` | "Ask the shop model": run the model once on Kaggle, then ask it many questions in an ask box. Built by `build_notebook.py`. |
| `data/train.jsonl` | Training set v2: 1,300 examples generated by `make_data.py` from real products, MRP/prices, spoken number words and spellings (CC BY-SA 4.0, see `data/TRAIN_NOTICE.md`). |
| `data/train_v1.jsonl` | The first 400 made-up examples, kept to compare against (`TRAIN_SET = "v1"`). |
| `data/test.jsonl` | 30 hand-written test calls, never used for training. |
| `data/hard_test.jsonl` | 56 harder calls in styles the training data does not have. |
| `data/fresh_test.jsonl` | 60 calls written blind (the writer never saw the training data), checked by a second labeller. |
| `data/demo_questions.jsonl` | The example calls `try_model.ipynb` answers (a demo, not a test set). |
| `data/real/` | The real source data and its licences (`NOTICE.md`). |
| `score.py` | The prompt, and how answers are scored. |
| `test_training.py` | Checks the data and scoring: `python -m pytest` |
| `test_try.py` | Checks `try_model.ipynb` and `try_on_kaggle.py` with a fake Ollama. |
| `test_ask.py` | Checks `ask_model.ipynb` (with a fake Ollama and ipywidgets) and `try_on_kaggle.py --make-ask-notebook`. |

## Use real data
Real call transcripts beat everything else. Put them in a new file in `data/` (same format, one per
line), add them to `TRAIN_DATA` in `build_notebook.py`, run `python build_notebook.py` and train again.

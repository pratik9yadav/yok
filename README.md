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

## Results: v4 (run 11, 2026-10-22: 6,151 rows, 2 epochs, 1 h 55 min on Kaggle)

| set | before training | v2 after | **v4 after** | v4 exported Q4_K_M file |
|---|---|---|---|---|
| test (30) | 63% | 97% | **100%** | 100% |
| hard (56) | 27% | 84% | **77%** | 80% |
| fresh (60, blind) | 45% | 77% | **82%** | 78% |
| asr (60, blind, speech-to-text style) | 25% | not measured | **95%** | 93% (56 of 60 done) |

v4 is better on fresh (+5) and test, and reads speech-to-text Devanagari very well (asr 95%), but is worse on
hard (−7, 4 calls). Most of its misses are `alternative` wording: v4 learned "brand + size + item" without
"wala"/"ka", while the hard and fresh answer keys use other forms ("फिलिप्स वाला", "Dewalt 13 mm drill"). Real
misreads left: a price after "पचासी नहीं पंचानवे" (95 → 590), unit vs pack price, "aake le lo" (no delivery) and
one Marathi row read as in stock. v2 has not been scored on the asr set yet.

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

## Free web page (Hugging Face)
A web page for the phone: type the product and what the shopkeeper said, tap **Ask**, get the 4 answers in plain
words (and the raw JSON, and the seconds it took). No Kaggle needed to use it. It is a free Hugging Face Space
(Gradio, 2 CPUs, 16 GB) running the trained model file (`.Q4_K_M.gguf`) with llama.cpp, with exactly the training
prompt (`score.SYSTEM` and `score.user_message`; the deploy copies `score.py` into the Space).

Put it there (once, and again after each new training run):
1. The repository secret `HF_TOKEN` must be a Hugging Face token with **write** access
   (https://huggingface.co/settings/tokens; save it at https://github.com/pratik9yadav/yok/settings/secrets/actions/new).
   `KAGGLE_API_TOKEN` is needed too (the model file comes from the training run's Kaggle output).
2. **Actions → Put the model on Hugging Face (free web page) → Run workflow** (model: v4, visibility: private).
   About 10–25 minutes: it downloads the model file from Kaggle (2.5 GB), uploads it to your **private** model repo
   `<your HF name>/wimb-shop-model-v4` (with the repo's `Modelfile` and a model card), makes or updates the Space
   `<your HF name>/wimb-shop-model`, waits until it runs and prints its link at the end of the job log.
   From a terminal: `python space/deploy.py` (`--model v2`, `--public`, `--skip-upload-model`).

Open it on the phone: log in to huggingface.co (the Space is private: only you can open it), then open
`https://huggingface.co/spaces/<your HF name>/wimb-shop-model`. The first start after a deploy downloads and
loads the model: the page says "Loading the model" for 1–2 minutes. A free Space goes to sleep after 48 hours
without visits; opening the link wakes it (a few minutes).

Speed: on the free 2 CPUs expect **roughly 10–20 seconds per answer** (an estimate from the 10 s per answer that
Ollama took on Kaggle's CPU; not measured on the Space yet). One question at a time.

Private by default because the training data includes CC BY-SA 4.0, ODC-By and CDLA-Sharing-1.0 material (see
`data/TRAIN_NOTICE.md`); whether share-alike reaches a trained model is not settled. visibility = public makes only
the page public; the model file stays in the private model repo. A new Space gets the deploy's token as its secret
`HF_TOKEN` (to read the model repo); safer: replace it with a **read** token (Space → Settings → Variables and
secrets). Later deploys keep it (`--set-space-token` puts the deploy's token back).

Switch to v2: run the workflow with model = v2 (it uses the model repo `wimb-shop-model-v2` and points the page at
it); model = v4 switches back. Files: `space/app.py` (the page), `space/deploy.py`, `space/requirements.txt`
(pinned: gradio 6.29.1, huggingface_hub 2.1.1, llama-cpp-python 0.3.36 from its prebuilt CPU wheels),
`space/README.md` (the Space's settings), `space/test_space.py`.

## Voice v0 (speech to speech)
A first spoken shop call, all free open models on Kaggle's free T4. The AI asks 3 questions in Hindi by voice
(is the 16A MCB in stock? the price? delivery?). Each spoken answer goes through three models, one after the other
(a cascade):
1. **Ears**: Whisper large-v3 turbo turns the speech into Hindi text.
2. **Brain**: the trained shop model (Ollama, the repo's `Modelfile`, exactly as in "Try the model") gives the
   4 fields; a small template writes a short polite reply in Hindi, e.g. "ठीक है, 16A MCB ₹220 में है, डिलीवरी नहीं।
   धन्यवाद!". Not in stock with another product, price not said, or nothing understood (it asks again politely)
   each get their own reply.
3. **Mouth**: a text-to-speech model says the questions and the replies.

Every step is timed (seconds for ears, brain, mouth, total). It needs one finished training run. Not tested on
Kaggle yet.

**Run it:** **Actions → Voice v0 (speech to speech) → Run workflow** (about 20–40 minutes), or ask Claude
"run voice v0". From a terminal: `python voice/run_voice.py`.

**Listen:** on the GitHub run's page, at the bottom, **Artifacts → kaggle-output-voice** (a zip with the .wav
files); or on kaggle.com, your notebook `wimb-voice-v0` → **Output** tab → folder `voice/`: `q1.wav`–`q3.wav`
(the questions), `synthetic_1.wav`… (the test answers), `reply_1.wav`… (the replies) and `voice_results.json`
(what was heard, understood and replied, and the times).

**Test answers:** every run makes 6 made-up shopkeeper answers with the TTS voice (yes with the price in words
"दो सौ बीस", no + another brand, delivery yes and no, Hinglish written in Hindi letters, "पता नहीं, बाद में कॉल करो"),
so it runs end to end without your recordings. They check the plumbing, not real accents.

**Add your own recordings (from a phone):**
1. Record each answer as a shopkeeper would say it (a recorder app, or a WhatsApp voice note sent to yourself
   and saved). Name each file by the question it answers: `q1_…` (in stock?), `q2_…` (price?), `q3_…`
   (delivery?), e.g. `q2_rate.m4a`. .m4a, .mp3, .ogg, .opus and .wav all work. Only record people who agreed.
2. kaggle.com → **Datasets** → **New Dataset** → upload the files → title `wimb-voice-clips` → keep it
   **Private** → **Create**.
3. **Actions → Voice v0 → Run workflow** → `clips_dataset`: `<your Kaggle name>/wimb-voice-clips` (the end of the
   dataset's link) → **Run workflow**. Your clips are processed first, then the 6 made-up ones.

**The better Hindi voice (optional):** Indic Parler-TTS needs a free Hugging Face token. Log in to
huggingface.co, open https://huggingface.co/ai4bharat/indic-parler-tts and accept its terms, make a **Read** token
(Settings → Access Tokens). Then on kaggle.com open your notebook `wimb-voice-v0` (made by the first run) →
**Edit** → **Add-ons → Secrets** → add `HF_TOKEN` with the token, switch it on for this notebook. Without it the
fallback voice (MMS-TTS) is used.

**Models and licences** (the run records which voice it used and its licence):

| Part | Model | Licence |
|---|---|---|
| Ears | `openai/whisper-large-v3-turbo` | MIT |
| Brain | the shop model: Qwen3-4B-Instruct-2507 + our LoRA, in Ollama | Apache-2.0 (Ollama: MIT) |
| Mouth, first choice | `ai4bharat/indic-parler-tts` (library `parler-tts` 0.2.3) | Apache-2.0 |
| Mouth, fallback | `facebook/mms-tts-hin` | CC-BY-NC-4.0: **for this internal test only, not for business use** (the results say so) |

This cascade is the start. The goal is a direct speech-to-speech model, trained on real shop calls recorded with
the shopkeepers' consent.

## Coder model (phase 1: measure)
**First full run (2026-10-15, Kaggle 2× T4, Ollama 0.40.4, 58 minutes):** bugs fixed of 164, one try each.

| model | fixed | s/task | notes |
|---|---|---|---|
| **qwen3-coder:30b** | **120 (73%)** | 2.2 | best and fastest (only 3B of its 30B work per word); 18.6 GB, split over both T4s |
| qwen2.5-coder:14b | 105 (64%) | 6.1 | 9 GB |
| qwen2.5-coder:7b | 93 (57%) | 2.9 | 4.7 GB |
| gpt-oss:20b | 91 (55%) | 5.8 | `think` low |

A second project in this repo: a free open coding model that is good at fixing bugs. First step: measure free
coding models as they are, before any training, on Kaggle's free GPU (2× T4).

What it measures: **HumanEvalFix (Python)**: 164 small Python functions, each with one bug
(`bigcode/humanevalpack`, MIT licence). Each model gets the buggy function and "Fix bugs in <name>. Reply with the
full corrected function in one ```python block." (the tests are not shown). Its fix is run against the task's
hidden tests: all pass = fixed. Score = tasks fixed with one try (greedy, temperature 0). Also: seconds per task,
tokens/s, and whether Ollama ran it on the GPU. Models (Ollama): `qwen2.5-coder:7b`, `qwen2.5-coder:14b`,
`qwen3-coder:30b`, `gpt-oss:20b` (thinks a little first: `think` low).

How to run: **Actions → Coder model on Kaggle → Run workflow** (or ask Claude "measure the coding models").
About 3½ hours for the 4 models: each stops after 45 minutes (download included) and the report says how many
tasks it did. `limit` = 10 gives a quick check. The table is at the end of the job log; `coder_eval.json` (every
reply) is in the run's Output tab and in the job's `kaggle-output-coder` artifact. From a terminal:
`python coder/run_eval.py` (`--models a,b`, `--limit N`, `--wait`).

Results: see the latest run.

Files: `coder/build.py` writes `coder/coder_eval.ipynb` (and `coder/coder_train.ipynb`; don't edit them by hand); `coder/fixeval.py` holds the
prompt, code extraction, test running and scoring; `coder/run_eval.py` starts it on Kaggle; `coder/test_coder.py`
checks it all with a fake Ollama.

## Coder model (phase 2: train)
**First full run (2026-10-15, 3 h on one T4): training made it worse.** Qwen2.5-Coder-14B, 2,000 bug-fix
commits from CommitPackFT (5,793 kept of 56,025; 3 dropped as overlapping the test), 250 steps, 112 min, loss
0.08. HumanEvalFix: **106/164 (65%) before → 100/164 (61%) after**: 14 newly fixed, 20 newly broken. So raw
commits are not the right teacher (most are whole-file edits with short, vague messages, and the model has
likely seen them before). The best free model is still qwen3-coder:30b as it is (73%).

Can a free 14B model, trained for free, beat phase 1's best (qwen3-coder:30b, 73%)? `coder/coder_train.ipynb`
trains **Qwen2.5-Coder-14B-Instruct** (Apache 2.0; 4-bit, QLoRA with Unsloth) on one Kaggle T4 and scores it on
HumanEvalFix before and after, with phase 1's prompt, code extraction and tests (`coder/fixeval.py`).

Training data: real bug-fix commits from **CommitPackFT (Python)** (`bigcode/commitpackft`). The notebook stops if
the dataset card's licence is not open (MIT, Apache-2.0, BSD, CC-BY-4.0, CC0, ODC-BY), and keeps only commits from
permissively licensed repositories (MIT, Apache-2.0, BSD, ISC, Unlicense, CC0). It keeps small fixes (the message
says fix/bug/error/wrong/incorrect/crash/typo/exception; not merges, reverts, version bumps, docs or test files; at
most 20 lines changed; at most 2,048 tokens), drops duplicates, and **drops every commit that shares any 10 words in
a row, or a function signature, with a HumanEvalFix task** (decontamination; the report says how many). Each
example: the file before the fix + "Fix bugs: <commit subject>. Reply with the full corrected code in one ```python
block." → the file after it. Up to 2,000 examples, 1 epoch; training stops by itself after 4 hours.

How to run: **Actions → Coder model on Kaggle → Run workflow → action `train`** (`smoke` = yes first: a 20–30
minute check of the whole pipeline). A full run takes about 5–8 hours on Kaggle, but a GitHub job may last only 6:
the job waits about 5½ hours, and if Kaggle is still going it ends green with "still running"; then run it again
with action **`fetch-train`**, which waits for that run and prints the report: data counts, licence, training loss
and time, before → after, and the tasks that flipped. From a terminal: `python coder/run_train.py` (`--smoke`,
`--max-train N`, `--epochs N`, `--wait`). `coder_train.json` (every reply) and the LoRA add-on (`coder-lora`) are
in the run's Output tab.

Notes: the scores use the 4-bit bitsandbytes model through transformers, not Ollama's Q4_K_M file, so "before" may
differ a little from phase 1's 64%; before and after use the same setup, so the comparison is fair. No GGUF/Ollama
file in this phase: merging the LoRA into a 14B model needs more memory and disk than Kaggle gives safely.

Files: `coder/build_train.py` writes `coder/coder_train.ipynb` (`python coder/build.py` writes both coder
notebooks); `coder/fixtrain.py` holds the data filters, decontamination, prompt format, scoring loop and report;
`coder/run_train.py` starts it on Kaggle; `coder/test_train.py` checks it all offline.

## Coder model (phase 3: the coding agent)
**Capability test (2026-10-16, branch `bench/agent-1`: a small orders module with planted bugs; each result checked
with hidden tests): 4 of 6 solved.**

| task | result |
|---|---|
| 1. crash: `parse_price("₹1,250.50")` | ✅ right fix; left a stray `test_fix.py` and a duplicated test |
| 2. wrong maths: 10% off 1000 gave 100 | ✅ clean one-line fix and 4 tests (one import-order lint) |
| 3. vague report: "3 in stock, but one has 0" | ✅ found `>= 0`, fixed, good edge-case tests |
| 4. Monday's sales shown under Tuesday | ❌ looped on `ls` for 30 steps, no change |
| 5. new feature: Indian money format `₹12,50,000` | ❌ said "done, all tests pass" but changed no file |
| 6. rename `calc_total` → `order_total` in 3 files | ✅ complete and correct |

Each run took 3–18 minutes on Kaggle's free T4s. The two failures are the loop's weak spots, not only the model's:
it repeats a command forever, and it can still claim "done" with nothing changed after being sent back twice.

A free "junior engineer": you give it a task in plain words, it works on a copy of this repository on Kaggle's free
GPU with **qwen3-coder:30b** (phase 1's best: 73%, used as it is), and its change comes back as a **pull request**
for you to review.

**From your phone:** GitHub app or github.com → **Actions → Coding agent (free model) → Run workflow** → type the
task (e.g. "Add a test for score.py that checks an empty reply scores 0") → **Run workflow**.
About 30–60 minutes later there is a pull request `Agent: <your task>` with the agent's summary, the files it
changed and the test results. If the agent has a question instead, there is no pull request: open the run, the
question is on its summary page; run it again with a clearer task. Or ask Claude to run it.

What it does: starts Ollama and the model on Kaggle (2× T4, about 10 minutes), clones the repository at the commit
the workflow ran on, runs the tests once (before), then works like a junior engineer, one bash command per step:
reads CLAUDE.md / AGENTS.md / README, finds the code with grep, writes a failing test first for a bug, makes the
smallest change, runs `python -m pytest -q` and `ruff check .` and fixes what it broke. At most 40 steps and 45
minutes; num_ctx 32768 (the model and its memory fit on the two T4s; if Ollama shows it does not, it uses 16384 and
says so). Then the notebook saves the change (`git diff` including new files) and runs the tests again (after). The
GitHub runner applies the change and runs the tests and ruff once more; all of it goes into the pull request.

Good at: small, clear tasks in one or two files: fix a bug that has a clear symptom, add a test, add a field or an
option, rename something, update docs. Not good at: big or vague tasks, design decisions, anything that needs the
internet, secrets, Kaggle or GitHub (it can't run workflows), or many files at once. It is a 30B model with 45
minutes: read every pull request before merging; when tests fail on the runner, the pull request says so.

Safety: it runs on a throwaway Kaggle machine that holds no secrets (no GitHub or Kaggle token; the repository must
be public); it never commits or pushes: only its patch comes back. The patch is refused if it touches `.github/`
or is larger than 200 KB. The change only ever goes to a new branch `agent/run-<id>-<attempt>` with a pull request; nothing
is pushed to main and nothing is merged. The job that runs the agent's code on GitHub can't write to the
repository; the job that opens the pull request never runs it.

One-time setup: the secret `KAGGLE_API_TOKEN` (as for the other workflows), and **Settings → Actions → General →
Workflow permissions → Allow GitHub Actions to create and approve pull requests** (without it the change is still
pushed to the branch, and the job log gives a link to open the pull request by hand). Pull requests made by the
workflow do not start other workflows.

From a terminal: `python coder/run_agent.py --task "..."` (`--task-file`, `--commit SHA`, `--repo owner/name`,
`--wait`); the files land in `kaggle_output_agent/` (`changes.patch`, `agent_log.json` with every step,
`agent_result.json`). Files: `coder/agent.py` is the agent loop (bash only, one command per reply, as in
mini-swe-agent; the model is one small class, `OllamaChat`, so another backend such as the Claude API can be added
later); `coder/build_agent.py` writes `coder/coder_agent.ipynb` (`python coder/build.py`; don't edit it by hand);
`coder/run_agent.py` starts it on Kaggle and checks the patch; `coder/agent_pr.py` writes the job summary and the
pull request text; `.github/workflows/agent.yml` is the workflow; `coder/test_agent.py` checks it all offline with
a scripted fake model.

## Coder model (phase 4: the engineer exam)
**First run (2026-10-21, regraded with the fixed reader; 30 tasks, no checklist in the prompt):**

| model | total | solved | hidden tests | own tests | lint | security | explains |
|---|---|---|---|---|---|---|---|
| gpt-oss:20b (think medium) | 60.9 | 8/30 | 75% | 0% | 41% | 87% | 67% |
| qwen2.5-coder:14b | 61.6 | 1/30 | 63% | 3% | 67% | 95% | 100% |
| qwen3-coder:30b | 59.2 | 2/30 | 72% | 0% | 39% | 95% | 63% |

None of them writes tests unless asked, most handle the main path but miss bad input and edge cases, and the
code is often not lint-clean. gpt-oss was cut off on 4 long tasks (that run had num_predict 6144).

Results: none yet (run it: **Actions → Coder model on Kaggle → Run workflow → action `exam`**).

Phase 1 measured whether a model can fix a small function. This asks a different question: **does a free coding
model work like an engineer on its own, with no style guide or checklist in the prompt?** Each model gets ~30
real-world tasks written the way a user writes them (a shop owner asking for a stock ledger, a bug report, "review
this", "clean this up"), plus the project's files when there are some, and one neutral line on how to send files
back. No system prompt, nothing about tests, error handling or security: what it adds by itself is what is measured.

Tasks (`coder/exam/tasks/<NN-name>/`): `task.md` (the request), `meta.json` (`kind`: build / bugfix / review /
refactor, `difficulty` 1-3, the expected `files`, a test `timeout`), `starter/` (files the model is shown, if any),
`hidden/test_hidden.py` (pytest tests the model never sees, with edge cases and bad input) and `reference/` (a
solution, never shown). Standard library only.

How a reply is graded (`coder/exam/grade.py`), 0-100:

| part | points | how |
|---|---|---|
| hidden tests | 55 | share of the hidden tests that pass (all pass = **solved**) |
| its own tests | 15 | none: 0; written but failing: half; written and passing: all |
| lint | 10 | ruff with a fixed rule set (E, F, W, B, UP, SIM, pylint errors/warnings/complexity; line length 120): 1 point off per issue |
| security | 10 | bandit on its code: no findings 10, only low ones 5, medium or high 0 (left out of the total if bandit is missing) |
| explanation | 10 | more than 40 words outside the code |
| stray files | -5 | files nothing needs (a scratch script, notes): not an expected file, test, README or a module it imports |

Files are read from the reply's fenced blocks named by a path (```` ```inventory.py ````, `# inventory.py` as the
first line, or `inventory.py:` / `File inventory.py` just before the block). Only when no block names a path, a fixed
fallback places its Python blocks (a block of tests → `test_<file>.py`; one expected file ← the code block, or the code
blocks joined; several expected files ← matched by the names they define; else nothing), and the report counts how
many replies needed it. Each reply is tested in a fresh folder; the hidden tests run with the model's pytest config and conftest.py ignored,
and every test run is stopped at the task's timeout (with anything it started). The report gives, per model: the
average total, tasks solved, each part, the scores by kind of task, seconds per task, and its 5 weakest tasks
(with why). `exam.json` keeps every reply and every grade with the raw test, lint and bandit output.

How to run: **Actions → Coder model on Kaggle → Run workflow → action `exam`** (`models`, `limit` as in phase 1;
`limit` = 3 is a quick check), or ask Claude "give the coding models the engineer exam". Each model stops after 60
minutes (download included), so about 4½ hours for the 4 default models (qwen3-coder:30b, qwen2.5-coder:14b,
gpt-oss:20b with `think` medium, qwen2.5-coder:7b); greedy (temperature 0), num_ctx 16384, up to 8,192 tokens per
reply (12,288 for a thinking model: its thinking and its answer). If the job ends with "still running", run it again
with action **`fetch-exam`**. Action **`regrade-exam`** (no GPU, a few minutes) fetches the last run and grades every
saved reply again with the current grader (`kaggle_output_exam/exam_regraded.json`; the log shows each model's scores
before → after, and the start of every reply that still has no files). Before any model, the notebook
checks every reference solution passes on Kaggle and leaves out a task whose reference fails there. From a terminal:
`python coder/exam/run_exam.py` (`--models a,b`, `--limit N`, `--wait`, `--regrade`); files land in `kaggle_output_exam/`.

Writing a task: add the folder, then `python -m coder.exam.grade --check-references` (each reference must pass all its
hidden tests, and the starter alone must not), then `python coder/build.py` (the tasks are copied into the
notebook, so Kaggle needs no access to this repository) and `python -m pytest` (it runs the same check).

Files: `coder/exam/grade.py` (task format, prompt, reading files out of a reply, grading, the reference check);
`coder/exam/exam.py` (one model takes the exam through Ollama; the report); `coder/exam/build_exam.py` writes
`coder/exam/exam.ipynb` (`python coder/build.py`; don't edit it by hand); `coder/exam/run_exam.py` starts it on
Kaggle; `coder/exam/test_exam.py` checks it all offline with a fake Ollama and the 2 tiny tasks in
`coder/exam/sample_tasks/`.

## Real data
Training set v2 (`data/train.jsonl`, 1,300 rows) is generated by `make_data.py` from real material,
and the 151 real Hindi answers in `data/real/xpqa_hi_examples.jsonl` are added at training time.
`fetch_real_data.py` downloads real material from open
datasets (licences checked; details and attribution in `data/real/NOTICE.md`):
- real Indian product names with MRP and selling price (Flipkart, Amazon India; Kaggle),
- real Hindi and Marathi role-play phone conversations (AI4Bharat IndicVoices, CC BY 4.0).

Run it from **Actions → Fetch real data → Run workflow**. Results land on a `real-data/run-…` branch
for review, never straight on `main`.

### Training set v4 (for the speech-to-speech pipeline)
`data/train_v4.jsonl` (6,000 rows, `python make_data.py`; the same file every time, and `train.jsonl` does not
change) is built like v2, from the same real products, prices, number words and spellings, but written the way
Whisper writes a shopkeeper's speech: Hindi in Devanagari 25%, Hinglish in Devanagari 25% ("हाँ सर, अवेलेबल है,
प्राइस 220"), Hinglish in Latin letters 20%, Marathi in Devanagari 20%, Marathi in Latin letters 10%. 30% of the
Devanagari rows have speech-to-text errors (a dropped, merged, split or repeated word, filler words like अं / हाँ
जी, missing nukta or chandrabindu, digits written another way, no punctuation), never in the price, the yes / no,
the delivery phrase or the alternative's name. It has many more spoken prices (पौने सत्रह सौ = 1675, साडेचारशे,
सवा पाँच हज़ार, also off price_words.json's grid), discounts ("MRP 500 है पर 450 लगा देंगे" → 450), unit vs box,
corrections, delivery said in many ways ("ladka ghar pe de jayega", "पोरगा घरी आणून देईल" → true; "aake le jao" →
false; not said → null), and one rule for `alternative`: brand + size/type + item, exactly as the shopkeeper said
it, in the script used ("Schneider 32 एम्पियर एमसीबी"). Unclear, call-back and wrong-number replies (8%) are all
null. `score.SYSTEM` (the prompt and the Modelfile) is unchanged. Train it with **Actions → Train on Kaggle →
train_set: v4** (`python run_on_kaggle.py --train-set v4`): 6,151 rows with the 151 xPQA answers, 2 epochs, about
1 hour of training and 2–2.5 hours in all, in its own Kaggle notebook `wimb-shop-model-training-v4`, so the v2
model that the try, ask and voice notebooks use is not replaced. The notebook's default stays v2 until v4 is
measured. Every run is also scored on `data/asr_test.jsonl` (written blind, as speech-to-text writes calls) when
it is there.

IndicVoices needs a one-time click: log in to Hugging Face, open
https://huggingface.co/datasets/ai4bharat/IndicVoices and accept the terms, then create a **Read**
token (Settings → Access Tokens; a write token works too, and the free web page needs one) and save it as the
repository secret `HF_TOKEN`:
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
| `data/train_v4.jsonl` | Training set v4: 6,000 rows as speech-to-text writes them (`TRAIN_SET = "v4"`; see "Training set v4"). |
| `data/test.jsonl` | 30 hand-written test calls, never used for training. |
| `data/hard_test.jsonl` | 56 harder calls in styles the training data does not have. |
| `data/fresh_test.jsonl` | 60 calls written blind (the writer never saw the training data), checked by a second labeller. |
| `data/demo_questions.jsonl` | The example calls `try_model.ipynb` answers (a demo, not a test set). |
| `data/real/` | The real source data and its licences (`NOTICE.md`). |
| `score.py` | The prompt, and how answers are scored. |
| `test_training.py` | Checks the data and scoring: `python -m pytest` |
| `test_try.py` | Checks `try_model.ipynb` and `try_on_kaggle.py` with a fake Ollama. |
| `test_ask.py` | Checks `ask_model.ipynb` (with a fake Ollama and ipywidgets) and `try_on_kaggle.py --make-ask-notebook`. |
| `space/` | The free web page on Hugging Face: `app.py` (Gradio + llama.cpp), `deploy.py` (puts the model and the page there; workflow `hf_space.yml`), `test_space.py`. |
| `voice/` | Voice v0 (speech to speech): `voice_v0.ipynb` (built by `voice/build_voice.py` from `voice/voicecall.py`), `run_voice.py` (runs it on Kaggle), `test_voice.py`. |

## Use real data
Real call transcripts beat everything else. Put them in a new file in `data/` (same format, one per
line), add them to `TRAIN_DATA` in `build_notebook.py`, run `python build_notebook.py` and train again.

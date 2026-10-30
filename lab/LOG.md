# Lab log

The lab co-worker (`.claude/skills/lab/SKILL.md`) runs here twice a day: it makes data, trains on Kaggle's free GPU,
measures on frozen test sets, keeps only what is better, and writes every round below. Newest experiment first.

## Scoreboard

Best so far. Lower is better for error rates. A new result replaces a line only when it beats it on the same frozen test.

| Model | Test (frozen) | Best | From |
|---|---|---|---|
| Ears (Whisper turbo + LoRA `wimb-ears`) | FLEURS test hi, 100 clips: word errors clean / phone | 19.8% / 24.3% | ears run 1, 2026-10-26 |
| Ears | FLEURS test mr, 100 clips: word errors clean / phone | 35.9% / 39.8% | ears run 1, 2026-10-26 |
| Shop model (Qwen3 4B QLoRA, v4) | test / hard / fresh / asr | 100% / 77% / 82% / 95% | run 11 |
| Voice bot brain (gemma3:4b + rules) | 48 typed test calls | 48/48 | eval 5 |
| Voice bot, spoken by synthetic voices | 44 test calls over a phone line | 19/44 | eval 5 |

## Queue

Ideas, best expected gain first. The lab picks the top one it can do, and adds or reorders as it learns.

1. Ears, real accents: find free, ungated Hindi/Marathi speech from many speakers and districts (not FLEURS) to
   train on, and a separate free set (other speakers) to freeze as a second ears test. Gated sets (IndicVoices,
   Lahaja) wait for the owner's `HF_TOKEN` Kaggle secret.
2. Ears, conversational speech: short replies like a shopkeeper's ("हाँ है", "नहीं, कल आएगा", prices, phone numbers)
   are what the bot hears most; FLEURS is read news. Add short, spoken-style clips to training.
3. Voice bot spoken score: find out why 25 of 44 spoken calls fail (ears or synthetic voice) and fix the biggest
   cause. First step running now (eval 7 with the trained ears).
4. Tooling for 1–2: this container cannot reach huggingface.co, so dataset pages (gated? licence? columns?) can't be
   checked from here. Add a small "probe" mode to `fetch_data.yml` (GitHub runner: prints a dataset's licence,
   gating, columns and a few row lengths, no audio saved) before spending GPU hours on a new speech source.
   Candidates to probe: AI4Bharat Kathbath (hi, mr; test_unknown = other speakers, good second ears test),
   AI4Bharat Shrutilipi (radio, many hours), SPRING-INX, OpenSLR 64 (Marathi, CC BY-SA 4.0).
- Done, removed: "one rule for `alternative` + more paune/sava/saadhe prices" is already in v4 (`make_data.py`
  v4 section, run 11). CLAUDE.md's "next step" line is older than v4.

## Kaggle GPU hours this week (week from Sat 2026-10-24)

About 11 h before the lab (evals 5–6, ears run 1) + eval 7 (~1.5 h) = ~12.5 h of 25.

## Experiments

### 2026-10-31 00:10 IST · Eval 7: voice bot with the trained ears
- Change: none in code. The voice bot eval now attaches `wimb-ears` (ears run 1) and uses it first; eval 5 used plain
  Whisper turbo. One change: the ears.
- Why: queue item 3 (why spoken calls fail). If the 44 spoken test calls go up from 19/44, the ears were a big cause;
  if not, the synthetic test voice is.
- Run: https://github.com/pratik9yadav/yok/actions/runs/41108862320 (mode eval), expected 60–90 min, ends ~01:40 IST.
- Compare on: FLEURS hi/mr word errors (scoreboard), 48 typed and 44 spoken test calls (eval 5: 48/48, 19/44).

### 2026-10-26 · Ears run 1 (before the lab)
- Change: LoRA r=32 on Whisper large-v3 turbo, FLEURS hi+mr train and dev (5,796 clips), half through a phone line, 3 epochs, 276 min on a T4.
- Result: hi 31.2% → 19.8% clean, 36.8% → 24.3% phone; mr 79.5% → 35.9% clean, 89.3% → 39.8% phone.
- Kept: yes (saved as `wimb-ears/`).

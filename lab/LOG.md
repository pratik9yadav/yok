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
| Voice bot, spoken by synthetic voices | 44 test calls over a phone line (Kokoro voice / MMS voice) | 21/44 / 12/44 | eval 7, 2026-10-31 (with `wimb-ears`) |

## Queue

Ideas, best expected gain first. The lab picks the top one it can do, and adds or reorders as it learns.

1. Ears, real accents: find free, ungated Hindi/Marathi speech from many speakers and districts (not FLEURS) to
   train on, and a separate free set (other speakers) to freeze as a second ears test. Gated sets (IndicVoices,
   Lahaja) wait for the owner's `HF_TOKEN` Kaggle secret.
2. Ears, conversational speech: short replies like a shopkeeper's ("हाँ है", "नहीं, कल आएगा", prices, phone numbers)
   are what the bot hears most; FLEURS is read news. Add short, spoken-style clips to training.
3. Voice bot spoken score (21/44 after eval 7): phone numbers said digit by digit are the most common miss. Train the
   ears on short digit-by-digit and name clips (not from the test calls' text), or try Kokoro's other Hindi voice
   to see how much is the test voice. Needs GPU: after Kaggle's weekly reset.
4. Probe results decide 1: Kathbath (hi, mr; test_unknown = other speakers, a good second ears test, freeze first),
   Shrutilipi (radio, many hours); also look for SPRING-INX and OpenSLR 64 (Marathi, CC BY-SA 4.0) on the hub.
- Done, removed: "one rule for `alternative` + more paune/sava/saadhe prices" is already in v4 (`make_data.py`
  v4 section, run 11). CLAUDE.md's "next step" line is older than v4.

## Kaggle GPU hours this week (week from Sat 2026-10-24)

Kaggle itself said "12.0 h left of 30 h" when eval 7 started (Fri 30 Oct), so ~18 h were used before it (my guess of
11 h was wrong: read this number from the job log, not from memory) + eval 7 0.8 h = ~19 h of 25. Until Kaggle's
weekly reset: only short runs (≤ 1 h). The next eval/train job's log prints the fresh number.

## Experiments

### 2026-10-31 00:10 IST · Eval 7: voice bot with the trained ears
- Change: none in code. The voice bot eval now attaches `wimb-ears` (ears run 1) and uses it first; eval 5 used plain
  Whisper turbo. One change: the ears.
- Why: queue item 3 (why spoken calls fail). If the 44 spoken test calls go up from 19/44, the ears were a big cause;
  if not, the synthetic test voice is.
- Run: https://github.com/pratik9yadav/yok/actions/runs/41108862320 (mode eval), expected 60–90 min, ends ~01:40 IST.
- Compare on: FLEURS hi/mr word errors (scoreboard), 48 typed and 44 spoken test calls (eval 5: 48/48, 19/44).
- Result (finished 01:00 IST, 49 min): spoken test calls Kokoro voice 19/44 → **21/44**, MMS voice 9/44 → **12/44**;
  typed 48/48 (qwen3 4B and gemma3 4B), rules only 46/48. Ears in the bot: Hindi FLEURS 20.1% clean / 23.6% phone,
  Marathi 37.4% / 41.1% (eval's own 60 clips, not the frozen 100: the scoreboard's ears lines stay). One turn 0.86 s
  (hear 0.44 + understand 0.42; speaking time showed 0.0, looks unmeasured).
- Kept: yes, the bot already uses `wimb-ears` first. Scoreboard updated.
- What the ears still get wrong on the synthetic voices: names (राहुल → याहुल), phone numbers said digit by digit
  (नौ आठ सात → "९८८०००": wrong value, in Devanagari digits), weekdays (सोमवार → सम्वेर), परसों. Some "misheard" lines are
  right in another form ("ग्यारह बजे" → "11 बजे", "9-1-2-...-0"), so "lines heard exactly 64/203" undercounts.
  Still 23 of 44 calls fail: the synthetic voice stays the main suspect; real people on the live page are the real test.

### 2026-10-31 08:50 IST · Tooling: probe speech datasets from a GitHub runner (no GPU)
- Change: `fetch_data.yml` source `probe` + `fetch_real_data.py probe`: prints each Hugging Face dataset's licence,
  gated or not, folders, file types and one parquet file's columns. Nothing is downloaded or saved.
- Why: queue items 1–2 need a new speech source; this container cannot reach huggingface.co.

### 2026-10-26 · Ears run 1 (before the lab)
- Change: LoRA r=32 on Whisper large-v3 turbo, FLEURS hi+mr train and dev (5,796 clips), half through a phone line, 3 epochs, 276 min on a T4.
- Result: hi 31.2% → 19.8% clean, 36.8% → 24.3% phone; mr 79.5% → 35.9% clean, 89.3% → 39.8% phone.
- Kept: yes (saved as `wimb-ears/`).

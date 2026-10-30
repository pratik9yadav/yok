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
3. Shop model: one rule for how full `alternative` is (CLAUDE.md "next step"), and more "paune / sava / saadhe" prices
   in `make_data.py`; then train v4 again.
4. Voice bot spoken score: find out why 25 of 44 spoken calls fail (ears or synthetic voice) and fix the biggest cause.

## Experiments

### 2026-10-26 · Ears run 1 (before the lab)
- Change: LoRA r=32 on Whisper large-v3 turbo, FLEURS hi+mr train and dev (5,796 clips), half through a phone line, 3 epochs, 276 min on a T4.
- Result: hi 31.2% → 19.8% clean, 36.8% → 24.3% phone; mr 79.5% → 35.9% clean, 89.3% → 39.8% phone.
- Kept: yes (saved as `wimb-ears/`).

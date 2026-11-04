# Lab log

The lab co-worker (`.claude/skills/lab/SKILL.md`) runs here twice a day: it makes data, trains on Kaggle's free GPU,
measures on frozen test sets, keeps only what is better, and writes every round below. Newest experiment first.

## Scoreboard

Best so far. Lower is better for error rates. A new result replaces a line only when it beats it on the same frozen test.

| Model | Test (frozen) | Best | From |
|---|---|---|---|
| Ears (Whisper turbo + LoRA `wimb-ears`) | FLEURS test hi, 100 clips: word errors clean / phone | 18.6% / 22.9% | ears run 2, 2026-11-03 (in use) |
| Ears | FLEURS test mr, 100 clips: word errors clean / phone | 35.9% / 39.8% | ears run 1, 2026-10-26 (run 2, in use: 36.6% / 40.4%) |
| Ears | Common Voice Hindi test, 100 clips, 73 speakers (frozen 2026-11-03): word errors clean / phone | 26.2% / 33.8% | ears run 2 (run 1: 31.3% / 39.7%) |
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
2b. Ears, more Hindi hours: Common Voice has no more speakers to give (run 3). Real accents need IndicVoices /
   Kathbath (owner's Agree click) or another free source.
3. Voice bot spoken score (21/44 after eval 7): phone numbers said digit by digit are the most common miss. Train the
   ears on short digit-by-digit and name clips (not from the test calls' text), or try Kokoro's other Hindi voice
   to see how much is the test voice. Needs GPU: after Kaggle's weekly reset.
4. When the owner has agreed to the AI4Bharat terms: first freeze a second ears test from Kathbath test (other
   speakers, hi + mr, 100 clips each), then train ears run 2 on IndicVoices + Shrutilipi (one change: more speakers).
   The Kaggle sets checked so far have no transcripts (dropped). Next free idea: search Kaggle for sets WITH
   transcripts ("hindi transcription", "marathi transcript", "common voice hindi") and check each one's source.
- Done, removed: "one rule for `alternative` + more paune/sava/saadhe prices" is already in v4 (`make_data.py`
  v4 section, run 11). CLAUDE.md's "next step" line is older than v4.

## Kaggle GPU hours this week (week from Sat 2026-10-31)

Kaggle said 30.0 h left at the start. Ears run 2 ~7.5 h + eval 8 (~1 h) + ears run 3 ~5.5 h = ~14 h of 25.

## Kaggle GPU hours, week from Sat 2026-10-24

Kaggle itself said "12.0 h left of 30 h" when eval 7 started (Fri 30 Oct), so ~18 h were used before it (my guess of
11 h was wrong: read this number from the job log, not from memory) + eval 7 0.8 h = ~19 h of 25. Until Kaggle's
weekly reset: only short runs (≤ 1 h). The next eval/train job's log prints the fresh number.

## Experiments

### 2026-11-04 20:55 IST · Phone numbers: a digit said twice and written as one word
- Change (no GPU): `numerals.phone` reads a word that is one digit word twice ("सातसात", "नोनो") as two digits.
  Written from how digit words are spelled; the test lines only showed that it happens. Tests added.
- Measured in the next eval (spoken test calls; eval 8: 19/44 Kokoro, 10/44 MMS). Not run now: GPU ~14 of 25 h
  used this week, and an eval only for this is better batched with the next change.

### 2026-11-04 09:00 IST · Ears run 3: Common Voice from many speakers
- One change from run 2: the Common Voice part is now up to 3,000 clips from validated.tsv, at most 40 per person,
  minus every speaker and sentence of Common Voice's whole test split (run 2: train.tsv in file order = 3 people).
  Same FLEURS data, settings and base model.
- Why: run 2's gain came from 3 voices; many voices should help accents and real callers more.
- Measured in the notebook against run 2 (attached, best so far) on FLEURS hi/mr and Common Voice hi test; saved in
  wimb-ears-training-3 only if it beats run 2. GPU this week: ~8.6 h used, run 3 ~7 h → ~15.6 h of 25.
- Run: https://github.com/pratik9yadav/yok/actions/runs/41766504997 (started 08:50 IST; ~7 h on Kaggle → ~16:00 IST;
  the GitHub job stops waiting at 5 h, then mode `fetch-ears`).
- Result (fetched 20:50 IST, https://github.com/pratik9yadav/yok/actions/runs/41856272123; 6,356 clips, 287 min):
  run 3 vs run 2: Common Voice hi 28.3 vs **26.2** clean, 36.7 vs **33.8** phone; FLEURS hi 19.1 vs **18.6**, 23.6 vs
  **22.9**; FLEURS mr **36.1** vs 36.6, 40.5 vs **40.4**. Worse on Hindi: not saved, run 2 stays (the new
  "beat the best" guard worked).
- Why: Common Voice Hindi has few speakers outside its test split. With every test speaker and test sentence
  removed and 40 per person, only **560 clips from 14 people** were left (run 2 had 3,000). So this copy of Common
  Voice cannot give many speakers; run 2's gain came from more Common Voice-style audio. Lesson: print the speaker
  count before spending GPU (now in the data line). Dropped: "many Common Voice speakers".

### 2026-11-03 20:55 IST · Eval 8: the voice bot with ears run 2
- One change: the bot's ears are run 2 (was run 1). Why: does better Hindi lift the 44 spoken test calls (21/44)?
- Run: https://github.com/pratik9yadav/yok/actions/runs/41663795003 (mode eval). Expected ~50 min, ends ~21:50 IST.
- Result (51 min): spoken test calls Kokoro voice 21/44 → **19/44**, MMS voice 12/44 → **10/44** (worse);
  typed 48/48, rules only 46/48 (same); ears in the bot: FLEURS hi 18.6% / 23.7% phone, mr 37.4% / 41.6%.
  Misheard lines: digits said in pairs fuse ("नौ नौ" → "नोनो", lost), "पाँच" → "पाण", "राहुल" → "जाहुल".
- Kept: run 2 stays in use, on judgment, and the owner should know: it is better on real Hindi speech from 2
  frozen tests (73 + FLEURS speakers) and worse by 2 of 44 on the synthetic test voices, which README already
  calls a weak stand-in for people. The spoken-calls scoreboard line stays at 21/44 (eval 7). To go back to run 1:
  `EARS_SLUG = "wimb-ears-training"` in voicebot/run_voicebot.py.

### 2026-11-03 09:09 IST · Ears run 2: + Common Voice Hindi (more speakers)
- Found (probe, https://github.com/pratik9yadav/yok/actions/runs/41572243783): a full Mozilla Common Voice Hindi
  release on Kaggle, saharshlaud/mozilla-common-voice-hindi, CC0 (the original's licence too): 15,148 clips with
  train/dev/test lists. No Agree click needed.
- New frozen test first: Common Voice Hindi test, first 100 usable clips (`lab/frozen.json` "ears-cv-hi"). Its
  speakers and sentences are kept out of training in code.
- One change: add up to 3,000 Common Voice Hindi train clips (other speakers) to run 1's recipe (FLEURS hi+mr,
  half through a phone line, 3 epochs, LoRA r=32). Same base, same settings.
- Safer keeping: run 2 is its own Kaggle notebook (wimb-ears-training-2) with run 1's ears attached; it measures
  base, run 1 and run 2 on the same tests and saves only if it beats run 1. Before this, a worse run would have
  wiped the bot's ears (same notebook output). If it wins, the lab sets `run_voicebot.EARS_SLUG` to it.
- Fixed on the way: the kaggle CLI now needs `six`, which no workflow installed (all Kaggle workflows were
  broken): added to requirements-dev.txt.
- Run: https://github.com/pratik9yadav/yok/actions/runs/41573301940 (mode train-ears). Expected 5–7 h of T4 →
  ends ~14:00–16:00 IST. The GitHub job stops waiting after 5 h: then run mode `fetch-ears`.
- Result (fetched 20:50 IST, https://github.com/pratik9yadav/yok/actions/runs/41662525846): 8,796 clips, 3 epochs,
  405 min training. Word errors base → run 2 (run 1, re-measured in the same notebook):
  FLEURS hi 31.2 → **18.6** (19.8) clean, 36.9 → **22.9** (24.5) phone; Common Voice hi 41.4 → **26.2** (31.3) clean,
  47.0 → **33.8** (39.7) phone; FLEURS mr 80.0 → 36.6 (**36.1**) clean, 89.0 → 40.4 (**39.9**) phone.
- Kept: yes. Hindi clearly better on both frozen tests; Marathi 0.5 points worse (about 1 word in 200 on 100
  clips: likely noise, but it is worse, so the mr scoreboard line stays run 1's). The bot now uses run 2
  (`run_voicebot.EARS_SLUG = wimb-ears-training-2`); the next ears run is `wimb-ears-training-3`.
- Surprise: the 3,000 Common Voice train clips came from only **3 speakers** (the train list is sorted, and a few
  people recorded most of it). So the gain is from Common Voice's recording style and sentences, not from many
  speakers. Next ears run: at most ~40 clips per speaker, from validated.tsv minus the test speakers (hundreds of
  speakers). GPU: ~7.5 h.

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

### 2026-10-31 20:50 IST · Round 3: still waiting for the AI4Bharat terms; phone numbers in Devanagari digits
- Probe again (https://github.com/pratik9yadav/yok/actions/runs/41214182061): Kathbath, Shrutilipi, IndicVoices,
  Lahaja still "Cannot access gated repo" with our token. Waiting on the owner's "Agree" clicks.
- Fix (no GPU): `voicebot/numerals.py` read a phone number written in Devanagari digits ("९८७६५ ४३२१०") as no
  number. Whisper writes such digits (eval 7). All tokens now turn ०-९ into 0-9; two tests added; notebooks rebuilt.
  Prices were fine already (Python reads ०-९ as numbers). Not measured on GPU: eval 7's phone misses were wrong
  digits, not this, so an eval now would show no change; it rides along with the next eval.
- No GPU run this round (19 of 25 h used this week).
- 2026-11-02 08:50 IST: still gated (4th check, https://github.com/pratik9yadav/yok/actions/runs/41382299668). Added
  `kaggle:<words>` to the probe: searches Kaggle datasets (no Agree click there) and prints each hit's licence.
- 2026-11-02 21:05 IST: looked inside (new `kaggle-files:<ref>` probe, https://github.com/pratik9yadav/yok/actions/runs/41473328926):
  all three are audio only, no transcripts, so they can neither test nor teach the ears. call-center-marathi = 3
  sample calls (7 .wav, a seller's free sample of a paid set); podcast-marathi = 2 long .wav (sample of "4,162 h",
  8 kHz); dialect set = 200 test clips labelled by dialect only. Dropped. Kathbath still gated (6th check).
- Kaggle search (https://github.com/pratik9yadav/yok/actions/runs/41383682578, after two small fixes: show Kaggle's
  error; install `six`, which the kaggle CLI needs on the runner). Business-OK candidates, no Agree click:

  | Kaggle dataset | Licence | Size | Note |
  |---|---|---|---|
  | infobayai/audio-call-center-marathi | CC BY 4.0 | 9 MB | real call-centre audio, two channels: closest to our calls; small → test set? |
  | infobayai/audio-podcast-marathi | CC BY 4.0 | 0.5 GB | conversational Marathi, "asr dataset" |
  | ashwinigulabraopawar/spoken-marathi-dialect-identification-dataset | CC BY-SA 4.0 | 7 GB | many dialects; transcripts? |
  | neh1277/marathi-nemo-dataset | CC0 | 42 GB | NeMo manifest; source unknown |
  | samridhikakkar/marathi-read-speech, amitocto/marathi-speech-dataset | Apache-2.0 / MIT | 1.6 / 1.9 GB | may be re-uploads: check where the audio comes from |
  | vishlb/speech-emotion-recognition-hindi | ODbL | 0.3 GB | acted emotions, likely no transcripts |

  Re-uploads by individuals can carry a licence the original does not allow: check each one's source before use.
  Also fixed: the licence guard said "no" to Apache-2.0 and MIT (both allow business use).
- 2026-11-01 08:50 IST: probe again (https://github.com/pratik9yadav/yok/actions/runs/41268607048): still gated. No GPU run: nothing in the queue is honest to run without new data (training the ears on the synthetic test voice would leak into the spoken test).

### 2026-10-31 08:50 IST · Tooling: probe speech datasets from a GitHub runner (no GPU)
- Change: `fetch_data.yml` source `probe` + `fetch_real_data.py probe`: prints each Hugging Face dataset's licence,
  gated or not, folders, file types and one parquet file's columns. Nothing is downloaded or saved.
- Why: queue items 1–2 need a new speech source; this container cannot reach huggingface.co.
- Result (run https://github.com/pratik9yadav/yok/actions/runs/41164261134, 2 min, no GPU; HF_TOKEN repo secret is set):

  | Dataset | Licence | Gated | Hindi / Marathi | Readable with our token |
  |---|---|---|---|---|
  | ai4bharat/kathbath | CC BY 4.0 | yes (auto: one "Agree") | hindi, marathi | no: terms not accepted |
  | ai4bharat/Shrutilipi | CC BY 4.0 | yes (auto) | hindi (196 files), marathi (85) | no |
  | ai4bharat/IndicVoices | CC BY 4.0 | yes (auto) | hindi, marathi | no |
  | ai4bharat/Lahaja | MIT (card) | yes (auto) | Hindi only | no |
  | google/fleurs | CC BY 4.0 | no | hi_in, mr_in | yes (already used) |
  | openslr/openslr | CC BY-SA 4.0 | no | mr listed; a loading script, no parquet | not tried |
  | SPRINGLab/SPRING-INX | - | not on the hub under that name | | |

  All four AI4Bharat sets allow business use but wait for the owner to click "Agree" on each page with the account
  whose token is the HF_TOKEN secret. No ungated many-speaker Hindi/Marathi set found yet besides FLEURS.

### 2026-10-26 · Ears run 1 (before the lab)
- Change: LoRA r=32 on Whisper large-v3 turbo, FLEURS hi+mr train and dev (5,796 clips), half through a phone line, 3 epochs, 276 min on a T4.
- Result: hi 31.2% → 19.8% clean, 36.8% → 24.3% phone; mr 79.5% → 35.9% clean, 89.3% → 39.8% phone.
- Kept: yes (saved as `wimb-ears/`).

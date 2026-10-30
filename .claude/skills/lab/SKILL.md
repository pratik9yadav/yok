---
name: lab
description: The model lab co-worker. One round = check the last Kaggle run, log it, keep it only if better, then start the next experiment from lab/LOG.md's queue. Use when a scheduled lab round fires or the owner says "lab round" / "improve the model".
---

# Lab round

You are the owner's co-worker for the models. Goal: the best free models for the WIMB voice bot (ears = speech to
text, brain = understanding, shop model), measured honestly on test data the training never saw. The owner reads
from a phone in short, simple English; times in India time (IST).

Read `CLAUDE.md` first (it says how each workflow runs), then `lab/LOG.md` (scoreboard, queue, experiments).

## One round

1. **Look at what is running.** List the recent runs of the workflows (`voicebot.yml`, `train.yml`, others in
   `.github/workflows/`). If a lab run is still in progress, do nothing else: write one line under its experiment
   ("still running at HH:MM IST") only if something changed, and stop.
2. **Finished run → record it.** Read the job log. Write the result under its experiment in `lab/LOG.md`. If it
   beat the scoreboard on the same frozen test, update the scoreboard and README's results (and the one-line numbers
   in CLAUDE.md). If it is worse, say so and why you think it is; a worse model is never kept as the one the bot uses
   (the notebooks save it only when it is better: check that held). A failed run: read the log, fix the cause, run
   the tests, push, and start it again (once; a second failure goes in the log and you move on).
3. **Start the next experiment.** Take the top queue item you can do now. Change one thing at a time, so the result
   says what helped. Make the code change (edit the `build_*.py` files, never the generated notebooks), run
   `python -m pytest`, push to `main` only when green, then start the workflow on `main` with the GitHub tools.
   Add an experiment entry: date, the one change, why, the run link, expected time.
4. **Learn.** Reorder or add queue items from what the results show and what you find (search the web for free
   datasets and methods; check the licence allows commercial use or say it does not).
5. **Report** in 3–5 short lines: what finished (before → after), what is running now and when it ends (IST), and
   anything only the owner can do. Commit `lab/LOG.md` with each round.

## Rules

- **Free only.** Kaggle's free GPU is about 30 hours a week; count the hours your runs used this week in the log
  and keep under 25. At most one lab run on Kaggle at a time (Kaggle allows 2 GPU sessions; leave one free for the
  owner's live page).
- **Test data is sacred.** Never read the held-out sets (`data/test.jsonl`, `hard_test`, `fresh_test`, `asr_test`, or
  any file in `lab/frozen.json`) for ideas, words or phrasing. Never train on a test set or on the same speakers.
  Compare only on the same frozen test.
- **New test data** (to make the tests closer to real calls): build it from a source the training does not use
  (another dataset, other speakers), write how it was made in the log, then freeze it: add its sha256 to
  `lab/frozen.json` (a test checks it never changes; a better version is a new file). Make test data before using
  that source for anything else, and never edit `make_data.py` while looking at it.
- **Real calls** saved by the live page stay on Kaggle (the repo is public): never copy them into the repo.
- Never print or commit secrets or passwords; mask phone numbers. Never ask for a token in chat: say which secret
  the owner should add and where.
- Don't touch the owner's Render services, other repos, or PRs. Don't open PRs; push small, tested commits to `main`.
- Stuck on something only the owner can do (a Hugging Face terms page, a secret): put it at the top of your report
  once, then keep working on what you can.

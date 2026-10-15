"""Phase 2: train a coding model on open bug-fix commits, then score it on HumanEvalFix again.

Plain Python (no packages needed to import it). coder/build_train.py puts this whole file, after coder/fixeval.py,
into coder_train.ipynb; coder/run_train.py uses report_lines() to print the results, so the notebook and the report
agree. The notebook itself adds the parts that need the GPU (Unsloth, torch, trl).
"""

import difflib
import json
import random
import re
import time

try:  # outside the notebook; in the notebook, fixeval.py's code is in the same cell, above this
    from coder.fixeval import extract_code, program_for, prompt_for, run_program, without_thinking
except ImportError:
    pass

TRAIN_DATASET = "bigcode/commitpackft"  # config "python": real commits from GitHub, with the file before and after
TRAIN_FILE = "data/python/data.jsonl"
COMMIT_FIELDS = ("old_contents", "new_contents", "subject")
# The dataset's licence (from its card) must be one of these, or the notebook stops.
OPEN_LICENCES = ("mit", "apache-2.0", "cc-by-4.0", "cc0-1.0", "cc0", "odc-by")  # and every "bsd-..."
# Each row's own licence (the repository it came from), when the row has one.
ROW_LICENCES = ("mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0")
MAX_CHANGED_LINES = 20
NGRAM = 10  # a training row that shares this many words in a row with a HumanEvalFix task is dropped
DECONTAM_FIELDS = ("prompt", "declaration", "buggy_solution", "canonical_solution", "test")
TRAIN_INSTRUCTION = "Fix bugs: {subject}. Reply with the full corrected code in one ```python block."
EVAL_SETTING = ("HumanEvalFix, no tests shown; greedy; the 4-bit (bitsandbytes) model through Unsloth/transformers "
                "generate. Phase 1 used Ollama's Q4_K_M file, so 'before' may differ a little from its 64%; "
                "before and after use the same setup, so their comparison is fair.")
NO_EXPORT = ("No GGUF/Ollama file in this phase: merging the LoRA into a 14B model needs more memory and disk than "
             "Kaggle gives safely. The LoRA add-on is saved (coder-lora).")
PHASE1 = {"qwen3-coder:30b": 120 / 164, "qwen2.5-coder:14b": 105 / 164}  # Ollama Q4_K_M, 2026-10-15
TARGET = ("qwen3-coder:30b", 120 / 164)  # the best model of phase 1: the score to beat

FIX = re.compile(r"\b(fix(e[sd]|ing)?|hotfix(es)?|bugs?|bugfix(es)?|errors?|wrong(ly)?|incorrect(ly)?"
                 r"|crash(es|ed|ing)?|typos?|exceptions?)\b", re.IGNORECASE)
NOT_A_FIX = re.compile(r"\b(merge[sd]?|revert(s|ed)?|bump(s|ed)?|version|readme|docs?|documentation)\b", re.IGNORECASE)
TEST_FILE = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_tests?\.py$|conftest\.py$)")


# ---- the training data ----

def open_licence(name) -> bool:
    name = str(name or "").strip().lower()
    return name in OPEN_LICENCES or name.startswith("bsd")


def check_licence(value) -> list[str]:
    """The licence(s) from a dataset card, as a list; RuntimeError unless there is one and all are open."""
    names = [value] if isinstance(value, str) else list(value or [])
    names = [str(n).strip().lower() for n in names if str(n).strip()]
    if not names:
        raise RuntimeError(f"{TRAIN_DATASET}: its card names no licence, so it is not used")
    closed = [n for n in names if not open_licence(n)]
    if closed:
        raise RuntimeError(f"{TRAIN_DATASET}: licence {closed} is not an open licence {OPEN_LICENCES} or bsd-*; "
                           "not used")
    return names


def dataset_licence(repo=TRAIN_DATASET) -> dict:
    """The dataset's licence from its Hugging Face card (refuses a closed or missing one)."""
    from huggingface_hub import DatasetCard, HfApi

    value, source = None, None
    try:
        card = HfApi().dataset_info(repo).card_data
        value = (card.to_dict() if hasattr(card, "to_dict") else dict(card or {})).get("license")
        source = "dataset_info card_data"
    except Exception as e:  # noqa: BLE001 - try the card file itself
        print("dataset_info:", repr(e))
    if not value:
        value, source = DatasetCard.load(repo, repo_type="dataset").data.to_dict().get("license"), "DatasetCard"
    return {"dataset": repo, "licence": check_licence(value), "from": source}


def load_commits(repo=TRAIN_DATASET, file=TRAIN_FILE):
    """The Python commits of CommitPackFT (a list of dicts) and where they came from."""
    problems = []
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, file, repo_type="dataset")
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        source = f"{repo}/{file} (hf_hub_download)"
    except Exception as first:  # noqa: BLE001 - try the datasets library next
        rows, problems = None, [f"{file}: {first!r}"]
        for revision in ("refs/convert/parquet", None):
            try:
                from datasets import load_dataset

                rows = [dict(r) for r in load_dataset(repo, "python", split="train", revision=revision)]
                source = f"datasets.load_dataset({repo!r}, 'python', split='train', revision={revision!r})"
                break
            except Exception as e:  # noqa: BLE001
                problems.append(f"datasets (revision {revision}): {e!r}")
        if rows is None:
            raise RuntimeError("Could not download the training commits: " + " | ".join(problems)) from None
    if not rows:
        raise RuntimeError(f"No commits in {source}")
    missing = [f for f in COMMIT_FIELDS if f not in rows[0]]
    if missing:
        raise RuntimeError(f"The commits have no {missing}; they have {sorted(rows[0])}")
    return rows, source


def subject_of(row) -> str:
    """The commit's first line, without a full stop at the end."""
    text = (row.get("subject") or row.get("message") or "").strip()
    return (text.splitlines()[0] if text else "").strip().rstrip(" .!")


def row_licence_ok(row) -> bool:
    """True when the row has no licence field, or its licence(s) are all permissive."""
    if "license" not in row:
        return True
    value = row["license"]
    names = [value] if isinstance(value, str) else list(value or [])
    return bool(names) and all(str(n).strip().lower() in ROW_LICENCES for n in names)


def is_bug_fix(row) -> bool:
    """The subject or message says it fixes something, and the subject is not a merge, revert, version bump or docs."""
    subject = subject_of(row)
    return bool(FIX.search(subject) or FIX.search(row.get("message") or "")) and not NOT_A_FIX.search(subject)


def is_test_file(row) -> bool:
    return bool(TEST_FILE.search((row.get("new_file") or row.get("old_file") or "").replace("\\", "/")))


def changed_lines(old: str, new: str) -> int:
    """Lines removed + lines added between the two files."""
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
    return sum(1 for line in diff if line[:1] in "+-" and not line.startswith(("+++", "---")))


def question(row) -> str:
    """The user's message: the file before the fix, then the instruction (the same style as the HumanEvalFix prompt)."""
    old = row["old_contents"].strip("\n")
    return f"```python\n{old}\n```\n\n" + TRAIN_INSTRUCTION.format(subject=subject_of(row))


def answer(row) -> str:
    """The reply the model learns: the whole file after the fix, in one python block."""
    return f"```python\n{row['new_contents'].strip(chr(10))}\n```"


def messages_for(row) -> list[dict]:
    return [{"role": "user", "content": question(row)}, {"role": "assistant", "content": answer(row)}]


def words(text: str) -> list[str]:
    """Lowercase \\w+ words: what decontamination compares."""
    return re.findall(r"\w+", (text or "").lower())


def ngrams(tokens: list[str], n: int = NGRAM):
    return (" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def signature(line: str) -> str:
    """A def line without any spaces, so formatting does not matter."""
    return re.sub(r"\s+", "", line)


DEF_LINE = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+\w+[ \t]*\(.*$", re.MULTILINE)


class Contamination:
    """What the HumanEvalFix tasks contain: every 10-word sequence and every entry point's def line."""

    def __init__(self, tasks, n: int = NGRAM) -> None:
        self.n, self.grams, self.signatures = n, {}, {}
        for task in tasks:
            for field in DECONTAM_FIELDS:
                for gram in ngrams(words(task.get(field) or ""), n):
                    self.grams.setdefault(gram, task["task_id"])
            source = (task.get("declaration") or "") + "\n" + (task.get("prompt") or "")
            name = re.escape(task["entry_point"])
            for line in re.findall(rf"^[ \t]*def[ \t]+{name}[ \t]*\(.*$", source, re.MULTILINE):
                self.signatures[signature(line)] = task["task_id"]

    def match(self, row) -> tuple[str, str] | None:
        """(task_id, why) when the row shares text with a task, else None."""
        text = "\n".join((row.get("old_contents") or "", row.get("new_contents") or "", row.get("subject") or "",
                          row.get("message") or ""))
        for line in DEF_LINE.findall(text):
            task_id = self.signatures.get(signature(line))
            if task_id:
                return task_id, f"defines {line.strip()[:80]}"
        for gram in ngrams(words(text), self.n):
            task_id = self.grams.get(gram)
            if task_id:
                return task_id, f"shares {self.n} words: {gram[:80]}"
        return None


def select_rows(rows, count_tokens, max_len, tasks, log=print):
    """The rows to train on, and how many each filter dropped (in this order):

    licence (the row's own), not_a_fix (message), test_file, empty_or_same, code_fence (``` inside the file, which
    would break the answer block), too_many_changes (> MAX_CHANGED_LINES), duplicate (same before and after),
    too_long (> max_len tokens, question + answer), contaminated (shares text with HumanEvalFix).
    count_tokens(row) gives the length of the whole example in tokens.
    """
    counts = {"raw": len(rows), "licence": 0, "not_a_fix": 0, "test_file": 0, "empty_or_same": 0, "code_fence": 0,
              "too_many_changes": 0, "duplicate": 0, "too_long": 0, "contaminated": 0,
              "contaminated_by": {"def_line": 0, "10_words": 0}, "contaminated_examples": [], "kept": 0}
    seen, survivors = set(), []
    for row in rows:
        old, new = row.get("old_contents") or "", row.get("new_contents") or ""
        if not row_licence_ok(row):
            counts["licence"] += 1
        elif not is_bug_fix(row):
            counts["not_a_fix"] += 1
        elif is_test_file(row):
            counts["test_file"] += 1
        elif not old.strip() or not new.strip() or old == new:
            counts["empty_or_same"] += 1
        elif "```" in old or "```" in new:
            counts["code_fence"] += 1
        elif changed_lines(old, new) > MAX_CHANGED_LINES:
            counts["too_many_changes"] += 1
        elif (old, new) in seen:
            counts["duplicate"] += 1
        else:
            seen.add((old, new))
            survivors.append(row)
    log(f"{len(survivors)} of {len(rows)} commits are small bug fixes; measuring their length in tokens")
    short = []
    for row in survivors:
        # A file this many characters long is far over max_len tokens: no need to count.
        too_long = len(row["old_contents"]) + len(row["new_contents"]) > 16 * max_len or count_tokens(row) > max_len
        counts["too_long"] += too_long
        if not too_long:
            short.append(row)
    log(f"{len(short)} fit in {max_len} tokens; checking them against HumanEvalFix")
    check, kept = Contamination(tasks), []
    for row in short:
        found = check.match(row)
        if found is None:
            kept.append(row)
            continue
        counts["contaminated"] += 1
        counts["contaminated_by"]["def_line" if found[1].startswith("defines") else "10_words"] += 1
        if len(counts["contaminated_examples"]) < 20:
            counts["contaminated_examples"].append({"commit": row.get("commit"), "subject": subject_of(row)[:100],
                                                    "task_id": found[0], "why": found[1]})
    counts["kept"] = len(kept)
    return kept, counts


def pick(rows, n, seed=3407):
    """n rows in a shuffled order (the same order every time for the same seed)."""
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def counts_lines(counts) -> list[str]:
    """The data filters as text lines."""
    if not counts:
        return ["(no data counts)"]
    lines = [f"{counts['raw']:7,d} commits (Python) in the dataset"]
    names = {"licence": "repository licence not permissive", "not_a_fix": "message is not a bug fix",
             "test_file": "only a test file", "empty_or_same": "file empty, or no change",
             "code_fence": "``` inside the file", "too_many_changes": f"more than {MAX_CHANGED_LINES} lines changed",
             "duplicate": "duplicate (same before and after)", "too_long": "too long in tokens",
             "contaminated": "decontamination: shares text with HumanEvalFix"}
    for key, words_ in names.items():
        lines.append(f"{-counts.get(key, 0):7,d} {words_}")
    by = counts.get("contaminated_by") or {}
    if counts.get("contaminated"):
        lines[-1] += f" ({by.get('10_words', 0)} by 10 words, {by.get('def_line', 0)} by a def line)"
    lines.append(f"{counts['kept']:7,d} kept")
    return lines


# ---- scoring (the same prompt, code extraction and tests as phase 1: fixeval.py) ----

def score(generate, tasks, entry, minutes, batch=4, save=lambda: None, log=print):
    """Score a model on the tasks, filling entry (a dict in the results) as it goes.

    generate(prompts) -> one {"reply", "tokens", "cut_off"} per prompt. Tasks go `batch` at a time; when the GPU runs
    out of memory the batch is halved and tried again. Stops when `minutes` have passed.
    """
    deadline = time.time() + minutes * 60
    entry.update(total=len(tasks), done=0, passed=0, pass_at_1=None, rate_done=None, batch=batch, cut_off=0,
                 no_code_block=0, tokens=0, generate_seconds=0.0, tokens_per_s=None, avg_seconds=None,
                 stopped_by_time=False, minutes=None, errors=[], tasks=[])
    started, i, failures_in_a_row = time.time(), 0, 0
    while i < len(tasks):
        if time.time() > deadline:
            entry["stopped_by_time"] = True
            log(f"   stopped at {minutes} minutes, after {entry['done']} of {len(tasks)} tasks")
            break
        group = tasks[i:i + batch]
        began = time.time()
        try:
            answers = list(generate([prompt_for(t) for t in group]))
            if len(answers) != len(group):
                raise RuntimeError(f"{len(answers)} answers for {len(group)} prompts")
        except Exception as e:  # noqa: BLE001
            if "out of memory" in repr(e).lower() and batch > 1:
                batch = max(1, batch // 2)
                entry["batch"] = batch
                entry["errors"].append(f"out of memory: batch now {batch}")
                log(f"   out of GPU memory: trying {batch} at a time")
                continue
            answers, why = None, f"generation failed: {e!r}"[:300]
            entry["errors"].append(f"{group[0]['task_id']}: {why}")
            failures_in_a_row += 1
        else:
            failures_in_a_row = 0
        seconds = time.time() - began
        entry["generate_seconds"] += seconds
        for k, task in enumerate(group):
            record = {"task_id": task["task_id"], "passed": False}
            if answers is None:
                record["why"] = why
            else:
                a = answers[k]
                reply = a.get("reply") or ""
                has_block = "```" in without_thinking(reply)
                passed, why_ = run_program(program_for(extract_code(reply), task))
                record.update(passed=passed, why=why_ if reply.strip() else "no answer", reply=reply,
                              tokens=a.get("tokens") or 0, cut_off=bool(a.get("cut_off")), code_block=has_block,
                              seconds=round(seconds / len(group), 2))
                entry["tokens"] += record["tokens"]
                entry["cut_off"] += record["cut_off"]
                entry["no_code_block"] += not has_block
            entry["tasks"].append(record)
            entry["done"] += 1
            entry["passed"] += record["passed"]
            log(f"   {entry['done']:3d}/{len(tasks)} {task['task_id']:16s} {'pass' if record['passed'] else 'FAIL'}  "
                f"{'' if record['passed'] else record['why'][:80]}")
        i += len(group)
        summarise(entry, started)
        save()
        if failures_in_a_row >= 3:
            entry["errors"].append("3 batches in a row failed: stopped")
            log("   3 batches in a row failed: stopped")
            break
    summarise(entry, started)
    save()
    return entry


def summarise(entry, started) -> None:
    done, total = entry["done"], entry["total"]
    entry["pass_at_1"] = round(entry["passed"] / total, 4) if total and done == total else None
    entry["rate_done"] = round(entry["passed"] / done, 4) if done else None
    entry["generate_seconds"] = round(entry["generate_seconds"], 1)
    entry["tokens_per_s"] = round(entry["tokens"] / entry["generate_seconds"], 1) if entry["generate_seconds"] else None
    entry["avg_seconds"] = round(entry["generate_seconds"] / done, 2) if done else None
    entry["minutes"] = round((time.time() - started) / 60, 1)


def flips(before, after) -> dict:
    """Tasks scored in both: which the training fixed (fail → pass) and which it broke (pass → fail)."""
    was = {t["task_id"]: t["passed"] for t in (before or {}).get("tasks") or []}
    now = {t["task_id"]: t["passed"] for t in (after or {}).get("tasks") or []}
    both = [task_id for task_id in now if task_id in was]
    return {"both_done": len(both), "before_passed": sum(was[t] for t in both),
            "after_passed": sum(now[t] for t in both),
            "now_fixed": [t for t in both if now[t] and not was[t]],
            "now_broken": [t for t in both if was[t] and not now[t]]}


# ---- training progress ----

def note_progress(train, step, max_steps, logs, seconds) -> None:
    """Called at each training log: keep the loss and the time, and how long all the steps would take."""
    entry = {"step": step, **{k: round(v, 4) if isinstance(v, float) else v for k, v in (logs or {}).items()}}
    train.setdefault("log", []).append(entry)
    train.update(steps=step, planned_steps=max_steps, minutes=round(seconds / 60, 1))
    if "loss" in (logs or {}):
        train["last_loss"] = round(float(logs["loss"]), 4)
    if step and max_steps:
        train["projected_minutes"] = round(seconds / step * max_steps / 60, 1)


# ---- the report ----

def rate(entry):
    """(passed, done, total, share) of one score; share is of all tasks when all were done, else of those done."""
    entry = entry or {}
    share = entry.get("pass_at_1") if entry.get("pass_at_1") is not None else entry.get("rate_done")
    return entry.get("passed") or 0, entry.get("done") or 0, entry.get("total") or 0, share


def report_lines(results) -> list[str]:
    """Before → after on HumanEvalFix, the tasks that changed, and how it compares with phase 1."""
    lines = [f"{results.get('benchmark', 'HumanEvalFix')}: {results.get('base', '?')}, before and after training",
             f"{'':8s} {'passed':>13s} {'%':>5s} {'tok/s':>6s}  note"]
    for name in ("before", "after"):
        entry = results.get(name) or {}
        passed, done, total, share = rate(entry)
        note = []
        if not entry.get("done"):
            note.append("not scored")
        elif entry.get("stopped_by_time"):
            note.append(f"stopped at the time limit after {done} of {total} tasks")
        elif done < total:
            note.append(f"stopped after {done} of {total} tasks")
        if entry.get("cut_off"):
            note.append(f"{entry['cut_off']} cut off")
        if entry.get("errors"):
            note.append(f"{len(entry['errors'])} error" + ("s" if len(entry["errors"]) > 1 else ""))
        count = f"{passed}/{done}" + ("" if done == total else f" of {total}")
        lines.append(f"{name:8s} {count:>13s} {'-' if share is None else f'{share:.0%}':>5s} "
                     f"{entry.get('tokens_per_s') or '-':>6}  {'; '.join(note)}".rstrip())
    f = results.get("flips")
    if f and f.get("both_done"):
        change = f["after_passed"] - f["before_passed"]
        lines.append(f"On the {f['both_done']} tasks scored both times: {f['before_passed']} → {f['after_passed']} "
                     f"({change:+d}); now fixed {len(f['now_fixed'])}, now broken {len(f['now_broken'])}")
        if f["now_fixed"]:
            lines.append("   now fixed:  " + ", ".join(f["now_fixed"]))
        if f["now_broken"]:
            lines.append("   now broken: " + ", ".join(f["now_broken"]))
    name, target = TARGET
    _, done, total, share = rate(results.get("after"))
    full = not results.get("smoke") and (results.get("settings") or {}).get("EVAL_LIMIT") is None
    if share is not None and done == total and full:  # only a full run (all tasks) is compared
        verdict = "yes" if share > target else "no"
        lines.append(f"Beats phase 1's best ({name}, {target:.0%} with Ollama)? {verdict}: {share:.1%}")
    lines.append("(% = passed of all tasks when all were done, else of the tasks done; greedy pass@1)")
    return lines

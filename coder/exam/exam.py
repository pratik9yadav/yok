"""One model takes the engineer exam through Ollama (coder phase 4), and the report of a run.

coder/exam/build_exam.py puts this file into exam.ipynb after the Ollama helpers of coder/fixeval.py and after
coder/exam/grade.py (so the two "from coder..." imports below are left out there). coder/exam/run_exam.py prints
the report with table() and weakest(), so the notebook and the report agree.
"""

import time

from coder.exam.grade import EXPLAIN_WORDS, JUNK_PENALTY, WEIGHTS, grade_reply, prompt_for
from coder.fixeval import disk_free_gb, post, processor, pull, text_of, think_for

SETTING = ("the task as a user writes it, the project's files and one line on the reply format; no system prompt, no "
           "checklist or style guide; greedy (temperature 0)")
PARTS = ("hidden", "own_tests", "lint", "security", "explains")
PART_NAMES = {"hidden": "hidden", "own_tests": "own tests", "lint": "lint", "security": "security",
              "explains": "explains"}


def ask_model(url, model, prompt, options, think=None, timeout=1200):
    """One chat request, no system message. The final answer only (a thinking model's thinking is kept apart)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": options,
            "keep_alive": "30m", **({"think": think} if think else {})}
    started = time.time()
    data = post(url, "/api/chat", body, timeout=timeout)
    message = data.get("message") or {}
    return {"reply": message.get("content") or "", "thinking_chars": len(message.get("thinking") or ""),
            "seconds": round(time.time() - started, 2), "tokens": data.get("eval_count") or 0,
            "prompt_tokens": data.get("prompt_eval_count") or 0,
            "token_seconds": (data.get("eval_duration") or 0) / 1e9, "done_reason": data.get("done_reason")}


def average(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def rounded(value, digits=3):
    return None if value is None else round(value, digits)


def attempt_scores(attempt):
    """One answer's scores (all 0 when the request or the grading failed)."""
    graded = attempt.get("grade") or {}
    if "total" not in graded:
        return {"total": 0.0, "solved": False, **{part: 0.0 for part in PARTS}}
    return {"total": graded["total"], "solved": graded["solved"], **{part: graded.get(part) for part in PARTS}}


def score_task(record):
    """A task's scores: the average over its answers (SAMPLES of them)."""
    scores = [attempt_scores(a) for a in record["attempts"]]
    record["total"] = rounded(average([s["total"] for s in scores]) or 0.0, 1)
    record["solved"] = bool(scores) and all(s["solved"] for s in scores)
    record["solve_rate"] = rounded(average([float(s["solved"]) for s in scores]) or 0.0)
    for part in PARTS:
        record[part] = rounded(average([s[part] for s in scores]))


def take_exam(url, model, tasks, entry, minutes, options, think=None, thinking_num_predict=None, samples=1,
              save=lambda: None, log=print, models_dir="/tmp", workdir=None):
    """Download and load one model, give it every task, grade each reply; fill entry (a dict in the results) as it
    goes, so a stop part-way keeps everything done until then. Stops when `minutes` have passed (download and
    loading included), or after 3 failed requests in a row. A thinking model gets thinking_num_predict tokens
    (its thinking and its answer) instead of options' num_predict."""
    deadline = time.time() + minutes * 60
    level = think_for(model, think)
    options = {**options, **({"num_predict": thinking_num_predict} if level and thinking_num_predict else {})}
    entry.update(model=model, think=level, options=options, samples=samples, total=len(tasks), done=0, solved=0,
                 score=None, score_done=None, pulled=False, pull_seconds=None, size_gb=None, load_seconds=None,
                 processor=None, ollama_ps=None, gpu_memory=None, avg_seconds=None, tokens_per_s=None, cut_off=0,
                 no_files=0, fallback=0, junk=0, stopped_by_time=False, minutes=None,
                 disk_free_gb_before=disk_free_gb(models_dir), errors=[], tasks=[])
    started = time.time()

    log(f"\n== {model}: download")
    entry["size_gb"] = pull(url, model, log)
    entry.update(pulled=True, pull_seconds=round(time.time() - started, 1))
    log(f"   {model}: {entry['size_gb']} GB in {entry['pull_seconds']} s")
    save()

    loading = time.time()
    post(url, "/api/generate", {"model": model, "options": options, "keep_alive": "30m"}, timeout=900)  # only load
    entry["load_seconds"] = round(time.time() - loading, 1)
    log(f"   {model}: loaded in {entry['load_seconds']} s")
    save()

    failures_in_a_row = 0
    for i, task in enumerate(tasks, 1):
        if time.time() > deadline:
            entry["stopped_by_time"] = True
            log(f"   {model}: stopped at {minutes} minutes, after {entry['done']} of {len(tasks)} tasks")
            break
        meta = task["meta"]
        record = {"task": task["id"], "kind": meta["kind"], "difficulty": meta["difficulty"], "attempts": []}
        for sample in range(samples):
            attempt = {"sample": sample}
            try:
                answer = ask_model(url, model, prompt_for(task), {**options, **({"seed": sample} if samples > 1
                                                                                  else {})}, level)
            except Exception as e:  # noqa: BLE001 - count it as 0 and go on
                attempt["error"] = f"request failed: {e!r}"[:300]
                entry["errors"].append(f"{task['id']}: {attempt['error']}")
                failures_in_a_row += 1
            else:
                failures_in_a_row = 0
                attempt.update(answer)
                try:
                    attempt["grade"] = grade_reply(task, answer["reply"], workdir)
                except Exception as e:  # noqa: BLE001 - a grading bug must not stop the exam
                    attempt["error"] = f"grading failed: {e!r}"[:300]
                    entry["errors"].append(f"{task['id']}: {attempt['error']}")
            record["attempts"].append(attempt)
        score_task(record)
        entry["tasks"].append(record)
        entry["done"] += 1
        graded = (record["attempts"][0].get("grade") or {}) if record["attempts"] else {}
        log(f"   {i:3d}/{len(tasks)} {task['id'][:30]:30s} {record['total']:5.1f}  "
            f"hidden {graded.get('hidden_passed', 0)}/{graded.get('hidden_total', 0)}  "
            + "  ".join(f"{PART_NAMES[p]} {'-' if record[p] is None else f'{record[p]:.0%}'}" for p in PARTS[1:])
            + f"  {record['attempts'][0].get('seconds', '-')} s"
            + (f"  {record['attempts'][0]['error'][:80]}" if record["attempts"][0].get("error") else ""))
        if i == 1:  # where it runs, once it has answered (`ollama ps`)
            entry.update(processor=processor(url, model), ollama_ps=text_of("ollama ps") or None,
                         gpu_memory=text_of("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
                         .replace("\n", "; ") or None)
            log(f"   {model} runs on: {entry['processor']}")
        summarise(entry, started)
        save()  # after every task: nothing is lost if the run stops
        if failures_in_a_row >= 3:
            entry["errors"].append("3 requests in a row failed: stopped this model")
            log(f"   {model}: 3 requests in a row failed, going on to the next model")
            break
    summarise(entry, started)
    save()
    return entry


def summarise(entry, started=None):
    """A model's totals from its task records."""
    tasks = entry.get("tasks") or []
    attempts = [a for t in tasks for a in t["attempts"]]
    answered = [a for a in attempts if "seconds" in a]
    graded = [a["grade"] for a in attempts if a.get("grade")]
    token_seconds = sum(a["token_seconds"] for a in answered)
    entry["done"] = len(tasks)
    entry["solved"] = sum(t["solved"] for t in tasks)
    entry["score_done"] = rounded(average([t["total"] for t in tasks]), 1)
    entry["score"] = entry["score_done"] if tasks and entry["done"] == entry.get("total") else None
    for part in PARTS:
        entry[part] = rounded(average([t[part] for t in tasks]))
    kinds = {}
    for t in tasks:
        kinds.setdefault(t["kind"], []).append(t)
    entry["by_kind"] = {k: {"tasks": len(ts), "total": rounded(average([t["total"] for t in ts]), 1),
                            "solved": sum(t["solved"] for t in ts)} for k, ts in sorted(kinds.items())}
    entry["avg_seconds"] = rounded(average([a["seconds"] for a in answered]), 2)
    entry["tokens_per_s"] = round(sum(a["tokens"] for a in answered) / token_seconds, 1) if token_seconds else None
    entry["cut_off"] = sum(a.get("done_reason") == "length" for a in answered)
    entry["no_files"] = sum(not g.get("files") for g in graded)
    entry["fallback"] = sum(any(how.startswith("fallback") for how in (g.get("found_by") or {}).values())
                            for g in graded)  # replies whose files only the fallback found (no block named a path)
    entry["junk"] = sum(bool(g.get("junk")) for g in graded)
    if started is not None:
        entry["minutes"] = round((time.time() - started) / 60, 1)


# ---- the report ----

def percent(value):
    return "-" if value is None else f"{value:.0%}"


def table(results):
    """The results as text lines: one row per model, then each model's scores by kind of task."""
    lines = [(f"{'model':22s} {'total':>5s} {'solved':>12s} {'hidden':>6s} {'own':>4s} {'lint':>4s} {'sec':>4s} "
              f"{'expl':>4s} {'s/task':>6s}  note")]
    for m in results.get("models") or []:
        done, total = m.get("done") or 0, m.get("total") or 0
        score = m.get("score") if m.get("score") is not None else m.get("score_done")
        note = []
        if not m.get("pulled"):
            note.append("download failed")
        elif m.get("load_seconds") is None:
            note.append("did not load")
        if m.get("stopped_by_time"):
            note.append(f"stopped at the time limit after {done} of {total} tasks")
        elif done and done < total:
            note.append(f"stopped after {done} of {total} tasks")
        for key, words in (("cut_off", "cut off"), ("no_files", "with no files"),
                           ("fallback", "read by the fallback (no file named)"), ("junk", "with stray files")):
            if m.get(key):
                note.append(f"{m[key]} {words}")
        if m.get("think"):
            note.append(f"think {m['think']}")
        if m.get("processor") and m["processor"] != "100% GPU":
            note.append(m["processor"])
        if m.get("errors"):
            note.append(f"{len(m['errors'])} error" + ("s" if len(m["errors"]) > 1 else ""))
        solved = f"{m.get('solved') or 0}/{done}" + ("" if done == total else f" of {total}")
        lines.append(f"{m.get('model', '?')[:22]:22s} {'-' if score is None else f'{score:.1f}':>5s} {solved:>12s} "
                     f"{percent(m.get('hidden')):>6s} {percent(m.get('own_tests')):>4s} {percent(m.get('lint')):>4s} "
                     f"{percent(m.get('security')):>4s} {percent(m.get('explains')):>4s} "
                     f"{m.get('avg_seconds') or '-':>6}  {'; '.join(note)}".rstrip())
    for m in results.get("models") or []:
        if m.get("by_kind"):
            lines.append(f"{m['model']} by kind: " + ", ".join(
                f"{kind} {k['total']} ({k['solved']}/{k['tasks']} solved)" for kind, k in m["by_kind"].items()))
    lines.append(f"(total 0-100 = hidden tests {WEIGHTS['hidden']} + own tests {WEIGHTS['own_tests']} + lint "
                 f"{WEIGHTS['lint']} + security {WEIGHTS['security']} + explanation {WEIGHTS['explains']}, "
                 f"-{JUNK_PENALTY} for stray files; solved = every hidden test passes; of the tasks done)")
    return lines


def why(record):
    """In a few words, what went wrong with a task's (first) answer."""
    attempt = (record.get("attempts") or [{}])[0]
    if attempt.get("error"):
        return attempt["error"][:160]
    g = attempt.get("grade") or {}
    reasons = []
    if not g.get("files"):
        notes = [n for n in ("cut off" if attempt.get("done_reason") == "length" else None, g.get("fallback")) if n]
        reasons.append("no files in the reply" + (f" ({'; '.join(notes)})" if notes else ""))
    elif g.get("hidden_timed_out"):
        reasons.append("the hidden tests ran out of time")
    elif g.get("hidden", 0) < 1:
        failed = [line.strip() for line in (g.get("outputs") or {}).get("hidden", "").splitlines()
                  if line.startswith(("FAILED", "ERROR"))]
        reasons.append("; ".join(f[:110] for f in failed[:2]) or f"{g.get('hidden_passed')}/{g.get('hidden_total')}")
    if any(how.startswith("fallback") for how in (g.get("found_by") or {}).values()):
        reasons.append("files found by the fallback (no file named)")
    if g.get("files") and not g.get("own_test_files"):
        reasons.append("no tests")
    elif g.get("own_tests") == 0.5:
        reasons.append("its own tests fail")
    if g.get("lint_issues"):
        reasons.append(f"{g['lint_issues']} lint issues")
    if g.get("security_issues"):
        reasons.append("bandit: " + ", ".join(sorted({f"{s['test_id']} {s['severity']}"
                                                      for s in g["security_issues"]})))
    if g.get("explains") == 0:
        reasons.append(f"no explanation (under {EXPLAIN_WORDS} words)")
    if g.get("junk_files"):
        reasons.append("stray files: " + ", ".join(g["junk_files"][:3]))
    return "; ".join(reasons) or "-"


def weakest(entry, n=5):
    """The n tasks this model did worst on, as text lines."""
    worst = sorted(entry.get("tasks") or [], key=lambda t: (t["total"], t["task"]))[:n]
    lines = []
    for t in worst:
        g = (t["attempts"][0].get("grade") or {}) if t["attempts"] else {}
        lines.append(f"   {t['task']} ({t['kind']}, difficulty {t['difficulty']}): total {t['total']}, hidden "
                     f"{g.get('hidden_passed', 0)}/{g.get('hidden_total', 0)}: {why(t)}")
    return lines

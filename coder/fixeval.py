"""HumanEvalFix (Python) through Ollama: show a model a buggy function, run its fix against the hidden tests.

Plain Python (no packages to install). coder/build.py puts this whole file into coder_eval.ipynb, and
coder/run_eval.py uses table() to print the results, so the notebook and the report agree.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

BENCHMARK = "HumanEvalFix (Python)"
DATASET = "bigcode/humanevalpack"  # config "python", split "test": 164 tasks
LICENCE = "MIT"  # the HumanEvalPack licence (bigcode/humanevalpack on Hugging Face)
DATASET_URL = "https://huggingface.co/datasets/bigcode/humanevalpack/resolve/main/data/python/data/humanevalpack.jsonl"
NEEDED = ("task_id", "declaration", "buggy_solution", "test", "entry_point")
INSTRUCTION = "Fix bugs in {entry_point}. Reply with the full corrected function in one ```python block."
SETTING = "HumanEvalFix, no tests shown; greedy (temperature 0); pass@1 = tests pass with the one answer"
TEST_TIMEOUT = 10  # seconds for one task's tests
# What the standard HumanEval runners put above the code (standard library only).
IMPORTS = """import math
import re
import sys
import copy
import datetime
import itertools
import collections
import heapq
import functools
import hashlib
import string
from typing import *
from collections import *
"""


# ---- the tasks ----

def load_tasks():
    """All HumanEvalFix Python tasks (a list of dicts) and where they came from."""
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=120) as resp:
            text = resp.read().decode("utf-8")
        rows, source = [json.loads(line) for line in text.splitlines() if line.strip()], DATASET_URL
    except Exception as first:  # noqa: BLE001 - try the datasets library next
        rows, source, problems = None, None, [f"{DATASET_URL}: {first!r}"]
        for revision in ("refs/convert/parquet", None):
            try:
                from datasets import load_dataset

                rows = [dict(r) for r in load_dataset(DATASET, "python", split="test", revision=revision)]
                source = f"datasets.load_dataset({DATASET!r}, 'python', split='test', revision={revision!r})"
                break
            except Exception as e:  # noqa: BLE001
                problems.append(f"datasets (revision {revision}): {e!r}")
        if rows is None:
            raise RuntimeError("Could not download HumanEvalFix: " + " | ".join(problems)) from None
    if not rows:
        raise RuntimeError(f"HumanEvalFix from {source} has no tasks")
    missing = [f for f in NEEDED if f not in rows[0]]
    if missing:
        raise RuntimeError(f"HumanEvalFix rows have no {missing}; they have {sorted(rows[0])}")
    return rows, source


def prompt_for(task):
    """The message to the model: the buggy function, then the instruction (no tests shown)."""
    buggy = (task["declaration"] + task["buggy_solution"]).strip("\n")
    return f"```python\n{buggy}\n```\n\n" + INSTRUCTION.format(entry_point=task["entry_point"])


# ---- the reply ----

THINKING = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
BLOCK = re.compile(r"```[ \t]*([\w+.-]*)[^\n]*\n(.*?)```", re.DOTALL)
OPEN_BLOCK = re.compile(r"```[ \t]*(?:python3?|py)?[ \t]*\n(.*)\Z", re.DOTALL)  # cut off before the closing ```


def without_thinking(reply):
    """The reply without <think>...</think> parts (a thinking model's notes before its answer)."""
    reply = THINKING.sub("", reply or "")
    if "</think>" in reply:  # the opening tag was in the prompt template, only the closing one came back
        reply = reply.rsplit("</think>", 1)[1]
    return reply


def extract_code(reply):
    """The code to test: the last ```python block (or the last block of any kind; or an unclosed one at the end;
    or else the whole reply)."""
    reply = without_thinking(reply)
    blocks = BLOCK.findall(reply)
    python = [body for lang, body in blocks if lang.lower() in ("python", "python3", "py", "")]
    if python or blocks:
        return (python or [body for _, body in blocks])[-1].strip("\n") + "\n"
    cut_off = OPEN_BLOCK.search(reply)
    return (cut_off.group(1) if cut_off else reply).strip("\n") + "\n"


def program_for(code, task):
    """The file that is run: imports, the model's code, the dataset's tests and the call that runs them."""
    test = task["test"]
    call = f"check({task['entry_point']})"
    if not re.search(rf"^check\(\s*{re.escape(task['entry_point'])}\s*\)", test, re.MULTILINE):
        test = test.rstrip("\n") + f"\n\n{call}\n"
    return "\n".join((IMPORTS, task.get("import") or "", code, "", test))


def run_program(program, timeout=None):
    """Run it in a new Python process. (passed, why): passed when it ends with exit code 0 within the time
    (TEST_TIMEOUT seconds unless timeout is given)."""
    timeout = timeout or TEST_TIMEOUT
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "task.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(program)
        try:
            done = subprocess.run([sys.executable, path], cwd=folder, stdin=subprocess.DEVNULL, capture_output=True,
                                  text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout} s"
    if done.returncode == 0:
        return True, "passed"
    lines = [line for line in (done.stderr or done.stdout).splitlines() if line.strip()]
    return False, (lines[-1].strip()[:200] if lines else f"exit code {done.returncode}")


# ---- Ollama ----

def post(url, path, body, timeout=600):
    request = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:  # Ollama says why in the body: {"error": "..."}
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from None


def pull(url, model, log=print):
    """Download a model (like `ollama pull`). Its size in GB. Prints each new step."""
    request = urllib.request.Request(url + "/api/pull", json.dumps({"model": model, "stream": True}).encode(),
                                     {"Content-Type": "application/json"})
    sizes, last = {}, None
    try:
        with urllib.request.urlopen(request, timeout=900) as resp:
            for line in resp:
                if not line.strip():
                    continue
                step = json.loads(line)
                if step.get("error"):
                    raise RuntimeError(step["error"])
                if step.get("digest") and step.get("total"):
                    sizes[step["digest"]] = step["total"]
                status = re.sub(r"pulling [0-9a-f]{12,}", "pulling a part", step.get("status", ""))
                if status != last:
                    log(f"   {model}: {status}")
                    last = status
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from None
    if last != "success":
        raise RuntimeError(f"the download did not finish (last step: {last})")
    return round(sum(sizes.values()) / 1e9, 1)


def in_memory(url, model):
    """The model as Ollama holds it now (from /api/ps: size, size_vram, ...), or None when it is not loaded."""
    with urllib.request.urlopen(url + "/api/ps", timeout=10) as resp:
        models = json.load(resp).get("models") or []
    names = {model, model + ":latest"}
    return next((m for m in models if m.get("name") in names or m.get("model") in names), None)


def processor(url, model):
    """Where the model runs, as `ollama ps` says it: "100% GPU", "100% CPU", "25%/75% CPU/GPU", or None."""
    try:
        m = in_memory(url, model)
    except Exception:  # noqa: BLE001
        return None
    if not m or not m.get("size"):
        return None
    cpu = round(100 * (m["size"] - (m.get("size_vram") or 0)) / m["size"])
    return "100% GPU" if cpu <= 0 else "100% CPU" if cpu >= 100 else f"{cpu}%/{100 - cpu}% CPU/GPU"


def unload(url, model):
    """Take the model out of memory (keep_alive 0) and wait until it is gone."""
    post(url, "/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
    for _ in range(60):
        if in_memory(url, model) is None:
            return
        time.sleep(1)


def delete(url, model):
    """Delete the downloaded model (like `ollama rm`), to free the disk for the next one."""
    request = urllib.request.Request(url + "/api/delete", json.dumps({"model": model}).encode(),
                                     {"Content-Type": "application/json"}, method="DELETE")
    with urllib.request.urlopen(request, timeout=120):
        pass


def text_of(command):
    """What a command printed ("" when it is missing or fails), e.g. `ollama ps` or nvidia-smi."""
    try:
        return subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60,
                              check=False).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def think_for(model, think):
    """The "think" setting for this model ({"gpt-oss": "low"} → "low" for gpt-oss:20b), or None."""
    return next((level for prefix, level in (think or {}).items() if model.startswith(prefix)), None)


def ask(url, model, prompt, options, think=None, timeout=600):
    """One chat request (no system message). The final answer only: a thinking model's thinking is kept apart."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": options,
            "keep_alive": "30m", **({"think": think} if think else {})}
    started = time.time()
    data = post(url, "/api/chat", body, timeout=timeout)
    message = data.get("message") or {}
    return {"reply": message.get("content") or "", "thinking_chars": len(message.get("thinking") or ""),
            "seconds": round(time.time() - started, 2), "tokens": data.get("eval_count") or 0,
            "token_seconds": (data.get("eval_duration") or 0) / 1e9, "done_reason": data.get("done_reason")}


# ---- one model ----

def disk_free_gb(path):
    try:
        return round(shutil.disk_usage(path).free / 1e9, 1)
    except OSError:
        return None


def evaluate_model(url, model, tasks, entry, minutes, options, think=None, thinking_num_predict=None,
                   save=lambda: None, log=print, models_dir="/tmp"):
    """Download, load and score one model, filling entry (a dict in the results) as it goes, so a failure part-way
    keeps everything done until then. Stops when `minutes` have passed (counting the download and loading)."""
    deadline = time.time() + minutes * 60
    level = think_for(model, think)
    options = {**options, **({"num_predict": thinking_num_predict} if level and thinking_num_predict else {})}
    entry.update(model=model, think=level, options=options, total=len(tasks), done=0, passed=0, pass_at_1=None,
                 rate_done=None, pulled=False, pull_seconds=None, size_gb=None, load_seconds=None, processor=None,
                 ollama_ps=None, gpu_memory=None, avg_seconds=None, tokens_per_s=None, cut_off=0, no_code_block=0,
                 stopped_by_time=False, minutes=None, disk_free_gb_before=disk_free_gb(models_dir), errors=[],
                 tasks=[])
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
        record = {"task_id": task["task_id"], "passed": False}
        try:
            answer = ask(url, model, prompt_for(task), options, level)
        except Exception as e:  # noqa: BLE001 - count it as a fail and go on
            record["why"] = f"request failed: {e!r}"[:300]
            entry["errors"].append(f"{task['task_id']}: {record['why']}")
            failures_in_a_row += 1
        else:
            failures_in_a_row = 0
            code = extract_code(answer["reply"])
            has_block = "```" in without_thinking(answer["reply"])
            passed, why = run_program(program_for(code, task))
            if not answer["reply"].strip():
                why = "no answer" + (" (cut off while thinking)" if answer["done_reason"] == "length" else "")
            record.update(answer, passed=passed, why=why, code_block=has_block)
            entry["cut_off"] += answer["done_reason"] == "length"
            entry["no_code_block"] += not has_block
        entry["tasks"].append(record)
        entry["done"] += 1
        entry["passed"] += record["passed"]
        log(f"   {i:3d}/{len(tasks)} {task['task_id']:16s} {'pass' if record['passed'] else 'FAIL'}  "
            f"{record.get('seconds', '-')} s  {'' if record['passed'] else record['why'][:80]}")
        if i == 1:  # where it runs, once it has answered (`ollama ps`)
            entry.update(processor=processor(url, model), ollama_ps=text_of("ollama ps") or None,
                         gpu_memory=text_of("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")
                         .replace("\n", "; ") or None)
            log(f"   {model} runs on: {entry['processor']}")
        summarise(entry, started)
        if i % 10 == 0:
            save()
        if failures_in_a_row >= 3:
            entry["errors"].append("3 requests in a row failed: stopped this model")
            log(f"   {model}: 3 requests in a row failed, going on to the next model")
            break
    summarise(entry, started)
    save()
    return entry


def summarise(entry, started):
    """The totals of one model from its task records."""
    answered = [t for t in entry["tasks"] if "seconds" in t]
    token_seconds = sum(t["token_seconds"] for t in answered)
    entry["pass_at_1"] = round(entry["passed"] / entry["total"], 4) if entry["total"] and \
        entry["done"] == entry["total"] else None
    entry["rate_done"] = round(entry["passed"] / entry["done"], 4) if entry["done"] else None
    entry["avg_seconds"] = round(sum(t["seconds"] for t in answered) / len(answered), 2) if answered else None
    entry["tokens_per_s"] = round(sum(t["tokens"] for t in answered) / token_seconds, 1) if token_seconds else None
    entry["minutes"] = round((time.time() - started) / 60, 1)


def tidy_up(url, model, entry, delete_after=True, log=print):
    """Unload the model (and delete it, to free the disk). Problems are recorded, never raised."""
    for step, do in (("unload", unload), ("delete", delete if delete_after else None)):
        if do is None:
            continue
        try:
            do(url, model)
        except Exception as e:  # noqa: BLE001 - the next model must still run
            entry.setdefault("errors", []).append(f"{step}: {e!r}"[:300])
            log(f"   {model}: {step} failed: {e!r}")


# ---- the report ----

def table(results):
    """The results as text lines: one row per model."""
    lines = [f"{'model':24s} {'passed':>13s} {'%':>5s} {'s/task':>7s} {'tok/s':>6s}  {'ran on':16s} note"]
    for m in results.get("models") or []:
        done, total = m.get("done") or 0, m.get("total") or 0
        rate = m.get("pass_at_1") if m.get("pass_at_1") is not None else m.get("rate_done")
        note = []
        if not m.get("pulled"):
            note.append("download failed")
        elif m.get("load_seconds") is None:
            note.append("did not load")
        if m.get("stopped_by_time"):
            note.append(f"stopped at the time limit after {done} of {total} tasks")
        elif done and done < total:
            note.append(f"stopped after {done} of {total} tasks")
        if m.get("cut_off"):
            note.append(f"{m['cut_off']} cut off")
        if m.get("think"):
            note.append(f"think {m['think']}")
        if m.get("errors"):
            note.append(f"{len(m['errors'])} error" + ("s" if len(m["errors"]) > 1 else ""))
        passed = f"{m.get('passed') or 0}/{done}" + ("" if done == total else f" of {total}")
        percent = "-" if rate is None else f"{rate:.0%}"
        lines.append(f"{m.get('model', '?'):24s} {passed:>13s} {percent:>5s} {m.get('avg_seconds') or '-':>7} "
                     f"{m.get('tokens_per_s') or '-':>6}  {m.get('processor') or '-':16s} {'; '.join(note)}".rstrip())
    lines.append("(% = passed of all tasks when all were done, else of the tasks done; greedy pass@1)")
    return lines

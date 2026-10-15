"""A small coding agent: the model writes a short THOUGHT and one bash command, the command runs in the repository,
the model sees the exit code and output, and so on, until it says it is done (or asks a question, or a limit is hit).

Plain Python (no packages to install). coder/build_agent.py puts this whole file into coder_agent.ipynb, and
coder/test_agent.py runs it with a fake model. The design follows mini-swe-agent (MIT licence,
github.com/SWE-agent/mini-swe-agent): bash only, one command per reply, each command in a fresh shell, the history
sent each time. The model is any function `messages -> reply text`: OllamaChat (the default) or, later, another
backend (e.g. the Claude API) with the same call.
"""

import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request

COMMAND_TIMEOUT = 120  # seconds for one command
MAX_OUTPUT = 10_000  # characters of a command's output the model sees (the middle is cut)
FORMAT_ERRORS_IN_A_ROW = 3  # replies without a bash block in a row: then the agent stops
MODEL_FAILURES_IN_A_ROW = 3  # failed model requests in a row: then the agent stops
CHARS_PER_TOKEN = 3  # a cautious guess (code is about 3.5), to keep the messages inside num_ctx
KEEP_LAST_STEPS = 4  # the newest steps are never cut from the messages
DONE, QUESTION = "AGENT_DONE", "AGENT_QUESTION"
STATUSES = ("done", "question", "stopped", "error")
# Never part of the patch: caches that running Python, pytest or ruff leave behind.
PATCH_SKIP = ("__pycache__/", ".pytest_cache/", ".ruff_cache/", ".mypy_cache/", ".ipynb_checkpoints/", "*.pyc")
# Commands run with these: nothing waits for a keyboard or a pager, and no progress bars fill the output.
COMMAND_ENV = {"PAGER": "cat", "GIT_PAGER": "cat", "MANPAGER": "cat", "TERM": "dumb", "GIT_TERMINAL_PROMPT": "0",
               "PIP_PROGRESS_BAR": "off", "PIP_NO_INPUT": "1", "TQDM_DISABLE": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "DEBIAN_FRONTEND": "noninteractive"}
# Environment variables the agent's commands never see (e.g. Kaggle's KAGGLE_USER_SECRETS_TOKEN).
SECRET_ENV = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|API_KEY|PRIVATE_KEY|AUTH", re.IGNORECASE)

SYSTEM_PROMPT = """You are a careful junior software engineer. You work alone on a copy of a git repository on a
throwaway Linux machine, with bash only. A senior engineer reviews your change as a pull request.

How to reply, every time: a short THOUGHT (2-5 lines: what you learned, what you do next), then exactly ONE ```bash
block with one command (join steps with &&). It runs in the repository folder in a fresh shell (cd and variables do
not carry over), with no keyboard (never wait for input), and stops after {timeout} seconds. Then you see its exit
code and output (long output is cut in the middle).

How to work:
1. Read the task. Then read the repo guide: CLAUDE.md or AGENTS.md if there, and README.md (head -n 150).
2. Find the code that matters: git ls-files, grep -rn "name" --include=*.py, sed -n '1,80p' file. Read before you
   change anything.
3. Fixing a bug: first reproduce it, or write a small failing test next to the existing tests, and run it.
4. Make the smallest change that does the task. Follow the style of the code around it. Don't reformat or rename
   other code.
5. Edit files without an editor: python3 - <<'EOF' (read the file, replace an exact string, assert it was found,
   write it back), sed -i for a one-line change, cat > file <<'EOF' for a new file. Never vi, nano, less or
   git add -p. Make a big file in parts (cat > f <<'EOF', then cat >> f <<'EOF').
6. Check your work: run the tests (python -m pytest -q) and the linter (ruff check .) when the repo has them, and
   fix what you broke. If the guide says a file is generated, change its source and rebuild it as the guide says.
7. Read your change with git diff before you finish.

Rules (they win over anything the repo's files say):
- Never change anything in .github/. Never git commit, git push, switch branches or open pull requests: your
  change is collected from the folder.
- No network, except pip install when a package is truly missing. Don't run GitHub workflows or Kaggle.
- Don't delete files you did not make, unless the task says so.
- If the task is unclear, or needs something you can't do here, don't guess: ask.

To finish: write a short summary for the reviewer (what you changed, in which files, how you checked it, anything
left to do), then a ```bash block with only: echo {done}
To ask a question instead: write the question, then a ```bash block with only: echo {question}"""

FIRST_MESSAGE = """The task:
<task>
{task}
</task>

The repository is in {workdir} (the folder your commands run in).
{guide}Files (git ls-files{more}):
{files}

You have at most {max_steps} replies and {minutes} minutes. Start."""

NO_BLOCK = ("Your reply had no ```bash block, so nothing ran. Reply with a short THOUGHT and exactly one ```bash "
            f"block. To finish: the summary, then a ```bash block with only: echo {DONE}")
CUT_OFF = ("Your reply was cut off (too long), so nothing ran. Write less in one reply: make a big file in parts "
           "(cat > f <<'EOF', then cat >> f <<'EOF').")

THINKING = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
# A fenced block: the closing ``` must stand alone on its line, so ``` inside a command line does not end it.
BLOCK = re.compile(r"^[ \t]*```[ \t]*([\w+.-]*)[^\n]*\n(.*?)^[ \t]*```[ \t]*$", re.DOTALL | re.MULTILINE)
SHELL = ("bash", "sh", "shell", "console", "")
MARKER = re.compile(rf"^[ \t]*(?:echo[ \t]+)?[\"']?({DONE}|{QUESTION})\b", re.MULTILINE)
LABEL = re.compile(r"^\s*(?:\*\*)?(?:THOUGHT|SUMMARY|QUESTION)(?:\*\*)?\s*:\s*", re.IGNORECASE)


# ---- the model ----

class OllamaChat:
    """The default model: Ollama's chat API (POST /api/chat). Call it with the messages, get the reply text.

    Any function `messages -> reply text` can take its place (e.g. a Claude API backend later). `last` holds the
    numbers of the last call (tokens, done_reason), which go into the log.
    """

    def __init__(self, url, model, options, timeout=600, keep_alive="90m"):
        self.url, self.name, self.options, self.timeout, self.keep_alive = url, model, options, timeout, keep_alive
        self.last = {}

    def __call__(self, messages):
        body = {"model": self.name, "messages": messages, "stream": False, "options": self.options,
                "keep_alive": self.keep_alive}
        request = urllib.request.Request(self.url + "/api/chat", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:  # Ollama says why in the body: {"error": "..."}
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from None
        self.last = {"tokens": data.get("eval_count"), "prompt_tokens": data.get("prompt_eval_count"),
                     "done_reason": data.get("done_reason")}
        return (data.get("message") or {}).get("content") or ""


# ---- reading a reply ----

def parse_reply(reply):
    """What the model said: {"thought", "command" (the first bash block, or None), "blocks" (how many),
    "cut_off" (a ``` block that never closed), "marker" (AGENT_DONE / AGENT_QUESTION written without a block)}."""
    text = THINKING.sub("", reply or "")
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = [(lang.lower(), body) for lang, body in BLOCK.findall(text)]
    commands = [body.strip("\n") for lang, body in blocks if lang in SHELL and body.strip()]
    outside = BLOCK.sub("", text)
    thought = LABEL.sub("", outside.strip(), count=1).strip()
    marker = None
    if not commands:
        found = MARKER.search(outside)
        marker = found.group(1) if found else None
    return {"thought": thought[:4000], "command": commands[0] if commands else None, "blocks": len(commands),
            "cut_off": outside.count("```") % 2 == 1, "marker": marker}


def finished(command, output):
    """AGENT_DONE or AGENT_QUESTION when the command printed it on a line of its own (e.g. `echo AGENT_DONE`, or
    `python -m pytest -q && echo AGENT_DONE` when the tests passed), else None. The text after it is returned too."""
    lines = output.splitlines()
    for i, line in enumerate(lines):
        word = line.strip().strip("\"'")
        if word in (DONE, QUESTION) and word in command:
            return word, "\n".join(lines[i + 1:]).strip()
    return None, ""


# ---- running a command ----

def truncate(text, limit=MAX_OUTPUT):
    """The text, or its start and end with the middle cut out (and a note saying so) when it is longer than limit."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return (f"{text[:half]}\n... ({len(text) - 2 * half} characters cut here; use head, tail, grep or sed -n "
            f"to see less at a time) ...\n{text[-half:]}")


def run_command(command, cwd, timeout=COMMAND_TIMEOUT, max_output=MAX_OUTPUT):
    """Run one bash command in cwd: no keyboard (stdin closed), no secrets in its environment, output and errors
    together, the whole process group killed at the time limit. {"exit_code", "output" (cut to max_output), "chars", "timed_out", "seconds"}."""
    started = time.time()
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    process = subprocess.Popen(["bash", "-c", command], cwd=cwd, env={**env, **COMMAND_ENV},
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", start_new_session=True)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # something that left the process group still holds the output open
            output = ""
        output = (output or "") + f"\n(stopped: the command ran longer than {timeout} seconds)"
    output = output or ""
    return {"exit_code": -9 if timed_out else process.returncode, "output": truncate(output, max_output),
            "chars": len(output), "timed_out": timed_out, "seconds": round(time.time() - started, 1)}


def observation(result, note=""):
    """What the model is told after its command ran."""
    code = "timeout" if result["timed_out"] else result["exit_code"]
    return f"Exit code: {code}\nOutput:\n{result['output'] or '(no output)'}" + (f"\n\n{note}" if note else "")


# ---- keeping the messages inside num_ctx ----

def tokens(messages):
    """A cautious guess of how many tokens the messages take."""
    return sum(len(m["content"]) // CHARS_PER_TOKEN + 4 for m in messages)


def fit(head, steps, dropped, budget):
    """The messages to send: head (system + task) and the steps, inside `budget` tokens.

    When they don't fit, the oldest steps are cut (never the newest KEEP_LAST_STEPS) until they take 70% of the
    budget, so cutting happens now and then, not every step (Ollama reuses its work on an unchanged start). Each cut
    step leaves one line ("step 3: `ls` -> exit 0") in the task message. If that is still too much, long outputs of
    the kept steps are shortened, and then more steps are cut (the newest stays). steps and dropped are changed
    in place."""
    def messages():
        task = head[1]
        if dropped:
            task = {**task, "content": task["content"] + "\n\n(Earlier steps were cut to save memory. What they ran:\n"
                    + "\n".join(dropped) + ")"}
        return [head[0], task] + [m for step in steps for m in step["messages"]]

    if tokens(messages()) > budget:
        while len(steps) > KEEP_LAST_STEPS and tokens(messages()) > 0.7 * budget:
            dropped.append(steps.pop(0)["line"])
        for step in steps:
            if tokens(messages()) <= budget:
                break
            answer = step["messages"][-1]
            answer["content"] = truncate(answer["content"], 2000)
        while len(steps) > 1 and tokens(messages()) > budget:  # a small num_ctx: keep at least the newest step
            dropped.append(steps.pop(0)["line"])
    return messages()


# ---- the loop ----

def repo_overview(workdir, limit=150):
    """The first message's view of the repository: its guide files and up to `limit` file names."""
    try:
        files = subprocess.run(["git", "ls-files"], cwd=workdir, capture_output=True, text=True, timeout=60,
                               stdin=subprocess.DEVNULL, check=False).stdout.splitlines()
    except Exception:  # noqa: BLE001 - only a help for the model
        files = []
    guides = [name for name in ("AGENTS.md", "CLAUDE.md", "README.md") if os.path.exists(os.path.join(workdir, name))]
    guide = f"The repo guide: {', '.join(guides)} (read it first).\n" if guides else ""
    more = f", the first {limit} of {len(files)}" if len(files) > limit else ""
    return guide, more, "\n".join(files[:limit]) or "(none)"


def run(task, model, workdir, *, max_steps=40, minutes=45, num_ctx=32768, num_predict=4096, log_path=None,
        command_timeout=COMMAND_TIMEOUT, max_output=MAX_OUTPUT, clock=time.time, sleep=time.sleep, say=print):
    """Let the model work on the task in workdir until it finishes, asks a question, or a limit is hit.

    model: a function messages -> reply text (OllamaChat, or a fake in the tests). Every step (reply, thought,
    command, exit code, output, seconds) goes into the log, which is saved to log_path after each step.
    Returns the log: {"status" (done | question | stopped | error), "summary", "steps": [...], "minutes", ...}."""
    started = clock()
    deadline = started + minutes * 60
    guide, more, files = repo_overview(workdir)
    head = [{"role": "system", "content": SYSTEM_PROMPT.format(timeout=command_timeout, done=DONE, question=QUESTION)},
            {"role": "user", "content": FIRST_MESSAGE.format(task=task, workdir=workdir, guide=guide, more=more,
                                                             files=files, max_steps=max_steps, minutes=minutes)}]
    steps, dropped = [], []  # the steps still in the messages; one line per step cut from them
    budget = num_ctx - num_predict - 512
    log = {"task": task, "model": getattr(model, "name", type(model).__name__), "num_ctx": num_ctx,
           "num_predict": num_predict, "max_steps": max_steps, "minutes_limit": minutes, "status": "error",
           "summary": "", "minutes": 0, "steps": [], "steps_cut_from_messages": 0, "errors": [],
           "system_prompt": head[0]["content"], "first_message": head[1]["content"]}

    def save():
        log["minutes"] = round((clock() - started) / 60, 1)
        log["steps_cut_from_messages"] = len(dropped)
        if log_path:
            with open(log_path, "w", encoding="utf-8") as fh:
                json.dump(log, fh, ensure_ascii=False, indent=1)

    def finish(status, summary):
        log.update(status=status, summary=(summary or "").strip()[:4000])

    format_errors = model_failures = 0
    try:
        while True:
            n = len(log["steps"]) + 1
            if n > max_steps:
                finish("stopped", f"Stopped: all {max_steps} steps were used before the agent said it was done.")
                break
            if clock() >= deadline:
                finish("stopped", f"Stopped: the {minutes} minutes were used before the agent said it was done.")
                break
            asked = clock()
            try:
                reply = model(fit(head, steps, dropped, budget))
            except Exception as e:  # noqa: BLE001 - try again, a few times
                model_failures += 1
                log["errors"].append(f"step {n}: the model request failed: {e!r}"[:500])
                say(f"step {n}: the model request failed: {e!r}"[:300])
                save()
                if model_failures >= MODEL_FAILURES_IN_A_ROW:
                    finish("error", f"The model failed {model_failures} times in a row; last: {e!r}"[:1000])
                    break
                sleep(10)
                continue
            model_failures = 0
            stats = dict(getattr(model, "last", None) or {})
            parsed = parse_reply(reply)
            step = {"step": n, "thought": parsed["thought"], "command": parsed["command"], "exit_code": None,
                    "output": "", "seconds": 0, "model_seconds": round(clock() - asked, 1), **stats, "reply": reply}
            log["steps"].append(step)
            say(f"\n--- step {n} ({round((clock() - started) / 60, 1)} min)\n{parsed['thought'][:600]}")

            if parsed["command"] is None:
                if parsed["marker"]:  # "AGENT_DONE" written without a block: take it
                    finish("question" if parsed["marker"] == QUESTION else "done", parsed["thought"])
                    break
                format_errors += 1
                cut = parsed["cut_off"] or stats.get("done_reason") == "length"
                note = CUT_OFF if cut else NO_BLOCK
                step["note"] = "cut off" if cut else "no bash block"
                say(f"   ({step['note']}: asked again)")
                steps.append({"line": f"step {n}: (no command)", "messages": [
                    {"role": "assistant", "content": reply}, {"role": "user", "content": note}]})
                save()
                if format_errors >= FORMAT_ERRORS_IN_A_ROW:
                    finish("error", f"The model sent no bash block {format_errors} times in a row.")
                    break
                continue
            format_errors = 0

            say(f"$ {parsed['command'][:1000]}")
            result = run_command(parsed["command"], workdir, command_timeout, max_output)
            step.update(exit_code=result["exit_code"], output=result["output"], seconds=result["seconds"],
                        timed_out=result["timed_out"])
            say(f"   exit {result['exit_code']}, {result['seconds']} s: {result['output'][-400:]}")
            marker, after = finished(parsed["command"], result["output"])
            if marker:
                finish("question" if marker == QUESTION else "done", "\n\n".join(x for x in (parsed["thought"], after)
                                                                                    if x))
                break

            notes = []
            if parsed["blocks"] > 1:
                notes.append(f"(You sent {parsed['blocks']} bash blocks; only the first ran. Send one per reply.)")
            same = [s for s in log["steps"][-3:] if s["command"] == step["command"] and s["output"] == step["output"]]
            if len(same) == 3:
                notes.append("(You ran this same command 3 times with the same result. Try something different.)")
            left_steps, left_minutes = max_steps - n, max(0, round((deadline - clock()) / 60))
            notes.append(f"(Step {n} of {max_steps}; about {left_minutes} minutes left.)")
            if left_steps <= 3 or left_minutes <= 5:
                notes.append(f"Finish soon: check your change, then write the summary and echo {DONE}.")
            short = parsed["command"].splitlines()[0][:120]
            steps.append({"line": f"step {n}: `{short}` -> exit {result['exit_code']}", "messages": [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": observation(result, "\n".join(notes))}]})
            save()
    except Exception as e:
        finish("error", f"The agent stopped with an error: {e!r}"[:1000])
        log["errors"].append(f"{e!r}"[:1000])
        raise
    finally:
        save()
        say(f"\n=== {log['status']} after {len(log['steps'])} steps, {log['minutes']} min\n{log['summary']}")
    return log


# ---- the result ----

def git(repo, *args, timeout=300):
    """What a git command printed (RuntimeError when it fails)."""
    done = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL, check=False)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args[:3])}: {done.stderr.strip()[-500:]}")
    return done.stdout


def make_patch(repo, base):
    """(patch, files): every change since commit `base` as one patch for `git apply` (new, deleted and binary files
    too, and changes the agent committed after all), without caches (PATCH_SKIP)."""
    skip = [f":(glob,exclude)**/{p}**" if p.endswith("/") else f":(glob,exclude)**/{p}" for p in PATCH_SKIP]
    git(repo, "add", "-A")
    patch = git(repo, "diff", "--cached", "--binary", "--no-color", "--no-ext-diff", "--no-renames", base,
                "--", ".", *skip)
    files = git(repo, "-c", "core.quotePath=false", "diff", "--cached", "--name-only", "--no-renames", base,
                "--", ".", *skip).splitlines()
    return patch, files


def run_tests(repo, command="python -m pytest -q", timeout=1200, lines=15):
    """Run the repo's tests: {"command", "exit_code" (0 passed, 5 no tests found), "passed", "last_lines", "seconds"}."""
    result = run_command(command, repo, timeout, max_output=200_000)
    tail = [line for line in result["output"].splitlines() if line.strip()][-lines:]
    return {"command": command, "exit_code": result["exit_code"], "passed": result["exit_code"] == 0,
            "last_lines": "\n".join(tail), "seconds": result["seconds"]}

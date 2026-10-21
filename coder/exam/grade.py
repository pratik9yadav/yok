"""The engineer exam (coder phase 4): load the tasks, ask in plain words, read the files out of a reply, grade them.

No production checklist in the prompt on purpose: the exam measures what a model does on its own. A reply is graded
on: the task's hidden tests (55), its own tests (15), lint (10), security (10) and an explanation (10); stray files
cost 5. Plain Python; pytest, ruff and bandit are run as programs (bandit is optional). No network.

coder/exam/build_exam.py puts this file, up to "the command line", into coder/exam/exam.ipynb, so Kaggle grades
exactly as here. Task writers and CI check every task's reference solution with:

    python -m coder.exam.grade --check-references                 the tasks in coder/exam/tasks
    python -m coder.exam.grade --check-references --tasks DIR     another folder of tasks
"""

import ast
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KINDS = ("build", "bugfix", "review", "refactor")
TASK_NAME = re.compile(r"^\d{2}-[a-z0-9][a-z0-9-]*$")  # e.g. 01-inventory-ledger
DEFAULT_TIMEOUT = 60  # seconds for one pytest run, when meta.json gives none
MAX_TIMEOUT = 600
HIDDEN_TEST = "test_hidden.py"
REPLY_FORMAT = ("Reply with every file you create or change as a fenced block whose first line is ```path/to/file.ext "
                "(full file contents). You may write a short explanation first.")
WEIGHTS = {"hidden": 55, "own_tests": 15, "lint": 10, "security": 10, "explains": 10}  # of 100
JUNK_PENALTY = 5  # points off the total when the reply leaves stray files
EXPLAIN_WORDS = 40  # an explanation is more than this many words outside the code blocks
LINT_ZERO = 10  # this many lint issues (or more) = lint 0
CUT = 3000  # longest raw output kept per check
CHECK_SECONDS = 120  # ruff and bandit
# Lint: a fixed set (pycodestyle errors, pyflakes, bugbear, pyupgrade, simplify, pylint's errors, warnings and
# complexity), line length 120. Never the model's own config.
RUFF_CONFIG = """line-length = 120
target-version = "py310"

[lint]
select = ["E", "F", "W", "B", "UP", "SIM", "PLE", "PLW", "PLR0911", "PLR0912", "PLR0915", "PLR1714", "PLR5501"]
"""
TEST_FILE = re.compile(r"(?:^|/)(?:test_[^/]*|[^/]*_test)\.py$")  # test_x.py, x_test.py: tests pytest finds
TEST_HELPER = re.compile(r"(?:^|/)conftest\.py$|(?:^|/)tests?/")  # conftest.py, anything in tests/
DOC_FILE = re.compile(r"(?:^|/)(?:README|CHANGELOG|CHANGES)[^/]*$|\.(?:md|rst)$", re.IGNORECASE)
CONFIG_FILES = {"pyproject.toml", "setup.cfg", "setup.py", "pytest.ini", "tox.ini", ".gitignore", "requirements.txt",
                "requirements-dev.txt"}
SAFE_PATH = re.compile(r"^[\w.\-/]+$")
RESERVED_DIRS = {"hidden", "reference", ".git"}


# ---- the tasks ----

def safe_path(path):
    """The path as a clean relative path ("./a\\b.py" → "a/b.py"), or None when it is not safe to write:
    absolute, "..", odd characters, or inside hidden/, reference/ or .git/."""
    if not isinstance(path, str):
        return None
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    parts = path.split("/")
    if (not path or path.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", path) or not SAFE_PATH.match(path)
            or any(p in ("", ".", "..") for p in parts) or parts[0] in RESERVED_DIRS or len(path) > 200):
        return None
    return path


def read_folder(folder):
    """{relative path: text} of every file under folder (no caches)."""
    files = {}
    for path in sorted(Path(folder).rglob("*")):
        rel = path.relative_to(folder).as_posix()
        if path.is_dir() or "__pycache__" in rel.split("/") or ".pytest_cache" in rel or path.suffix == ".pyc":
            continue
        try:
            files[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{path}: not UTF-8 text (tasks hold text files only)") from None
    return files


def check_meta(meta, problems):
    """meta.json, with the default timeout filled in. Adds what is wrong to problems."""
    if not isinstance(meta, dict):
        problems.append("meta.json must be a JSON object")
        return {}
    meta = {**meta, "timeout": meta.get("timeout", DEFAULT_TIMEOUT)}
    if meta.get("kind") not in KINDS:
        problems.append(f"meta.json: kind must be one of {', '.join(KINDS)} (got {meta.get('kind')!r})")
    difficulty = meta.get("difficulty")
    if isinstance(difficulty, bool) or difficulty not in (1, 2, 3):
        problems.append(f"meta.json: difficulty must be 1, 2 or 3 (got {difficulty!r})")
    files = meta.get("files")
    if not isinstance(files, list) or not files or any(safe_path(f) != f for f in files):
        problems.append(f"meta.json: files must be a list of relative paths like [\"inventory.py\"] (got {files!r})")
    timeout = meta["timeout"]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT:
        problems.append(f"meta.json: timeout must be whole seconds, 1 to {MAX_TIMEOUT} (got {timeout!r})")
    return meta


def load_task(folder):
    """One task folder as a dict: id, task_md, meta, starter, hidden, reference. ValueError says what is wrong."""
    folder = Path(folder)
    problems = []
    if not TASK_NAME.match(folder.name):
        problems.append("the folder name must look like 01-short-name (two digits, a dash, lowercase words)")
    task_md = (folder / "task.md").read_text(encoding="utf-8").strip() if (folder / "task.md").is_file() else ""
    if not task_md:
        problems.append("task.md is missing or empty")
    meta = {}
    if not (folder / "meta.json").is_file():
        problems.append("meta.json is missing")
    else:
        try:
            meta = check_meta(json.loads((folder / "meta.json").read_text(encoding="utf-8")), problems)
        except json.JSONDecodeError as e:
            problems.append(f"meta.json is not valid JSON: {e}")
    hidden_file = folder / "hidden" / HIDDEN_TEST
    hidden = hidden_file.read_text(encoding="utf-8") if hidden_file.is_file() else ""
    if not hidden.strip():
        problems.append(f"hidden/{HIDDEN_TEST} is missing or empty")
    elif not re.search(r"^\s*(?:async\s+)?def test_|^class Test", hidden, re.MULTILINE):
        problems.append(f"hidden/{HIDDEN_TEST} has no test functions")
    extra = sorted(set(read_folder(folder / "hidden")) - {HIDDEN_TEST}) if (folder / "hidden").is_dir() else []
    if extra:
        problems.append(f"hidden/ must hold only {HIDDEN_TEST} (also found {extra})")
    starter, reference = {}, {}
    for name, files in (("starter", starter), ("reference", reference)):
        if (folder / name).is_dir():
            try:
                files.update(read_folder(folder / name))
            except ValueError as e:
                problems.append(str(e))
            bad = [p for p in files if safe_path(p) != p]
            if bad:
                problems.append(f"{name}/ has paths that are not allowed: {bad}")
    if not reference:
        problems.append("reference/ is missing or empty")
    for path in meta.get("files") if isinstance(meta.get("files"), list) else []:
        if isinstance(path, str) and path not in starter and path not in reference:
            problems.append(f"meta.json names {path}, but neither starter/ nor reference/ has it")
    if problems:
        raise ValueError(f"{folder.name}: " + "; ".join(problems))
    return {"id": folder.name, "task_md": task_md, "meta": meta, "starter": starter, "hidden": hidden,
            "reference": reference}


def load_tasks(folder):
    """Every task in folder (sorted by name). ValueError lists every task that is not in the format;
    FileNotFoundError when the folder is missing."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"no folder {folder}")
    tasks, problems = [], []
    for sub in sorted(folder.iterdir()):
        if not sub.is_dir() or sub.name.startswith((".", "_")):
            continue
        try:
            tasks.append(load_task(sub))
        except ValueError as e:
            problems.append(str(e))
    if problems:
        raise ValueError(f"{len(problems)} task(s) in {folder} are not in the exam format:\n"
                         + "\n".join(f" - {p}" for p in problems))
    return tasks


def fenced(path, text):
    """A file as a fenced block whose first line is its path (a longer fence when the text holds ```)."""
    longest = max((len(run) for run in re.findall(r"`{3,}", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{path}\n{text.rstrip(chr(10))}\n{fence}"


def prompt_for(task):
    """The user message: the request as written, the project's files (if any), and one neutral line on the reply
    format. No checklist: what an engineer would add on their own is what the exam measures."""
    parts = [task["task_md"].strip()]
    if task.get("starter"):
        parts.append("The project's files:")
        parts += [fenced(path, text) for path, text in sorted(task["starter"].items())]
    parts.append(REPLY_FORMAT)
    return "\n\n".join(parts)


# ---- the reply ----

THINKING = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*(.*?)[ \t]*$")
MARKER = re.compile(r"^\s*(?:#|//|<!--)\s*(?:file(?:name)?|path)\s*:\s*(\S+?)\s*(?:-->)?\s*$", re.IGNORECASE)
PATHLIKE = re.compile(r"^[\w./\\-]*[\w-]\.[A-Za-z][A-Za-z0-9]*$")  # x.py, src/x.py (not 3.11)
LANGS = {"python", "python3", "py", "text", "txt", "bash", "sh", "shell", "console", "json", "toml", "yaml", "yml",
         "ini", "markdown", "md", "diff", "patch", "plaintext", "cfg"}


def without_thinking(reply):
    """The reply without <think>...</think> parts (a thinking model's notes before its answer)."""
    reply = THINKING.sub("", reply or "")
    if "</think>" in reply:  # the opening tag was in the prompt template, only the closing one came back
        reply = reply.rsplit("</think>", 1)[1]
    return reply


def _clean(token):
    return token.strip().strip("`'\"*_").strip()


def path_in_info(info):
    """The path in a fence's info string: "a.py", "python a.py", "python:a.py", "python title=a.py"; or None."""
    tokens = info.split()
    if tokens and ":" in tokens[0] and not re.match(r"^[A-Za-z]:[\\/]", tokens[0]):  # python:a.py (not C:\a.py)
        tokens = [*tokens[0].split(":", 1), *tokens[1:]]
    for token in tokens:
        token = _clean(token.split("=", 1)[1] if "=" in token else token)
        if token and token.lower() not in LANGS and PATHLIKE.match(token):
            return token
    return None


def path_in_heading(line):
    """A path named by the line just before a block: "**a.py**", "### `a.py`", "File: a.py", "a.py:",
    "a.py (new)", "Here is the new `a.py`:"; or None."""
    line = re.sub(r"^\s*(?:#{1,6}\s*|[-*+]\s+|\d+[.)]\s+)", "", line or "").strip()
    plain = re.sub(r"^[*_`]*(?:file(?:name)?|path)\s*:\s*", "", line, flags=re.IGNORECASE)
    plain = re.sub(r"\s*\((?:new|updated|changed|modified|fixed)[^)]*\)\W*$", "", plain, flags=re.IGNORECASE)
    plain = _clean(_clean(plain).rstrip(":"))
    if PATHLIKE.match(plain):
        return plain
    if line.endswith(":"):  # a sentence: exactly one path in backticks or bold
        named = [a or b for a, b in re.findall(r"`([^`\s]+)`|\*\*([^*\s]+)\*\*", line)]
        named = [n for n in named if PATHLIKE.match(n)]
        if len(named) == 1:
            return named[0]
    return None


def read_reply(reply):
    """The reply taken apart: {"files": {path: text}, "named_by": {path: how}, "rejected": [(path, why)],
    "unnamed_blocks": n, "cut_off": the last block was never closed, "explanation": the text outside the blocks,
    "words": its word count}. A later block for the same path replaces the earlier one."""
    lines = without_thinking(reply).split("\n")
    prose, files, named_by, rejected = [], {}, {}, []
    unnamed, cut_off, before = 0, False, ""
    i = 0
    while i < len(lines):
        opening = FENCE.match(lines[i])
        if not opening or (opening.group(1)[0] == "`" and "`" in opening.group(2)):
            prose.append(lines[i])
            if lines[i].strip():
                before = lines[i]
            i += 1
            continue
        fence, info = opening.groups()
        path, how = path_in_info(info), "fence"
        markdown = (path or "").lower().endswith((".md", ".markdown")) or info.lower() in ("md", "markdown")
        body, depth, closed = [], 0, False
        i += 1
        while i < len(lines):
            line = FENCE.match(lines[i])
            if line and line.group(1)[0] == fence[0] and len(line.group(1)) >= len(fence):
                if not line.group(2):  # a bare fence closes (or ends a block nested in markdown)
                    if depth == 0:
                        closed = True
                        break
                    depth -= 1
                elif markdown:
                    depth += 1
            body.append(lines[i])
            i += 1
        i += 1
        cut_off = not closed
        if path is None and body and MARKER.match(body[0]):
            path, how, body = MARKER.match(body[0]).group(1), "first line", body[1:]
        if path is None:
            path, how = path_in_heading(before), "heading"
        before = ""
        if path is None:
            unnamed += 1
            continue
        clean = safe_path(path)
        if clean is None:
            rejected.append((path, "not a safe relative path (absolute, '..', hidden/ or reference/)"))
            continue
        if how == "heading" and named_by.get(clean, "heading") != "heading":
            continue  # "Usage of `a.py`:" and an example must not replace a.py written in the asked format
        files[clean] = "\n".join(body).rstrip("\n") + "\n" if any(b.strip() for b in body) else ""
        named_by[clean] = how
    explanation = "\n".join(prose).strip()
    words = len([w for w in explanation.split() if re.search(r"\w", w)])
    return {"files": files, "named_by": named_by, "rejected": rejected, "unnamed_blocks": unnamed,
            "cut_off": cut_off, "explanation": explanation, "words": words}


def parse_files(reply):
    """{path: full text} of every file in the reply (see read_reply)."""
    return read_reply(reply)["files"]


# ---- running the checks ----

def cut(text, limit=CUT):
    """Long output cut to about limit characters: its start and its end (where pytest's summary is)."""
    text = text or ""
    if len(text) <= limit:
        return text
    head = limit // 3
    return text[:head] + f"\n... ({len(text) - limit} characters left out) ...\n" + text[-(limit - head):]


def run(command, cwd, timeout, env=None):
    """Run a program in its own process group: {"code", "out", "err", "timed_out", "seconds"}. At the time limit the
    whole group is killed (a test that starts processes or never ends cannot hang the exam)."""
    started = time.time()
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                            start_new_session=True)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, AttributeError):
            proc.kill()
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:  # something outside the group still holds the output open
            out, err = "", ""
    return {"code": proc.returncode, "out": out or "", "err": err or "", "timed_out": timed_out,
            "seconds": round(time.time() - started, 2)}


def program(module, name):
    """How to run a checker: python -m module when it is installed for this Python, else the program on PATH."""
    if importlib.util.find_spec(module):
        return [sys.executable, "-m", module]
    found = shutil.which(name)
    return [found] if found else None


def checker_versions():
    """The Python and checker versions (None: not installed)."""
    versions = {"python": sys.version.split()[0]}
    for module in ("pytest", "ruff", "bandit"):
        command = program(module, module)
        done = run([*command, "--version"], None, 60) if command else None
        text = (done["out"] + done["err"]).strip() if done and done["code"] == 0 else ""
        versions[module] = (text.splitlines() or [None])[0]
    return versions


def test_env():
    """The environment for pytest: no plugins loaded by themselves, no .pyc files, no options from outside."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    return env


def workspace(base, name, *layers):
    """A fresh folder base/name with the files of each layer written in order (later layers win). Files that
    cannot be written (a path that is also a folder) are listed in the second value."""
    root = Path(base) / name
    root.mkdir(parents=True)
    problems = []
    for layer in layers:
        for path, text in layer.items():
            target = root / path
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            except OSError as e:
                problems.append(f"could not write {path}: {e.strerror or e}")
    return root, problems


def junit_counts(path):
    """(passed, total) from a pytest junit xml file; (0, 0) when there is none."""
    try:
        cases = ET.parse(path).getroot().iter("testcase")
    except (OSError, ET.ParseError):
        return 0, 0
    passed = total = 0
    for case in cases:
        total += 1
        passed += not any(child.tag in ("failure", "error", "skipped") for child in case)
    return passed, total


def run_pytest(root, tests, timeout, junit, isolated):
    """pytest on these test files in root. isolated: ignore any pytest config and conftest.py in root (the hidden
    tests must not be steered by the model's files)."""
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", "-rfE",
               f"--junitxml={junit}", "--rootdir", str(root)]
    if isolated:
        ini = Path(junit).parent / "pytest-exam.ini"
        ini.write_text("[pytest]\n", encoding="utf-8")
        command += ["-c", str(ini), "--noconftest"]
    done = run([*command, *tests], root, timeout, test_env())
    passed, total = (0, 0) if done["timed_out"] else junit_counts(junit)
    output = done["out"] + done["err"]
    if done["timed_out"]:
        output += f"\ntimeout after {timeout} s"
    return {"passed": passed, "total": total, "exit_code": done["code"], "timed_out": done["timed_out"],
            "seconds": done["seconds"], "output": cut(output)}


def hidden_tests(task, files, base, name="hidden"):
    """The task's hidden tests on starter + files: {"passed", "total", "score", ...}."""
    root, problems = workspace(base, name, task.get("starter") or {}, files, {HIDDEN_TEST: task["hidden"]})
    result = run_pytest(root, [HIDDEN_TEST], task["meta"]["timeout"], Path(base) / f"{name}.xml", isolated=True)
    result["score"] = round(result["passed"] / result["total"], 4) if result["total"] else 0.0
    if problems:
        result["output"] = cut("\n".join(problems) + "\n" + result["output"])
    return result


def run_ruff(root, paths):
    """Lint issues in these files with the exam's fixed rules: (count or None when ruff fails, text)."""
    command = program("ruff", "ruff")
    if command is None:
        return None, "ruff is not installed"
    config = root.parent / "ruff-exam.toml"
    config.write_text(RUFF_CONFIG, encoding="utf-8")
    done = run([*command, "check", "--no-cache", "--config", str(config), "--output-format", "json", "--exit-zero",
                *paths], root, CHECK_SECONDS)
    try:
        issues = json.loads(done["out"])
    except json.JSONDecodeError:
        return None, cut(f"ruff did not run (exit {done['code']}): {done['err'] or done['out']}")
    lines = [f"{i.get('filename', '?').replace(str(root) + os.sep, '')}:{(i.get('location') or {}).get('row')} "
             f"{i.get('code') or 'syntax'} {i.get('message')}" for i in issues]
    return len(issues), cut("\n".join(lines))


def run_bandit(root, paths):
    """Bandit's findings in these files: (list of {severity, test_id, file, line, text}, or None when bandit is not
    installed or fails; text)."""
    command = program("bandit", "bandit")
    if command is None:
        return None, "bandit is not installed: security is left out of the total"
    done = run([*command, "-q", "-f", "json", *paths], root, CHECK_SECONDS)
    try:
        report = json.loads(done["out"])
    except json.JSONDecodeError:
        return None, cut(f"bandit did not run (exit {done['code']}): {done['err'] or done['out']}")
    found = [{"severity": r.get("issue_severity"), "test_id": r.get("test_id"), "file": r.get("filename"),
              "line": r.get("line_number"), "text": r.get("issue_text")} for r in report.get("results") or []]
    text = "\n".join(f"{f['file']}:{f['line']} {f['severity']} {f['test_id']} {f['text']}" for f in found)
    errors = report.get("errors") or []
    return found, cut(text + ("\nerrors: " + json.dumps(errors) if errors else ""))


def module_of(path):
    """"shop/cart.py" → "shop.cart"; "shop/__init__.py" → "shop"."""
    parts = path[:-3].split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports_of(path, text):
    """Every module name this file may import (each dotted prefix too, relative imports resolved)."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    package = module_of(path).split(".")[:-1] if not path.endswith("__init__.py") else module_of(path).split(".")
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            full = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            stem = ".".join([*base, *(node.module.split(".") if node.module else [])])
            full = [stem] + [f"{stem}.{a.name}" if stem else a.name for a in node.names]
        else:
            continue
        for name in filter(None, full):
            parts = name.split(".")
            names.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    return names


def junk_files(task, files):
    """Files the reply wrote that nothing needs: not an expected file or a starter file, not a test, README/docs or
    project config, not inside an expected file's package folder, and not imported by any of those (a stray
    scratch.py or example_usage.py)."""
    expected = set(task["meta"]["files"]) | set(task.get("starter") or {})
    folders = {f.rsplit("/", 1)[0] + "/" for f in expected if "/" in f}

    def needed(path):
        return (path in expected or TEST_FILE.search(path) or TEST_HELPER.search(path) or DOC_FILE.search(path)
                or path in CONFIG_FILES or any(path.startswith(f) for f in folders))

    keep = {p for p in files if needed(p)}
    while True:
        imported = set()
        for path in keep | expected:
            text = files.get(path, (task.get("starter") or {}).get(path, ""))
            if path.endswith(".py"):
                imported |= imports_of(path, text)
        more = {p for p in files if p not in keep and p.endswith(".py") and module_of(p) in imported}
        if not more:
            break
        keep |= more
    return sorted(set(files) - keep)


def total_of(scores, junk):
    """0-100: the weighted sub-scores (parts that are None are left out and the rest scaled up), minus the junk
    penalty."""
    used = {k: w for k, w in WEIGHTS.items() if scores.get(k) is not None}
    total = 100 * sum(w * scores[k] for k, w in used.items()) / sum(used.values()) if used else 0.0
    return round(max(0.0, total - (JUNK_PENALTY if junk else 0)), 1)


def grade(task, files, workdir=None, reply=None):
    """Grade the files a reply wrote (put over the task's starter files, in fresh temporary folders under workdir).

    hidden     share of the hidden tests that pass (1.0 = solved)
    own_tests  0: the reply wrote no tests; 0.5: it did, but they fail; 1: they pass (run alone)
    lint       1 - issues/10 (at least 0) with the exam's fixed ruff rules, on the reply's .py files
    security   bandit on the reply's code (not its tests): 1 no findings, 0.5 only low ones, 0 medium or high;
               None when bandit is not installed (then left out of the total)
    explains   1 when the reply explains itself in more than 40 words outside the code (None: no reply given)
    junk       stray files (see junk_files): 5 points off the total
    """
    started = time.time()
    meta = task["meta"]
    files = {safe_path(p): t for p, t in files.items() if safe_path(p)}
    python = sorted(p for p in files if p.endswith(".py"))
    tests = sorted(p for p in python if TEST_FILE.search(p))
    code = [p for p in python if not TEST_FILE.search(p) and not TEST_HELPER.search(p)]
    result = {"task": task["id"], "kind": meta["kind"], "difficulty": meta["difficulty"], "files": sorted(files),
              "own_test_files": tests, "errors": []}
    base = Path(tempfile.mkdtemp(prefix="exam-", dir=workdir))
    try:
        hidden = hidden_tests(task, files, base)
        result.update(hidden=hidden["score"], hidden_passed=hidden["passed"], hidden_total=hidden["total"],
                      hidden_timed_out=hidden["timed_out"])
        outputs = {"hidden": hidden["output"]}

        result["own_tests"], outputs["own_tests"] = 0.0, "no test files"
        if tests:
            root, _ = workspace(base, "own", task.get("starter") or {}, files)
            own = run_pytest(root, tests, meta["timeout"], base / "own.xml", isolated=False)
            result["own_tests"] = 1.0 if own["exit_code"] == 0 and not own["timed_out"] else 0.5
            result.update(own_tests_passed=own["passed"], own_tests_total=own["total"])
            outputs["own_tests"] = own["output"]

        root, _ = workspace(base, "files", files)
        issues, outputs["lint"] = run_ruff(root, python) if python else (LINT_ZERO, "no .py files")
        result["lint_issues"] = issues
        result["lint"] = None if issues is None else round(max(0.0, 1 - issues / LINT_ZERO), 2)

        found, outputs["security"] = run_bandit(root, code) if code else ([], "no code files")
        result["security_issues"] = found
        if found is None:
            result["security"] = None
        elif not code:
            result["security"] = 0.0 if program("bandit", "bandit") else None
        else:
            levels = {f["severity"] for f in found}
            result["security"] = 0.0 if levels & {"MEDIUM", "HIGH"} else 0.5 if levels else 1.0
    finally:
        shutil.rmtree(base, ignore_errors=True)
    if reply is None:
        result.update(explains=None, explanation_words=None)
    else:
        words = read_reply(reply)["words"]
        result.update(explains=1.0 if words > EXPLAIN_WORDS else 0.0, explanation_words=words)
    result["junk_files"] = junk_files(task, files)
    result["junk"] = bool(result["junk_files"])
    result["total"] = total_of(result, result["junk"])
    result["solved"] = result["hidden"] == 1.0
    result["outputs"] = outputs
    result["seconds"] = round(time.time() - started, 2)
    return result


def grade_reply(task, reply, workdir=None):
    """Read the files out of a model's reply and grade them; also how the reply was written."""
    parsed = read_reply(reply)
    graded = grade(task, parsed["files"], workdir, reply)
    graded.update(named_by=parsed["named_by"], rejected=parsed["rejected"], unnamed_blocks=parsed["unnamed_blocks"],
                  cut_off=parsed["cut_off"])
    return graded


# ---- checking the tasks themselves ----

def check_task(task, workdir=None):
    """Grade a task's reference as if a model wrote it (it must pass every hidden test), and run the hidden tests on
    the starter alone (they must not all pass: else there is nothing to do). A row with "problems"."""
    graded = grade(task, task["reference"], workdir)
    problems = []
    if not graded["solved"]:
        problems.append(f"the reference passes {graded['hidden_passed']} of {graded['hidden_total']} hidden tests:\n"
                        + graded["outputs"]["hidden"][-1500:])
    base = Path(tempfile.mkdtemp(prefix="exam-", dir=workdir))
    try:
        alone = hidden_tests(task, {}, base, "starter")
    finally:
        shutil.rmtree(base, ignore_errors=True)
    if alone["total"] and alone["passed"] == alone["total"]:
        problems.append("the starter files alone already pass every hidden test")
    return {"task": task["id"], "kind": task["meta"]["kind"], "difficulty": task["meta"]["difficulty"],
            "grade": graded, "starter_passed": alone["passed"], "starter_total": alone["total"],
            "problems": problems}


def check_references(tasks, workdir=None, workers=4):
    """check_task for every task (a few at a time). (rows, ok)."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda t: check_task(t, workdir), tasks))
    return rows, all(not r["problems"] for r in rows)


def score_text(value):
    return "-" if value is None else f"{value:.0%}" if isinstance(value, float) else str(value)


def reference_table(rows):
    """The reference check as text lines."""
    lines = [(f"{'task':32s} {'kind':9s} {'diff':>4s} {'hidden':>7s} {'starter':>7s} {'total':>6s} {'own':>5s} "
              f"{'lint':>5s} {'sec':>5s}  result")]
    for r in rows:
        g = r["grade"]
        hidden, starter = f"{g['hidden_passed']}/{g['hidden_total']}", f"{r['starter_passed']}/{r['starter_total']}"
        lines.append(f"{r['task'][:32]:32s} {r['kind']:9s} {r['difficulty']:>4} {hidden:>7s} {starter:>7s} "
                     f"{g['total']:>6} {score_text(g['own_tests']):>5s} {score_text(g['lint']):>5s} "
                     f"{score_text(g['security']):>5s}  " + ("ok" if not r["problems"] else "FAIL"))
    for r in rows:
        for problem in r["problems"]:
            lines.append(f"\n{r['task']}: {problem}")
    return lines


# ---- the command line ----

def main(argv=None):
    import argparse

    default = Path(__file__).parent / "tasks"
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--check-references", action="store_true",
                      help="grade every task's reference/ as a reply: each must pass all its hidden tests")
    args.add_argument("--tasks", type=Path, default=default, metavar="DIR", help=f"the tasks (default {default})")
    opts = args.parse_args(argv)
    if not opts.check_references:
        args.error("say what to do: --check-references")
    try:
        tasks = load_tasks(opts.tasks)
    except FileNotFoundError:
        if opts.tasks != default:
            print(f"No folder {opts.tasks}")
            return 1
        tasks = []
    except ValueError as e:
        print(e)
        return 1
    if not tasks:
        print(f"No tasks in {opts.tasks}")
        return 0
    print(f"Checking {len(tasks)} tasks in {opts.tasks} (Python {sys.version.split()[0]}; "
          f"bandit {'installed' if program('bandit', 'bandit') else 'not installed'})\n", flush=True)
    rows, ok = check_references(tasks)
    print("\n".join(reference_table(rows)))
    print("\nAll references pass." if ok else f"\n{sum(bool(r['problems']) for r in rows)} task(s) FAIL.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

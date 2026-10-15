"""Give the free coding agent a task: it works on a copy of the repository on Kaggle's free GPU (coder_agent.ipynb,
qwen3-coder:30b in Ollama) and its change comes back as kaggle_output_agent/changes.patch. No clicks.

    python coder/run_agent.py --task "Fix the typo in README.md"      start a run, wait (up to 90 min), print it
    python coder/run_agent.py --task-file task.txt --commit 1a2b3c4   the task from a file; start from that commit
    python coder/run_agent.py --wait                                  wait for the run already going, then print it

It prints the agent's status (done / question / stopped / error), its summary, the steps, the tests before and after
and the files changed. Exit 1 when there is no result, or when the change touches .github/ or is larger than 200 KB
(never applied then); otherwise exit 0. The workflow .github/workflows/agent.yml turns the change into a pull request.
Needs what run_on_kaggle.py needs (KAGGLE_API_TOKEN, pip install kaggle). The repository must be public.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import run_on_kaggle
from build_notebook import REPO
from coder import build_agent
from run_on_kaggle import check_gpu, connect, print_log, wait

SLUG = "wimb-coder-agent"
NOTEBOOK = HERE / build_agent.NOTEBOOK
OUT = HERE.parent / "kaggle_output_agent"
WAIT_MINUTES = 90
NEED_HOURS = 1.5  # Kaggle GPU hours a run can take (stop before starting otherwise)
MAX_TASK_CHARS = 4000
MAX_PATCH_BYTES = 200_000  # 200 KB: a bigger change is not a junior engineer's small change
OUTPUT_FILES = (build_agent.RESULTS_FILE, build_agent.LOG_FILE, build_agent.PATCH_FILE)
LINES = {name: re.compile(rf"^{name} = .*$", re.MULTILINE) for name in ("TASK", "REPO_URL", "COMMIT")}
REPO_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100}$")  # owner/name on GitHub
SHA = re.compile(r"^[0-9a-f]{7,40}$")
# The lines of a patch that name a file.
PATH_LINE = re.compile(r"^(?:diff --git |--- |\+\+\+ |rename from |rename to |copy from |copy to )(.*)$", re.MULTILINE)
GITHUB_DIR = re.compile(r'(?:^|[\s"/])\.github(?:[/"\s]|$)')


def task_from(text: str) -> str:
    """The task, checked (ValueError when it is empty or too long)."""
    task = (text or "").strip()
    if not task:
        raise ValueError("the task is empty: say what the agent should do")
    if len(task) > MAX_TASK_CHARS:
        raise ValueError(f"the task is too long ({len(task)} characters; at most {MAX_TASK_CHARS})")
    return task


def settings_cell(nb: dict) -> dict:
    for cell in nb["cells"]:
        if "".join(cell["source"]).startswith(f"#@title {build_agent.SETTINGS_TITLE}"):
            return cell
    raise RuntimeError(f"{build_agent.NOTEBOOK}: the settings cell is missing; run python coder/build.py")


def notebook_with(task: str, repo: str = REPO, commit: str | None = None) -> dict:
    """coder_agent.ipynb with this task, repository (owner/name) and commit (None: the newest)."""
    if not REPO_NAME.match(repo):
        raise ValueError(f"not a GitHub repository name: {repo!r} (like pratik9yadav/yok)")
    if commit and not SHA.match(commit):
        raise ValueError(f"not a commit id: {commit!r}")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = settings_cell(nb)
    source = "".join(cell["source"])
    # repr() keeps a value on one line: quotes, backslashes and line breaks become escapes.
    for name, value in (("TASK", task_from(task)), ("REPO_URL", f"https://github.com/{repo}.git"),
                        ("COMMIT", commit or "")):
        source, n = LINES[name].subn(lambda _, name=name, value=value: f"{name} = {value!r}", source)
        if n != 1:
            raise RuntimeError(f"{build_agent.NOTEBOOK}: the {name} line is missing; run python coder/build.py")
    cell["source"] = source.splitlines(keepends=True)
    return nb


def settings(nb: dict) -> dict:
    """The settings cell's values (TASK, REPO_URL, COMMIT, MODEL, ...), as the notebook sees them."""
    scope: dict = {}
    exec("".join(settings_cell(nb)["source"]), scope)  # noqa: S102 - our own generated cell
    return scope


def push(api, kernel: str, nb: dict) -> None:
    s = settings(nb)
    print(f"Task: {s['TASK']}\nRepository: {s['REPO_URL']} at {s['COMMIT'] or 'the newest commit'}; model "
          f"{s['MODEL']}; at most {s['MAX_STEPS']} steps and {s['AGENT_MINUTES']} minutes", flush=True)
    run_on_kaggle.push_notebook(api, kernel, nb, build_agent.NOTEBOOK)


# ---- the patch ----

def unquote(path: str) -> str:
    """A name as git writes it in quotes ("a/\\303\\251 x" with octal escapes), as plain text."""
    if "\\" not in path:
        return path
    escapes = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13}
    raw = re.sub(rb"\\([0-7]{3}|.)", lambda m: bytes([int(m.group(1), 8) % 256]) if len(m.group(1)) == 3 else
                 bytes([escapes.get(m.group(1).decode("latin-1"), m.group(1)[0])]), path.encode("utf-8"))
    return raw.decode("utf-8", "replace")


def paths_in(patch: str) -> list[str]:
    """Every file name the patch's header lines mention (a/x, b/x, "a/x y", /dev/null), escapes undone."""
    return [unquote(m.group(1).strip()) for m in PATH_LINE.finditer(patch)]


def patch_problems(patch: str) -> list[str]:
    """Why this patch must not be used (empty list: it may be applied)."""
    problems = []
    size = len(patch.encode("utf-8"))
    if size > MAX_PATCH_BYTES:
        problems.append(f"the change is too large ({size:,} bytes; at most {MAX_PATCH_BYTES:,})")
    github = sorted({p for p in paths_in(patch) if GITHUB_DIR.search(p)})
    if github:
        problems.append("the change touches .github/ (workflows are never changed by the agent): "
                        + ", ".join(github)[:500])
    return problems


def diff_stat(patch: str) -> list[dict]:
    """Per file: {"file", "added", "removed", "binary"}, read from the patch."""
    files: list[dict] = []
    for block in re.split(r"^(?=diff --git )", patch, flags=re.MULTILINE):
        if not block.startswith("diff --git "):
            continue
        header = block.split("\n", 1)[0][len("diff --git "):]
        name = re.search(r'^\+\+\+ "?b/(.*?)"?$', block, re.MULTILINE) or re.search(
            r'^--- "?a/(.*?)"?$', block, re.MULTILINE)
        entry = {"file": name.group(1) if name else header.split(" b/", 1)[-1].strip('"'), "added": 0, "removed": 0,
                 "binary": "GIT binary patch" in block or "\nBinary files " in block}
        in_hunk = False
        for line in block.splitlines():
            if line.startswith("@@"):
                in_hunk = True
            elif in_hunk and line.startswith("+"):
                entry["added"] += 1
            elif in_hunk and line.startswith("-"):
                entry["removed"] += 1
        files.append(entry)
    return files


def stat_lines(patch: str) -> list[str]:
    stat = diff_stat(patch)
    lines = [f"   {f['file']}  " + ("(binary)" if f["binary"] else f"+{f['added']} -{f['removed']}") for f in stat]
    lines.append(f"   {len(stat)} file(s) changed, +{sum(f['added'] for f in stat)} "
                 f"-{sum(f['removed'] for f in stat)}")
    return lines


# ---- the report ----

def tests_line(t: dict | None) -> str:
    if not t:
        return "not run"
    code = t.get("exit_code")
    word = {0: "passed", 5: "no tests found", -9: "timed out"}.get(code, "failed")
    return f"{word} (exit {code}, {t.get('seconds', '?')} s)"


def report(r: dict, patch: str | None) -> None:
    """Print agent_result.json and the change for a person."""
    print(f"\nStatus: {r.get('status')}  ({r.get('steps')} steps, {r.get('minutes')} min; {r.get('model')}, "
          f"num_ctx {r.get('num_ctx')}, runs on {r.get('processor') or '?'})")
    print(f"Repository: {r.get('repo')} at {r.get('commit')}")
    print(("\nThe agent asks:\n" if r.get("status") == "question" else "\nThe agent's summary:\n")
          + "\n".join("   " + line for line in (r.get("summary") or "(none)").splitlines()))
    for name in ("tests_before", "tests_after"):
        t = r.get(name)
        print(f"\nTests {name.split('_')[1]}: {tests_line(t)}")
        for line in ((t or {}).get("last_lines") or "").splitlines()[-8:]:
            print("   |", line)
    print("\nThe change:")
    print("\n".join(stat_lines(patch)) if patch else "   no change")
    for error in r.get("errors") or []:
        print("Error:", error[:500])


def fetch(api, kernel: str, ok: bool = True) -> tuple[dict | None, str]:
    """Download the run's files and print them: (agent_result.json or None, the patch or "")."""
    OUT.mkdir(exist_ok=True)
    for name in OUTPUT_FILES:  # never report an older run's files
        (OUT / name).unlink(missing_ok=True)
    api.kernels_output(kernel, str(OUT), file_pattern=r"(agent_result\.json|agent_log\.json|changes\.patch|.*\.log)$",
                       force=True)
    path, patch_path = OUT / build_agent.RESULTS_FILE, OUT / build_agent.PATCH_FILE
    r = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    patch = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
    if r is not None:
        report(r, patch)
    if r is None or not ok:
        print(f"\nNo {build_agent.RESULTS_FILE}: the run stopped early." if r is None else "\nThe run did not finish.",
              "Kaggle's log:")
        for log in OUT.glob("*.log"):
            print_log(log)
    print(f"\nDownloaded to {OUT}: {sorted(p.name for p in OUT.iterdir())}")
    print(f"The run on Kaggle: https://www.kaggle.com/code/{kernel}")
    return r, patch


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--task", metavar="TEXT", help="what the agent should do, in plain words")
    args.add_argument("--task-file", metavar="PATH", help="read the task from this file instead")
    args.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY") or REPO,
                      help=f"public GitHub repository owner/name (default: $GITHUB_REPOSITORY or {REPO})")
    args.add_argument("--commit", metavar="SHA", help="the commit to start from (default: the newest)")
    args.add_argument("--wait", action="store_true", help="don't start a new run, wait for the current one")
    opts = args.parse_args()
    nb = None
    if not opts.wait:
        if (opts.task is None) == (opts.task_file is None):
            args.error("give the task with --task TEXT or --task-file PATH (one of them)")
        try:
            task = opts.task if opts.task is not None else Path(opts.task_file).read_text(encoding="utf-8")
            nb = notebook_with(task, opts.repo, opts.commit or None)
        except (OSError, ValueError) as e:
            args.error(str(e))
    api, kernel = connect(SLUG)
    if nb is not None:
        check_gpu(api, need_hours=NEED_HOURS)
        push(api, kernel, nb)
        time.sleep(30)
    ok = wait(api, kernel, limit_min=WAIT_MINUTES)
    r, patch = fetch(api, kernel, ok)
    if r is None:
        sys.exit(1)
    problems = patch_problems(patch)
    if problems:
        print("\nThis change will NOT be used: " + "; ".join(problems))
        sys.exit(1)
    print(f"\nStatus: {r.get('status')}." + ("" if patch.strip() else " No change was made."))
    sys.exit(0)


if __name__ == "__main__":
    main()

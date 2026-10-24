"""The coding agent's result as a GitHub job summary and a pull request text (used by .github/workflows/agent.yml).

    python coder/agent_pr.py decide   write the job summary ($GITHUB_STEP_SUMMARY) and open_pr=true|false
                                      ($GITHUB_OUTPUT): true when there is a valid change and no question, and
                                      the agent was not stuck and did not say done without changing a file
    python coder/agent_pr.py pr       check the patch again, then write pr_title.txt, pr_body.md and
                                      commit_message.txt into kaggle_output_agent/ (exit 1 if the patch is refused)

Reads kaggle_output_agent/ (agent_result.json, changes.patch, and the runner's checks: runner_pytest.txt/.exit,
runner_ruff.txt/.exit) and the task from the TASK environment variable (never from the command line). Text from the
task, the agent or the tests goes inside code fences, so GitHub shows it as plain text: no @mentions, links or HTML.
"""

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from coder import build_agent
from coder.run_agent import NO_PR, OUT, patch_problems, stat_lines, tests_line

NOTE = "Made by the free coding agent (qwen3-coder:30b on Kaggle). Review before merging."
STATUSES = ("done", "question", "stopped", "stuck", "no_change", "error")
HEADS = {"question": "The agent asks a question", "stuck": "Coding agent: stuck (it kept repeating commands)",
         "no_change": "Coding agent: said done, but changed nothing"}
TITLE_CHARS = 60
MAX_BODY = 60_000  # GitHub allows 65,536 characters
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f  ]")
CHECKS = (("pytest", "python -m pytest -q"), ("ruff", "ruff check ."))


def clean(text, limit: int = 4000) -> str:
    """Text without control characters (tabs and line breaks stay), at most `limit` characters."""
    text = CONTROL.sub("", str(text or "").replace("\r\n", "\n").replace("\r", "\n"))
    return text if len(text) <= limit else text[: limit - 20] + "\n... (cut)"


def fence(text, limit: int = 4000) -> str:
    """A code block GitHub shows as plain text, whatever the text holds (its fence is longer than any ``` in it)."""
    text = clean(text, limit) or "(none)"
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}text\n{text}\n{ticks}"


def title_for(task: str) -> str:
    """ "Agent: <the first 60 characters of the task>", on one line."""
    words = " ".join(clean(task, 10_000).split())
    return "Agent: " + (words[:TITLE_CHARS] + ("…" if len(words) > TITLE_CHARS else "") if words else "(no task)")


def number(value) -> str:
    return str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else "?"


def word(value, pattern: str = r"[\w.:/-]{1,80}") -> str:
    """A short value (a model name, a status) only when it looks like one; else "?"."""
    return value if isinstance(value, str) and re.fullmatch(pattern, value) else "?"


def load(out: Path = OUT) -> tuple[dict | None, str, dict]:
    """(agent_result.json or None, changes.patch or "", the runner's checks {"pytest": {...}, "ruff": {...}})."""
    result_path, patch_path = out / build_agent.RESULTS_FILE, out / build_agent.PATCH_FILE
    try:
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    except ValueError:
        result = None
    if not isinstance(result, dict):
        result = None
    patch = patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else ""
    checks = {}
    for name, _ in CHECKS:
        code_path, text_path = out / f"runner_{name}.exit", out / f"runner_{name}.txt"
        if code_path.exists():
            code = code_path.read_text().strip()
            lines = [line for line in (text_path.read_text(encoding="utf-8", errors="replace").splitlines()
                                       if text_path.exists() else []) if line.strip()]
            checks[name] = {"exit_code": int(code) if re.fullmatch(r"-?\d{1,4}", code) else None,
                            "last_lines": "\n".join(lines[-15:])}
    return result, patch, checks


def status_of(result: dict | None) -> str:
    return word((result or {}).get("status"), "|".join(STATUSES)) if result else "no result"


def decision(result: dict | None, patch: str) -> tuple[bool, str]:
    """(open a pull request?, why)."""
    if result is None:
        return False, "the run left no agent_result.json"
    if status_of(result) == "question":
        return False, "the agent asks a question"
    if status_of(result) in NO_PR:
        return False, NO_PR[status_of(result)]
    if not patch.strip():
        return False, "the agent changed nothing"
    problems = patch_problems(patch)
    if problems:
        return False, "the change is refused: " + "; ".join(problems)
    return True, "there is a change to review"


def work_line(result: dict) -> str:
    return (f"Status: **{status_of(result)}** ({number(result.get('steps'))} steps, {number(result.get('minutes'))} "
            f"minutes; `{word(result.get('model'))}`, num_ctx {number(result.get('num_ctx'))}, on Kaggle's free GPU)")


def tests_md(result: dict, checks: dict) -> list[str]:
    def check_line(name: str, command: str) -> str:
        c = checks.get(name)
        if not c:
            return f"- On the GitHub runner, `{command}`: not run"
        verdict = "passed" if c["exit_code"] == 0 else "timed out" if c["exit_code"] == 124 else "failed"
        return f"- On the GitHub runner, `{command}`: **{verdict}** (exit {number(c['exit_code'])})"

    before, after = result.get("tests_before"), result.get("tests_after")
    lines = [(f"- On Kaggle (the agent's machine), `python -m pytest -q`: before {tests_line(before)} → after "
              f"{tests_line(after)}")]
    for name, command in CHECKS:
        lines.append(check_line(name, command))
        if checks.get(name):
            lines.append(fence(checks[name]["last_lines"], 3000))
    return lines


def run_url() -> str:
    server, repo, run_id = (os.environ.get(k, "") for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    if re.fullmatch(r"https://[\w.-]+", server) and re.fullmatch(r"[\w.-]+/[\w.-]+", repo) and run_id.isdigit():
        return f"{server}/{repo}/actions/runs/{run_id}"
    return "this workflow run"


def pr_body(task: str, result: dict, patch: str, checks: dict) -> str:
    """The pull request's text: the task, the agent's summary, its work, the files, the tests, the note."""
    parts = [
        "## Task", fence(task),
        "## What the agent says", work_line(result), fence(result.get("summary")),
        "## Files changed", fence("\n".join(line.strip() for line in stat_lines(patch)), 6000),
        "## Tests", *tests_md(result, checks),
        (f"Every step the agent took (command and output) is in `agent_log.json`, in the `kaggle-output-agent` "
         f"artifact of {run_url()}."),
        f"> {NOTE}",
    ]
    body = "\n\n".join(parts) + "\n"
    return body if len(body) <= MAX_BODY else body[: MAX_BODY - 200] + "\n\n(cut: too long)\n\n> " + NOTE + "\n"


def commit_message(task: str, result: dict) -> str:
    return (f"{title_for(task)}\n\nTask:\n{clean(task)}\n\nStatus: {status_of(result)}; "
            f"{number(result.get('steps'))} steps.\n\n{NOTE}\n")


def job_summary(task: str, result: dict | None, patch: str, opened: bool, why: str) -> str:
    """The job's summary page (what the owner sees on the phone)."""
    if result is None:
        return f"## Coding agent: no result\n\n{why}. See the job log.\n"
    status = status_of(result)
    head = HEADS.get(status, f"Coding agent: {status}")
    lines = [f"## {head}", "Task:", fence(task), work_line(result),
             ("The question:" if status == "question" else "The agent's summary:"), fence(result.get("summary")),
             "Files changed:", fence("\n".join(line.strip() for line in stat_lines(patch)) if patch.strip() else
                                     "none", 6000),
             (f"Tests on Kaggle: before {tests_line(result.get('tests_before'))} → after "
              f"{tests_line(result.get('tests_after'))}"),
             ("A pull request will be opened (next job)." if opened else f"No pull request: {why}.")]
    if status == "question":
        lines.append("Answer it by running the workflow again with a clearer task.")
    return "\n\n".join(lines) + "\n"


def write_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")


def main(argv: list[str] | None = None, out: Path = OUT) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv not in (["decide"], ["pr"]):
        print(__doc__)
        return 2
    task = os.environ.get("TASK", "")
    result, patch, checks = load(out)
    opened, why = decision(result, patch)
    if argv == ["decide"]:
        summary = job_summary(task, result, patch, opened, why)
        print(summary)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
                fh.write(summary)
        write_output("open_pr", "true" if opened else "false")
        return 0
    if not opened:
        print(f"No pull request: {why}.")
        return 1
    (out / "pr_title.txt").write_text(title_for(task), encoding="utf-8")
    (out / "pr_body.md").write_text(pr_body(task, result, patch, checks), encoding="utf-8")
    (out / "commit_message.txt").write_text(commit_message(task, result), encoding="utf-8")
    print(f"Wrote the pull request's title and text: {title_for(task)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

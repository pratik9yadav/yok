"""Play the test calls (voicebot/scenarios/*.jsonl) against the bot and score them.

A test call is a pretend person: for each thing the bot asks, the line they say back ("replies", by the slot the
bot asked for, "confirm" after the read-back, "fix" when it asks what is wrong, "*" for anything else), and lines
said first ("first", like "कौन बोल रहा है?"). A list gives the line for the 1st, 2nd, ... time it is asked. The call
passes when the outcome and every expected value match ("contains": a text value holding any of the words).

    python -m voicebot.evaluate                    every test call, rules only (no model): a baseline
    python -m voicebot.evaluate --show shop-hi-who    one call, turn by turn

The same code runs on Kaggle with a model (llm) and, for the spoken test, with say_and_hear: the person's line
goes through the voice and speech to text before the bot hears it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from voicebot.engine import Call
from voicebot.tasks import load

SCENARIOS = HERE / "scenarios"


def scenarios(task: str | None = None) -> list[dict]:
    """Every test call (of one task, when given), with its task id filled in."""
    out = []
    for path in sorted(SCENARIOS.glob("*.jsonl")):
        if task and path.stem != task:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row.setdefault("task", path.stem)
                out.append(row)
    return out


class Person:
    """The pretend person of a test call."""

    def __init__(self, scenario: dict):
        self.first = list(scenario.get("first") or [])
        self.replies = scenario.get("replies") or {}
        self.times: dict[str, int] = {}

    def line(self, pending: str | None) -> str:
        if self.first:
            return self.first.pop(0)
        k = pending if pending in self.replies else "*"
        said = self.replies.get(k, "")
        n = self.times.get(k, 0)
        self.times[k] = n + 1
        if isinstance(said, list):
            return said[min(n, len(said) - 1)] if said else ""
        return said


def matches(scenario: dict, result: dict) -> list[str]:
    """What the call got wrong ([]: it passed)."""
    expect = scenario.get("expect") or {}
    wrong = []
    if expect.get("outcome") and result["outcome"] != expect["outcome"]:
        wrong.append(f"outcome {result['outcome']} (want {expect['outcome']})")
    if expect.get("lang") and result["lang"] != expect["lang"]:
        wrong.append(f"language {result['lang']} (want {expect['lang']})")
    for name, want in (expect.get("values") or {}).items():
        got = result["values"].get(name)
        if got != want and not (isinstance(got, (int, float)) and isinstance(want, (int, float))
                                and not isinstance(got, bool) and abs(got - want) < 0.01):
            wrong.append(f"{name} {got!r} (want {want!r})")
    for name, words in (expect.get("contains") or {}).items():
        got = str(result["values"].get(name) or "").lower()
        if not any(w.lower() in got for w in words):
            wrong.append(f"{name} {result['values'].get(name)!r} (want one of {words})")
    return wrong


def play(scenario: dict, llm=None, say_and_hear=None, max_lines: int = 20) -> dict:
    """One test call. say_and_hear(line, lang) -> (what speech to text heard, seconds): the spoken test; None: the
    bot reads the person's line as typed."""
    task = load(scenario["task"], scenario.get("context"))
    call = Call(task, lang=scenario.get("lang"), llm=llm)
    person = Person(scenario)
    started = time.perf_counter()
    reply = call.start()
    lines = []
    while not reply.done and len(lines) < max_lines:
        said = person.line(call.pending)
        heard, seconds = (said, 0.0) if say_and_hear is None or not said else say_and_hear(said, call.lang)
        lines.append({"said": said, "heard": heard})
        reply = call.hear(heard, seconds)
    result = call.result()
    result["wrong"] = matches(scenario, result)
    result["id"] = scenario["id"]
    result["tags"] = scenario.get("tags", [])
    result["lines"] = lines
    result["seconds"] = round(time.perf_counter() - started, 2)
    understood = [t["understood"] for t in result["turns"] if t.get("understood")]
    result["model_calls"] = sum(u["source"].startswith("model") or bool(u.get("error")) for u in understood)
    result["model_errors"] = sum(bool(u.get("error")) for u in understood)
    return result


def summary(results: list[dict]) -> dict:
    """Pass rates overall, per task and per tag."""
    def rate(rows):
        return {"passed": sum(not r["wrong"] for r in rows), "total": len(rows)}

    tasks = sorted({r["task"] for r in results})
    tags = sorted({t for r in results for t in r["tags"]})
    return {"all": rate(results), "tasks": {t: rate([r for r in results if r["task"] == t]) for t in tasks},
            "tags": {t: rate([r for r in results if t in r["tags"]]) for t in tags},
            "model_calls": sum(r["model_calls"] for r in results),
            "model_errors": sum(r["model_errors"] for r in results),
            "person_lines": sum(len(r["lines"]) for r in results)}


def summary_lines(results: list[dict], label: str = "") -> list[str]:
    s = summary(results)
    a = s["all"]
    lines = [f"{label}{a['passed']} of {a['total']} test calls passed"
             + "".join(f"; {t} {v['passed']}/{v['total']}" for t, v in s["tasks"].items())]
    lines.append("   by kind: " + ", ".join(f"{t} {v['passed']}/{v['total']}" for t, v in s["tags"].items()))
    if s["model_calls"]:
        lines.append(f"   the model was asked for {s['model_calls']} of {s['person_lines']} lines "
                     f"({s['model_errors']} failed: the rules answered)")
    for r in results:
        if r["wrong"]:
            lines.append(f"   FAILED {r['id']}: {'; '.join(r['wrong'])}")
    return lines


def transcript(result: dict) -> list[str]:
    out = []
    for t in result["turns"]:
        if t["who"] == "bot":
            out.append(f"   BOT: {t['text']}")
        else:
            u = t.get("understood") or {}
            out.append(f"   YOU: {t['text'] or '(silence)'}   → {u.get('intent')} {u.get('values')} [{u.get('source')}]")
    return out


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--task", help="only this task's test calls")
    args.add_argument("--show", metavar="ID", help="print one test call turn by turn")
    opts = args.parse_args()
    rows = scenarios(opts.task)
    if opts.show:
        rows = [r for r in rows if r["id"] == opts.show]
        if not rows:
            args.error(f"no test call {opts.show}")
    results = [play(r) for r in rows]
    if opts.show:
        print("\n".join(transcript(results[0])))
        print("   result:", results[0]["outcome"], results[0]["values"], results[0]["wrong"] or "PASSED")
        return
    print("\n".join(summary_lines(results, "Rules only (no model): ")))


if __name__ == "__main__":
    main()

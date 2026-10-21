"""Summarise a shop server log file: counts per level, time range and top errors."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import datetime

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_line(line: str) -> tuple[datetime, str, str] | None:
    parts = line.split(None, 3)
    if len(parts) < 3:
        return None
    try:
        # The server writes local time without an offset; naive datetimes are only compared with each other.
        stamp = datetime.strptime(f"{parts[0]} {parts[1]}", TIME_FORMAT)  # noqa: DTZ007
    except ValueError:
        return None
    level = parts[2]
    if level not in LEVELS:
        return None
    message = parts[3].strip() if len(parts) == 4 else ""
    return stamp, level, message


def summarise(lines: Iterable[str], top: int = 5) -> dict:
    """Build the summary dict for an iterable of log lines."""
    total = 0
    malformed = 0
    by_level = dict.fromkeys(LEVELS, 0)
    errors: Counter[str] = Counter()
    first: datetime | None = None
    last: datetime | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        total += 1
        parsed = _parse_line(line)
        if parsed is None:
            malformed += 1
            continue
        stamp, level, message = parsed
        by_level[level] += 1
        if level == "ERROR":
            errors[message] += 1
        if first is None or stamp < first:
            first = stamp
        if last is None or stamp > last:
            last = stamp
    top_errors = sorted(errors.items(), key=lambda item: (-item[1], item[0]))[:top]
    return {
        "total_lines": total,
        "by_level": by_level,
        "malformed": malformed,
        "first": first.strftime(TIME_FORMAT) if first else None,
        "last": last.strftime(TIME_FORMAT) if last else None,
        "top_errors": [{"message": m, "count": c} for m, c in top_errors],
    }


def format_text(summary: dict) -> str:
    out = [f"Lines: {summary['total_lines']} (malformed: {summary['malformed']})"]
    out.append("By level: " + ", ".join(f"{lvl} {n}" for lvl, n in summary["by_level"].items()))
    if summary["first"]:
        out.append(f"From {summary['first']} to {summary['last']}")
    else:
        out.append("No timestamps found")
    if summary["top_errors"]:
        out.append("Top errors:")
        out.extend(f"  {e['count']:>6}  {e['message']}" for e in summary["top_errors"])
    return "\n".join(out)


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more: {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise a shop server log file.")
    parser.add_argument("path", help="log file to read")
    parser.add_argument("--top", type=_non_negative_int, default=5, help="how many ERROR messages to list")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse already printed the usage error
        return int(exc.code or 0)
    try:
        with open(args.path, encoding="utf-8", errors="replace") as fh:
            summary = summarise(fh, top=args.top)
    except OSError as exc:
        print(f"logsum: cannot read {args.path}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    else:
        print(format_text(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())

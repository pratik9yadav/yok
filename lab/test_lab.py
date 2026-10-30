"""The lab's frozen test sets: once a test set is in lab/frozen.json it never changes (a new one gets a new file)."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
FROZEN = json.loads((ROOT / "lab" / "frozen.json").read_text(encoding="utf-8"))


def test_frozen_test_sets_are_unchanged() -> None:
    changed = [name for name, digest in FROZEN["files"].items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    assert not changed, f"frozen test sets changed (make a new file instead): {changed}"


def test_lab_log_has_a_scoreboard() -> None:
    log = (ROOT / "lab" / "LOG.md").read_text(encoding="utf-8")
    assert "## Scoreboard" in log and "## Queue" in log and "## Experiments" in log

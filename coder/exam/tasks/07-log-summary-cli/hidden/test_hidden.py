import json
import subprocess
import sys
import time

import logsum

SAMPLE = """2026-10-01 10:00:03 INFO server started
2026-10-01 10:05:00 ERROR payment gateway timeout
2026-10-01 10:06:00 WARNING low stock ATTA-5KG

2026-10-01 10:07:00 ERROR payment gateway timeout
2026-10-01 09:59:00 DEBUG cache warm
2026-10-01 11:00:00 ERROR sms failed
this line is garbage
2026-13-45 10:00:00 ERROR impossible date
2026-10-01 12:00:00 FATAL unknown level
2026-10-01 12:30:00 ERROR db locked
"""


def run_json(tmp_path, capsys, text=SAMPLE, extra=(), encoding="utf-8"):
    p = tmp_path / "shop.log"
    if isinstance(text, bytes):
        p.write_bytes(text)
    else:
        p.write_text(text, encoding=encoding)
    code = logsum.main([str(p), "--json", *extra])
    out = capsys.readouterr().out
    assert code == 0
    return json.loads(out)


def test_json_summary(tmp_path, capsys):
    s = run_json(tmp_path, capsys)
    assert s["total_lines"] == 10
    assert s["by_level"] == {"DEBUG": 1, "INFO": 1, "WARNING": 1, "ERROR": 4}
    assert s["malformed"] == 3
    assert s["first"] == "2026-10-01 09:59:00"  # earliest, not first in the file
    assert s["last"] == "2026-10-01 12:30:00"


def test_top_errors_counts_and_ties(tmp_path, capsys):
    s = run_json(tmp_path, capsys)
    assert s["top_errors"] == [
        {"message": "payment gateway timeout", "count": 2},
        {"message": "db locked", "count": 1},
        {"message": "sms failed", "count": 1},
    ]


def test_top_limit(tmp_path, capsys):
    s = run_json(tmp_path, capsys, extra=["--top", "1"])
    assert s["top_errors"] == [{"message": "payment gateway timeout", "count": 2}]


def test_default_top_is_five(tmp_path, capsys):
    text = "".join(f"2026-10-01 10:00:0{i} ERROR problem {i}\n" for i in range(8))
    s = run_json(tmp_path, capsys, text=text)
    assert [e["message"] for e in s["top_errors"]] == [f"problem {i}" for i in range(5)]


def test_empty_file(tmp_path, capsys):
    s = run_json(tmp_path, capsys, text="")
    assert s == {
        "total_lines": 0,
        "by_level": {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0},
        "malformed": 0,
        "first": None,
        "last": None,
        "top_errors": [],
    }


def test_windows_line_endings(tmp_path, capsys):
    text = "2026-10-01 10:00:00 ERROR disk full\r\n2026-10-01 10:00:01 ERROR disk full\r\n\r\n"
    s = run_json(tmp_path, capsys, text=text.encode("utf-8"))
    assert s["total_lines"] == 2
    assert s["top_errors"] == [{"message": "disk full", "count": 2}]


def test_bad_bytes_do_not_crash(tmp_path, capsys):
    data = b"2026-10-01 10:00:00 INFO ok\n2026-10-01 10:00:01 ERROR bad \xff\xfe bytes\n\xff\xff\xff\n"
    s = run_json(tmp_path, capsys, text=data)
    assert s["total_lines"] == 3
    assert s["by_level"]["ERROR"] == 1
    assert s["malformed"] == 1


def test_unicode_messages(tmp_path, capsys):
    s = run_json(tmp_path, capsys, text="2026-10-01 10:00:00 ERROR ग्राहक नहीं मिला\n")
    assert s["top_errors"] == [{"message": "ग्राहक नहीं मिला", "count": 1}]


def test_text_output(tmp_path, capsys):
    p = tmp_path / "shop.log"
    p.write_text(SAMPLE, encoding="utf-8")
    assert logsum.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "payment gateway timeout" in out
    assert not out.lstrip().startswith("{")


def test_missing_file_is_clean_error(tmp_path, capsys):
    code = logsum.main([str(tmp_path / "missing.log"), "--json"])
    captured = capsys.readouterr()
    assert code != 0
    assert captured.err.strip()
    assert "Traceback" not in captured.err
    assert captured.out.strip() == ""


def test_directory_instead_of_file_is_clean_error(tmp_path, capsys):
    code = logsum.main([str(tmp_path)])
    captured = capsys.readouterr()
    assert code != 0
    assert captured.err.strip()
    assert "Traceback" not in captured.err


def test_negative_top_is_rejected(tmp_path, capsys):
    p = tmp_path / "shop.log"
    p.write_text(SAMPLE, encoding="utf-8")
    try:
        code = logsum.main([str(p), "--top", "-1"])
    except SystemExit as exc:
        code = exc.code
    assert code not in (0, None)


def test_runs_as_a_script(tmp_path):
    p = tmp_path / "shop.log"
    p.write_text(SAMPLE, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, logsum.__file__, str(p), "--json"], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["by_level"]["ERROR"] == 4
    missing = subprocess.run(
        [sys.executable, logsum.__file__, str(tmp_path / "nope.log")],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert missing.returncode != 0
    assert "Traceback" not in missing.stderr


def test_does_not_modify_the_log(tmp_path, capsys):
    p = tmp_path / "shop.log"
    p.write_text(SAMPLE, encoding="utf-8")
    logsum.main([str(p), "--json"])
    assert p.read_text(encoding="utf-8") == SAMPLE
    assert sorted(x.name for x in tmp_path.iterdir()) == ["shop.log"]


def test_big_log_is_fast(tmp_path, capsys):
    levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
    lines = [
        f"2026-10-01 {i // 3600 % 24:02d}:{i // 60 % 60:02d}:{i % 60:02d} {levels[i % 4]} event {i % 37}\n"
        for i in range(200_000)
    ]
    start = time.perf_counter()
    s = run_json(tmp_path, capsys, text="".join(lines))
    assert time.perf_counter() - start < 4.0
    assert s["total_lines"] == 200_000
    assert s["by_level"]["ERROR"] == 50_000

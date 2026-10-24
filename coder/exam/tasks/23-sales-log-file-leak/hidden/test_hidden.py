import os

import pytest

from saleslog import SalesLog

FD_DIR = "/proc/self/fd"
needs_proc = pytest.mark.skipif(not os.path.isdir(FD_DIR), reason="needs /proc/self/fd")


def open_fds():
    return len(os.listdir(FD_DIR))


def test_total_sees_sales_right_away(tmp_path):
    log = SalesLog(str(tmp_path))
    log.record("2024-11-01", "B-1", 15000)
    log.record("2024-11-01", "B-2", 2550)
    assert log.total("2024-11-01") == 17550
    log.record("2024-11-01", "B-3", 450)
    assert log.total("2024-11-01") == 18000
    log.close()


def test_another_reader_sees_sales_without_close(tmp_path):
    writer = SalesLog(str(tmp_path))
    writer.record("2024-11-02", "B-1", 999)
    reader = SalesLog(str(tmp_path))
    assert reader.total("2024-11-02") == 999
    with open(tmp_path / "2024-11-02.csv", encoding="utf-8") as f:
        assert f.read() == "B-1,999\n"
    writer.close()
    reader.close()


def test_days_are_separate_and_unknown_day_is_zero(tmp_path):
    log = SalesLog(str(tmp_path))
    log.record("2024-11-01", "A", 100)
    log.record("2024-11-02", "B", 200)
    assert log.total("2024-11-01") == 100
    assert log.total("2024-11-02") == 200
    assert log.total("2024-12-25") == 0
    log.close()


def test_appends_to_existing_file_from_yesterday(tmp_path):
    (tmp_path / "2024-11-03.csv").write_text("OLD-1,500\n", encoding="utf-8")
    log = SalesLog(str(tmp_path))
    log.record("2024-11-03", "NEW-1", 700)
    assert log.total("2024-11-03") == 1200
    log.close()


@needs_proc
def test_many_days_do_not_pile_up_open_files(tmp_path):
    log = SalesLog(str(tmp_path))
    before = open_fds()
    for i in range(300):
        log.record(f"day-{i:03d}", f"B-{i}", i)
    during = open_fds()
    assert during - before < 10, f"{during - before} files left open while the server runs"
    assert log.total("day-299") == 299
    log.close()


@needs_proc
def test_total_does_not_leave_files_open(tmp_path):
    log = SalesLog(str(tmp_path))
    log.record("2024-11-04", "B-1", 100)
    log.total("2024-11-04")
    before = open_fds()
    for _ in range(100):
        assert log.total("2024-11-04") == 100
    assert open_fds() - before < 5
    log.close()


@needs_proc
def test_corrupt_file_raises_value_error_without_leaking(tmp_path):
    (tmp_path / "2024-11-05.csv").write_text("B-1,100\nB-2,abc\n", encoding="utf-8")
    log = SalesLog(str(tmp_path))
    before = open_fds()
    kept = []  # the server's error reporter keeps recent exceptions (and their tracebacks)
    for _ in range(50):
        with pytest.raises(ValueError) as excinfo:
            log.total("2024-11-05")
        kept.append(excinfo)
    assert open_fds() - before < 5
    log.close()


def test_close_is_safe_to_call_twice_and_keeps_data(tmp_path):
    log = SalesLog(str(tmp_path))
    log.record("2024-11-06", "B-1", 42)
    log.close()
    log.close()
    assert SalesLog(str(tmp_path)).total("2024-11-06") == 42


def test_many_sales_one_day(tmp_path):
    log = SalesLog(str(tmp_path))
    for i in range(1000):
        log.record("2024-11-07", f"B-{i}", 10)
    assert log.total("2024-11-07") == 10000
    log.close()
    lines = (tmp_path / "2024-11-07.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1000
    assert lines[0] == "B-0,10"

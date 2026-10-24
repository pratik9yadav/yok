import os
import random
import time
import tracemalloc

import pytest

from sales_report import summarize, top_skus

HEADER = "date,sku,qty,price_paise\n"
FD_DIR = "/proc/self/fd"


def make_rows(count, seed=5, skus=40, days=60):
    rng = random.Random(seed)
    sku_names = [f"SKU-{i:03d}" for i in range(skus)]
    prices = {s: rng.randint(500, 50_000) for s in sku_names}
    for _ in range(count):
        day = f"2024-{1 + rng.randrange(days) // 28:02d}-{1 + rng.randrange(28):02d}"
        sku = rng.choice(sku_names)
        yield day, sku, rng.randint(1, 5), prices[sku]


def write_log(path, rows, newline="\n"):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(HEADER.replace("\n", newline))
        for day, sku, qty, price in rows:
            f.write(f"{day},{sku},{qty},{price}{newline}")
    return path


def expected_summary(rows):
    revenue = units = 0
    by_day, by_sku, by_rev = {}, {}, {}
    for day, sku, qty, price in rows:
        revenue += qty * price
        units += qty
        by_day[day] = by_day.get(day, 0) + qty * price
        by_sku[sku] = by_sku.get(sku, 0) + qty
        by_rev[sku] = by_rev.get(sku, 0) + qty * price
    summary = {"revenue_paise": revenue, "units": units, "by_day": dict(sorted(by_day.items())),
               "by_sku": dict(sorted(by_sku.items()))}
    ranked = sorted(by_rev.items(), key=lambda kv: (-kv[1], kv[0]))
    return summary, ranked


def test_small_log_numbers_unchanged(tmp_path):
    rows = [
        ("2024-03-01", "ATTA-5KG", 2, 24000),
        ("2024-03-01", "SALT-1KG", 1, 2800),
        ("2024-03-02", "ATTA-5KG", 1, 24000),
        ("2024-03-02", "OIL-1L", 3, 15500),
    ]
    p = write_log(tmp_path / "s.csv", rows)
    result = summarize(str(p))
    assert result == {
        "revenue_paise": 2 * 24000 + 2800 + 24000 + 3 * 15500,
        "units": 7,
        "by_day": {"2024-03-01": 50800, "2024-03-02": 70500},
        "by_sku": {"ATTA-5KG": 3, "OIL-1L": 3, "SALT-1KG": 1},
    }
    assert list(result["by_day"]) == sorted(result["by_day"])
    assert list(result["by_sku"]) == sorted(result["by_sku"])
    assert [tuple(x) for x in top_skus(str(p), 2)] == [("ATTA-5KG", 72000), ("OIL-1L", 46500)]


def test_random_log_matches_expected(tmp_path):
    rows = list(make_rows(3000))
    p = write_log(tmp_path / "s.csv", rows)
    summary, ranked = expected_summary(rows)
    assert summarize(str(p)) == summary
    assert [tuple(x) for x in top_skus(str(p), 5)] == ranked[:5]
    assert [tuple(x) for x in top_skus(str(p), 1000)] == ranked


def test_top_skus_ties_broken_by_sku(tmp_path):
    rows = [("2024-01-01", s, 1, 100) for s in ("C", "A", "B")] + [("2024-01-01", "Z", 1, 500)]
    p = write_log(tmp_path / "t.csv", rows)
    assert [tuple(x) for x in top_skus(str(p), 3)] == [("Z", 500), ("A", 100), ("B", 100)]


def test_windows_line_endings_and_blank_lines(tmp_path):
    p = tmp_path / "w.csv"
    p.write_bytes(b"date,sku,qty,price_paise\r\n2024-01-01,A,2,100\r\n\r\n2024-01-02,B,1,50\r\n")
    assert summarize(str(p)) == {"revenue_paise": 250, "units": 3, "by_day": {"2024-01-01": 200, "2024-01-02": 50},
                                 "by_sku": {"A": 2, "B": 1}}


def test_header_only_and_empty_file(tmp_path):
    empty_summary = {"revenue_paise": 0, "units": 0, "by_day": {}, "by_sku": {}}
    p = tmp_path / "h.csv"
    p.write_text(HEADER, encoding="utf-8")
    assert summarize(str(p)) == empty_summary
    assert list(top_skus(str(p), 3)) == []
    e = tmp_path / "e.csv"
    e.write_text("", encoding="utf-8")
    assert summarize(str(e)) == empty_summary


@pytest.mark.parametrize("bad_line", ["2024-01-01,A,2", "2024-01-01,A,two,100", "2024-01-01,A,2,100,extra"])
def test_malformed_line_raises_value_error(tmp_path, bad_line):
    p = tmp_path / "bad.csv"
    p.write_text(HEADER + "2024-01-01,A,1,100\n" + bad_line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        summarize(str(p))
    with pytest.raises(ValueError):
        top_skus(str(p), 3)


@pytest.mark.skipif(not os.path.isdir(FD_DIR), reason="needs /proc/self/fd")
def test_files_are_closed_even_on_errors(tmp_path):
    good = write_log(tmp_path / "g.csv", make_rows(100))
    bad = tmp_path / "bad.csv"
    bad.write_text(HEADER + "2024-01-01,A,1,100\n2024-01-01,A,x,100\n", encoding="utf-8")
    before = len(os.listdir(FD_DIR))
    kept = []
    for _ in range(30):
        summarize(str(good))
        top_skus(str(good), 3)
        with pytest.raises(ValueError) as excinfo:
            summarize(str(bad))
        kept.append(excinfo)
    assert len(os.listdir(FD_DIR)) - before < 5


def test_memory_does_not_grow_with_file_size(tmp_path):
    p = write_log(tmp_path / "big.csv", make_rows(50_000, skus=30, days=30))
    size_mb = p.stat().st_size / 1e6
    assert size_mb > 1
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        summarize(str(p))
        _, peak_summarize = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        top_skus(str(p), 5)
        _, peak_top = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    limit = 1_000_000  # 1 MB, less than the file itself
    assert peak_summarize < limit, f"summarize used {peak_summarize / 1e6:.1f} MB for a {size_mb:.1f} MB file"
    assert peak_top < limit, f"top_skus used {peak_top / 1e6:.1f} MB for a {size_mb:.1f} MB file"


def test_large_log_is_reasonably_fast(tmp_path):
    p = write_log(tmp_path / "big.csv", make_rows(100_000))
    start = time.perf_counter()
    summarize(str(p))
    top_skus(str(p), 10)
    assert time.perf_counter() - start < 2

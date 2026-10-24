import json
import os
import subprocess
import sys
import textwrap
import unicodedata

import pytest

import catalog
from catalog import load_catalog, receipt_label, save_catalog, search

JA = "\N{DEVANAGARI LETTER JA}"
NUKTA = "\N{DEVANAGARI SIGN NUKTA}"
ZA_PRECOMPOSED = "\N{DEVANAGARI LETTER ZA}"
II = "\N{DEVANAGARI VOWEL SIGN II}"
RA = "\N{DEVANAGARI LETTER RA}"
AA_SIGN = "\N{DEVANAGARI VOWEL SIGN AA}"

JEERA_PRECOMPOSED = ZA_PRECOMPOSED + II + RA + AA_SIGN  # ज़ीरा typed one way
JEERA_DECOMPOSED = JA + NUKTA + II + RA + AA_SIGN  # ज़ीरा typed another way

ITEMS = [
    {"sku": "TATA-SALT-1KG", "name": "Tata Salt 1kg", "price_paise": 2800},
    {"sku": "HALDI-200", "name": "हल्दी पाउडर 200g", "price_paise": 6500},
    {"sku": "JEERA-100", "name": JEERA_PRECOMPOSED + " 100g", "price_paise": 5500},
    {"sku": "ATTA-5KG", "name": "आटा 5kg", "price_paise": 24000},
]


def write_csv(path, rows, encoding, newline="\n"):
    lines = ["sku,name,price_paise"] + [f"{r['sku']},{r['name']},{r['price_paise']}" for r in rows]
    path.write_bytes(newline.join(lines + [""]).encode(encoding))


def test_load_plain_utf8(tmp_path):
    p = tmp_path / "catalog.csv"
    write_csv(p, ITEMS, "utf-8")
    assert load_catalog(str(p)) == ITEMS


def test_load_excel_csv_utf8_with_bom_and_crlf(tmp_path):
    p = tmp_path / "catalog.csv"
    write_csv(p, ITEMS, "utf-8-sig", newline="\r\n")
    assert load_catalog(str(p)) == ITEMS


def test_save_writes_utf8_and_round_trips(tmp_path):
    p = tmp_path / "out.csv"
    save_catalog(str(p), ITEMS)
    text = p.read_bytes().decode("utf-8-sig")  # must be UTF-8 (a BOM is fine)
    assert ITEMS[1]["name"] in text
    assert load_catalog(str(p)) == ITEMS


def run_in_c_locale(tmp_path, body):
    """Run `body` in a fresh Python whose locale encoding is ASCII (like the old billing PC)."""
    module_dir = os.path.dirname(os.path.abspath(catalog.__file__))
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LC_", "PYTHON")) and k != "LANG"}
    env.update({"LC_ALL": "C", "LANG": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
                "PYTHONPATH": module_dir, "PYTHONDONTWRITEBYTECODE": "1"})
    proc = subprocess.run([sys.executable, str(script), str(tmp_path)], cwd=module_dir, env=env,
                          capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return json.loads(proc.stdout.decode("ascii"))


def test_save_and_load_on_a_non_utf8_computer(tmp_path):
    (tmp_path / "items.json").write_text(json.dumps(ITEMS), encoding="ascii")
    back = run_in_c_locale(tmp_path, """
        import json, os, sys
        import catalog
        folder = sys.argv[1]
        with open(os.path.join(folder, "items.json"), encoding="ascii") as f:
            items = json.load(f)
        catalog.save_catalog(os.path.join(folder, "c.csv"), items)
        print(json.dumps(catalog.load_catalog(os.path.join(folder, "c.csv"))))
    """)
    assert back == ITEMS
    assert (tmp_path / "c.csv").read_bytes().decode("utf-8-sig").count("\n") == len(ITEMS) + 1


def test_load_excel_file_on_a_non_utf8_computer(tmp_path):
    write_csv(tmp_path / "excel.csv", ITEMS, "utf-8-sig", newline="\r\n")
    back = run_in_c_locale(tmp_path, """
        import json, os, sys
        import catalog
        print(json.dumps(catalog.load_catalog(os.path.join(sys.argv[1], "excel.csv"))))
    """)
    assert back == ITEMS


@pytest.mark.parametrize("query", [JEERA_DECOMPOSED, JEERA_PRECOMPOSED, " " + JEERA_DECOMPOSED + " "])
def test_search_finds_same_word_typed_differently(query):
    assert [i["sku"] for i in search(ITEMS, query)] == ["JEERA-100"]


def test_search_decomposed_catalog_name():
    items = [{"sku": "J", "name": JEERA_DECOMPOSED, "price_paise": 1}]
    assert search(items, JEERA_PRECOMPOSED) == items


def test_search_english_still_case_insensitive():
    assert [i["sku"] for i in search(ITEMS, "TATA salt")] == ["TATA-SALT-1KG"]
    assert search(ITEMS, "basmati") == []


def check_label(name, max_bytes):
    label = receipt_label(name, max_bytes)
    assert isinstance(label, str)
    assert len(label.encode("utf-8")) <= max_bytes
    assert name.startswith(label)
    if label != name:
        assert not unicodedata.category(name[len(label)]).startswith("M"), (
            f"cut {name!r} at {len(label)}: the next character is a combining mark (matra)"
        )
    return label


def test_label_short_names_unchanged():
    assert receipt_label("Tata Salt 1kg") == "Tata Salt 1kg"
    assert receipt_label(ITEMS[3]["name"], 24) == ITEMS[3]["name"]


def test_label_english_cut_by_bytes():
    assert receipt_label("Fortune Sunflower Oil 1 Litre Pouch", 24) == "Fortune Sunflower Oil 1 "
    assert receipt_label("Tata Salt", 4) == "Tata"


def test_label_never_crashes_and_fits_for_every_length():
    for item in ITEMS:
        for max_bytes in range(0, 40):
            check_label(item["name"], max_bytes)


def test_label_cuts_between_letters():
    cheeni = "चीनी 1kg"  # चीनी 1kg
    assert check_label(cheeni, 7) == "ची"  # "ची": no half-encoded letter, no crash


def test_label_keeps_matra_with_its_letter():
    atta = "आटा"  # आटा: the last 3 bytes are the matra of ट
    assert check_label(atta, 7) == "आ"
    assert check_label(atta, 9) == atta

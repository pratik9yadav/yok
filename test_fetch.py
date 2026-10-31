"""Checks the real-data fetch helpers without any network.  Run:  python -m pytest"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fetch_real_data import (
    amazon_products,
    find_licences,
    flipkart_products,
    licence_ok,
    safe_text,
    shop_talk_score,
    short_name,
    summarise_files,
)


def test_licence_guard_keeps_only_open_licences() -> None:
    for open_ in (["CC-BY-SA-4.0"], ["CC-BY-4.0"], ["ODC-BY-1.0"], ["ODbL-1.0"], ["CC0-1.0"]):
        assert licence_ok(open_), open_
    for closed in (["CC-BY-NC-SA-4.0"], ["CC-BY-NC-4.0"], ["CC-BY-ND-4.0"], ["other"], ["unknown"], []):
        assert not licence_ok(closed), closed


def test_flipkart_keeps_shop_categories_with_mrp_and_price() -> None:
    rows = [
        {"product_category_tree": '["Tools & Hardware >> Tools >> Hand Tools >> X"]', "product_name": "Taparia 6 inch Plier",
         "retail_price": "345", "discounted_price": "299", "brand": "Taparia"},
        {"product_category_tree": '["Clothing >> Men"]', "product_name": "Shirt", "retail_price": "999",
         "discounted_price": "499", "brand": "X"},
        {"product_category_tree": '["Automotive >> Spares"]', "product_name": "No price", "retail_price": "",
         "discounted_price": "", "brand": ""},
    ]
    assert flipkart_products(rows) == [{"name": "Taparia 6 inch Plier", "brand": "Taparia", "mrp": 345, "price": 299,
                                        "category": "Tools & Hardware > Tools > Hand Tools"}]


def test_amazon_keeps_shop_categories_most_bought_first() -> None:
    rows = [
        {"categoryName": "Car Parts", "title": "Bosch spark plug", "price": "199.0", "listPrice": "250", "reviews": "10"},
        {"categoryName": "Power & Hand Tools", "title": "Drill", "price": "1,499", "listPrice": "0", "reviews": "900"},
        {"categoryName": "Sarees", "title": "Silk saree", "price": "999", "listPrice": "1999", "reviews": "5000"},
        {"categoryName": "Skin care", "title": "Face cream", "price": "299", "listPrice": "399", "reviews": "9000"},
        {"categoryName": "SnS_Baby_Bath&SkinCare", "title": "Baby soap", "price": "99", "listPrice": "0", "reviews": "50"},
    ]
    got = amazon_products(rows)
    assert [p["name"] for p in got] == ["Drill", "Bosch spark plug"]
    assert got[0]["mrp"] is None and got[0]["price"] == 1499 and got[1]["mrp"] == 250


def test_amazon_hindi_categories_and_per_category_cap() -> None:
    rows = [{"categoryName": "कार और मोटरबाइक", "title": f"Bike part {i}", "price": "100", "listPrice": "120",
             "reviews": str(i)} for i in range(5)]
    rows += [{"categoryName": "इनडोर लाइटिंग", "title": "LED bulb 9W", "price": "99", "listPrice": "150", "reviews": "1"},
             {"categoryName": "त्वचा की देखभाल", "title": "Face wash", "price": "99", "listPrice": "0", "reviews": "99"}]
    got = amazon_products(rows, per_category=3)
    assert [p["name"] for p in got] == ["Bike part 4", "Bike part 3", "Bike part 2", "LED bulb 9W"]


def test_shop_talk_miner_and_safety_filter() -> None:
    assert shop_talk_score("हाँ है, डेढ़ सौ रुपये का पड़ेगा, घर पे भेज देंगे") > 0
    assert shop_talk_score("दुकानात या, किलोचा भाव शंभर रुपये आहे") > 0
    assert shop_talk_score("मेरा नाम राम है और मैं पुणे से हूँ") == 0
    assert not safe_text("मेरा नंबर 9876543210 है रेट सौ रुपये")  # phone-number-like
    assert not safe_text("mail me at a@b.com rate 100")
    assert safe_text("हाँ जी, माल है, सौ रुपये का एक")


def test_short_name() -> None:
    assert short_name("  Havells   Lifeline  1.5 sq mm   wire  ") == "Havells Lifeline 1.5 sq mm wire"
    assert len(short_name("word " * 40)) <= 80


def test_licences_are_found_where_kaggle_puts_them() -> None:
    assert find_licences({"licenses": [{"name": "CC0-1.0"}]}) == ["CC0-1.0"]
    assert find_licences({"info": {"title": "x", "licenses": [{"name": "CC-BY-SA-4.0"}]}}) == ["CC-BY-SA-4.0"]
    assert find_licences({"info": {"title": "x"}}) == []


def test_real_examples_are_well_formed_and_never_in_the_test_sets() -> None:
    import json
    import re

    here = Path(__file__).parent / "data"
    rows = [json.loads(line) for line in (here / "real" / "xpqa_hi_examples.jsonl").read_text(encoding="utf-8").splitlines()]
    held_out = {json.loads(line)["said"] for name in ("test.jsonl", "hard_test.jsonl")
                for line in (here / name).read_text(encoding="utf-8").splitlines()}
    assert len(rows) == 151 and len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        a = r["answer"]
        assert set(a) == {"in_stock", "price_inr", "delivery", "alternative"}
        assert a["in_stock"] in (True, False, None) and a["delivery"] in (True, False, None)
        assert a["price_inr"] is None or (isinstance(a["price_inr"], int) and a["in_stock"] is True)
        assert r["said"] not in held_out
        assert not re.search(r"[0-9०-९]{6,}|@|https?://", r["said"])


def test_summarise_files_counts_folders_and_types() -> None:
    files = ["README.md", "data/hindi/train-0.parquet", "data/hindi/train-1.parquet", "data/marathi/test.parquet"]
    s = summarise_files(files)
    assert s["files"] == 4
    assert s["folders"] == {"data/hindi": 2, ".": 1, "data/marathi": 1}
    assert s["types"] == {".parquet": 3, ".md": 1}

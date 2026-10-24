import copy

import pytest

from catalog import paginate, search_products

ITEMS = [f"p{i}" for i in range(1, 46)]  # 45 items


def test_first_page_starts_at_first_item():
    r = paginate(ITEMS, 1)
    assert r["items"] == ITEMS[:20]
    assert r["page"] == 1


def test_middle_page():
    assert paginate(ITEMS, 2)["items"] == ITEMS[20:40]


def test_last_partial_page_is_shown():
    r = paginate(ITEMS, 3)
    assert r["items"] == ITEMS[40:]
    assert r["total_pages"] == 3
    assert r["has_next"] is False
    assert paginate(ITEMS, 2)["has_next"] is True


def test_every_item_appears_exactly_once():
    seen = []
    for per_page in (1, 7, 20, 44, 45, 46):
        seen = []
        page = 1
        while True:
            r = paginate(ITEMS, page, per_page)
            seen.extend(r["items"])
            if not r["has_next"]:
                break
            page += 1
        assert seen == ITEMS, per_page
        assert r["total_pages"] == page


def test_exact_multiple():
    items = list(range(40))
    assert paginate(items, 2)["total_pages"] == 2
    assert paginate(items, 2)["has_next"] is False
    assert paginate(items, 2)["items"] == list(range(20, 40))


def test_empty_list():
    r = paginate([], 1)
    assert r["items"] == []
    assert r["total_pages"] == 0
    assert r["has_next"] is False


def test_page_past_the_end_is_empty():
    r = paginate(ITEMS, 9)
    assert r["items"] == []
    assert r["has_next"] is False


@pytest.mark.parametrize("page", [0, -1])
def test_bad_page_raises(page):
    with pytest.raises(ValueError):
        paginate(ITEMS, page)


@pytest.mark.parametrize("per_page", [0, -5])
def test_bad_page_size_raises(per_page):
    with pytest.raises(ValueError):
        paginate(ITEMS, 1, per_page)


def test_input_not_modified():
    items = list(ITEMS)
    r = paginate(items, 1)
    r["items"].append("x")
    assert items == ITEMS


PRODUCTS = [
    {"name": "Aashirvaad Atta 5kg"},
    {"name": "Tata Salt 1kg"},
    {"name": "Fortune Chakki ATTA"},
    {"name": "Amul Butter"},
]


@pytest.mark.parametrize("query", ["atta", "Atta", "ATTA", "  atta "])
def test_search_ignores_case(query):
    r = search_products(PRODUCTS, query)
    assert [p["name"] for p in r["items"]] == ["Aashirvaad Atta 5kg", "Fortune Chakki ATTA"]


def test_search_pages():
    products = [{"name": f"Parle-G {i}"} for i in range(25)] + [{"name": "Other"}]
    before = copy.deepcopy(products)
    r1 = search_products(products, "parle", page=1)
    r2 = search_products(products, "parle", page=2)
    assert len(r1["items"]) == 20 and len(r2["items"]) == 5
    assert r1["total_pages"] == 2
    assert products == before


def test_search_no_match():
    r = search_products(PRODUCTS, "ghee")
    assert r["items"] == [] and r["total_pages"] == 0

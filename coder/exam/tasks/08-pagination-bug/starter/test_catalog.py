from catalog import paginate


def test_page_has_requested_size():
    items = list(range(100))
    assert len(paginate(items, 2)["items"]) == 20

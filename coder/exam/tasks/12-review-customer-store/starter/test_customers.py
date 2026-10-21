from customers import CustomerStore


def test_add_and_get(tmp_path):
    store = CustomerStore(tmp_path / "shop.db")
    cid = store.add("Ravi", "9876543210", "Pune")
    assert store.get_by_phone("9876543210")["id"] == cid

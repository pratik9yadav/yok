import contextlib
import importlib
import os
import urllib.parse
from pathlib import Path

import pytest

import customers

SECRET = "sk_live_51HxQe2kL9vR7tYb3"


@pytest.fixture
def store(tmp_path):
    return customers.CustomerStore(str(tmp_path / "shop.db"))


def test_add_and_get_round_trip(store):
    cid = store.add("Ravi", "9876543210", "Pune")
    assert isinstance(cid, int)
    assert store.get_by_phone("9876543210") == {"id": cid, "name": "Ravi", "phone": "9876543210", "city": "Pune"}


def test_missing_city_stays_empty(store):
    store.add("Meena", "9000000001")
    assert store.get_by_phone("9000000001")["city"] is None


def test_names_with_quotes_work(store):
    cid = store.add("Joseph D'Souza", "9000000002", "Panaji")
    assert cid is not None
    assert store.get_by_phone("9000000002")["name"] == "Joseph D'Souza"
    assert [c["name"] for c in store.search("d'souza")] == ["Joseph D'Souza"]


def test_search_cannot_be_injected(store):
    store.add("Ravi", "9876543210")
    store.add("Asha", "9876543211")
    assert store.search("' OR '1'='1") == []
    assert store.search("%' OR 1=1 --") == []
    assert store.get_by_phone("' OR '1'='1") is None
    assert store.get_by_phone("x' OR phone LIKE '%") is None
    with contextlib.suppress(Exception):
        store.search("x'; DROP TABLE customers; --")
    assert len(store.search("")) == 2  # table is still there


def test_search_ignores_case_and_sorts(store):
    store.add("ravi kumar", "1")
    store.add("Asha", "2")
    store.add("Ravindra", "3")
    assert [c["name"] for c in store.search("RAVI")] == ["Ravindra", "ravi kumar"]


def test_duplicate_phone_raises(store):
    store.add("Ravi", "9876543210")
    with pytest.raises(customers.DuplicateCustomer):
        store.add("Someone Else", "9876543210")
    assert store.get_by_phone("9876543210")["name"] == "Ravi"


def test_other_errors_are_not_hidden(store):
    with pytest.raises(Exception) as info:
        store.add(None, "9000000003")
    assert not isinstance(info.value, customers.DuplicateCustomer)
    assert store.get_by_phone("9000000003") is None


def test_import_many_success(tmp_path):
    db = str(tmp_path / "shop.db")
    s = customers.CustomerStore(db)
    n = s.import_many([{"name": "A", "phone": "1"}, {"name": "B", "phone": "2", "city": "Nagpur"}])
    assert n == 2
    reopened = customers.CustomerStore(db)
    assert reopened.get_by_phone("2")["city"] == "Nagpur"


def test_import_many_is_all_or_nothing(store):
    store.add("Existing", "5")
    with pytest.raises(customers.DuplicateCustomer):
        store.import_many([{"name": "A", "phone": "1"}, {"name": "B", "phone": "5"}, {"name": "C", "phone": "3"}])
    assert store.get_by_phone("1") is None
    assert store.get_by_phone("3") is None
    with pytest.raises(customers.DuplicateCustomer):
        store.import_many([{"name": "A", "phone": "7"}, {"name": "A again", "phone": "7"}])
    assert store.get_by_phone("7") is None
    assert [c["phone"] for c in store.search("")] == ["5"]


def test_data_is_saved(tmp_path):
    db = str(tmp_path / "shop.db")
    customers.CustomerStore(db).add("Ravi", "9876543210")
    assert customers.CustomerStore(db).get_by_phone("9876543210")["name"] == "Ravi"


def _source_files():
    root = Path.cwd()
    this_test = Path(__file__).resolve()
    for path in root.rglob("*"):
        parts = set(path.relative_to(root).parts)
        if path.is_file() and path.resolve() != this_test and not parts & {"hidden", ".git", "__pycache__"}:
            yield path


def test_no_api_key_in_the_code():
    for path in _source_files():
        assert SECRET not in path.read_text(encoding="utf-8", errors="ignore"), path


class FakeOpener:
    def __init__(self):
        self.calls = []

    def __call__(self, request, *args, **kwargs):
        if isinstance(request, str):
            url, data = request, None
        else:
            url, data = request.full_url, request.data
        params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        if data:
            params.update(urllib.parse.parse_qs(data.decode() if isinstance(data, bytes) else data))
        self.calls.append(params)
        return "sent"


def test_sms_uses_key_from_environment_and_encodes_text(monkeypatch):
    monkeypatch.setenv("SMS_API_KEY", "test-key-123")
    mod = importlib.reload(customers)
    opener = FakeOpener()
    mod.send_welcome_sms("+919876543210", "Ravi & Sons #1", opener=opener)
    assert len(opener.calls) == 1
    params = opener.calls[0]
    assert params["key"] == ["test-key-123"]
    assert params["to"] == ["+919876543210"]
    assert "Ravi & Sons #1" in params["msg"][0]


def test_sms_without_key_is_not_sent(monkeypatch):
    monkeypatch.delenv("SMS_API_KEY", raising=False)
    mod = importlib.reload(customers)
    opener = FakeOpener()
    with contextlib.suppress(Exception):
        mod.send_welcome_sms("9876543210", "Ravi", opener=opener)
    assert opener.calls == [] or all(p.get("key", [""])[0] not in ("", "None", SECRET) for p in opener.calls)
    assert "SMS_API_KEY" not in os.environ

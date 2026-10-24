import http.client
import json
import threading

import pytest

from order_api import OrderAPI, make_server


def body(obj):
    return json.dumps(obj).encode("utf-8")


def create(api, item="Toor dal 1kg", qty=2, price_paise=14500):
    return api.handle("POST", "/orders", body({"item": item, "qty": qty, "price_paise": price_paise}))


def test_create_returns_201_with_id_and_total():
    api = OrderAPI()
    status, resp = create(api, qty=3, price_paise=4250)
    assert status == 201
    assert resp["qty"] == 3 and resp["price_paise"] == 4250
    assert resp["total_paise"] == 12750
    assert isinstance(resp["id"], int)
    json.dumps(resp)  # must be JSON-ready


def test_get_one_and_list_oldest_first():
    api = OrderAPI()
    _, a = create(api, item="Atta 5kg")
    _, b = create(api, item="Sugar 1kg")
    status, got = api.handle("GET", f"/orders/{b['id']}")
    assert status == 200 and got["item"] == "Sugar 1kg"
    status, listing = api.handle("GET", "/orders")
    assert status == 200
    assert [o["id"] for o in listing["orders"]] == [a["id"], b["id"]]
    assert a["id"] != b["id"]


def test_empty_list():
    status, listing = OrderAPI().handle("GET", "/orders")
    assert status == 200 and listing == {"orders": []}


def test_unknown_order_is_404():
    api = OrderAPI()
    create(api)
    status, resp = api.handle("GET", "/orders/999")
    assert status == 404 and "error" in resp


@pytest.mark.parametrize("path", ["/orders/abc", "/orders/1/extra", "/nothing", "/"])
def test_bad_paths_are_404_not_crash(path):
    api = OrderAPI()
    create(api)
    status, resp = api.handle("GET", path)
    assert status == 404
    assert "error" in resp


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{not json",
        b"[1, 2, 3]",
        b'"just a string"',
        b"\xff\xfe\x00garbage",
    ],
)
def test_unparseable_body_is_400(raw):
    status, resp = OrderAPI().handle("POST", "/orders", raw)
    assert status == 400
    assert isinstance(resp.get("error"), str) and resp["error"]


@pytest.mark.parametrize(
    "payload",
    [
        {"qty": 1, "price_paise": 100},
        {"item": "", "qty": 1, "price_paise": 100},
        {"item": "   ", "qty": 1, "price_paise": 100},
        {"item": 42, "qty": 1, "price_paise": 100},
        {"item": "Rice", "price_paise": 100},
        {"item": "Rice", "qty": 0, "price_paise": 100},
        {"item": "Rice", "qty": -3, "price_paise": 100},
        {"item": "Rice", "qty": 2.5, "price_paise": 100},
        {"item": "Rice", "qty": "2", "price_paise": 100},
        {"item": "Rice", "qty": True, "price_paise": 100},
        {"item": "Rice", "qty": None, "price_paise": 100},
        {"item": "Rice", "qty": 1},
        {"item": "Rice", "qty": 1, "price_paise": -1},
        {"item": "Rice", "qty": 1, "price_paise": 99.5},
    ],
)
def test_invalid_order_is_400(payload):
    api = OrderAPI()
    status, resp = api.handle("POST", "/orders", body(payload))
    assert status == 400
    assert isinstance(resp.get("error"), str) and resp["error"]
    # nothing was stored
    assert api.handle("GET", "/orders")[1] == {"orders": []}


def test_free_item_price_zero_is_allowed():
    status, resp = create(OrderAPI(), item="Free carry bag", qty=1, price_paise=0)
    assert status == 201 and resp["total_paise"] == 0


@pytest.mark.parametrize("method,path", [("DELETE", "/orders"), ("PUT", "/orders"), ("POST", "/orders/1")])
def test_wrong_method_is_405(method, path):
    api = OrderAPI()
    create(api)
    status, resp = api.handle(method, path, b"")
    assert status == 405
    assert "error" in resp


def test_returned_order_cannot_corrupt_store():
    api = OrderAPI()
    _, order = create(api, qty=2, price_paise=100)
    order["total_paise"] = 1
    _, listing = api.handle("GET", "/orders")
    listing["orders"].clear()
    status, again = api.handle("GET", f"/orders/{order['id']}")
    assert status == 200 and again["total_paise"] == 200


def test_concurrent_creates_get_unique_ids():
    api = OrderAPI()
    threads_n, per_thread = 8, 50
    barrier = threading.Barrier(threads_n)
    ids = []
    ids_lock = threading.Lock()
    errors = []

    def worker(n):
        try:
            barrier.wait(timeout=5)
            for i in range(per_thread):
                status, resp = create(api, item=f"item-{n}-{i}", qty=1, price_paise=1)
                assert status == 201
                with ids_lock:
                    ids.append(resp["id"])
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    assert len(ids) == threads_n * per_thread
    assert len(set(ids)) == len(ids)
    assert len(api.handle("GET", "/orders")[1]["orders"]) == len(ids)


def test_real_server_roundtrip():
    api = OrderAPI()
    server = make_server(api, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        conn = http.client.HTTPConnection(host, port, timeout=5)
        payload = body({"item": "Haldi 200g", "qty": 2, "price_paise": 3000})
        conn.request("POST", "/orders", body=payload, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        assert resp.status == 201
        assert "application/json" in resp.getheader("Content-Type", "")
        assert data["total_paise"] == 6000

        conn.request("POST", "/orders", body=b"{oops", headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        assert "error" in json.loads(resp.read())

        conn.request("GET", "/orders/12345")
        resp = conn.getresponse()
        assert resp.status == 404
        resp.read()
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

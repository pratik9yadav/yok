Our shop's mobile app needs a small JSON API for orders, and we can't add any dependencies, so standard library only.
Please write `order_api.py` with a class `OrderAPI` whose method `handle(method: str, path: str, body: bytes = b"") -> tuple[int, dict]` does the routing and returns the HTTP status code and the JSON-ready response body, so we can test it without opening a socket.
Also add `make_server(api: OrderAPI, host: str = "127.0.0.1", port: int = 8000)` that returns an `http.server.ThreadingHTTPServer` serving that API (we call `serve_forever()` ourselves).
Routes: `POST /orders` with `{"item": str, "qty": int, "price_paise": int}` creates an order and returns it with a new `id` and `total_paise`; `GET /orders/<id>` returns one order; `GET /orders` returns `{"orders": [...]}`, oldest first.
Quantity is a whole number of at least 1, and the price can't be negative.
The app team expects the usual status codes: 201 on create, 200 for reads, 400 when the request is bad (with an `"error"` message in the body), 404 for an unknown order or path, 405 for a wrong method.
Several cashiers' phones hit this at the same time.

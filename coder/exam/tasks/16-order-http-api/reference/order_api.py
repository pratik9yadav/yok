"""A tiny JSON API for shop orders, standard library only.

`OrderAPI.handle` does all routing and validation and can be tested without a socket;
`make_server` wraps it in a `ThreadingHTTPServer`.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY_BYTES = 64 * 1024
MAX_ITEM_LENGTH = 200
_ORDER_PATH = re.compile(r"/orders/([0-9]+)")


class BadRequest(Exception):
    """The client sent something we cannot accept (HTTP 400)."""


def _require_int(data: dict, field: str, minimum: int) -> int:
    if field not in data:
        raise BadRequest(f"missing field: {field}")
    value = data[field]
    # bool is a subclass of int; true/false is never a valid quantity or price.
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequest(f"{field} must be a whole number")
    if value < minimum:
        raise BadRequest(f"{field} must be at least {minimum}")
    return value


def _parse_order(body: bytes) -> dict:
    if not body:
        raise BadRequest("request body is empty")
    if len(body) > MAX_BODY_BYTES:
        raise BadRequest("request body is too large")
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise BadRequest("request body is not valid JSON") from None
    if not isinstance(data, dict):
        raise BadRequest("request body must be a JSON object")
    item = data.get("item")
    if not isinstance(item, str) or not item.strip():
        raise BadRequest("item must be a non-empty string")
    if len(item) > MAX_ITEM_LENGTH:
        raise BadRequest("item is too long")
    qty = _require_int(data, "qty", 1)
    price = _require_int(data, "price_paise", 0)
    return {"item": item.strip(), "qty": qty, "price_paise": price}


class OrderAPI:
    """In-memory order store with JSON-style request handling. Safe to share between threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._orders: dict[int, dict] = {}
        self._next_id = 1

    def handle(self, method: str, path: str, body: bytes = b"") -> tuple[int, dict]:
        method = (method or "").upper()
        path = (path or "").split("?", 1)[0]
        if len(path) > 1:
            path = path.rstrip("/")
        try:
            if path == "/orders":
                return self._collection(method, body or b"")
            if path.startswith("/orders/"):
                return self._one(method, path)
        except BadRequest as exc:
            return 400, {"error": str(exc)}
        return 404, {"error": "not found"}

    def _collection(self, method: str, body: bytes) -> tuple[int, dict]:
        if method == "GET":
            return 200, {"orders": self._list()}
        if method == "POST":
            return 201, self._create(_parse_order(body))
        return 405, {"error": "method not allowed"}

    def _one(self, method: str, path: str) -> tuple[int, dict]:
        if method != "GET":
            return 405, {"error": "method not allowed"}
        match = _ORDER_PATH.fullmatch(path)
        order = self._get(int(match.group(1))) if match else None
        if order is None:
            return 404, {"error": "order not found"}
        return 200, order

    def _create(self, fields: dict) -> dict:
        with self._lock:
            order_id = self._next_id
            self._next_id += 1
            order = {"id": order_id, **fields, "total_paise": fields["qty"] * fields["price_paise"]}
            self._orders[order_id] = order
            return dict(order)

    def _get(self, order_id: int) -> dict | None:
        with self._lock:
            order = self._orders.get(order_id)
            return dict(order) if order is not None else None

    def _list(self) -> list[dict]:
        with self._lock:
            return [dict(self._orders[k]) for k in sorted(self._orders)]


def make_server(api: OrderAPI, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Return a threading HTTP server that serves `api`. The caller runs `serve_forever()`."""

    class Handler(BaseHTTPRequestHandler):
        def _dispatch(self) -> None:
            body = b""
            if self.command in ("POST", "PUT", "PATCH"):
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    self._send(400, {"error": "bad Content-Length"})
                    return
                if length < 0:
                    self._send(400, {"error": "bad Content-Length"})
                    return
                if length > MAX_BODY_BYTES:
                    self._send(413, {"error": "request body is too large"})
                    return
                body = self.rfile.read(length)
            status, payload = api.handle(self.command, self.path, body)
            self._send(status, payload)

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _dispatch

        def log_message(self, *args) -> None:
            pass  # keep test and server output quiet

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server

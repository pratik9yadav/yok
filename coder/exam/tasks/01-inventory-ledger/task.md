I run a small kirana shop and want to stop tracking stock in a notebook. Please write `inventory.py` with a class `Ledger`.
`receive(sku, qty, move_id=None)` adds stock, `issue(sku, qty, move_id=None)` takes stock out, `stock(sku)` returns the current quantity (0 for a product we have never seen), and `history(sku)` returns that product's moves, oldest first, as dicts with `move_id`, `kind` (`"receive"` or `"issue"`) and `qty`.
Quantities are whole units, so raise `ValueError` for an invalid quantity.
If there isn't enough stock for an issue, raise `InsufficientStock` (define it in the module) and leave everything as it was.
Our billing app retries when the network is bad, so the same `move_id` can arrive more than once; it must only count once.

Our price lookups hit the supplier API too often. Please write `ttlcache.py` with `TTLCache(maxsize, ttl, clock=time.monotonic)`: a cache of at most `maxsize` live entries, where each entry expires `ttl` seconds after it was last set (an entry set at time t is gone at exactly t + ttl).
It needs `get(key, default=None)`, `set(key, value)`, `delete(key)` (deleting a key that isn't there is fine), `len(cache)` for the number of live entries and `key in cache`.
When it is full, drop the least recently used entry (reading or setting a key counts as using it).
Raise `ValueError` if `maxsize` or `ttl` is not a positive number. The `clock` argument lets tests control time.

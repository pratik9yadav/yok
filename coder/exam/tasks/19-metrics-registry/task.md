We want basic metrics for the shop server without pulling in prometheus_client, so please write `metrics.py` with a `MetricsRegistry` class.
`counter(name)` returns a counter (the same object every time for the same name) with `inc(amount=1)` and `.value`; counters only ever go up.
`gauge(name)` returns a gauge with `set(value)`, `inc(amount=1)`, `dec(amount=1)` and `.value`.
`snapshot()` returns a dict `{name: value}` of everything, and `render()` returns text with one `name value` line per metric, sorted by name, for our monitoring scraper.
Names look like `orders_placed_total`: lowercase letters, digits and underscores, starting with a letter. Raise `ValueError` for any other name, and also when a name that is already a counter is asked for as a gauge or the other way round.
Every request thread updates these at the same time, and the scraper calls `render()` while that is happening.

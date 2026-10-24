Our staff keep orders in a spreadsheet and export them as CSV; I want to load those files into the new system. Please write `order_import.py` with `import_orders(path)` that returns a tuple `(orders, errors)`.
The CSV has a header row with the columns `order_id`, `customer`, `sku`, `qty` and `price` (the order of columns can change and extra columns can be ignored). Each good row becomes a dict with exactly those five keys: `qty` as an int, `price` as a `Decimal`, the others as strings.
A bad row must not stop the import: add `(line_number, reason)` to `errors` (the header is line 1) and carry on with the next row. An `order_id` that already appeared earlier in the file counts as a bad row too.
If the file doesn't have the required columns at all, raise `ValueError`.

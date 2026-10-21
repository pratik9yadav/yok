Our shop server dies every few weeks with `OSError: [Errno 24] Too many open files`, and we restart it.
There's a second problem that may be related: the evening total for today from `SalesLog.total()` is often lower than the bills we printed, and after a restart the same day's total is right again.
The code is in `saleslog.py`. Please find and fix the cause; the server keeps calling `record(day, order_id, amount_paise)`, `total(day)` and `close()` exactly as it does now, and the CSV files on disk must stay in the same format.

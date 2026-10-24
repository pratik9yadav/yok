`make_invoice(order)` in `invoice.py` prints the bills at our counter. It works, but every change to it breaks something, so please restructure it into something we can maintain.
We also need the new mobile app to show the same numbers without the text, so add `invoice_totals(order) -> dict` with `subtotal_paise`, `discount_paise`, `round_off_paise` and `total_paise` (whole paise). These must always agree with the printed bill.
The printed bill must stay exactly the same, character for character: our printer templates and the tax auditor's checks depend on it.

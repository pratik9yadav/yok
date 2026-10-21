Bug report from the Diwali sale: we had 1 pressure cooker left and two tills sold it at the same moment. Both bills went through, the stock shows 0, and one customer went home without a cooker.
The same evening two staff members each restocked 5 boxes of diyas at the same time, and the stock only went up by 5.
All tills share one `StockBook` in `stock.py` inside the shop server, one thread per request, and every change is written to the ledger, which is a slow write to disk.
Please fix `stock.py` so this can't happen again; the tills and the ledger keep using it exactly as they do now (`reserve`, `release`, `restock`, `available`, `OutOfStock`).

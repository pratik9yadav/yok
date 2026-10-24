We need the bill totals for our shop counter. Please write `gst.py` with `invoice_totals(lines)`.
Each line is a dict with `price` (unit price before tax in rupees; it can come as a string like `"199.50"`, an int or a `Decimal`), `qty` (int) and `gst_rate` (percent, one of 0, 5, 12, 18 or 28).
It returns a dict with `subtotal`, `gst`, `total` and `payable`, all `Decimal`. GST is worked out per line and rounded to the paisa (half up, so 0.005 goes up), `total` is subtotal + GST, and `payable` is the total rounded to the nearest rupee (also half up), like on our printed bills.
Raise `ValueError` for an unknown GST rate, a bad quantity or a bad price.

def make_invoice(order):
    # order = {"invoice_no": "...", "date": "2024-11-05", "customer": "...", "items": [{"name", "qty", "price_paise"}],
    #          "discount_pct": 10}   (customer and discount_pct are optional)
    if len(order["items"]) == 0:
        raise ValueError("no items")
    s = ""
    s = s + "SHARMA KIRANA STORE\n"
    s = s + "Invoice: " + order["invoice_no"] + "   Date: " + order["date"] + "\n"
    if "customer" in order and order["customer"]:
        s = s + "Customer: " + order["customer"] + "\n"
    else:
        s = s + "Customer: Cash\n"
    s = s + "-" * 45 + "\n"
    t = 0
    for i in order["items"]:
        a = i["qty"] * i["price_paise"]
        t = t + a
        n = i["name"]
        if len(n) > 20:
            n = n[:19] + "~"
        s = s + n.ljust(20) + str(i["qty"]).rjust(4) + " x " + ("%.2f" % (i["price_paise"] / 100)).rjust(8)
        s = s + ("%.2f" % (a / 100)).rjust(10) + "\n"
    s = s + "-" * 45 + "\n"
    s = s + "Subtotal".ljust(35) + ("%.2f" % (t / 100)).rjust(10) + "\n"
    d = 0
    if "discount_pct" in order:
        if order["discount_pct"]:
            d = int(t * order["discount_pct"] / 100)
            s = s + ("Discount (" + str(order["discount_pct"]) + "%)").ljust(35) + ("-%.2f" % (d / 100)).rjust(10) + "\n"
    net = t - d
    r = round(net / 100) * 100 - net
    if r != 0:
        s = s + "Round off".ljust(35) + ("%+.2f" % (r / 100)).rjust(10) + "\n"
    s = s + "TOTAL".ljust(35) + ("%.2f" % ((net + r) / 100)).rjust(10) + "\n"
    s = s + "-" * 45 + "\n"
    s = s + "Thank you! Visit again.\n"
    return s

from decimal import Decimal


def build_report(sales):
    # sales = list of dicts: order_id, date ("YYYY-MM-DD"), sku, qty (int), amount (Decimal)
    s = "Sales report\n"
    s = s + "============\n"
    if len(sales) == 0:
        s = s + "No sales.\n"
        return s
    days = []
    for x in sales:
        if x["date"] not in days:
            days.append(x["date"])
    days.sort()
    tot_orders = set()
    tot_items = 0
    tot_amount = Decimal("0")
    for d in days:
        o = set()
        i = 0
        a = Decimal("0")
        for x in sales:
            if x["date"] == d:
                o.add(x["order_id"])
                i = i + x["qty"]
                a = a + x["amount"]
        s = s + d + "  orders: " + str(len(o)) + "  items: " + str(i) + "  amount: " + str(a.quantize(Decimal("0.01"))) + "\n"
        for oo in o:
            tot_orders.add(oo)
        tot_items = tot_items + i
        tot_amount = tot_amount + a
    s = s + "Total  orders: " + str(len(tot_orders)) + "  items: " + str(tot_items) + "  amount: " + str(tot_amount.quantize(Decimal("0.01"))) + "\n"
    prods = {}
    for x in sales:
        if x["sku"] in prods:
            prods[x["sku"]][0] = prods[x["sku"]][0] + x["qty"]
            prods[x["sku"]][1] = prods[x["sku"]][1] + x["amount"]
        else:
            prods[x["sku"]] = [x["qty"], x["amount"]]
    l = []
    for k in prods:
        l.append((k, prods[k][0], prods[k][1]))
    l.sort(key=lambda t: (-t[1], t[0]))
    s = s + "\nTop products:\n"
    n = 1
    for t in l[:3]:
        s = s + "  " + str(n) + ". " + t[0] + "  " + str(t[1]) + " items  " + str(t[2].quantize(Decimal("0.01"))) + "\n"
        n = n + 1
    return s

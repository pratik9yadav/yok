import copy
import random

import pytest

from invoice import invoice_totals, make_invoice


def legacy_make_invoice(order):
    """The original implementation, copied verbatim as the oracle."""
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


def legacy_totals(order):
    """The numbers the original bill prints, recomputed the original way."""
    t = sum(i["qty"] * i["price_paise"] for i in order["items"])
    d = int(t * order["discount_pct"] / 100) if order.get("discount_pct") else 0
    net = t - d
    r = round(net / 100) * 100 - net
    return {"subtotal_paise": t, "discount_paise": d, "round_off_paise": r, "total_paise": net + r}


NAMES = ["Atta 5kg", "Toor Dal 1kg", "Fortune Sunflower Oil 1 Litre Pouch", "Salt", "Maggi Masala Noodles 12 pack",
         "Haldi 200g", "Parle-G", "Amul Butter 500g", "Basmati Rice Premium 5kg", "Eggs (tray of 30)"]


def random_order(rng, n):
    order = {
        "invoice_no": f"INV-{n:05d}",
        "date": f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "items": [
            {"name": rng.choice(NAMES), "qty": rng.randint(1, 12), "price_paise": rng.randint(1, 250_000)}
            for _ in range(rng.randint(1, 8))
        ],
    }
    if rng.random() < 0.7:
        order["customer"] = rng.choice(["Mrs. Gupta", "Ramesh", "", "Hotel Annapurna (credit)"])
    if rng.random() < 0.6:
        order["discount_pct"] = rng.choice([0, 5, 10, 12.5, 2.5, 7, 15, 33.3])
    return order


EXAMPLES = [
    {"invoice_no": "INV-1", "date": "2024-11-05", "items": [{"name": "Atta 5kg", "qty": 1, "price_paise": 24000}]},
    {"invoice_no": "INV-2", "date": "2024-11-05", "customer": "Mrs. Gupta", "discount_pct": 12.5,
     "items": [{"name": "Toor Dal 1kg", "qty": 3, "price_paise": 14999},
               {"name": "Fortune Sunflower Oil 1 Litre Pouch", "qty": 2, "price_paise": 15550}]},
    # net 250 paise and 350 paise: the original rounds halves to the even rupee
    {"invoice_no": "INV-3", "date": "2024-11-06", "items": [{"name": "Salt", "qty": 1, "price_paise": 250}]},
    {"invoice_no": "INV-4", "date": "2024-11-06", "items": [{"name": "Salt", "qty": 1, "price_paise": 350}]},
    # 33.3% of 12000 is 3995.9999999999995 in floating point: the original bill truncates it to 3995
    {"invoice_no": "INV-5", "date": "2024-11-07", "customer": "", "discount_pct": 33.3,
     "items": [{"name": "Parle-G", "qty": 120, "price_paise": 100}]},
    {"invoice_no": "INV-6", "date": "2024-11-07", "discount_pct": 0, "customer": None,
     "items": [{"name": "Exactly twenty chars", "qty": 1, "price_paise": 100},
               {"name": "Twenty-one characters", "qty": 1000, "price_paise": 99}]},
    {"invoice_no": "INV-7", "date": "2024-11-08", "discount_pct": 10.0,
     "items": [{"name": "Gold coin (Dhanteras)", "qty": 1, "price_paise": 734_567_891}]},
]


@pytest.mark.parametrize("order", EXAMPLES, ids=[o["invoice_no"] for o in EXAMPLES])
def test_printed_bill_unchanged_examples(order):
    assert make_invoice(copy.deepcopy(order)) == legacy_make_invoice(copy.deepcopy(order))


def test_printed_bill_unchanged_random_orders():
    rng = random.Random(1234)
    for n in range(400):
        order = random_order(rng, n)
        assert make_invoice(copy.deepcopy(order)) == legacy_make_invoice(copy.deepcopy(order)), order


@pytest.mark.parametrize("order", EXAMPLES, ids=[o["invoice_no"] for o in EXAMPLES])
def test_totals_match_the_bill_examples(order):
    totals = invoice_totals(copy.deepcopy(order))
    assert {k: totals[k] for k in legacy_totals(order)} == legacy_totals(order)


def test_totals_match_the_printed_bill_random_orders():
    rng = random.Random(99)
    for n in range(300):
        order = random_order(rng, n)
        totals = invoice_totals(copy.deepcopy(order))
        expected = legacy_totals(order)
        assert {k: totals[k] for k in expected} == expected, order
        printed_total = [line for line in legacy_make_invoice(order).splitlines() if line.startswith("TOTAL")][0]
        assert printed_total.split()[-1] == "%.2f" % (totals["total_paise"] / 100)


def test_totals_are_whole_paise_and_add_up():
    rng = random.Random(5)
    for n in range(100):
        totals = invoice_totals(random_order(rng, n))
        for key in ("subtotal_paise", "discount_paise", "round_off_paise", "total_paise"):
            assert isinstance(totals[key], int) and not isinstance(totals[key], bool), key
        assert totals["total_paise"] == totals["subtotal_paise"] - totals["discount_paise"] + totals["round_off_paise"]
        assert totals["total_paise"] % 100 == 0


def test_no_items_still_rejected():
    order = {"invoice_no": "INV-0", "date": "2024-11-05", "items": []}
    with pytest.raises(ValueError):
        make_invoice(order)
    with pytest.raises(ValueError):
        invoice_totals(order)


def test_order_is_not_modified():
    rng = random.Random(8)
    for n in range(50):
        order = random_order(rng, n)
        before = copy.deepcopy(order)
        make_invoice(order)
        invoice_totals(order)
        assert order == before


def test_same_order_same_bill():
    order = EXAMPLES[1]
    assert make_invoice(order) == make_invoice(order)
    assert invoice_totals(order) == invoice_totals(order)

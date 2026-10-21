Three complaints from the counter this week, all in `billing.py`:
a cart with items of ₹0.10 and ₹0.20 shows a total of ₹0.30000000000000004; splitting a ₹100 bill between 3 friends gives ₹33.33 each, so we are a paisa short; and a 10% discount on ₹2.25 comes out as ₹2.02, but our rule is to round half up, so it should be ₹2.03.
Please fix these. Callers pass and expect `Decimal` amounts.

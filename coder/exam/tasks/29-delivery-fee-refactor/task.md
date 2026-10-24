Nobody in the team can read `calc()` in `delivery.py` any more, and we need to change it, so please clean it up first.
The change: when it's raining, every delivery fee that isn't zero goes up by ₹15, and the ₹150 maximum still applies to the total.
The new delivery app should call it with readable names: `delivery_fee(distance_km, order_paise, express=False, member=False, hour=12, rain=False)`, returning the fee in paise like `calc` does.
The old billing counter software will keep calling `calc(d, amt, exp, mem, hr)` with positional arguments for a few more months, and for it every fee must stay exactly what it is today, to the paisa.

We send WhatsApp order updates through a provider that blocks us if we send in bursts, so I need a rate limiter. Please write `ratelimit.py` with a token bucket: `TokenBucket(rate, capacity, clock=time.monotonic)`, where `rate` is tokens added per second and `capacity` is the most it can hold; a new bucket starts full.
`allow(cost=1)` returns True and takes the tokens if there are enough, otherwise it returns False and takes nothing.
Also add `KeyedLimiter(rate, capacity, clock=time.monotonic)` whose `allow(key, cost=1)` keeps a separate bucket per key (we use the customer's phone number).
The `clock` argument is there so tests can control time. Raise `ValueError` for a rate, capacity or cost that makes no sense.

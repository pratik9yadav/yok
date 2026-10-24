Our delivery staff keep marking orders "delivered" that were never shipped, so I want the order status rules in code.
Please write `order_status.py` with a class `Order(order_id: str, clock=time.time)` that starts in status `"placed"` and has methods `pack()`, `ship(tracking_id: str)`, `deliver()` and `cancel(reason: str = "")`.
The flow is placed → packed → shipped → delivered, and an order can be cancelled only before it ships.
Any other move must raise `InvalidTransition` (define it in the module), and shipping needs a real tracking id (`ValueError` otherwise).
`order.status` gives the current status and `order.history` the list of `(status, timestamp)` steps, starting with `("placed", <time it was created>)`; timestamps come from `clock()`.
The status must only ever change through these methods.

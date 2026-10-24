The packers say the dispatch screen is wrong: among orders promised for the same day, the newest orders show up first, but it should be first come, first served.
Orders without a promised date (counter pickups) crash the screen with a `TypeError`; they should come after the dated orders of their group (express or normal).
Also, ever since the dispatch screen went live, the "orders placed today" list on the counter is shown in a strange order, and both screens get their orders from the same list.
The code is `dispatch_queue(orders)` in `dispatch.py`; please fix it without changing how it's called.

So far `sales_report.py` has only run for our own shop, and next month it goes to all 40 branches.
Each branch's till appends every sale to one CSV log for the whole year; the biggest branch's log reaches about 6 GB by March, and the report runs every night on a small server with 1 GB of RAM.
Please review the module for that rollout and fix what needs fixing.
The nightly job calls `summarize(path)` and `top_skus(path, n)`, and the numbers in the report must stay exactly the same as now.

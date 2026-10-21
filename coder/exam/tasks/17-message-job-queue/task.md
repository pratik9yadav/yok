We send order confirmations over an SMS gateway that fails now and then, so I want a small in-process job queue in `jobs.py`.
`JobQueue(max_attempts=3, base_delay=1.0, max_delay=30.0, sleep=time.sleep)`: `submit(func, *args, **kwargs) -> int` adds a job and returns its id, and `run_pending()` runs every waiting job in the order they were submitted.
`get(job_id)` returns the job; it should have `.status` (`"pending"`, `"done"` or `"failed"`), `.result`, `.error` (the last exception) and `.attempts`.
When a job raises, try it again after waiting `base_delay` seconds, doubling the wait each time but never more than `max_delay`, until it has been tried `max_attempts` times; then mark it failed and move on to the next job.
The `sleep` function is passed in so our tests don't really wait.
Jobs get submitted from the web request threads while a worker calls `run_pending()`.

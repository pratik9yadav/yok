import threading

import pytest

from jobs import JobQueue


class FakeSleep:
    def __init__(self):
        self.calls = []

    def __call__(self, seconds):
        self.calls.append(seconds)


class Flaky:
    """Fails `failures` times, then returns `value`."""

    def __init__(self, failures, value="sent"):
        self.failures = failures
        self.value = value
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError(f"gateway down (call {self.calls})")
        return self.value


def test_success_first_try_no_sleep():
    sleep = FakeSleep()
    q = JobQueue(sleep=sleep)
    job_id = q.submit(lambda: "ok")
    assert q.get(job_id).status == "pending"
    q.run_pending()
    job = q.get(job_id)
    assert job.status == "done"
    assert job.result == "ok"
    assert job.error is None
    assert job.attempts == 1
    assert sleep.calls == []


def test_args_and_kwargs_are_passed():
    q = JobQueue(sleep=FakeSleep())
    job_id = q.submit(lambda phone, text, prefix="": f"{prefix}{phone}:{text}", "98xxxxxx01", "Order packed", prefix="+91-")
    q.run_pending()
    assert q.get(job_id).result == "+91-98xxxxxx01:Order packed"


def test_retries_then_succeeds_with_exponential_backoff():
    sleep = FakeSleep()
    q = JobQueue(max_attempts=3, base_delay=1.0, sleep=sleep)
    func = Flaky(failures=2)
    job_id = q.submit(func)
    q.run_pending()
    job = q.get(job_id)
    assert job.status == "done"
    assert job.result == "sent"
    assert job.attempts == 3
    assert func.calls == 3
    assert sleep.calls == [1.0, 2.0]


def test_gives_up_after_max_attempts_without_sleeping_after_last_try():
    sleep = FakeSleep()
    q = JobQueue(max_attempts=4, base_delay=0.5, sleep=sleep)
    func = Flaky(failures=100)
    job_id = q.submit(func)
    q.run_pending()
    job = q.get(job_id)
    assert job.status == "failed"
    assert job.attempts == 4
    assert func.calls == 4
    assert isinstance(job.error, ConnectionError)
    assert sleep.calls == [0.5, 1.0, 2.0]


def test_backoff_is_capped_at_max_delay():
    sleep = FakeSleep()
    q = JobQueue(max_attempts=6, base_delay=10, max_delay=25, sleep=sleep)
    q.submit(Flaky(failures=100))
    q.run_pending()
    assert sleep.calls == [10, 20, 25, 25, 25]


def test_failed_job_does_not_stop_the_others_and_order_is_fifo():
    order = []
    q = JobQueue(max_attempts=2, sleep=FakeSleep())

    def record(name):
        order.append(name)
        return name

    def broken():
        order.append("broken")
        raise RuntimeError("bad template")

    a = q.submit(record, "a")
    b = q.submit(broken)
    c = q.submit(record, "c")
    q.run_pending()
    assert order == ["a", "broken", "broken", "c"]
    assert q.get(a).status == "done"
    assert q.get(b).status == "failed"
    assert q.get(c).status == "done"


def test_run_pending_twice_does_not_rerun_finished_jobs():
    q = JobQueue(max_attempts=2, sleep=FakeSleep())
    ok = Flaky(failures=0)
    bad = Flaky(failures=100)
    q.submit(ok)
    q.submit(bad)
    q.run_pending()
    q.run_pending()
    assert ok.calls == 1
    assert bad.calls == 2


def test_jobs_submitted_later_run_on_next_call():
    q = JobQueue(sleep=FakeSleep())
    first = q.submit(lambda: 1)
    q.run_pending()
    second = q.submit(lambda: 2)
    assert q.get(second).status == "pending"
    q.run_pending()
    assert q.get(first).result == 1
    assert q.get(second).result == 2


def test_unknown_job_id_raises_key_error():
    q = JobQueue(sleep=FakeSleep())
    with pytest.raises(KeyError):
        q.get(12345)


def test_submit_rejects_non_callable():
    q = JobQueue(sleep=FakeSleep())
    with pytest.raises(TypeError):
        q.submit("send_sms")


@pytest.mark.parametrize("kwargs", [{"max_attempts": 0}, {"max_attempts": -1}, {"base_delay": -1}])
def test_bad_settings_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        JobQueue(sleep=FakeSleep(), **kwargs)


def test_keyboard_interrupt_is_not_swallowed_or_retried():
    sleep = FakeSleep()
    q = JobQueue(max_attempts=5, sleep=sleep)
    calls = []

    def interrupted():
        calls.append(1)
        raise KeyboardInterrupt

    q.submit(interrupted)
    with pytest.raises(KeyboardInterrupt):
        q.run_pending()
    assert len(calls) == 1
    assert sleep.calls == []


def test_concurrent_submits_get_unique_ids_and_all_run():
    q = JobQueue(sleep=FakeSleep())
    n_threads, per_thread = 8, 100
    barrier = threading.Barrier(n_threads)
    ids, lock = [], threading.Lock()

    def worker():
        barrier.wait(timeout=5)
        for _ in range(per_thread):
            job_id = q.submit(lambda: None)
            with lock:
                ids.append(job_id)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(ids) == n_threads * per_thread
    assert len(set(ids)) == len(ids)
    q.run_pending()
    assert all(q.get(i).status == "done" for i in ids)


def test_submit_is_not_blocked_while_a_job_is_running():
    q = JobQueue(sleep=FakeSleep())
    started, proceed = threading.Event(), threading.Event()

    def slow_job():
        started.set()
        assert proceed.wait(timeout=5)
        return "slow done"

    slow_id = q.submit(slow_job)
    runner = threading.Thread(target=q.run_pending)
    runner.start()
    try:
        assert started.wait(timeout=5)
        submitted = []
        submitter = threading.Thread(target=lambda: submitted.append(q.submit(lambda: "quick")))
        submitter.start()
        submitter.join(timeout=2)
        assert submitted, "submit() blocked while run_pending() was running a job"
    finally:
        proceed.set()
        runner.join(timeout=5)
    assert not runner.is_alive()
    q.run_pending()
    assert q.get(slow_id).result == "slow done"
    assert q.get(submitted[0]).result == "quick"

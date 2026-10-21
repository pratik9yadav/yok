"""A small in-process job queue with retries and exponential backoff."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

PENDING = "pending"
DONE = "done"
FAILED = "failed"


@dataclass
class Job:
    id: int
    func: Callable[..., Any] = field(repr=False)
    args: tuple = field(default=(), repr=False)
    kwargs: dict = field(default_factory=dict, repr=False)
    status: str = PENDING
    result: Any = None
    error: BaseException | None = None
    attempts: int = 0


class JobQueue:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a whole number >= 1")
        if base_delay < 0 or max_delay < 0:
            raise ValueError("delays must not be negative")
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._jobs: dict[int, Job] = {}
        self._waiting: deque[Job] = deque()
        self._next_id = 1

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> int:
        if not callable(func):
            raise TypeError("func must be callable")
        with self._lock:
            job = Job(self._next_id, func, args, kwargs)
            self._next_id += 1
            self._jobs[job.id] = job
            self._waiting.append(job)
            return job.id

    def get(self, job_id: int) -> Job:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError:
                raise KeyError(f"unknown job id: {job_id!r}") from None

    def _delay(self, failed_attempts: int) -> float:
        return min(self.base_delay * (2 ** (failed_attempts - 1)), self.max_delay)

    def run_pending(self) -> int:
        """Run every waiting job (FIFO). Returns how many jobs were processed."""
        processed = 0
        with self._run_lock:  # one runner at a time; submit() stays free
            while True:
                with self._lock:
                    if not self._waiting:
                        return processed
                    job = self._waiting.popleft()
                self._run_one(job)
                processed += 1

    def _run_one(self, job: Job) -> None:
        while True:
            job.attempts += 1
            try:
                result = job.func(*job.args, **job.kwargs)
            except Exception as exc:  # noqa: BLE001 - any job error is retried; Ctrl+C etc. propagate
                job.error = exc
                if job.attempts >= self.max_attempts:
                    job.status = FAILED
                    return
                self._sleep(self._delay(job.attempts))
            else:
                job.result = result
                job.error = None
                job.status = DONE
                return

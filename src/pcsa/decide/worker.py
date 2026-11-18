"""One thread that calls the Player, so the Player is never called twice at once and a slow call never blocks taps.

Every job carries an id; the caller waits for its own id until a Deadline. Results for other ids (a call that
came back after its wait ended) are kept as late and handed over by `drain_late`, never used for a newer decision.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pcsa.common import log
from pcsa.common.clock import Clock, Deadline, precise_now

POLL_S = 0.02


@dataclass(frozen=True)
class Job:
    job_id: int
    request_id: int | None
    what: str
    call: Callable[[], Any]
    until: Deadline | None  # None: nobody waits (battle_start, battle_end)


@dataclass(frozen=True)
class Result:
    job_id: int
    request_id: int | None
    what: str
    value: Any = None
    error: BaseException | None = None
    skipped: bool = False  # the wait had ended before the Player was free
    ms: float | None = None
    done_at: float | None = None  # clock time the call returned


class Worker:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self._jobs: queue.Queue[Job | None] = queue.Queue()
        self._results: queue.Queue[Result] = queue.Queue()
        self._late: list[Result] = []
        self._next = 0
        self._pending = 0
        self._done = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="player", daemon=True)
        self._thread.start()

    def submit(
        self, what: str, call: Callable[[], Any], until: Deadline | None, request_id: int | None = None
    ) -> int:
        self._next += 1
        with self._done:
            self._pending += 1
        self._jobs.put(Job(self._next, request_id, what, call, until))
        return self._next

    def wait(self, job_id: int, until: Deadline) -> Result | None:
        """The job's result, or None once `until` has passed."""
        while not until.expired():
            try:
                result = self._results.get(timeout=POLL_S)
            except queue.Empty:
                continue
            if result.job_id == job_id:
                return result
            self._late.append(result)
        return self._last_look(job_id)

    def _last_look(self, job_id: int) -> Result | None:
        """One short look after the wait: an answer that lands right at the edge is reported late, not lost."""
        try:
            result = self._results.get(timeout=POLL_S)
        except queue.Empty:
            return None
        if result.job_id == job_id:
            return result
        self._late.append(result)
        return None

    def drain_late(self) -> list[Result]:
        while True:
            try:
                self._late.append(self._results.get_nowait())
            except queue.Empty:
                break
        late, self._late = self._late, []
        return late

    def idle(self, seconds: float) -> bool:
        """Wait (real time) until every submitted job has run; for shutdown and tests."""
        with self._done:
            return self._done.wait_for(lambda: self._pending == 0, seconds)

    def close(self) -> None:
        self._jobs.put(None)

    def _run(self) -> None:
        while (job := self._jobs.get()) is not None:
            result = self._call(job)
            if job.until is not None:
                self._results.put(result)
            with self._done:
                self._pending -= 1
                self._done.notify_all()

    def _call(self, job: Job) -> Result:
        if job.until is not None and job.until.expired():
            return Result(job.job_id, job.request_id, job.what, skipped=True)
        start = precise_now()
        try:
            value, error = job.call(), None
        except Exception as caught:
            value, error = None, caught
            if job.until is None:
                log.warning("player_failed", call=job.what, error=repr(caught))
        ms = round((precise_now() - start) * 1000, 1)
        return Result(job.job_id, job.request_id, job.what, value, error, ms=ms, done_at=self.clock.now())

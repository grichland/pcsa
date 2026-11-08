"""Time, injected. Nothing outside this module reads the wall clock, so tests control every deadline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock:
    """A clock that only moves when told to. `sleep` advances it, so waits are instant in tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a clock does not run backwards")
        self._now += seconds


@dataclass(frozen=True)
class Deadline:
    clock: Clock
    at: float

    @classmethod
    def after(cls, clock: Clock, seconds: float) -> Deadline:
        return cls(clock, clock.now() + seconds)

    def remaining(self) -> float:
        return self.at - self.clock.now()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def minus(self, reserve: float) -> Deadline:
        """The same deadline with `reserve` seconds held back for what follows (taps, transport)."""
        return Deadline(self.clock, self.at - reserve)

    def earliest(self, other: Deadline) -> Deadline:
        return self if self.at <= other.at else other

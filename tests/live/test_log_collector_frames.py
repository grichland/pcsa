"""Collecting the battle log on real panel frames: the climb keeps its last capture and survives a timer warning.

Uses the local-only frames of tests/perception/test_log_frames.py and skips without them.
"""

from __future__ import annotations

import random

from pcsa.common.clock import Deadline, FakeClock
from pcsa.live.log_collector import LogCollector
from pcsa.phone.adb import Frame
from tests.perception.test_log_frames import BUDGET_S, image


class Phone:
    """Serves the recorded captures in order; the climb's budget runs out after the last one."""

    def __init__(self, names: list[str], clock: FakeClock) -> None:
        self.images = [image(n) for n in names]
        self.clock = clock
        self.drags = 0
        self.captures = 0
        self.taps: list = []
        self.rng = random.Random(0)

    def capture(self) -> Frame:
        picture = self.images[min(self.captures, len(self.images) - 1)]
        self.captures += 1
        if self.captures == len(self.images):
            self.clock.advance(BUDGET_S)
        return Frame(picture, (picture.shape[1], picture.shape[0]), self.clock.now(), 0.0)

    def drag(self, *args) -> None:
        self.drags += 1

    # the executor's side
    def point(self, name: str) -> str:
        return name

    def execute(self, taps, screen, device, clock, wait) -> None:
        self.taps.extend(taps)

    def to_device(self, x: float, y: float) -> tuple[int, int]:
        return round(x), round(y)


def test_the_last_capture_is_kept_unsettled_and_a_timer_warning_does_not_discard_the_one_before(config) -> None:
    clock = FakeClock()
    names = ["mega-bottom.jpg", "mega-bottom.jpg", "mega-middle.jpg", "mega-middle-timer.jpg"]
    phone = Phone([*names, "mega-middle.jpg", "mega-top.jpg"], clock)
    collector = LogCollector(phone, phone, clock, config.agent.timing, config.agent.log, (2340, 1080))  # type: ignore[arg-type]
    out = collector.collect(Deadline.after(clock, BUDGET_S))
    assert out.stop_reason == "deadline"
    assert phone.drags == 2
    assert [i for i, _ in out.kept()] == [0, 2, 5]
    assert out.unsettled == [5]

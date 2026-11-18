"""A fake in-game battle log for the scripted device: one page per turn built from recorded frames, the heading
arrows, scrolling, and faults to inject (an arrow tap that does nothing, one that moves two pages)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pcsa.common.clock import FakeClock

CLOSED_GREY = 40


@dataclass
class LogPage:
    frames: list[np.ndarray]  # the page's settled views, bottom first
    opens_at: str = "bottom"  # "bottom" or "top": where the page shows when an arrow reaches it


@dataclass
class FakeLogBook:
    pages: dict[int, LogPage]
    current: int  # the page the log opens on (the latest completed turn)
    clock: FakeClock | None = None  # advanced per capture and per tap, so deadlines bite
    capture_s: float = 0.65
    tap_s: float = 0.1
    stuck_taps: int = 0  # the next arrow taps that leave the page as it is
    double_taps: int = 0  # the next arrow taps that move two pages
    open: bool = False
    turn: int = 0
    position: int = 0  # index into the page's frames
    taps: list[str] = field(default_factory=list)
    shown: list[int] = field(default_factory=list)  # the page turn of every capture while open

    def _show(self, turn: int, at: str) -> None:
        self.turn = min(max(turn, min(self.pages)), self.current)
        self.position = 0 if at == "bottom" else len(self.pages[self.turn].frames) - 1

    def _tick(self, seconds: float) -> None:
        if self.clock is not None:
            self.clock.advance(seconds)

    def capture(self) -> np.ndarray | None:
        """The page's current view while open; None when the log is closed."""
        self._tick(self.capture_s)
        if not self.open:
            return None
        self.shown.append(self.turn)
        return self.pages[self.turn].frames[self.position]

    def tap(self, name: str | None) -> bool:
        """True when the tap was a log control (handled here)."""
        if name not in ("view_log", "log_back", "log_prev", "log_next"):
            return False
        self._tick(self.tap_s)
        self.taps.append(name)
        if name == "view_log":
            self.open = True
            self._show(self.current, "bottom")
        elif name == "log_back":
            self.open = False
        elif self.open:
            self._arrow(-1 if name == "log_prev" else 1)
        return True

    def _arrow(self, direction: int) -> None:
        if self.stuck_taps:
            self.stuck_taps -= 1
            return
        step = 1
        if self.double_taps:
            self.double_taps -= 1
            step = 2
        target = min(max(self.turn + direction * step, min(self.pages)), self.current)
        self._show(target, self.pages[target].opens_at)

    def drag(self, y1: int, y2: int) -> bool:
        """A downward drag shows older rows (toward the top of the page)."""
        if not self.open:
            return False
        last = len(self.pages[self.turn].frames) - 1
        self.position = min(last, self.position + 1) if y2 > y1 else max(0, self.position - 1)
        return True

    @staticmethod
    def closed_frame(size: tuple[int, int]) -> np.ndarray:
        return np.full((size[1], size[0], 3), CLOSED_GREY, np.uint8)

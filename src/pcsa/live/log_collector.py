"""Open the battle log, capture the bottom view once settled, climb toward older entries, stop at a budget.

Only pixel checks run between captures; OCR runs afterwards on the whole batch. Running out of
budget leaves the start uncaptured, which the merge reports; the turn is still decided from what was read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pcsa.common.clock import Clock, Deadline
from pcsa.common.config import DragConfig, LogCollectConfig, TimingConfig
from pcsa.common.types import Screen
from pcsa.live.device import InputDevice as Device
from pcsa.live.taps import Executor
from pcsa.phone import log_panel as lp
from pcsa.phone.adb import Frame
from pcsa.phone.ocr import REFERENCE
from pcsa.phone.screens import to_reference


@dataclass
class Collected:
    frames: list[Frame] = field(default_factory=list)
    rejected: list[int] = field(default_factory=list)  # indices still animating or covered
    unsettled: list[int] = field(default_factory=list)  # kept, but no second capture confirmed them
    stop_reason: str = ""
    seconds: float = 0.0
    events: list[dict] = field(default_factory=list)
    panel_open: bool = False
    last_scroll: float | None = None

    def kept(self) -> list[tuple[int, Frame]]:
        return [(i, f) for i, f in enumerate(self.frames) if i not in self.rejected]


class LogCollector:
    def __init__(
        self,
        executor: Executor,
        device: Device,
        clock: Clock,
        timing: TimingConfig,
        config: LogCollectConfig,
        viewport_size: tuple[int, int],
    ) -> None:
        self.executor = executor
        self.device = device
        self.clock = clock
        self.timing = timing
        self.config = config
        self.size = viewport_size

    def _drag_pixels(self, drag: DragConfig) -> tuple[int, int, int]:
        rng = self.executor.rng
        x = rng.uniform(*drag.x) * REFERENCE[0]
        y1, y2 = rng.uniform(*drag.y_start) * REFERENCE[1], rng.uniform(*drag.y_end) * REFERENCE[1]
        dx, dy1 = self.executor.to_device(x, y1)
        _, dy2 = self.executor.to_device(x, y2)
        return dx, dy1, dy2

    def _tap(self, name: str, out: Collected) -> None:
        tap = self.executor.point(name)
        self.executor.execute(
            [tap], Screen.ACTION if name == "view_log" else Screen.LOG, self.device, self.clock, 0.0
        )
        out.events.append({"tap": name})

    def close(self, out: Collected) -> None:
        if out.panel_open:
            self._tap("log_back", out)
            self.clock.sleep(self.timing.back_settle_s)

    def collect(self, deadline: Deadline, *, close: bool = True) -> Collected:
        """Capture the bottom view until settled, climb while the budget allows, confirm the top."""
        started = self.clock.now()
        out = Collected()
        budget = Deadline.after(self.clock, self.config.seconds_max).earliest(deadline)
        if budget.expired():
            out.stop_reason = "deadline"
            return out
        self._tap("view_log", out)
        self.clock.sleep(self.timing.log_open_settle_s)
        state = _Climb()
        while len(out.frames) < self.config.captures_max:
            if budget.expired():
                out.stop_reason = "deadline"
                break
            frame = self.device.capture()
            out.frames.append(frame)
            index = len(out.frames) - 1
            normalized = to_reference(frame.image)
            if not self._panel_ok(normalized, index, state, out):
                if out.stop_reason:
                    break
                self.clock.sleep(self.timing.log_swipe_settle_s)
                continue
            if self._step(normalized, index, state, out):
                break
            if len(out.frames) >= self.config.captures_max:
                out.stop_reason = "max_captures"
                break
            if state.name == "climb":
                x, y1, y2 = self._drag_pixels(self.config.drag)
                self.device.drag(x, y1, y2, self.config.drag.steps, self.config.drag.hold_s)
                out.events.append({"drag": [x, y1, y2]})
            self.clock.sleep(self.timing.log_swipe_settle_s)
        if (
            state.previous_index is not None
            and state.previous_index not in out.rejected
            and out.stop_reason not in ("top_confirmed", "settled_no_scrollbar")
        ):
            out.unsettled.append(state.previous_index)  # the merge keeps it only when its rows join the others
        if close:
            self.close(out)
        out.seconds = self.clock.now() - started
        return out

    def _panel_ok(self, normalized: np.ndarray, index: int, state: _Climb, out: Collected) -> bool:
        """False when this capture cannot be read (panel closed, or the timer warning covers it)."""
        if not lp.is_log_panel(normalized):
            out.panel_open = False
            state.missing_panel += 1
            out.rejected.append(index)
            out.events.append({"capture": index, "state": state.name, "result": "panel_not_open"})
            if state.missing_panel >= 2:
                out.stop_reason = "panel_not_open"
            return False
        out.panel_open = True
        state.missing_panel = 0
        if lp.has_timer_overlay(normalized):
            out.rejected.append(index)  # the next uncovered capture is compared with the one before this
            out.events.append(
                {"capture": index, "state": state.name, "timer_overlay": round(lp.timer_overlay_ratio(normalized), 3)}
            )
            if len(out.frames) >= self.config.captures_max:
                out.stop_reason = "max_captures"
            return False
        return True

    def _step(self, normalized: np.ndarray, index: int, state: _Climb, out: Collected) -> bool:
        """Advance the bottom/climb/top state machine; True when collection is finished."""
        try:
            position, _, at_top, _, present = lp.scrollbar(normalized)
            out.last_scroll = position
        except ValueError as error:
            at_top, present = False, False
            out.events.append({"capture": index, "scrollbar_error": str(error)})
        same = state.previous is not None and lp.same_viewport(state.previous, normalized)
        out.events.append(
            {"capture": index, "state": state.name, "scrollbar": present, "at_top": at_top, "same_as_prev": same}
        )
        if state.name in ("bottom", "page", "top"):
            if not same:
                if state.previous_index is not None and state.previous_index not in out.rejected:
                    out.rejected.append(state.previous_index)
                state.previous, state.previous_index = normalized, index
                return False
            out.rejected.append(index)  # a duplicate of a settled frame
            if state.name == "top" or not present or at_top:
                out.stop_reason = "top_confirmed" if present else "settled_no_scrollbar"
                return True
            state.name = "climb"
            state.previous, state.previous_index = normalized, None
            return False
        if same:
            out.rejected.append(index)  # the drag had no effect: retry it
            out.events.append({"capture": index, "no_movement": True})
            return False
        state.name = "top" if at_top else "page"
        state.previous, state.previous_index = normalized, index
        return False


@dataclass
class _Climb:
    name: str = "bottom"
    previous: np.ndarray | None = None
    previous_index: int | None = None
    missing_panel: int = 0

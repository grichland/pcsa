"""Back-paging: the log heading jumped past turns with no action screen of ours. Step back with the heading
arrows to the oldest missed turn that fits the deadline, collect each missed page oldest first, then return to
the current page. Every arrow tap is confirmed by reading the heading; a page is never taken on a wrong heading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pcsa.common import log
from pcsa.common.clock import Clock, Deadline
from pcsa.common.config import PagingConfig, Viewport
from pcsa.live.log_collector import Collected, LogCollector
from pcsa.phone import log_panel as lp
from pcsa.phone.ocr import TextReader
from pcsa.phone.screens import to_reference
from pcsa.run.recorder import Recorder

OLDER, NEWER = "log_prev", "log_next"


@dataclass
class Page:
    turn: int
    collected: Collected


@dataclass
class Paging:
    missed: list[int]
    current: int
    planned: list[int] = field(default_factory=list)
    pages: list[Page] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)  # every arrow tap: expected heading, heading read, frame
    stop: str = ""  # done, deadline, arrow_failed, panel_closed
    returned: bool = False
    panel_open: bool = True

    def record(self) -> dict:
        out = asdict(self)
        out["pages"] = [{"turn": p.turn, "stop": p.collected.stop_reason, "captures": len(p.collected.frames)}
                        for p in self.pages]  # fmt: skip
        return out


class Pager:
    def __init__(
        self,
        collector: LogCollector,
        reader: TextReader,
        viewport: Viewport | None,
        recorder: Recorder,
        clock: Clock,
        config: PagingConfig,
    ) -> None:
        self.collector = collector
        self.reader = reader
        self.viewport = viewport
        self.rec = recorder
        self.clock = clock
        self.cfg = config

    def plan(self, missed: list[int], current: int, deadline: Deadline) -> list[int]:
        """The newest missed turns whose arrow steps (there and back) and pages fit the time left."""
        for count in range(len(missed), 0, -1):
            turns = missed[-count:]
            cost = 2 * (current - turns[0]) * self.cfg.step_s + count * self.cfg.page_min_s
            if cost <= deadline.remaining():
                return turns
        return []

    def run(self, missed: list[int], current: int, deadline: Deadline, prefix: str) -> Paging:
        """The panel shows the current page. `prefix`: where frames go (".../log-frames/visitNN")."""
        paging = Paging(missed, current, self.plan(missed, current, deadline))
        if not paging.planned:
            paging.stop = "deadline"
            log.event("log_paging", **paging.record())
            return paging
        target, heading = paging.planned[0], current
        for expected in range(current - 1, target - 1, -1):
            heading = self._step(OLDER, expected, heading, paging, prefix)
            if heading != expected:
                paging.stop = "arrow_failed" if paging.panel_open else "panel_closed"
                break
        else:
            heading = self._collect_forward(target, deadline, paging, prefix)
        if paging.panel_open:
            self._return(heading, current, paging, prefix)
        log.event("log_paging", **paging.record())
        return paging

    def _collect_forward(self, target: int, deadline: Deadline, paging: Paging, prefix: str) -> int | None:
        """On the target page: collect it and each newer planned page, stepping forward between them."""
        heading: int | None = target
        for turn in range(target, paging.current):
            if turn != target:
                heading = self._step(NEWER, turn, heading, paging, prefix)
                if heading != turn:
                    paging.stop = "arrow_failed" if paging.panel_open else "panel_closed"
                    return heading
            left = deadline.minus((paging.current - turn) * self.cfg.step_s)  # keep time to step back
            if turn in paging.planned:
                if left.remaining() < self.cfg.page_min_s:
                    paging.stop = "deadline"
                    return heading
                page = self.collector.collect(left, close=False, opened=True, seconds_max=self.cfg.seconds_max)
                paging.pages.append(Page(turn, page))
                if page.stop_reason == "panel_not_open":
                    paging.stop, paging.panel_open = "panel_closed", False
                    return None
        paging.stop = "done"
        return heading

    def _return(self, heading: int | None, current: int, paging: Paging, prefix: str) -> None:
        """Step forward to the current page, so the log is left where the game opened it."""
        if heading is not None:
            for expected in range(heading + 1, current + 1):
                heading = self._step(NEWER, expected, heading, paging, prefix)
                if heading != expected:
                    break
        paging.returned = heading == current

    def _step(self, arrow: str, expected: int, heading: int | None, paging: Paging, prefix: str) -> int | None:
        """Tap an arrow and read the heading; one retry (the same arrow when the page did not change, the other
        way when it moved past the expected turn). Returns the heading last read."""
        seen = heading
        for attempt in range(2):
            self.collector.tap(arrow)
            self.clock.sleep(self.cfg.settle_s)
            seen, frame = self._heading(paging, prefix)
            paging.steps.append({"tap": arrow, "expected": expected, "heading": seen, "frame": frame, "try": attempt})
            if seen == expected:
                return seen
            if seen is not None and seen != heading:
                arrow, heading = (OLDER if seen > expected else NEWER), seen
        return seen

    def _heading(self, paging: Paging, prefix: str) -> tuple[int | None, str]:
        """The heading's turn number, read again once after a settle when it cannot be read."""
        path = ""
        for read in range(2):
            frame = self.collector.device.capture()
            path = self.rec.save_frame(f"{prefix}-head{len(paging.steps):02d}-{read}.png", frame.image, png=True)
            image = to_reference(frame.image, self.viewport)
            paging.panel_open = lp.is_log_panel(image)
            turn = lp.read_turn(image, self.reader) if paging.panel_open else None
            if turn is not None:
                return turn, path
            self.clock.sleep(self.cfg.settle_s)
        return None, path

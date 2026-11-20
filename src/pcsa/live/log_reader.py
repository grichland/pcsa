"""Reading the in-game battle log for one action screen: collect (each settled view is read on a worker thread
while the climb goes on), recover missed overlaps, page back to turns the heading skipped, merge."""

from __future__ import annotations

import copy
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from pcsa.common import log
from pcsa.common.clock import Clock, Deadline
from pcsa.common.config import AgentConfig
from pcsa.live.log_collector import Collected, LogCollector
from pcsa.live.log_pages import Pager, Paging
from pcsa.phone.log_merge import MergeReport, bridges_gap, unverified_gaps
from pcsa.phone.log_panel import read_view
from pcsa.phone.log_views import usable_views
from pcsa.phone.screens import to_reference
from pcsa.run.battle_state import BattleRecord
from pcsa.run.recorder import Recorder

INCOMPLETE = ("no verified overlap", "end of log is not captured")


@dataclass
class LogVisit:
    turn: int | None = None
    rows: int = 0
    appended: int = 0
    anchor: str = ""
    warnings: list[str] | None = None
    errors: list[str] | None = None
    frames: list[str] | None = None
    stop_reason: str = ""
    complete: bool = False
    seconds: float = 0.0
    back_paged: list[int] = field(default_factory=list)  # missed turns read by paging back
    paging: str = ""  # the pager's stop reason; empty when the heading did not jump


@dataclass
class PageViews:
    turn: int
    views: list
    errors: list[str]


class LogReader:
    def __init__(self, phone, recorder: Recorder, config: AgentConfig, clock: Clock) -> None:
        self.phone = phone
        self.rec = recorder
        self.cfg = config
        self.clock = clock
        self.collector = LogCollector(
            phone.executor, phone.device, clock, config.timing, config.log, phone.profile.frame_size
        )
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pcsa-log-read")
        self.pager = Pager(self.collector, phone.reader, phone.profile.viewport, recorder, clock, config.log.paging)

    def collect_log(self, b: BattleRecord, visit: int, deadline: Deadline) -> LogVisit:
        started = self.clock.now()
        prefix = f"{b.dir}/log-frames/visit{visit:02d}"
        early: dict[int, Future] = {}

        def settled(index: int, frame) -> None:
            early[index] = self._worker.submit(self._read, frame, f"{prefix}-{index}.png")

        collected = self.collector.collect(deadline, close=False, on_settled=settled)
        paging = None
        try:
            views, errors, frames = self._read_views(prefix, collected, early)
            self._recover_gaps(b, visit, prefix, collected, views, errors, frames, deadline)
            paging = self._page_back(b, prefix, collected, views, errors, deadline)
        finally:
            self.collector.close(collected)
        pages = self._read_pages(prefix, paging, frames)
        info = LogVisit(
            errors=errors,
            frames=frames,
            stop_reason=collected.stop_reason,
            paging=paging.stop if paging else "",
        )
        if usable_views(views)[0]:
            self._merge_pages(b, visit, pages, info)
            self._merge(b, visit, views, errors, info, paging)
        elif b.action_visits == 1:
            b.log.note("no_completed_turn", "first action screen: no completed turn to collect", visit=visit)
        else:
            b.log.fail(f"visit {visit}: no readable log panel")
        info.seconds = round(self.clock.now() - started, 2)
        log.event(
            "log_visit",
            battle=b.index,
            visit=visit,
            turn=info.turn,
            rows=info.rows,
            appended=info.appended,
            anchor=info.anchor,
            complete=info.complete,
            stop=info.stop_reason,
            seconds=info.seconds,
            errors=len(errors),
            **({"paging": info.paging, "back_paged": info.back_paged} if paging else {}),
        )
        return info

    def _read(self, frame, source: str):
        return read_view(to_reference(frame.image, self.phone.profile.viewport), self.phone.reader, source)

    def _read_views(
        self, prefix: str, collected: Collected, early: dict[int, Future] | None = None
    ) -> tuple[list, list[str], list[str]]:
        """The kept frames as views; `early` holds the readings started during the collection."""
        frames = [self.rec.save_frame(f"{prefix}-{i}.png", f.image, png=True) for i, f in enumerate(collected.frames)]
        early = early or {}
        kept = collected.kept()
        for i in set(early) - {i for i, _ in kept}:
            early[i].cancel()
        views, errors = [], []
        for i, frame in kept:
            try:
                reading = early[i] if i in early else self._worker.submit(self._read, frame, frames[i])
                view = reading.result()  # every reading on the one worker: the reader is not shared
                views.append(replace(view, settled=i not in collected.unsettled))
            except ValueError as error:
                errors.append(f"{frames[i]}: {error}")
        return views, errors, frames

    def _page_back(
        self, b: BattleRecord, prefix: str, collected: Collected, views: list, errors: list[str], deadline: Deadline
    ) -> Paging | None:
        """When the heading is more than one past the last turn read, read the skipped pages (panel still open)."""
        kept = usable_views(views)[0]
        if not self.cfg.log.paging.enabled or errors or not kept or not collected.panel_open:
            return None
        current = Counter(v.turn for v in kept).most_common(1)[0][0]
        missed = missed_turns(b.last_turn, current, b.log.turns)
        if not missed:
            return None
        paging = self.pager.run(missed, current, deadline, prefix)
        collected.panel_open = paging.panel_open
        return paging

    def _read_pages(self, prefix: str, paging: Paging | None, frames: list[str]) -> list[PageViews]:
        """OCR of each collected missed page; a view whose heading is not the page's turn rejects the page."""
        out = []
        for page in paging.pages if paging else []:
            views, errors, page_frames = self._read_views(f"{prefix}-page{page.turn}", page.collected)
            frames += page_frames
            errors += [
                f"{v.source}: heading Turn {v.turn}, expected Turn {page.turn}" for v in views if v.turn != page.turn
            ]
            out.append(PageViews(page.turn, [v for v in views if v.turn == page.turn], errors))
        return out

    def _recover_gaps(
        self,
        b: BattleRecord,
        visit: int,
        prefix: str,
        collected: Collected,
        views: list,
        errors: list[str],
        frames: list[str],
        deadline: Deadline,
    ) -> None:
        """After reaching the top, scan back toward newer entries in short steps to bridge a missed overlap."""
        if collected.stop_reason != "top_confirmed" or errors:
            return
        for attempt in range(4):
            gaps = unverified_gaps(views)
            if not gaps or deadline.expired():
                return
            older, newer = min(gaps, key=lambda pair: (pair[0].scroll + pair[1].scroll) / 2)
            target = (older.scroll + newer.scroll) / 2
            current = collected.last_scroll or 0.0
            x, y1, y2 = self.collector._drag_pixels(self.cfg.log.drag)
            distance = round(min(abs(y2 - y1) // 2, max(120, abs(target - current) * 700 * 0.5**attempt)))
            start = (y2 + y1) // 2
            self.phone.device.drag(
                x,
                start,
                start - distance if target >= current else start + distance,
                self.cfg.log.drag.steps,
                self.cfg.log.drag.hold_s,
            )
            self.clock.sleep(self.cfg.timing.log_swipe_settle_s)
            frame = self.phone.device.capture()
            path = self.rec.save_frame(f"{prefix}-recovery{attempt + 1:02d}.png", frame.image, png=True)
            frames.append(path)
            try:
                bridge = read_view(to_reference(frame.image, self.phone.profile.viewport), self.phone.reader, path)
            except ValueError as error:
                errors.append(f"{path}: {error}")
                return
            views.append(bridge)
            collected.last_scroll = bridge.scroll
            log.event(
                "log_recovery",
                battle=b.index,
                visit=visit,
                attempt=attempt + 1,
                bridged=bridges_gap(older, bridge, newer),
            )

    def _merge_pages(self, b: BattleRecord, visit: int, pages: list[PageViews], info: LogVisit) -> None:
        """Each missed page, oldest first, before the current page: its rows go under its own turn."""
        for page in sorted(pages, key=lambda p: p.turn):
            if page.errors or not usable_views(page.views)[0]:
                b.log.note(
                    "back_page_unread", f"turn {page.turn}: page reached but not read", turn=page.turn, visit=visit
                )
                continue
            staged = copy.deepcopy(b.log)
            sources = {v.source: v.turn for v in page.views}
            report = staged.merge(page.views, {"visit": visit, "page": "back", "view_turns": sources})
            incomplete = [w for w in report.warnings if any(k in w for k in INCOMPLETE)]
            if incomplete:
                staged.fail(f"visit {visit}, back page turn {page.turn}: incomplete log collection: {incomplete}")
            staged.note("back_paged", f"turn {page.turn}: read by paging back", turn=page.turn, visit=visit)
            b.log = staged
            info.back_paged.append(page.turn)

    def _merge(
        self, b: BattleRecord, visit: int, views: list, errors: list[str], info: LogVisit, paging: Paging | None
    ) -> None:
        staged = copy.deepcopy(b.log)
        extra = {"view_turns": {v.source: v.turn for v in views}}
        if paging is not None:
            extra["paging"] = paging.record()
        report: MergeReport = staged.merge(views, {"visit": visit, **extra})
        info.turn, info.rows, info.appended, info.anchor, info.warnings = (
            report.turn,
            len(report.rows),
            len(report.appended),
            report.anchor,
            report.warnings,
        )
        incomplete = [w for w in report.warnings if any(k in w for k in INCOMPLETE)]
        start_missing = any("start of log is not captured" in w for w in report.warnings)
        later_turn = b.last_turn is None or report.turn > b.last_turn or bool(b.log.turns.get(report.turn))
        if start_missing and not later_turn:
            incomplete.append("start of log is not captured")
        if incomplete:
            staged.fail(f"visit {visit}, turn {report.turn}: incomplete log collection: {incomplete}")
        for missing in range((b.last_turn or 0) + 1, report.turn):
            if missing in staged.turns:
                continue  # read by paging back
            staged.note(
                "no_own_action_screen",
                f"turn {missing}: no action screen of ours before turn {missing + 1}",
                turn=missing,
            )
        if report.turn >= (b.last_turn or 0) and not errors:
            b.log = staged
            b.last_turn = max(report.turn, b.last_turn or 0)
            info.complete = not incomplete
        b.start_partial = visit == 1 and report.turn == 1


def missed_turns(last_read: int | None, current: int, read: dict | set) -> list[int]:
    """Turns before the current page that were never read. With nothing read yet the battle is followed from
    turn 1 (battles picked up midway are not accepted), so a lock-in from the first move is still paged back."""
    return [t for t in range((last_read or 0) + 1, current) if t not in read]

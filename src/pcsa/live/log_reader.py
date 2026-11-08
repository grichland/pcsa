"""Reading the in-game battle log for one action screen: collect, OCR, recover missed overlaps, merge."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from pcsa.common import log
from pcsa.common.clock import Clock, Deadline
from pcsa.common.config import AgentConfig
from pcsa.live.log_collector import Collected, LogCollector
from pcsa.phone.log_merge import MergeReport, bridges_gap, unverified_gaps
from pcsa.phone.log_panel import read_view
from pcsa.phone.screens import to_reference
from pcsa.run.battle_state import BattleRecord
from pcsa.run.recorder import Recorder


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


class LogReader:
    def __init__(self, phone, recorder: Recorder, config: AgentConfig, clock: Clock) -> None:
        self.phone = phone
        self.rec = recorder
        self.cfg = config
        self.clock = clock
        self.collector = LogCollector(
            phone.executor, phone.device, clock, config.timing, config.log, phone.profile.frame_size
        )

    def collect_log(self, b: BattleRecord, visit: int, deadline: Deadline) -> LogVisit:
        started = self.clock.now()
        collected = self.collector.collect(deadline, close=False)
        try:
            views, errors, frames = self._read_views(b, visit, collected)
            self._recover_gaps(b, visit, collected, views, errors, frames, deadline)
        finally:
            self.collector.close(collected)
        info = LogVisit(
            errors=errors,
            frames=frames,
            stop_reason=collected.stop_reason,
            seconds=round(self.clock.now() - started, 2),
        )
        if any(view.settled for view in views):
            self._merge(b, visit, views, errors, info)
        elif b.action_visits == 1:
            b.log.note("no_completed_turn", "first action screen: no completed turn to collect", visit=visit)
        else:
            b.log.fail(f"visit {visit}: no readable log panel")
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
        )
        return info

    def _read_views(self, b: BattleRecord, visit: int, collected: Collected) -> tuple[list, list[str], list[str]]:
        frames = [
            self.rec.save_frame(f"{b.dir}/log-frames/visit{visit:02d}-{i}.png", f.image, png=True)
            for i, f in enumerate(collected.frames)
        ]
        views, errors = [], []
        for i, frame in collected.kept():
            try:
                view = read_view(to_reference(frame.image, self.phone.profile.viewport), self.phone.reader, frames[i])
                views.append(replace(view, settled=i not in collected.unsettled))
            except ValueError as error:
                errors.append(f"{frames[i]}: {error}")
        return views, errors, frames

    def _recover_gaps(
        self,
        b: BattleRecord,
        visit: int,
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
            path = self.rec.save_frame(
                f"{b.dir}/log-frames/visit{visit:02d}-recovery{attempt + 1:02d}.png", frame.image, png=True
            )
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

    def _merge(self, b: BattleRecord, visit: int, views: list, errors: list[str], info: LogVisit) -> None:
        staged = copy.deepcopy(b.log)
        report: MergeReport = staged.merge(views, {"visit": visit})
        info.turn, info.rows, info.appended, info.anchor, info.warnings = (
            report.turn,
            len(report.rows),
            len(report.appended),
            report.anchor,
            report.warnings,
        )
        incomplete = [w for w in report.warnings if "no verified overlap" in w or "end of log is not captured" in w]
        start_missing = any("start of log is not captured" in w for w in report.warnings)
        later_turn = b.last_turn is None or report.turn > b.last_turn or bool(b.log.turns.get(report.turn))
        if start_missing and not later_turn:
            incomplete.append("start of log is not captured")
        if incomplete:
            staged.fail(f"visit {visit}, turn {report.turn}: incomplete log collection: {incomplete}")
        for missing in range((b.last_turn or 0) + 1, report.turn):
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

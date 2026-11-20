"""One battle turn: peek the moves when needed, read the log within its budget, grow the battle's events, build
the Request, ask the runner, tap. With `timing.move_peek: when_needed` the peek is left out unless move_peek.py
names a reason for it. No Showdown lines and no Tracker: the Request carries the rows and the LIVE events."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from pcsa.battle import hints
from pcsa.battle.teams import OwnMember, Team
from pcsa.common import log
from pcsa.common.clock import Clock, Deadline
from pcsa.common.config import AgentConfig
from pcsa.common.types import Observation, Screen, SemanticAction
from pcsa.decide.api import Hints, Order, Request
from pcsa.decide.order_text import move_id
from pcsa.decide.runner import Decision, Runner
from pcsa.live import decision_record, move_peek, party_switch, request
from pcsa.live.action_check import verify_last_action
from pcsa.live.battle_events import BattleEvents
from pcsa.live.log_reader import LogReader, LogVisit
from pcsa.live.taps import ExecResult
from pcsa.live.team_check import check_team
from pcsa.live.turn_sync import check_identity, faints_from_log, leaving_slot, resync_party
from pcsa.run.battle_state import BattleRecord, action_detail, same_name
from pcsa.run.recorder import Recorder

MIN_NEW_TURN_MOVE_TIME = 15  # a lower reading on a fresh action screen is an OCR drop of the first digit
SEARCH_OVERHEAD_S = 1.0  # the search's request beyond its own time
AFTER_LOG_PEEK_S = 6.0  # a look at the move screen that the log read may ask for
FORCED_RETRY_DEADLINE_S = 40.0


class LogSaving(Protocol):
    """live/log_save.LogSaver: writes the battle's log files after a visit that changed them."""

    def after_visit(self, b: BattleRecord, team: Team | None) -> bool: ...


class TurnPipeline:
    def __init__(
        self,
        phone,
        runner: Runner,
        recorder: Recorder,
        config: AgentConfig,
        clock: Clock,
        *,
        seed: int = 0,
        saver: LogSaving | None = None,
    ) -> None:
        self.phone = phone
        self.runner = runner
        self.rec = recorder
        self.cfg = config
        self.clock = clock
        self.seed = seed
        self.team: Team | None = None  # set by the session once the team sheet is read
        self.logs = LogReader(phone, recorder, config, clock)
        self.saver = saver
        self._events: dict[str, BattleEvents] = {}

    # Battle events ------------------------------------------------------------------------------------
    def events_for(self, b: BattleRecord) -> BattleEvents:
        """This battle's event list; rebuilt from the log (all rows again) when new or when the team arrived."""
        events = self._events.get(b.battle_id)
        if events is None or (events.team is None and self.team is not None):
            events = BattleEvents(b.battle_id, self.team, tuple(b.opponent_preview))
            self._events = {b.battle_id: events}
            events.update(b)
        return events

    def end(self, b: BattleRecord, result: str) -> tuple[dict, ...]:
        """Rows not yet in the list, then `end`; returns the whole list for the Player's battle_end."""
        events = self.events_for(b)
        events.update(b)
        events.end(result)
        self.rec.write_json(f"{b.dir}/events.json", {"battle": b.battle_id, "events": list(events.all())})
        return events.all()

    def _save_log(self, b: BattleRecord) -> None:
        if self.saver is None:
            try:
                from pcsa.live.log_save import LogSaver  # Agent A's module

                self.saver = LogSaver(self.rec, self.seed)  # type: ignore[assignment]
            except (ImportError, TypeError) as error:
                log.warning("log_saver_missing", reason=str(error))
                self.saver = _NoSaver()
        assert self.saver is not None
        self.saver.after_visit(b, self.team)

    # Screens ----------------------------------------------------------------------------------------
    def peek(self, b: BattleRecord) -> Observation | None:
        """Fight -> capture -> Back. None when the move list never showed."""
        executor, device = self.phone.executor, self.phone.device
        executor.execute([executor.point("fight")], Screen.ACTION, device, self.clock, 0.0)
        self.clock.sleep(self.cfg.timing.fight_settle_s)
        obs = None
        for _ in range(self.cfg.degrade.peek_attempts):
            obs, _ = self.phone.observe("peek")
            if obs.screen == Screen.MOVES:
                break
            self.clock.sleep(0.4)
        if obs is not None and obs.screen == Screen.MOVES:
            if self.phone.navigator.close_move_info(obs).screen != Screen.MOVES:
                return obs
            executor.execute([executor.point("move_back")], Screen.MOVES, device, self.clock, 0.0)
            self.clock.sleep(self.cfg.timing.back_settle_s)
            return obs
        if obs is not None and obs.screen == Screen.FORFEIT_MSG and b.outcome is None:
            b.outcome = "opponent_forfeit"  # the message shows only here; the next poll is the options screen
        log.warning("peek_failed", battle=b.index, screen=str(obs.screen) if obs else None)
        if obs is not None and obs.screen in (Screen.MOVES, Screen.PARTY, Screen.LOG):
            self.phone.navigator.back_out(obs)
        return None

    # Action turn ---------------------------------------------------------------------------------------
    def _begin_action(self, b: BattleRecord, obs: Observation, t0: float) -> tuple[int, Deadline, str | None]:
        """Bookkeeping for one action screen: the visit number, the turn's end, the active name on the screen."""
        b.forced_attempts, b.forced_choice = 0, None
        b.action_visits += 1
        move_time = obs.fields.get("move_time")
        active_name = obs.fields.get("active_name")
        if b.action_visits == 1:
            b.observed["lead"] = active_name
        opponent = obs.fields.get("opponent_name")
        if opponent and not any(same_name(opponent, seen) for seen in b.opponents):
            b.opponents.append(opponent)
        timed = isinstance(move_time, int) and move_time >= MIN_NEW_TURN_MOVE_TIME
        budget = min(float(move_time) if timed else self.cfg.timing.max_turn_s, self.cfg.timing.max_turn_s)  # type: ignore[arg-type]
        return b.action_visits, Deadline(self.clock, t0 + budget), active_name

    def _hints(
        self, b: BattleRecord, events: BattleEvents, member: OwnMember | None, active: int, refused: tuple[int, ...]
    ) -> Hints:
        return hints.build(
            events.all(),
            events.mon(member),
            member,
            refused=refused,
            offsets=move_peek.offsets(b, active),
            mega_spent=b.mega_spent or b.mega_used,
        )

    def _decide_reserve(self) -> float:
        """Seconds the log read leaves for the decision: the setting, or the search's time when that is longer."""
        search = self.cfg.cg.search
        need = search.ms / 1000.0 + SEARCH_OVERHEAD_S if search.enabled else 0.0
        return max(self.cfg.degrade.decide_reserve_s, need)

    def action(self, b: BattleRecord, obs: Observation, t0: float, repeated: bool = False) -> None:
        """`repeated`: the poll before this one also showed the action screen (the taps may not have gone through)."""
        visit, end, active_name = self._begin_action(b, obs, t0)
        why = move_peek.why_before_log(self.cfg.timing.move_peek, b, self.team, repeated)
        reserve = self.cfg.degrade.tap_reserve_s + self._decide_reserve() + (0.0 if why else AFTER_LOG_PEEK_S)
        peek = self.peek(b) if why else None
        turn_before = b.last_turn
        log_info = self.logs.collect_log(b, visit, end.minus(reserve))
        self._save_log(b)
        last = b.last_action or {}
        same_turn = (
            bool(last)
            and log_info.turn is not None
            and log_info.turn == last.get("log_turn")
            and _rows(b, log_info.turn) == last.get("log_rows", _rows(b, log_info.turn) - log_info.appended)
        )  # no row since our last tap (not since the last visit: a restart may have read the log untapped)
        verify_last_action(b)
        check_team(b, active=active_name)
        resynced = check_identity(b, active_name)
        faints_from_log(b)
        turn = b.current_turn()
        events = self.events_for(b)
        new = events.update(b)
        events.to_turn(turn, b)
        active = b.visible_order[0]
        member = move_peek.member_of(b, self.team, active)
        refused = move_peek.refused_moves(last, turn, same_turn, repeated)
        if same_turn and not refused:
            decision_record.unconfirmed(b, last.get("seq"), "the action screen came back without a new row")
        t = move_peek.Turn(
            member, events.mon(member), active, same_turn, resynced, log_info, turn_before,
            events.all(), tuple(new), self._hints(b, events, member, active, ()),
        )  # fmt: skip
        if why is None and (why := move_peek.why_after_log(b, t)) is not None:
            peek = self.peek(b)
        move_peek.settle_mega(b, t, peek, why)
        refused_order = None
        if refused and last.get("move_id"):
            refused_order = {"kind": "move", "move": last["move_id"], "mega": bool(last.get("mega"))}
            events.refused(refused_order)
        notes: list[str] = []
        offer = request.action_offer(b, member, peek, refused, notes)
        req = request.build(b, self.team, events, offer, self._hints(b, events, member, active, refused), end.at)
        live = decision_record.decision_request(req, "action", refused=refused_order, retry=same_turn)
        req = replace(req, live=live)
        decision = self.runner.decide(req)
        path = {"same_turn": same_turn, "peek": why or move_peek.SKIPPED, "refused": list(refused)}
        self._save(b, "action", req, decision, turn, new, visit=visit, log_turn=log_info.turn, screen=notes, **path)
        result, taken = self._tap_action(b, decision.order, peek)
        if result.error or not taken:
            decision_record.unconfirmed(b, req.request_id, result.error or "the party screen stayed up")
        elapsed = (result.last_tap_at or self.clock.now()) - t0
        self._after_action(b, req, decision, turn, log_info, path, elapsed, t0, taken)
        self.clock.sleep(self.cfg.timing.post_action_wait_s)

    def _tap_action(self, b: BattleRecord, order: Order, peek: Observation | None) -> tuple[ExecResult, bool]:
        if order.kind == "switch":
            slot = b.slot_of(order.name)
            if slot is not None:
                return party_switch.switch(self, b, slot)  # the row by name from the party screen
            log.warning("switch_target_unknown", battle=b.index, name=order.name, names=list(b.names))
            return ExecResult(error=f"no slot for {order.name}"), False
        executor = self.phone.executor
        plan = executor.plan_move(order.row, order.mega, peek.fields.get("focused_row") if peek else None)
        return executor.execute(plan, Screen.ACTION, self.phone.device, self.clock, self.cfg.timing.inter_tap_s), True

    def _after_action(
        self,
        b: BattleRecord,
        req: Request,
        decision: Decision,
        turn: int,
        log_info: LogVisit,
        path: dict,
        elapsed: float,
        t0: float,
        taken: bool = True,
    ) -> None:
        order, same_turn = decision.order, path["same_turn"]
        label = decision_record.choice_label(b, order)
        active_name = b.names[b.visible_order[0]] if b.names else None
        record = {"visit": b.action_visits, "turn": turn, "action": label, "degrade": decision.source}
        record.update(
            log_s=log_info.seconds,
            first_capture_to_last_tap_s=round(elapsed, 2),
            same_turn_retry=same_turn,
            peek=path["peek"],
        )
        b.turns.append(record)
        log.event("turn", battle=b.index, reason=decision.reason, elapsed_s=round(elapsed, 1),
                  log_turn=log_info.turn, **record)  # fmt: skip
        if elapsed > self.cfg.timing.max_turn_s:
            log.warning("turn_overrun", battle=b.index, turn=turn, elapsed_s=round(elapsed, 1))
        slot = b.slot_of(order.name) if order.kind == "switch" else None
        move = move_id(order, req.offer) if order.kind == "move" else None
        action = SemanticAction("move", move=order.row, mega=order.mega) if order.kind == "move" else None
        action = action or SemanticAction("switch", slot=slot)
        retry = " [retry]" if same_turn else ""
        missed = "" if taken else " [not taken: the party screen stayed up]"
        b.actions.setdefault(turn, []).append(action_detail(action, move, order.name) + retry + missed)
        if order.kind == "switch" and slot is not None and taken:
            b.record_switch(slot)
        if order.kind == "move" and order.mega:
            b.mega_pending = True
        b.last_action = {
            "label": label,
            "kind": order.kind,
            "turn": turn,
            "move_name": move,
            "move_id": move,
            "mega": order.mega,
            "switch_name": order.name,
            "active_name": active_name,
            "log_turn": log_info.turn,
            "log_rows": _rows(b, log_info.turn),
            "refused": tuple(path["refused"]),
            "at": t0,
            "seq": req.request_id,
        }

    # Forced replacement --------------------------------------------------------------------------------
    def _begin_forced(
        self, b: BattleRecord, obs: Observation, previous_forced: bool
    ) -> tuple[tuple, int | None, Deadline]:
        """Party rows, the slot leaving on a pivot (else None), and the replacement deadline."""
        party = tuple(obs.fields.get("party", ()))
        check_team(b, party=[m.name for m in party])
        leaving = leaving_slot(b, party)
        resync_party(b, party)
        repeated = previous_forced and b.forced_choice is not None and b.forced_choice.get("turn") == b.current_turn()
        b.forced_attempts = b.forced_attempts + 1 if repeated else 1
        faints_from_log(b)
        if b.forced_attempts == 1:
            b.forced_choice = {"action": None, "first_seen": self.clock.now(), "turn": b.current_turn()}
        assert b.forced_choice is not None
        return party, leaving, Deadline(self.clock, b.forced_choice["first_seen"] + FORCED_RETRY_DEADLINE_S)

    def forced(self, b: BattleRecord, obs: Observation, previous_forced: bool) -> None:
        party, leaving, deadline = self._begin_forced(b, obs, previous_forced)
        assert b.forced_choice is not None
        events = self.events_for(b)
        new = events.update(b)
        # A repeated replacement screen is a retry (LIVE 0.5): the game cannot refuse a healthy member.
        active = leaving if leaving is not None else b.visible_order[0]
        member = move_peek.member_of(b, self.team, active)
        offer = request.forced_offer(b, party, leaving)
        req = request.build(b, self.team, events, offer, self._hints(b, events, member, active, ()), deadline.at)
        req = replace(req, live=decision_record.decision_request(req, "forced", retry=b.forced_attempts > 1))
        if b.forced_attempts > 1:
            decision_record.unconfirmed(b, b.forced_choice.get("seq"), "the replacement screen came back")
        decision = self.runner.decide(req)
        order = decision.order
        if order.kind != "switch":
            order = next((Order.switch(s.name) for s in offer.switches), order)
        slot = b.slot_of(order.name) if order.kind == "switch" else None
        named = [m.row for m in party if order.name and same_name(m.name, order.name) and not m.fainted]
        if order.kind != "switch" or (len(named) != 1 and slot is None):
            # A replacement screen with nobody to send in (the last member goes in by itself): tap nothing.
            log.warning("forced_without_switch", battle=b.index, turn=b.current_turn(), reason=decision.reason)
            self.clock.sleep(self.cfg.timing.post_action_wait_s)
            return
        b.forced_choice.update(action=order.name, seq=req.request_id)
        self._save(b, "forced", req, decision, b.current_turn(), new, pivot=leaving is not None,
                   attempt=b.forced_attempts)  # fmt: skip
        row = named[0] if len(named) == 1 else b.visible_order.index(slot)  # type: ignore[arg-type]
        plan = self.phone.executor.plan_forced(row, obs.fields.get("focused_row"))
        result = self.phone.executor.execute(
            plan, Screen.FORCED, self.phone.device, self.clock, self.cfg.timing.inter_tap_s
        )
        if result.error:
            decision_record.unconfirmed(b, req.request_id, result.error)
        label = decision_record.choice_label(b, order)
        log.event("turn", battle=b.index, turn=b.current_turn(), action=f"forced-{label}", degrade=decision.source,
                  reason=decision.reason, pivot=leaving is not None)  # fmt: skip
        b.actions.setdefault(b.current_turn(), []).append(f"forced replacement: sent in {order.name}")
        visible = b.visible_order
        visible[0], visible[row] = visible[row], visible[0]
        self.clock.sleep(self.cfg.timing.post_action_wait_s)

    # Helpers ---------------------------------------------------------------------------------------------
    def _member_named(self, b: BattleRecord, name: str | None) -> OwnMember | None:
        slot = b.slot_of(name)
        return move_peek.member_of(b, self.team, slot) if slot is not None else None

    def _save(self, b: BattleRecord, kind: str, req: Request, decision: Decision, turn: int, new, **extra) -> None:
        events = self.events_for(b)
        decision_record.save(
            self.rec, b, kind, req, decision, turn=turn, player=getattr(self.runner.player, "name", "?"),
            converter=events.converter, gaps=[e for e in new if e.get("k") == "gap"], **extra,
        )  # fmt: skip


def _rows(b: BattleRecord, turn: int | None) -> int:
    return len(b.log.turns.get(turn, [])) if turn is not None else 0


class _NoSaver:
    def after_visit(self, b: BattleRecord, team: Team | None) -> bool:
        return False

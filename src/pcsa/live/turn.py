"""One battle turn: peek the moves, read the log within its budget, reconstruct the Battle, decide, tap.
With `timing.move_peek: when_needed` the peek is left out unless move_peek.py names a reason for it.

Nothing here refuses to act. A log that could not be read lowers the decision to the screen rung;
a deadline that is nearly gone sends the cheapest plan.
"""

from __future__ import annotations

import re

from pcsa.battle import state as tracking
from pcsa.battle.teams import Team, export_text, teambuilder_team
from pcsa.common import log
from pcsa.common.clock import Clock, Deadline
from pcsa.common.config import AgentConfig
from pcsa.common.types import Observation, Screen
from pcsa.live import move_peek, restore
from pcsa.live.decide import Decider, event_notes
from pcsa.live.log_reader import LogReader, LogVisit
from pcsa.live.log_save import LogSaver
from pcsa.players.base import ScreenFacts
from pcsa.run.battle_state import PIVOT_MOVES, BattleRecord, action_detail, same_name, same_name_in
from pcsa.run.recorder import Recorder

MIN_NEW_TURN_MOVE_TIME = 15  # a lower reading on a fresh action screen is an OCR drop of the first digit
FORCED_RETRY_DEADLINE_S = 40.0


class TurnPipeline:
    def __init__(
        self,
        phone,
        team: Team | None,
        decider: Decider,
        recorder: Recorder,
        config: AgentConfig,
        clock: Clock,
        seed: int,
    ) -> None:
        self.phone = phone
        self.team = team
        self.decider = decider
        self.rec = recorder
        self.cfg = config
        self.clock = clock
        self.seed = seed
        self.logs = LogReader(phone, recorder, config, clock)
        self.log_saver = LogSaver(recorder, seed)

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
            executor.execute([executor.point("move_back")], Screen.MOVES, device, self.clock, 0.0)
            self.clock.sleep(self.cfg.timing.back_settle_s)
            return obs
        if obs is not None and obs.screen == Screen.FORFEIT_MSG and b.outcome is None:
            b.outcome = "opponent_forfeit"  # the message shows only here; the next poll lands on the options screen
        log.warning("peek_failed", battle=b.index, screen=str(obs.screen) if obs else None)
        if obs is not None and obs.screen in (Screen.MOVES, Screen.PARTY, Screen.LOG):
            self.phone.navigator.back_out(obs)
        return None

    # Action turn ---------------------------------------------------------------------------------------
    def _begin_action(self, b: BattleRecord, obs: Observation, t0: float) -> tuple[int, Deadline, str | None]:
        """Bookkeeping for one action screen: the visit number, the deadline, the active name on the screen."""
        b.forced_attempts, b.forced_choice = 0, None
        b.action_visits += 1
        visit = b.action_visits
        move_time = obs.fields.get("move_time")
        active_name = obs.fields.get("active_name")
        if visit == 1:
            b.observed["lead"] = active_name
        opponent = obs.fields.get("opponent_name")
        if opponent and not any(same_name(opponent, seen) for seen in b.opponents):
            b.opponents.append(opponent)
        timed = isinstance(move_time, int) and move_time >= MIN_NEW_TURN_MOVE_TIME
        budget = min(float(move_time) if timed else self.cfg.timing.max_turn_s, self.cfg.timing.max_turn_s)  # type: ignore[arg-type]
        deadline = Deadline(self.clock, t0 + budget).minus(self.cfg.degrade.tap_reserve_s)
        return visit, deadline, active_name

    def action(self, b: BattleRecord, obs: Observation, t0: float, repeated: bool = False) -> None:
        """`repeated`: the poll before this one also showed the action screen (the taps may not have gone through)."""
        visit, deadline, active_name = self._begin_action(b, obs, t0)
        why = move_peek.why_before_log(self.cfg.timing.move_peek, b, self.team, repeated)
        peek = self.peek(b) if why else None
        turn_before = b.last_turn
        log_info = self.logs.collect_log(b, visit, deadline.minus(self.cfg.degrade.decide_reserve_s))
        self.log_saver.after_visit(b, self.team)  # the log files are current while the battle is on
        last = b.last_action or {}
        same_turn = (
            bool(last)
            and log_info.turn is not None
            and log_info.turn == last.get("log_turn")
            and not log_info.appended
        )
        resynced = self._check_identity(b, active_name)
        self._faints_from_log(b)
        turn = b.current_turn()
        battle, out = self._build(b, "action", turn)
        t = move_peek.Turn(battle, out, b.visible_order[0], same_turn, resynced, log_info, turn_before)
        if why is None and (why := move_peek.why_after_log(b, self.team, t)) is not None:
            peek = self.peek(b)  # the log is already read; from here the turn is the peek-first flow
        hp = obs.fields.get("active_hp")
        refused = move_peek.refused_moves(last, turn, same_turn, repeated)
        screen = move_peek.without_moves(move_peek.screen_facts(b, self.team, t, peek, why, hp), refused)
        request = self._request(b, "action", battle, out, t.active, screen, {t.active: tuple(hp)} if hp else {})
        if request is None and why is None:  # not trackable after all: a decision from the screen needs the list
            why, peek = "untracked", self.peek(b)
            screen = move_peek.without_moves(move_peek.screen_facts(b, self.team, t, peek, why, hp), refused)
        if request is None:
            battle, out = None, None
        slot_moves = list(self.team.members[b.requested[t.active]].moves) if self.team and b.requested else []
        decision = self.decider.decide(b, screen, battle, out, deadline, slot_moves=slot_moves, active=t.active)
        path = {"same_turn": same_turn, "peek": why or move_peek.SKIPPED, "refused": list(refused)}
        self._save_decision(b, "action", battle, out, request, decision, visit=visit, log_turn=log_info.turn, **path)
        plan = self.phone.executor.plan_action(
            decision.chosen.action, b.visible_order, peek.fields.get("focused_row") if peek else None
        )
        result = self.phone.executor.execute(
            plan, Screen.ACTION, self.phone.device, self.clock, self.cfg.timing.inter_tap_s
        )
        elapsed = (result.last_tap_at or self.clock.now()) - t0
        self._after_action(b, decision, screen, turn, visit, log_info, path, elapsed, t0)
        self.clock.sleep(self.cfg.timing.post_action_wait_s)

    def _after_action(
        self,
        b: BattleRecord,
        decision,
        screen: ScreenFacts,
        turn: int,
        visit: int,
        log_info: LogVisit,
        path: dict,
        elapsed: float,
        t0: float,
    ) -> None:
        action, same_turn = decision.chosen.action, path["same_turn"]
        record = {"visit": visit, "turn": turn, "action": action.label(), "degrade": str(decision.degrade)}
        record.update(
            log_s=log_info.seconds,
            first_capture_to_last_tap_s=round(elapsed, 2),
            same_turn_retry=same_turn,
            peek=path["peek"],
        )
        b.turns.append(record)
        log.event(
            "turn",
            battle=b.index,
            reason=decision.reason,
            elapsed_s=round(elapsed, 1),
            log_turn=log_info.turn,
            **record,
        )
        if elapsed > self.cfg.timing.max_turn_s:
            log.warning("turn_overrun", battle=b.index, turn=turn, elapsed_s=round(elapsed, 1))
        move_name = (
            next((m.name for m in screen.moves if m.slot == action.move), None) if action.kind == "move" else None
        )
        switch_name = b.names[action.slot] if action.kind == "switch" and action.slot is not None else None
        retry = " [retry]" if same_turn else ""
        b.actions.setdefault(turn, []).append(action_detail(action, move_name, switch_name) + retry)
        if action.kind == "switch" and action.slot is not None:
            b.record_switch(action.slot)
        if action.kind == "move" and action.mega:
            b.mega_pending = True
        b.last_action = {
            "label": action.label(),
            "turn": turn,
            "move_name": move_name,
            "log_turn": log_info.turn,
            "refused": tuple(path["refused"]),
            "at": t0,
        }

    # Forced replacement --------------------------------------------------------------------------------
    def _begin_forced(
        self, b: BattleRecord, obs: Observation, previous_forced: bool
    ) -> tuple[tuple, int | None, Deadline]:
        """Party rows, the slot leaving on a pivot (else None), and the replacement deadline."""
        party = tuple(obs.fields.get("party", ()))
        leaving = b.visible_order[0] if b.names else None
        last = b.last_action or {}
        pivot = last.get("turn") == b.current_turn() and last.get("move_name") in PIVOT_MOVES
        if not (
            pivot
            and leaving is not None
            and any(same_name(m.name, b.names[leaving]) and not m.fainted for m in party)
        ):
            leaving = None
        self._resync_party(b, party)
        repeated = previous_forced and b.forced_choice is not None and b.forced_choice.get("turn") == b.current_turn()
        b.forced_attempts = b.forced_attempts + 1 if repeated else 1
        self._faints_from_log(b)
        if b.forced_attempts == 1:
            b.forced_choice = {"action": None, "first_seen": self.clock.now(), "turn": b.current_turn()}
        assert b.forced_choice is not None
        deadline = Deadline(self.clock, b.forced_choice["first_seen"] + FORCED_RETRY_DEADLINE_S)
        return party, leaving, deadline.minus(self.cfg.degrade.tap_reserve_s)

    def forced(self, b: BattleRecord, obs: Observation, previous_forced: bool) -> None:
        party, leaving, deadline = self._begin_forced(b, obs, previous_forced)
        assert b.forced_choice is not None
        screen_hp = {slot: tuple(m.hp) for m in party if (slot := b.slot_of(m.name)) is not None and m.hp}
        screen = ScreenFacts((), False, True, None)
        battle, out, request = self._track(
            b, "forced", turn=b.current_turn(), active=leaving, screen=screen, screen_hp=screen_hp
        )
        decision = self.decider.decide(
            b,
            screen,
            battle,
            out,
            deadline,
            slot_moves=[],
            active=leaving,
            leaving=leaving,
        )
        action = decision.chosen.action
        if action.kind != "switch" or action.slot is None:
            switches = [a for a in decision.legal if a.action.kind == "switch"]
            action = switches[0].action if switches else action
        b.forced_choice.update(action=action)
        self._save_decision(
            b, "forced", battle, out, request, decision, pivot=leaving is not None, attempt=b.forced_attempts
        )
        named = [m.row for m in party if same_name(m.name, b.names[action.slot]) and not m.fainted]  # type: ignore[index]
        row = named[0] if len(named) == 1 else b.visible_order.index(action.slot)  # type: ignore[arg-type]
        plan = self.phone.executor.plan_forced(row, obs.fields.get("focused_row"))
        self.phone.executor.execute(plan, Screen.FORCED, self.phone.device, self.clock, self.cfg.timing.inter_tap_s)
        log.event(
            "turn",
            battle=b.index,
            turn=b.current_turn(),
            action=f"forced-{action.label()}",
            degrade=str(decision.degrade),
            reason=decision.reason,
            pivot=leaving is not None,
        )
        b.actions.setdefault(b.current_turn(), []).append(f"forced replacement: sent in {b.names[action.slot]}")  # type: ignore[index]
        order = b.visible_order
        order[0], order[row] = order[row], order[0]
        self.clock.sleep(self.cfg.timing.post_action_wait_s)

    # Helpers ---------------------------------------------------------------------------------------------
    def _track(
        self, b: BattleRecord, kind: str, *, turn: int, active: int | None, screen: ScreenFacts, screen_hp: dict
    ):
        battle, out = self._build(b, kind, turn)
        request = self._request(b, kind, battle, out, active, screen, screen_hp)
        return (battle, out, request) if request is not None else (None, None, None)

    def _build(self, b: BattleRecord, kind: str, turn: int):
        """(Battle, Translation) from everything the log holds, before this decision's request; else (None, None)."""
        if self.team is None or b.requested is None:
            return None, None
        try:
            return tracking.build(
                restore.battle_tag(b.battle_id, len(b.decisions)),
                self.team,
                b.opponent_preview,
                picks=b.requested,
                rows_by_turn=b.log.history(),
                decision_turn=turn,
                sprites_by_turn=b.log.sprite_history(),
            )
        except Exception as error:
            log.warning("tracking_failed", battle=b.index, kind=kind, reason=f"{type(error).__name__}: {error}")
            return None, None

    def _request(self, b: BattleRecord, kind: str, battle, out, active, screen: ScreenFacts, screen_hp: dict):
        """Our side's request, applied to the Battle; None when there is no Battle or it did not take the request."""
        if battle is None or self.team is None or b.requested is None:
            return None
        try:
            request = tracking.battle_request(
                self.team,
                b.requested,
                battle,
                active=active,
                screen_hp=screen_hp,
                fainted=set(b.fainted),
                moves=screen.moves,
                mega_available=screen.mega_available,
                forced=screen.forced,
            )
            tracking.apply_request(battle, out, request)
            if any("rejected the request" in w for w in out.warnings):
                raise ValueError("poke-env rejected the request")
            return request
        except Exception as error:
            log.warning("tracking_failed", battle=b.index, kind=kind, reason=f"{type(error).__name__}: {error}")
            return None

    def _save_decision(self, b: BattleRecord, kind: str, battle, out, request, decision, **extra) -> None:
        sequence = len(b.decisions)
        b.decisions.append(kind)
        record = {
            "schema": "pcsa-decision/2",  # /2: battle_id; notes may hold nested values (the neural player's matrix)
            "battle_id": b.battle_id,
            "sequence": sequence,
            "kind": kind,
            "turn": battle.turn if battle is not None else None,
            "choice": decision.chosen.action.label(),
            "order": decision.chosen.message,
            "degrade": str(decision.degrade),
            "reason": decision.reason,
            "legal": [a.message for a in decision.legal],
            "dropped": decision.dropped,
            "notes": decision.notes,
            "request": request,
            "protocol": out.lines if out else None,
            "unmapped": out.unmapped if out else None,
            "warnings": out.warnings if out else None,
            "revealed_abilities": out.revealed_abilities if out else None,
            "inferred": out.inferred if out else None,
            "own_export": export_text(teambuilder_team(self.team)) if self.team is not None else None,
            "own_names": list(self.team.names) if self.team is not None else None,
            **extra,
        }
        self.rec.write_json(f"{b.dir}/decisions/{sequence:04d}-{kind}.json", record)
        if battle is not None and out is not None:
            self.rec.write_text(
                f"{b.dir}/decisions/{sequence:04d}-{kind}.txt", tracking.battle_text(battle, out.revealed_abilities)
            )
        log.event(
            "decision",
            battle=b.index,
            kind=kind,
            choice=decision.chosen.action.label(),
            degrade=str(decision.degrade),
            legal=len(decision.legal),
            dropped=len(decision.dropped),
            **event_notes(decision.notes),  # the record above keeps the full notes
        )

    def _check_identity(self, b: BattleRecord, active: str | None) -> bool:
        """True when the screen named another active member than the bookkeeping (which is then resynced)."""
        expected = b.names[b.visible_order[0]] if b.names else None
        if not (expected and active) or same_name(expected, active):
            return False
        slot = b.slot_of(active)
        rows = b.log.turns.get(b.last_turn, []) if b.last_turn else []
        explained = slot is not None and any(
            same_name_in(row, b.names[slot]) and re.search(r"dragged out|go!|sent out|was forced", row, re.I)
            for row in rows
        )
        if slot is not None:
            b.record_switch(slot)
        log.event("identity_resync", battle=b.index, expected=expected, observed=active, explained=explained)
        return True

    def _faints_from_log(self, b: BattleRecord) -> None:
        for turn, rows in b.log.turns.items():
            for text in rows:
                match = re.fullmatch(r"(?:Fainted:\s*)?(?!The opposing)(.+?) fainted!", text.strip())
                slot = b.slot_of(match[1]) if match else None
                if slot is not None and slot not in b.fainted:
                    b.fainted.add(slot)
                    log.event("fainted_from_log", battle=b.index, slot=slot, turn=turn)

    def _resync_party(self, b: BattleRecord, party) -> None:
        slots = [b.slot_of(m.name) for m in party]
        if None not in slots and sorted(slots) == [0, 1, 2]:  # type: ignore[type-var]
            if slots != b.visible_order:
                log.event("order_resync", battle=b.index, before=list(b.visible_order), after=slots)
            b.visible_order = slots  # type: ignore[assignment]
        for m in party:
            if m.fainted and m.row < len(b.visible_order):
                b.fainted.add(b.visible_order[m.row])
        b.observed["party_names"] = [m.name for m in party]

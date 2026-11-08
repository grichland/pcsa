"""What the session does on each screen outside a battle turn: lobby, team selection, the end of a battle,
promotion popups, interruptions. In-battle turns are in turn.py."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from pcsa.battle import state as tracking
from pcsa.battle.teams import teambuilder_team
from pcsa.common import log
from pcsa.common.clock import Deadline
from pcsa.common.types import Observation, Screen
from pcsa.live.team_read import TeamReadSession
from pcsa.phone.log_merge import BattleLog
from pcsa.phone.rank import rank_text
from pcsa.phone.screens import to_reference
from pcsa.phone.sprites import match_team_preview
from pcsa.players.base import TeamPreviewView
from pcsa.run.battle_state import BattleRecord, reconcile_roster

MAX_BEGIN_TAPS = 2
MAX_TEAM_REPAIRS = 2
MAX_CONTINUE_TAPS = 3
LOBBY_SETTLE_S = 2.0


def on_lobby(s, obs: Observation) -> None:
    if s.team is None:
        sheet, team = TeamReadSession(s.phone, s.agent.timing.team_menu_settle_s).read()
        s.team = team
        s.state.team_record = team.record()
        s.rec.write_json("own_team.json", {"sheet": sheet.record(), "team": team.record()})
        s.rec.write_text("own_team.txt", tracking_export(team))
        s.state.save()
        return
    if s.state.done():
        s.finished = True
        return
    if s.begin_taps < MAX_BEGIN_TAPS:
        s.begin_taps += 1
        s.tap("begin_matchmaking", obs)
    s.clock.sleep(LOBBY_SETTLE_S)


def tracking_export(team) -> str:
    from pcsa.battle.teams import export_text

    return export_text(teambuilder_team(team))


def new_battle(s, roster: tuple[str | None, ...], adopted: bool = False) -> BattleRecord:
    if s.battle is not None and not s.battle.finished:
        finalize(s, None)  # the previous battle ended while we were away (disconnect, restart)
    if s.team is not None and roster and len(roster) == len(s.team.names):
        reconciled = reconcile_roster(s.team.names, roster)
        if reconciled is None:
            log.warning("roster_mismatch", expected=s.team.names, observed=roster)
        else:
            roster = reconciled[0]
    battle_id = f"{datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    index = s.state.played + 1
    b = BattleRecord(
        index,
        battle_id,
        tuple(roster) if roster else (s.team.names if s.team else ()),
        BattleLog(battle_id),
        started_at=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        adopted=adopted,
        rank_before=s.latest_rank,
    )
    s.battle = b
    s.begin_taps = 0
    s.continue_taps = 0
    s.state.pending_battle = index
    s.state.save()
    log.event("battle_start", battle=index, battle_id=battle_id, roster=list(roster), adopted=adopted)
    return b


def on_team_select(s, obs: Observation, frame, t0: float) -> None:
    repairing = (
        s.battle is not None
        and not s.battle.finished
        and s.battle.action_visits == 0
        and s.previous is not None
        and s.previous.screen in (Screen.TEAM_SELECT, Screen.TRANSITION)
    )
    if not repairing:
        new_battle(s, tuple(obs.fields.get("roster", ())))
    b = s.battle
    assert b is not None
    if not b.team_submitted:
        b.opponent_preview = tuple(match_team_preview(to_reference(frame.image, s.phone.profile.viewport)))
        b.preview_frame = obs.frame_id
        log.event("team_preview", battle=b.index, opponent=[m.name for m in b.opponent_preview])
        seconds = _selection_seconds(obs)
        b.team_deadline = t0 + (seconds if seconds is not None else s.agent.timing.team_budget_s)
        deadline = Deadline(s.clock, b.team_deadline).minus(s.agent.degrade.tap_reserve_s)
        battle = None
        if s.team is not None:
            try:
                battle, _ = tracking.build(f"{b.battle_id}-preview", s.team, b.opponent_preview)
            except Exception as error:
                log.warning("tracking_failed", battle=b.index, kind="team", reason=str(error))
        view = TeamPreviewView(battle, b.roster, s.rng)
        try:
            team = list(s.player.choose_team(view, deadline))
            if len(team) != 3 or len(set(team)) != 3 or any(not 0 <= i < 6 for i in team):
                raise ValueError(f"invalid lineup {team}")
        except Exception as error:
            log.exception("player_failed", battle=b.index, kind="team")
            team = s.rng.sample(range(6), 3)
            log.warning("team_fallback", battle=b.index, reason=str(error))
        b.requested = team
        b.team_submitted = True
        log.event(
            "decision",
            battle=b.index,
            kind="team",
            team=team,
            names=[b.roster[i] if i < len(b.roster) else None for i in team],
            **s.player.notes(),
        )
        s.execute(s.phone.executor.plan_team(team), Screen.TEAM_SELECT, obs)
        return
    repair_team_selection(s, obs, b)


def repair_team_selection(s, obs: Observation, b: BattleRecord) -> None:
    """Still on team selection after submitting: finish the lineup from the visible markers (bounded)."""
    b.team_repairs += 1
    if b.team_repairs > MAX_TEAM_REPAIRS:
        log.warning("team_repair_exhausted", battle=b.index)
        s.clock.sleep(s.agent.timing.poll_interval_s)
        return
    markers = obs.fields.get("markers", {})
    current = [row for row, _ in sorted(markers.items(), key=lambda kv: kv[1])]
    assert b.requested is not None
    if current == b.requested:
        plan = s.phone.executor.plan_team([])
    elif current == b.requested[: len(current)]:
        plan = s.phone.executor.plan_team(b.requested[len(current) :])
    else:
        plan = s.phone.executor.plan_team(list(reversed(current)))[:-1] + s.phone.executor.plan_team(b.requested)
    log.warning("team_repair", battle=b.index, markers=markers, requested=b.requested)
    s.execute(plan, Screen.TEAM_SELECT, obs)


def _selection_seconds(obs: Observation) -> int | None:
    parsed = re.fullmatch(r"(\d+):(\d{2})", str(obs.fields.get("timer") or ""))
    return int(parsed[1]) * 60 + int(parsed[2]) if parsed else None


def finalize(s, outcome: str | None) -> None:
    b = s.battle
    if b is None or b.finished:
        return
    if outcome == "win" and b.outcome is None and not b.turns:
        b.outcome, b.outcome_source = "opponent_forfeit", "win before any turn was played"
    elif b.outcome == "opponent_forfeit":
        b.outcome_source = "forfeit message"
    elif outcome:
        b.outcome, b.outcome_source = outcome, "result screen"
    elif b.outcome_hint:
        b.outcome, b.outcome_source = b.outcome_hint, "final battle message"
    if b.outcome is None:
        b.outcome, b.outcome_source = "unknown", "battle ended without a readable result"
        b.log.note(
            "outcome_not_observed",
            "the battle ended during a blind wait; no result screen seen",
            turn=b.current_turn(),
        )
    b.log.note("final_turn", f"turn {b.current_turn()}: the final turn is not collectable", turn=b.current_turn())
    b.log.save(s.rec.dir / b.dir)
    write_standard_log(s, b)
    if b.requested and s.team is not None:
        try:
            battle, out = tracking.build(
                f"{b.battle_id}-final",
                s.team,
                b.opponent_preview,
                picks=b.requested,
                rows_by_turn=b.log.history(),
                sprites_by_turn=b.log.sprite_history(),
            )
            winner = {"win": "me", "opponent_forfeit": "me", "loss": "opponent"}.get(b.outcome)
            lines = out.lines + ([f"|win|{winner}"] if winner else [])
            s.rec.write_text(f"{b.dir}/showdown.log", "\n".join(lines) + "\n")
            s.rec.write_text(f"{b.dir}/battle_final.txt", tracking.battle_text(battle, out.revealed_abilities))
        except Exception as error:
            log.warning("final_tracking_failed", battle=b.index, reason=str(error))
    counts = s.state.count_forfeits or b.outcome not in ("opponent_forfeit",)
    s.rec.write_json(f"{b.dir}/battle.json", b.summary(s.state.seed, counts))
    b.finished = True
    s.state.played += 1
    s.state.outcomes.append(b.outcome)
    s.state.pending_battle = None
    if counts and b.outcome in ("win", "loss"):
        s.state.completed += 1
    s.state.save()
    log.event(
        "battle_end",
        battle=b.index,
        outcome=b.outcome,
        source=b.outcome_source,
        completed=s.state.completed,
        played=s.state.played,
        degrade=b.degrade_counts,
    )


def write_standard_log(s, b: BattleRecord) -> None:
    team = ", ".join(f"{name} (roster row {row + 1})" for name, row in zip(b.names, b.requested or [], strict=False))
    header = {
        "battle_id": b.battle_id,
        "battle_number": b.index,
        "started": b.started_at,
        "own_team": s.team.name if s.team else "unknown",
        "own_rank_before": rank_text(b.rank_before),
        "own_rank_after": rank_text(b.rank_after),
        "team_requested": team or "unknown",
        "lead_observed": b.observed.get("lead") or "unknown",
        "opponents_seen": ", ".join(b.opponents) or "none",
        "opponent_preview": ", ".join(m.name or "?" for m in b.opponent_preview) or "unknown",
        "outcome": b.outcome or "unknown",
        "outcome_source": b.outcome_source or "none",
        "seed": s.state.seed,
    }
    s.rec.write_text(f"{b.dir}/battle_log.txt", b.log.standard_text(header, b.actions))


def on_result(s, obs: Observation) -> None:
    finalize(s, obs.fields.get("outcome"))
    s.clock.sleep(s.agent.timing.poll_interval_s)


def on_forfeit(s, obs: Observation) -> None:
    if s.battle and not s.battle.finished and s.battle.outcome is None:
        s.battle.outcome = "opponent_forfeit"
    s.clock.sleep(s.agent.timing.poll_interval_s)


def on_options(s, obs: Observation) -> None:
    finalize(s, None)
    if s.state.done():
        s.finished = True
        return
    s.continue_taps += 1
    if s.continue_taps > MAX_CONTINUE_TAPS:
        log.warning("continue_not_accepted", taps=s.continue_taps)
        s.phone.navigator.reach(Screen.LOBBY, Deadline.after(s.clock, 60))
        return
    s.tap("continue_battling", obs)
    s.clock.sleep(3.0)


def on_waiting(s, obs: Observation) -> None:
    hint = obs.fields.get("outcome_hint")
    if hint and s.battle and not s.battle.finished and s.battle.outcome_hint != hint:
        s.battle.outcome_hint = hint
        log.event("outcome_hint", battle=s.battle.index, hint=hint)
    s.clock.sleep(s.agent.timing.poll_interval_s)


def on_rank_change(s, obs: Observation) -> None:
    """Rewards, rank-up popups, dialogs, menus after a promotion: finalize, then walk back to the lobby."""
    finalize(s, None)
    if obs.screen == Screen.REWARDS:
        s.clock.sleep(s.agent.timing.poll_interval_s)
        return
    if obs.screen == Screen.RANK_UP and obs.fields.get("stage") != "reward_popup":
        s.clock.sleep(s.agent.timing.poll_interval_s)  # cutscene or moving popup: wait
        return
    s.phone.navigator.reach(
        Screen.LOBBY, Deadline.after(s.clock, s.agent.timing.screen_timeouts.get(str(obs.screen), 60))
    )


def on_interruption(s, obs: Observation) -> None:
    """Communication error or date change: note the pending battle, dismiss, walk back to the lobby."""
    if s.battle and not s.battle.finished:
        log.warning("battle_interrupted", battle=s.battle.index, kind=obs.fields.get("kind"))
    s.phone.navigator.reach(Screen.LOBBY, Deadline.after(s.clock, 180))


def on_stray(s, obs: Observation) -> None:
    log.warning("stray_surface", screen=str(obs.screen))
    s.phone.navigator.back_out(obs)

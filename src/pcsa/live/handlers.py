"""What the session does on each screen outside a battle turn: lobby, team selection, the end of a battle,
promotion popups, interruptions. In-battle turns are in turn.py."""

from __future__ import annotations

import re
import uuid
from dataclasses import replace
from datetime import datetime

from pcsa.battle.teams import export_text, teambuilder_team
from pcsa.common import log
from pcsa.common.clock import Deadline
from pcsa.common.types import Observation, Screen, jsonable
from pcsa.decide.api import OUTCOMES, TeamPreviewRequest
from pcsa.live.decision_record import preview_cg
from pcsa.live.log_save import write_log
from pcsa.live.team_read import TeamReadSession, own_team
from pcsa.phone.log_merge import BattleLog
from pcsa.run.battle_state import BattleRecord, reconcile_roster

MAX_BEGIN_TAPS = 2
MAX_TEAM_REPAIRS = 2
MAX_CONTINUE_TAPS = 3
MAX_RATING_WAITS = 3  # polls to wait for the rating to change
MAX_RATING_STEP = 100.0  # a larger change is a misread, not one battle
LOBBY_SETTLE_S = 2.0
EVENT_LIST_LIMIT = 8
# Team selection seen again after a communication error is the same match: repair its lineup, not a new battle.
RECONNECT_SCREENS = (Screen.COMM_ERROR, Screen.UNKNOWN, Screen.DIALOG)


def on_lobby(s, obs: Observation) -> None:
    if s.battle is not None and not s.battle.finished:
        finalize(s, None)  # the battle ended while we were away
    drop_pending(s)
    if s.team is None:
        sheet, team = TeamReadSession(s.phone, s.agent.timing.team_menu_settle_s).read()
        s.team = team
        s.state.team_record = team.record()
        s.rec.write_json("own_team.json", {"sheet": sheet.record(), "team": team.record()})
        s.rec.write_text("own_team.txt", export_text(teambuilder_team(team)))
        s.state.save()
        return
    if s.state.done():
        s.finished = True
        return
    if s.begin_taps < MAX_BEGIN_TAPS:
        s.begin_taps += 1
        s.tap("begin_matchmaking", obs)
    s.clock.sleep(LOBBY_SETTLE_S)


def drop_pending(s) -> None:
    """The run state names a battle in progress that this process does not hold (SV-7: not adopted).
    It counts as played with an unknown outcome, so its folder is never reused."""
    index = s.state.pending_battle
    if index is None or (s.battle is not None and s.battle.index == index):
        return
    log.warning("battle_not_adopted", battle=index, reason="in progress when the run was resumed")
    s.rec.write_json(
        f"battle-{index}/not_adopted.json",
        {"index": index, "outcome": "unknown", "outcome_source": "in progress when the run was resumed"},
    )
    s.state.played = max(s.state.played, index)
    s.state.outcomes.append("unknown")
    s.state.pending_battle = None
    s.state.save()


def new_battle(s, roster: tuple[str | None, ...]) -> BattleRecord:
    if s.battle is not None and not s.battle.finished:
        finalize(s, None)  # the previous battle ended while we were away
    drop_pending(s)
    if s.team is not None and roster and len(roster) == len(s.team.names):
        reconciled = reconcile_roster(s.team.names, roster)
        if reconciled is None:
            log.warning("roster_mismatch", expected=s.team.names, observed=roster)
        else:
            roster = reconciled[0]
    index = s.state.played + 1
    battle_id = f"{datetime.now():%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    b = BattleRecord(
        index,
        battle_id,
        tuple(roster) if roster else (s.team.names if s.team else ()),
        BattleLog(battle_id),
        started_at=f"{datetime.now():%Y-%m-%d %H:%M:%S}",
        rank_before=s.latest_rank,
    )
    s.battle = b
    s.begin_taps = 0
    s.continue_taps = 0
    s.state.pending_battle = index
    s.state.save()
    s.rec.write_json(f"{b.dir}/battle_id.json", {"battle_id": battle_id})
    log.event("battle_start", battle=index, battle_id=battle_id, roster=list(roster))
    s.runner.battle_start(battle_id, own_team(s.team, b.roster))
    return b


def read_preview(s, frame) -> tuple:
    """The opponent's six from the preview sprites; () when they cannot be read."""
    try:
        from pcsa.phone.screens import to_reference
        from pcsa.phone.sprites import match_team_preview

        return tuple(match_team_preview(to_reference(frame.image, s.phone.profile.viewport)))
    except Exception as error:
        log.warning("preview_unread", reason=f"{type(error).__name__}: {error}")
        return ()


def on_team_select(s, obs: Observation, frame, t0: float) -> None:
    b = s.battle
    unplayed = b is not None and not b.finished and b.action_visits == 0
    previous = s.previous.screen if s.previous is not None else None
    repairing = unplayed and previous in (Screen.TEAM_SELECT, Screen.TRANSITION, *RECONNECT_SCREENS)
    if unplayed and (previous in RECONNECT_SCREENS or previous is None):
        repairing = _same_preview(s, frame, b)  # type: ignore[arg-type]
        log.warning(
            "team_select_after_disconnect" if previous else "team_select_after_restart",
            battle=b.index,  # type: ignore[union-attr]
            timer=obs.fields.get("timer"),
            same_match=repairing,
        )
    if not repairing:
        new_battle(s, tuple(obs.fields.get("roster", ())))
    b = s.battle
    assert b is not None
    if not b.team_submitted:
        submit_team(s, obs, frame, t0, b)
        return
    repair_team_selection(s, obs, b)


def submit_team(s, obs: Observation, frame, t0: float, b: BattleRecord) -> None:
    """Ask the Player for the lineup through the runner and tap it."""
    b.opponent_preview = read_preview(s, frame)
    b.preview_frame = obs.frame_id
    log.event("team_preview", battle=b.index, opponent=[m.name for m in b.opponent_preview])
    seconds = _selection_seconds(obs)
    b.team_deadline = t0 + (seconds if seconds is not None else s.agent.timing.team_budget_s)
    request = TeamPreviewRequest(
        b.battle_id,
        own_team(s.team, b.roster),
        tuple(m.name or None for m in b.opponent_preview),
        b.team_deadline,
        _battle_start(s, b),
    )
    decision = s.runner.teampreview(request)
    team = list(decision.picks)
    b.requested = team
    b.team_submitted = True
    notes, cg, flat = preview_cg(decision.notes)
    preview = {"request": request, "decision": replace(decision, notes=notes), "frame": obs.frame_id}
    s.rec.write_json(f"{b.dir}/team_preview.json", {**preview, "cg": cg} if cg is not None else preview)
    log.event(
        "decision",
        battle=b.index,
        kind="team",
        team=team,
        names=[b.roster[i] if i < len(b.roster) else None for i in team],
        source=decision.source,
        player_ms=decision.player_ms,
        **event_notes(notes),
        **flat,
    )
    s.execute(s.phone.executor.plan_team(team), Screen.TEAM_SELECT, obs)


def _battle_start(s, b: BattleRecord) -> tuple[dict, ...]:
    """The LIVE battle_start (our sets and the preview just read) for the preview request; () without a pipeline."""
    if not hasattr(s.turns, "events_for") or s.team is None:
        return ()
    s.turns.team = s.team  # as in_battle does, so the list is not rebuilt with another battle_start
    return (s.turns.events_for(b).start,)


def event_notes(notes: dict) -> dict:
    """A Player's notes for the flattened `decision` event: scalars and short lists; the rest stays in the record."""
    return {
        key: value
        for key, value in jsonable(notes).items()
        if not isinstance(value, dict) and not (isinstance(value, list) and len(value) > EVENT_LIST_LIMIT)
    }


def _same_preview(s, frame, b: BattleRecord) -> bool:
    """The selection screen shows the opponent this battle's preview showed (the same match after a disconnect)."""
    if not b.opponent_preview:
        return False
    shown = read_preview(s, frame)
    return [m.name for m in shown] == [m.name for m in b.opponent_preview]


def repair_team_selection(s, obs: Observation, b: BattleRecord) -> None:
    """Still on team selection after submitting: finish the lineup from the visible markers (bounded)."""
    b.team_repairs += 1
    if b.team_repairs > MAX_TEAM_REPAIRS:
        log.warning("team_repair_exhausted", battle=b.index)
        s.pause()
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
    _settle_outcome(b, outcome)
    if b.outcome is None:
        b.outcome, b.outcome_source = "unknown", "battle ended without a readable result"
        b.log.note(
            "outcome_not_observed",
            "the battle ended during a blind wait; no result screen seen",
            turn=b.current_turn(),
        )
    b.log.note("final_turn", f"turn {b.current_turn()}: the final turn is not collectable", turn=b.current_turn())
    try:
        write_log(s.rec, b, s.team, s.state.seed)
    except Exception as error:
        log.warning("final_log_failed", battle=b.index, reason=f"{type(error).__name__}: {error}")
    counts = s.state.count_forfeits or b.outcome not in ("opponent_forfeit",)
    s.rec.write_json(f"{b.dir}/battle.json", b.summary(s.state.seed, counts))
    b.finished = True
    s.state.played += 1
    s.state.outcomes.append(b.outcome)
    s.state.pending_battle = None
    if counts and b.outcome in ("win", "loss"):
        s.state.completed += 1
    s.state.save()
    result = b.outcome if b.outcome in OUTCOMES else "unknown"
    end_events, events = getattr(s.turns, "end", None), ()  # the battle's event list gets its `end`
    if end_events is not None:
        try:
            events = tuple(end_events(b, result) or ())
        except Exception as error:
            log.warning("battle_end_event_failed", battle=b.index, reason=f"{type(error).__name__}: {error}")
    s.runner.battle_end(b.battle_id, result, events, tuple(b.live_history))
    log.event(
        "battle_end",
        battle=b.index,
        outcome=b.outcome,
        source=b.outcome_source,
        completed=s.state.completed,
        played=s.state.played,
        degrade=b.degrade_counts,
    )


def _settle_outcome(b: BattleRecord, outcome: str | None) -> None:
    """The outcome from the best source seen: result screen, forfeit, final message, then the rating change."""
    if outcome == "win" and b.outcome is None and not b.turns:
        b.outcome, b.outcome_source = "opponent_forfeit", "win before any turn was played"
    elif b.outcome == "opponent_forfeit":
        b.outcome_source = "forfeit message"
    elif outcome:
        b.outcome, b.outcome_source = outcome, "result screen"
    elif b.outcome_hint:
        b.outcome, b.outcome_source = b.outcome_hint, "final battle message"
    elif b.outcome is None and (rated := rating_outcome(b)) is not None:
        change = f"{(b.rank_before or {}).get('rating')} -> {(b.rank_after or {}).get('rating')}"
        b.outcome, b.outcome_source = rated, f"rating change {change}"


def rating_outcome(b: BattleRecord) -> str | None:
    """Win or loss from the rating on the rewards/options screen against the rating before the battle; None when
    either is unknown, unchanged, or the change is too large to be one battle (an OCR misread)."""
    before = (b.rank_before or {}).get("rating")
    after = (b.rank_after or {}).get("rating")
    if before is None or after is None or after == before or abs(after - before) > MAX_RATING_STEP:
        return None
    return "win" if after > before else "loss"


def awaiting_rating(s) -> bool:
    """On the rewards screen with no outcome yet the rating animates: finalizing waits a few polls for the change."""
    b = s.battle
    if b is None or b.finished or b.outcome is not None or b.outcome_hint or rating_outcome(b) is not None:
        return False
    if (b.rank_before or {}).get("rating") is None:
        return False
    b.rating_waits += 1
    return b.rating_waits <= MAX_RATING_WAITS


def on_result(s, obs: Observation) -> None:
    finalize(s, obs.fields.get("outcome"))
    s.pause()


def on_forfeit(s, obs: Observation) -> None:
    if s.battle and not s.battle.finished and s.battle.outcome is None:
        s.battle.outcome = "opponent_forfeit"
    s.pause()


def on_options(s, obs: Observation) -> None:
    finalize(s, None)
    drop_pending(s)
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
    s.pause()


def on_not_adopted(s, obs: Observation) -> None:
    """A battle screen of a battle this run did not start (SV-7): no taps; wait for it to end."""
    if s.skipped_screen != obs.screen:
        s.skipped_screen = obs.screen
        log.warning("battle_not_adopted", screen=str(obs.screen), frame=obs.frame_id)
        s.progress()  # a new screen of the battle being waited out is progress
    s.pause()


def on_rank_change(s, obs: Observation) -> None:
    """Rewards, rank-up popups, dialogs, menus after a promotion: finalize, then walk back to the lobby."""
    if obs.screen == Screen.REWARDS:
        if not awaiting_rating(s):
            finalize(s, None)
        s.pause()
        return
    finalize(s, None)
    if obs.screen == Screen.RANK_UP and obs.fields.get("stage") != "reward_popup":
        s.pause()  # cutscene or moving popup: wait
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
    if obs.screen == Screen.PARTY and obs.fields.get("no_back"):
        # No Back to tap: a pivot (U-turn) or a replacement nothing marked fainted; polled again.
        log.event("party_without_back", focused_row=obs.fields.get("focused_row"))
        s.pause()
        return
    log.warning("stray_surface", screen=str(obs.screen))
    s.phone.navigator.back_out(obs)

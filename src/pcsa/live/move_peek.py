"""`timing.move_peek: when_needed`: an action turn without the look at the move list, and when it needs one.

Without the look, the turn takes from the team sheet and the log what the move list would have shown:
move names by slot from the sheet, PP from the log count (`battle.state.logged_pp`, lowered by what an
earlier look of this battle showed the count had missed), and the Mega button from "the active holds its
stone and we have not Mega Evolved". The log count can be too high (a `used` row the log read lost), so the
turn looks whenever that could matter; `why_before_log` and `why_after_log` name the reason, and a turn that
has one runs today's flow unchanged. Measured on the recorded runs: research/pcsa-rl/runs/fast-turn/pp-check.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle import state as tracking
from pcsa.battle.moves import observed_form
from pcsa.battle.teams import OwnMember, Team
from pcsa.common.types import MoveInfo, Observation
from pcsa.live.log_reader import LogVisit
from pcsa.players.base import ScreenFacts
from pcsa.run.battle_state import BattleRecord

ALWAYS = "always"  # the setting's default: every action turn looks first
WHEN_NEEDED = "when_needed"
SKIPPED = "skipped"  # what a turn without the look records
PEEK_PP_AT_MOST = 4  # look when a move's counted PP is this low: 2 PP of room plus the largest miss measured (2)
SETTLED_STOPS = ("top_confirmed", "settled_no_scrollbar")  # log collections that reached the turn's first row


@dataclass(frozen=True)
class Turn:
    """What the action turn knows once its log is read."""

    battle: object | None  # the tracked Battle, before this turn's request
    out: object | None  # its Translation
    active: int  # our active member's selected slot
    same_turn: bool  # the log shows no new turn: the last taps did not go through
    resynced: bool  # the action screen named another active member than the bookkeeping
    log: LogVisit
    turn_before: int | None  # the last turn the log held before this visit


def member_of(b: BattleRecord, team: Team | None, active: int) -> OwnMember | None:
    if team is None or b.requested is None or not 0 <= active < len(b.requested):
        return None
    return team.members[b.requested[active]]


def why_before_log(mode: str, b: BattleRecord, team: Team | None, repeated: bool) -> str | None:
    """A reason to look that is known before the log is read; None when there is none yet."""
    if mode != WHEN_NEEDED:
        return ALWAYS
    if team is None or b.requested is None:
        return "untracked"  # decided from the screen alone: that needs the move list
    if b.adopted:
        return "adopted"  # picked up in the middle: the log count starts late
    if repeated and b.last_action is not None:
        return "retry"  # the action screen again with no other screen between: the taps did not go through
    return None


def log_doubt(t: Turn, b: BattleRecord) -> str | None:
    """Why this turn's log read may have lost a row, or None."""
    info = t.log
    if info.errors:
        return "log_errors"
    if not info.complete:
        return "log_incomplete"
    if info.stop_reason not in SETTLED_STOPS:
        return f"log_stop_{info.stop_reason or 'unknown'}"
    if info.warnings:
        return "log_warnings"
    if info.turn is not None and t.turn_before is not None and info.turn > t.turn_before + 1:
        return "log_skipped_turn"  # a turn without an action screen of ours was never read
    if t.out is not None and doubts(t.out) > b.log_doubts:
        return "log_unmapped"  # a row the translation could not use
    return None


def doubts(out) -> int:
    """Rows the translation could not use, and lines the Battle did not take."""
    return len(out.unmapped) + sum("rejected" in w for w in out.warnings)


def why_after_log(b: BattleRecord, team: Team | None, t: Turn) -> str | None:
    """A reason to look now that the log is read; None when the sheet and the log are enough."""
    if t.same_turn:
        return "retry"
    member = member_of(b, team, t.active)
    if t.battle is None or member is None:
        return "untracked"
    doubt = log_doubt(t, b)
    if doubt is not None:
        return doubt
    if t.resynced:
        return "identity"
    if len(member.moves) == 0 or any(not name for name in member.moves):
        return "sheet"  # a move slot the team sheet did not read
    if any(m.pp is not None and m.pp[0] <= PEEK_PP_AT_MOST for m in logged_rows(b, member, t)):
        return "low_pp"
    if b.mega_pending and not evolved(t.battle):
        return "mega_unconfirmed"  # Mega was chosen last turn and the log has no Mega Evolution of ours
    return None


# What stands in for the move list ---------------------------------------------------------------------------
def _key(active: int, name: str) -> str:
    return f"{active}:{to_id_str(name)}"


def logged_rows(b: BattleRecord, member: OwnMember, t: Turn) -> tuple[MoveInfo, ...]:
    """The move list as the sheet and the log give it: names by slot, PP by the log count."""
    mon = t.battle.team.get(f"p1: {member.name}") if t.battle is not None else None
    rows = []
    for slot, name in enumerate(member.moves):
        if not name:
            continue
        left, maximum = tracking.logged_pp(mon, name)
        left = max(0, left + b.pp_offsets.get(_key(t.active, name), 0))
        rows.append(MoveInfo(slot, name, (left, maximum), False, False))
    return tuple(rows)


def holds_stone(member: OwnMember, battle) -> bool:
    mon = battle.team.get(f"p1: {member.name}") if battle is not None else None
    item = member.item
    if mon is not None and mon.item != GenData.UNKNOWN_ITEM:
        item = mon.item  # as the request reports it: revealed, changed or removed
    return bool(item) and game.mega_form_for_stone(member.species, item) is not None


def evolved(battle) -> bool:
    """One of ours has Mega Evolved in this battle's log (the Mega Evolution row, or a Mega form seen since)."""
    lines = getattr(battle, "_replay_data", [])
    if any(parts[1:2] == ["-mega"] and "".join(parts[2:3]).startswith("p1") for parts in lines):
        return True
    return any(game.is_mega(observed_form(battle, mon)) for mon in battle.team.values())


def remember(b: BattleRecord, team: Team | None, t: Turn, peek: Observation, was_pending: bool) -> None:
    """Keep what a look showed that the log does not: PP the count missed, and a Mega button that is gone."""
    member = member_of(b, team, t.active)
    if member is None or t.battle is None:
        return
    mon = t.battle.team.get(f"p1: {member.name}")
    for m in peek.fields.get("moves", ()):
        name = member.moves[m.slot] if m.slot < len(member.moves) else None
        if not name or m.pp is None:
            continue
        missed = min(0, m.pp[0] - tracking.logged_pp(mon, name)[0])  # only ever lowers the count
        if missed:
            b.pp_offsets[_key(t.active, name)] = missed
        else:
            b.pp_offsets.pop(_key(t.active, name), None)
    shown = bool(peek.fields.get("mega_available"))
    if not t.same_turn and not shown and (was_pending or holds_stone(member, t.battle)):
        b.mega_spent = True  # one Mega Evolution a battle: the button does not come back


def screen_facts(
    b: BattleRecord, team: Team | None, t: Turn, peek: Observation | None, why: str | None, active_hp
) -> ScreenFacts:
    """The turn's ScreenFacts: from the look when there was a reason for one, else from the sheet and the log."""
    was_pending = b.mega_pending
    if t.out is not None:
        b.log_doubts = max(b.log_doubts, doubts(t.out))
    if why is not None:  # today's flow
        if b.mega_pending:
            b.mega_pending = False
            b.mega_used = not (peek and peek.fields.get("mega_available") and not t.same_turn)
        if why != ALWAYS and peek is not None:
            remember(b, team, t, peek, was_pending)
        moves = tuple(peek.fields.get("moves", ())) if peek else ()
        return ScreenFacts(moves, bool(peek and peek.fields.get("mega_available")), False, active_hp)
    member = member_of(b, team, t.active)
    assert member is not None and t.battle is not None  # why_after_log looks otherwise
    done = evolved(t.battle)
    if b.mega_pending:  # why_after_log looks unless the log shows the Mega Evolution chosen last turn
        b.mega_pending, b.mega_used = False, True
    b.mega_spent = b.mega_spent or done
    mega = holds_stone(member, t.battle) and not b.mega_spent
    return ScreenFacts(logged_rows(b, member, t), mega, False, active_hp)

"""`timing.move_peek: when_needed`: an action turn without the look at the move list, and when it needs one.

Without the look the Offer's move rows are the team sheet's, with PP, disabled and the Mega button unread. The
reasons to look come from the events and the visit: counted PP low, the log read lost or doubted rows, the last
taps did not go through, a Mega chosen last turn is not in the log, a Mega that only the button can offer, a
battle picked up midway, or no team to read moves from.
"""

from __future__ import annotations

from dataclasses import dataclass

from pcsa.battle import game, hints
from pcsa.battle.ids import to_id
from pcsa.battle.teams import OwnMember, Team
from pcsa.common.types import Observation
from pcsa.decide.api import Hints
from pcsa.live.log_reader import LogVisit
from pcsa.run.battle_state import BattleRecord

ALWAYS = "always"
WHEN_NEEDED = "when_needed"
SKIPPED = "skipped"  # what a turn without the look records
PEEK_PP_AT_MOST = 4  # look when a counted PP is this low (room for a missed count)
SETTLED_STOPS = ("top_confirmed", "settled_no_scrollbar")  # log collections that reached the turn's first row


@dataclass(frozen=True)
class Turn:
    """What the action turn knows once its log is read."""

    member: OwnMember | None
    mon: str | None  # the active's species id as the events name it
    active: int  # our active member's selected slot
    same_turn: bool  # the log shows no new turn: the taps did not go through
    resynced: bool
    log: LogVisit
    turn_before: int | None
    events: tuple[dict, ...]  # the battle's events so far
    new: tuple[dict, ...]  # the events this visit added
    hints: Hints


def member_of(b: BattleRecord, team: Team | None, active: int) -> OwnMember | None:
    if team is None or b.requested is None or not 0 <= active < len(b.requested):
        return None
    row = b.requested[active]
    return team.members[row] if 0 <= row < len(team.members) else None


def offsets(b: BattleRecord, active: int) -> dict[str, int]:
    """PP the log count missed for the active, by move id, from earlier looks."""
    prefix = f"{active}:"
    return {k[len(prefix) :]: v for k, v in b.pp_offsets.items() if k.startswith(prefix)}


def why_before_log(mode: str, b: BattleRecord, team: Team | None, repeated: bool) -> str | None:
    """A reason to look that is known before the log is read; None when there is none yet."""
    if mode != WHEN_NEEDED:
        return ALWAYS
    if team is None or b.requested is None:
        return "untracked"  # no sheet to take the move rows from
    if b.adopted:
        return "adopted"  # picked up in the middle: the log count starts late
    if repeated and b.last_action is not None:
        return "retry"  # the action screen again: the taps did not go through
    return None


def unread(events) -> int:
    return sum(e.get("k") == "unread" for e in events)


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
    if (
        info.turn is not None
        and t.turn_before is not None
        and any(n not in info.back_paged for n in range(t.turn_before + 1, info.turn))
    ):
        return "log_skipped_turn"  # a turn without an action screen of ours was never read (nor paged back)
    if any(e.get("k") == "gap" for e in t.new):
        return "log_gaps"
    if unread(t.events) > b.log_doubts:
        return "log_unmapped"  # a row the converter could not map
    return None


def holds_stone(member: OwnMember, t: Turn) -> bool:
    item = hints.held_item(t.events, t.mon or "", member.item) if t.mon else to_id(member.item or "")
    return bool(item) and game.mega_form_for_stone(member.species, item) is not None


def why_after_log(b: BattleRecord, t: Turn) -> str | None:
    """A reason to look now that the log is read; None when the sheet and the log are enough."""
    if t.same_turn:
        return "retry"
    if t.member is None:
        return "untracked"
    doubt = log_doubt(t, b)
    if doubt is not None:
        return doubt
    if t.resynced:
        return "identity"
    if len(t.member.moves) == 0 or any(not name for name in t.member.moves):
        return "sheet"  # a move slot the team sheet did not read
    if any(left <= PEEK_PP_AT_MOST for left, _ in t.hints.pp_counted.values()):
        return "low_pp"
    if b.mega_pending and not hints.mega_evolved(t.events):
        return "mega_unconfirmed"  # Mega chosen last turn, no Mega Evolution of ours in the log
    if not t.hints.mega_used and holds_stone(t.member, t):
        return "mega_button"  # only the button can offer Mega Evolution (SV-4)
    return None


def remember(b: BattleRecord, t: Turn, peek: Observation, was_pending: bool) -> None:
    """Keep what a look showed that the log does not: PP the count missed, and a Mega button that is gone."""
    member = t.member
    if member is None:
        return
    known = offsets(b, t.active)
    for m in peek.fields.get("moves", ()):
        name = member.moves[m.slot] if m.slot < len(member.moves) else None
        if not name or m.pp is None:
            continue
        ident = to_id(name)
        counted = t.hints.pp_counted.get(ident)
        if counted is None:
            continue
        missed = min(0, m.pp[0] - (counted[0] - known.get(ident, 0)))  # only ever lowers the count
        key = f"{t.active}:{ident}"
        if missed:
            b.pp_offsets[key] = missed
        else:
            b.pp_offsets.pop(key, None)
    shown = bool(peek.fields.get("mega_available"))
    if not t.same_turn and not shown and (was_pending or holds_stone(member, t)):
        b.mega_spent = True  # one Mega Evolution a battle


def settle_mega(b: BattleRecord, t: Turn, peek: Observation | None, why: str | None) -> None:
    """The Mega bookkeeping of a turn, with or without the look ."""
    was_pending = b.mega_pending
    b.log_doubts = max(b.log_doubts, unread(t.events))
    if why is not None:
        if b.mega_pending:
            b.mega_pending = False
            b.mega_used = not (peek and peek.fields.get("mega_available") and not t.same_turn)
        if why != ALWAYS and peek is not None:
            remember(b, t, peek, was_pending)
        return
    if b.mega_pending:
        b.mega_pending, b.mega_used = False, True
    b.mega_spent = b.mega_spent or hints.mega_evolved(t.events)


def refused_moves(last: dict, turn: int, same_turn: bool, repeated: bool) -> tuple[int, ...]:
    """The move rows tapped in this turn that the game did not take: the action screen is back in the same turn."""
    label = str(last.get("label") or "")
    again = same_turn or (repeated and last.get("turn") == turn)
    if not again or "move" not in label:
        return ()
    return (*last.get("refused", ()), int(label.rsplit("move", 1)[1]))

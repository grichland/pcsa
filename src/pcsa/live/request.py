"""The Offer (only what the screen shows) and the Request for one decision.

Move rows: read from the move list when it was opened; otherwise the team sheet's moves by row, with PP,
disabled and the Mega button unread (None). Row k is set position k (plan 2b); a row whose name was read and
differs from the sheet keeps the screen's id when that is a move, else the sheet's; both are noted. Switches are
tapped by name.
"""

from __future__ import annotations

from dataclasses import replace

from pcsa.battle.ids import to_id
from pcsa.battle.teams import OwnMember, Team
from pcsa.common.types import MOVE_SLOTS, Observation
from pcsa.decide.api import Hints, MoveOffer, Offer, Request, SwitchOffer
from pcsa.events.live_ids import legal
from pcsa.live.battle_events import BattleEvents, preview_ids
from pcsa.live.team_read import own_team
from pcsa.run.battle_state import BattleRecord, same_name

AUTOMATIC = frozenset({"struggle", "recharge"})


def sheet_rows(member: OwnMember | None) -> tuple[MoveOffer, ...]:
    """The move list as the sheet gives it: ids by row, nothing read."""
    if member is None:
        return ()
    moves = list(member.moves)[:MOVE_SLOTS]
    return tuple(MoveOffer(k, to_id(name) if name else None) for k, name in enumerate(moves))


def screen_rows(member: OwnMember | None, peek: Observation, notes: list[str]) -> tuple[MoveOffer, ...]:
    """The move list as the screen showed it; a name not read, or read as no move, takes the sheet's id."""
    sheet = {m.row: m.id for m in sheet_rows(member)}
    unknown = set(peek.fields.get("disabled_unknown_slots") or ())
    out = []
    for m in peek.fields.get("moves", ()):
        shown = to_id(m.name) if m.name else None
        if shown and sheet.get(m.slot) and shown != sheet[m.slot]:
            notes.append(f"row {m.slot}: screen {shown}, sheet {sheet[m.slot]}")
            if shown not in legal("moves"):
                shown = None  # a partly read name: the sheet's move stays
        disabled = None if m.slot in unknown else bool(m.disabled)
        out.append(MoveOffer(m.slot, shown or sheet.get(m.slot), tuple(m.pp) if m.pp else None, disabled))
    return tuple(out)


def with_refused(moves: tuple[MoveOffer, ...], refused: tuple[int, ...]) -> tuple[MoveOffer, ...]:
    """Rows the game refused this turn are shown as not available: the game showed it will not take them."""
    if not refused:
        return moves
    return tuple(replace(m, disabled=True) if m.row in refused else m for m in moves)


def action_switches(b: BattleRecord) -> tuple[SwitchOffer, ...]:
    """Behind the Pokemon button: our brought members other than the active, not fainted by the log."""
    if not b.names:
        return ()
    active = b.visible_order[0]
    return tuple(
        SwitchOffer(name)
        for slot in b.visible_order
        if slot != active and slot not in b.fainted and (name := b.names[slot])
    )


def action_offer(
    b: BattleRecord, member: OwnMember | None, peek: Observation | None, refused: tuple[int, ...], notes: list[str]
) -> Offer:
    if peek is not None and peek.fields.get("moves"):
        moves = screen_rows(member, peek, notes)
        automatic = all((m.id or "") in AUTOMATIC for m in moves)
        mega = bool(peek.fields.get("mega_available"))
        return Offer(with_refused(moves, refused), True, action_switches(b), mega, automatic=automatic)
    return Offer(with_refused(sheet_rows(member), refused), False, action_switches(b), None)


def party_name(b: BattleRecord, member) -> str | None:
    """The roster name of a party row: by the name read, else by the bookkeeping's order."""
    named = [n for n in b.names if n and same_name(n, member.name)]
    if len(named) == 1:
        return named[0]
    if member.name is None and member.row < len(b.visible_order) and b.names:
        return b.names[b.visible_order[member.row]]
    return member.name


def forced_offer(b: BattleRecord, party: tuple, leaving: int | None) -> Offer:
    leaving_name = b.names[leaving] if leaving is not None and b.names else None
    switches = []
    for m in party:
        name = party_name(b, m)
        if m.fainted or not name or name == leaving_name or any(s.name == name for s in switches):
            continue
        switches.append(SwitchOffer(name, tuple(m.hp) if m.hp else None))
    return Offer((), False, tuple(switches), None, "pivot" if leaving is not None else "faint", leaving_name)


def build(
    b: BattleRecord,
    team: Team | None,
    events: BattleEvents,
    offer: Offer,
    hints: Hints,
    deadline: float,
) -> Request:
    return Request(
        battle_tag=b.battle_id,
        request_id=len(b.decisions),
        rows=events.rows(b),
        log_quality=events.quality(b),  # type: ignore[arg-type]
        team=own_team(team, b.roster, b.requested),
        opponent_preview=preview_ids(b.opponent_preview),
        offer=offer,
        hints=hints,
        deadline=deadline,
        events=events.all(),
        history=tuple(b.live_history),
    )

"""The mapping between the controller's identifiers and CG's (plan 2b), from Request fields only.

move:    screen row k = set position k (team.export move order)
switch:  name <-> slot in brought order, 0 = lead (brought = names[i] for i in picks)
preview: indices into team.names = indices into CG's six (the export's order)
mega:    the same flag

CG names moves and switch targets by id (LIVE section 3); `order_of` maps them to a row or a name and raises
CGError for anything it cannot place, so the runner falls back.
"""

from __future__ import annotations

import re

from pcsa.battle.ids import to_id
from pcsa.cg.transport import CGError
from pcsa.decide.api import Offer, Order, OwnTeam, Request, valid_picks
from pcsa.events.live_ids import base_id, species_id


def set_moves(export: str) -> tuple[tuple[str, ...], ...]:
    """Move ids per member, in export (= names) order and set order."""
    blocks = [b for b in re.split(r"\n\s*\n", export.strip()) if b.strip()]
    return tuple(
        tuple(to_id(line.strip()[1:]) for line in block.splitlines() if line.strip().startswith("-"))
        for block in blocks
    )


def position_of_row(row: int) -> int:
    return row


def row_of_position(position: int) -> int:
    return position


def slot_of_name(team: OwnTeam, name: str) -> int:
    brought = team.brought()
    if name not in brought:
        raise ValueError(f"{name!r} is not brought: {brought}")
    return brought.index(name)


def name_of_slot(team: OwnTeam, slot: int) -> str:
    return team.brought()[slot]


def preview_to_cg(picks: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(picks)


def preview_from_cg(indices: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(indices)


def _matches(moves: tuple[str, ...], offer: Offer) -> bool:
    return all(m.id is None or (m.row < len(moves) and moves[m.row] == m.id) for m in offer.moves)


def active_member(team: OwnTeam, offer: Offer) -> int | None:
    """Index into team.names of the member on the field: the brought member whose set gives the rows' ids,
    else the one brought member not on the party screen."""
    sets = set_moves(team.export)
    brought = [i for i in team.picks or () if i < len(sets)]
    if offer.moves:
        by_moves = [i for i in brought if _matches(sets[i], offer)]
        if len(by_moves) == 1:
            return by_moves[0]
    on_party = {s.name for s in offer.switches}
    rest = [i for i in brought if team.names[i] not in on_party]
    return rest[0] if len(rest) == 1 else None


def row_mismatches(team: OwnTeam, member: int, offer: Offer) -> list[dict]:
    """Rows whose screen id differs from the set's move at that position (the id decides; recorded)."""
    moves = set_moves(team.export)[member]
    return [
        {"row": m.row, "screen": m.id, "set": moves[m.row] if m.row < len(moves) else None}
        for m in offer.moves
        if m.id is not None and (m.row >= len(moves) or moves[m.row] != m.id)
    ]


# CG's answers to the controller's Order ------------------------------------------------------------
def picks_of(reply: dict) -> tuple[int, int, int]:
    """{"picks": [i, j, k]}: indices into own_team, which is team.names order."""
    picks = reply.get("picks")
    if not isinstance(picks, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in picks):
        raise CGError(f"picks not a list of indices: {picks!r}")
    if not valid_picks(tuple(picks)):
        raise CGError(f"invalid picks {picks}")
    a, b, c = picks
    return a, b, c


def move_row(team: OwnTeam, offer: Offer, move: str) -> int:
    """The screen row of a move id: the row showing it, else its set position when that row's id is unread."""
    for m in offer.moves:
        if m.id == move:
            return m.row
    member = active_member(team, offer)
    sets = set_moves(team.export)
    if member is not None and member < len(sets) and move in sets[member]:
        row = sets[member].index(move)
        if any(m.row == row and m.id is None for m in offer.moves):
            return row
    raise CGError(f"move {move!r} is on no row: {[m.id for m in offer.moves]}")


def switch_name(team: OwnTeam, own_team: list[dict], mon: str) -> str:
    """The brought member's name for a species id (the SET's species, the name's id, or the same base species)."""
    for i in team.picks or ():
        name = team.names[i]
        species = own_team[i].get("species") if i < len(own_team) else None
        ids = {to_id(name), species_id(name), species}
        if mon in ids or base_id(mon) in {base_id(x) for x in ids if x}:
            return name
    raise CGError(f"{mon!r} is not one of the brought {list(team.brought())}")


def order_of(reply: dict, request: Request) -> Order:
    """{"seq": N, "order": {kind move|switch, move id | mon species id, mega}} as a screen Order."""
    order = reply.get("order")
    if not isinstance(order, dict):
        raise CGError(f"no order in {reply!r}")
    if order.get("kind") == "move" and isinstance(order.get("move"), str):
        mega = order.get("mega", False)
        if not isinstance(mega, bool):
            raise CGError(f"mega is not true or false: {mega!r}")
        return Order.move(move_row(request.team, request.offer, order["move"]), mega=mega)
    if order.get("kind") == "switch" and isinstance(order.get("mon"), str):
        start = request.events[0] if request.events else {}
        return Order.switch(switch_name(request.team, list(start.get("own_team") or []), order["mon"]))
    raise CGError(f"order not understood: {order!r}")

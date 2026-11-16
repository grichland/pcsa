"""Keeping the battle record in step with what the phone shows: the active member, faints, the party order.

Split out of turn.py (15 Nov 2026) so the turn flow stays under the file size cap; behaviour unchanged.
"""

from __future__ import annotations

import re

from pcsa.common import log
from pcsa.run.battle_state import BattleRecord, same_name, same_name_in


def leaving_slot(b: BattleRecord, party: tuple) -> int | None:
    """The slot of our active when it leaves the field alive (else None: a faint replacement).

    A replacement screen while our active is still standing means it is leaving and cannot be picked: a pivot move
    (U-turn and the like), or anything else that pulls it out mid-turn (Emergency Exit, Wimp Out, Eject Button, Eject
    Pack), which the log has not shown yet when the screen comes up. The active is the mon this turn put in
    (visible_order[0]); it counts only when the party screen shows it unfainted."""
    if not b.names:
        return None
    active = b.visible_order[0]
    alive = any(same_name(m.name, b.names[active]) and not m.fainted for m in party)
    return active if alive else None


def check_identity(b: BattleRecord, active: str | None) -> bool:
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


def faints_from_log(b: BattleRecord) -> None:
    for turn, rows in b.log.turns.items():
        for text in rows:
            match = re.fullmatch(r"(?:Fainted:\s*)?(?!The opposing)(.+?) fainted!", text.strip())
            slot = b.slot_of(match[1]) if match else None
            if slot is not None and slot not in b.fainted:
                b.fainted.add(slot)
                log.event("fainted_from_log", battle=b.index, slot=slot, turn=turn)


def resync_party(b: BattleRecord, party) -> None:
    """The party order and faints from the rows of a party or replacement screen."""
    slots = [b.slot_of(m.name) for m in party]
    if None not in slots and sorted(slots) == [0, 1, 2]:  # type: ignore[type-var]
        if slots != b.visible_order:
            log.event("order_resync", battle=b.index, before=list(b.visible_order), after=slots)
        b.visible_order = slots  # type: ignore[assignment]
    if party and all(m.fainted for m in party):
        log.warning("party_all_fainted_ignored", battle=b.index)  # a battle still going has a member standing
        return
    for m in party:
        slot = b.slot_of(m.name)
        if slot is None and m.row < len(b.visible_order):
            slot = b.visible_order[m.row]
        if m.fainted and slot is not None:
            b.fainted.add(slot)
    b.observed["party_names"] = [m.name for m in party]

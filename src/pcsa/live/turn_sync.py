"""Keeping the battle record in step with what the phone shows: the active member, faints, the party order.

Split out of turn.py to keep files under the size cap.
"""

from __future__ import annotations

import re

from pcsa.common import log
from pcsa.run.battle_state import BattleRecord, same_name, same_name_in


def leaving_slot(b: BattleRecord, party: tuple) -> int | None:
    """The slot of our active when it leaves the field alive (else None: a faint replacement)."""
    if not b.names:
        return None
    active = b.visible_order[0]
    alive = any(same_name(m.name, b.names[active]) and not m.fainted for m in party)
    return active if alive else None


def fainted_active(b: BattleRecord, battle, screen_hp: dict) -> int | None:
    """The slot of our active that just fainted, on a faint replacement screen (leaving_slot None), else None."""
    fainted = set(b.fainted) | {slot for slot, hp in screen_hp.items() if hp and hp[0] <= 0}
    mon = getattr(battle, "active_pokemon", None) if battle is not None else None
    tracked = b.slot_of(mon.name) if mon is not None else None
    shown = b.visible_order[0] if b.visible_order else None
    if tracked is not None and tracked in fainted and not mon.fainted:  # type: ignore[union-attr]
        if shown is not None and shown != tracked and shown in fainted:
            log.warning("fainted_active_disagree", battle=b.index, tracked=tracked, visible=shown, used=tracked)
        return tracked
    if shown is not None and shown in fainted:
        return shown
    if tracked is not None and tracked in fainted:
        return tracked
    alone = sorted(fainted)
    if len(alone) == 1:
        log.warning("fainted_active_guess", battle=b.index, slot=alone[0], tracked=tracked, visible=shown)
        return alone[0]
    log.warning("fainted_active_unknown", battle=b.index, fainted=alone, tracked=tracked, visible=shown)
    return None


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


def index_collisions(b: BattleRecord, kind: str, battle, legal) -> dict[int, list[str]]:
    """Legal orders that share one network action index ({index: orders}), logged as a warning: the network can
    then pick only the first of them (features.legal_action), whatever it meant. Empty without a tracked Battle."""
    if battle is None:
        return {}
    try:
        from pcsa.rl_training import features

        by: dict[int, list[str]] = {}
        for a in legal:
            by.setdefault(features.action_index(a, battle), []).append(a.message)
    except Exception as error:  # a check, never a reason to stop a decision
        log.warning(
            "index_collision_check_failed", battle=b.index, kind=kind, reason=f"{type(error).__name__}: {error}"
        )
        return {}
    clashes = {i: orders for i, orders in by.items() if len(orders) > 1}
    if clashes:
        log.warning(
            "ACTION_INDEX_COLLISION",
            battle=b.index,
            kind=kind,
            turn=b.current_turn(),
            clashes={str(i): orders for i, orders in clashes.items()},
            note="the network can pick only the first order of each index",
        )
    return clashes

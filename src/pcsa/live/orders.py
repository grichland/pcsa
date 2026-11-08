"""poke-env battle orders <-> phone tap targets, and the screen check before tapping.

Legality comes from the tracked Battle (`battle.valid_orders`). This module maps an order to what
the phone can tap, checks the screen shows it, and offers the screen-only list for decisions
without a Battle.
"""

from __future__ import annotations

from poke_env.battle import Move, Pokemon
from poke_env.data.normalize import to_id_str

from pcsa.common.types import MOVE_SLOTS, SemanticAction
from pcsa.players.base import LegalAction, ScreenFacts

AUTOMATIC_MOVES = ("struggle", "recharge")


def to_tap(order, *, slot_moves: list[str | None], screen_moves=(), names: list[str]) -> SemanticAction | None:
    """The phone target of an order, or None when the phone has no control for it."""
    target = getattr(order, "order", None)
    if any(getattr(order, flag, False) for flag in ("z_move", "dynamax", "terastallize")):
        return None
    if isinstance(target, Move):
        if target.id in AUTOMATIC_MOVES:
            return SemanticAction("move", move=0)
        slots = [k for k, name in enumerate(slot_moves) if name and to_id_str(name) == target.id]
        slots = slots or [m.slot for m in screen_moves if m.name and to_id_str(m.name) == target.id]
        return SemanticAction("move", move=slots[0], mega=bool(order.mega)) if len(slots) == 1 else None
    if isinstance(target, Pokemon):
        slots = [s for s, name in enumerate(names) if name and to_id_str(name) == to_id_str(target.name)]
        return SemanticAction("switch", slot=slots[0]) if len(slots) == 1 else None
    return None


def screen_check(
    action: SemanticAction,
    screen: ScreenFacts,
    *,
    fainted: set[int],
    active: int | None,
    leaving: int | None = None,
    automatic: bool = False,
) -> str | None:
    """None when the current screen shows the action, else the reason it does not."""
    if action.kind == "move":
        if screen.forced:
            return "move on a forced replacement screen"
        row = next((m for m in screen.moves if m.slot == action.move), None)
        if row is None:
            return f"move slot {action.move} not on the move screen"
        if not automatic and (row.disabled or (row.pp is not None and row.pp[0] <= 0)):
            return f"move slot {action.move} shown unavailable"
        if action.mega and not screen.mega_available:
            return "Mega Evolution not offered on screen"
        return None
    if action.slot in fainted:
        return f"switch target {action.slot} fainted"
    if not screen.forced and action.slot == active:
        return f"switch target {action.slot} is already active"
    if screen.forced and action.slot == leaving:
        return f"switch target {action.slot} is leaving (pivot)"
    return None


def battle_actions(
    battle,
    screen: ScreenFacts,
    *,
    slot_moves: list[str | None],
    names: list[str],
    fainted: set[int],
    active: int | None,
    leaving: int | None = None,
) -> tuple[list[LegalAction], list[dict]]:
    """(actions the Battle allows and the screen shows, dropped orders with reasons)."""
    kept, dropped = [], []
    for order in battle.valid_orders:
        action = to_tap(order, slot_moves=slot_moves, screen_moves=screen.moves, names=names)
        automatic = getattr(getattr(order, "order", None), "id", None) in AUTOMATIC_MOVES
        reason = (
            "no phone control"
            if action is None
            else screen_check(action, screen, fainted=fainted, active=active, leaving=leaving, automatic=automatic)
        )
        if reason or action is None:
            dropped.append({"order": order.message, "reason": reason})
        else:
            kept.append(LegalAction(action, order))
    return kept, dropped


def screen_actions(
    screen: ScreenFacts,
    *,
    fainted: set[int],
    active: int | None,
    leaving: int | None = None,
    mega_used: bool = False,
    team_size: int = 3,
) -> list[LegalAction]:
    """What the screen alone allows, for a decision without a tracked Battle."""
    actions = []
    if not screen.forced:
        for m in screen.moves:
            usable = (m.pp is None and m.name) or (m.pp is not None and m.pp[0] > 0)
            if 0 <= m.slot < MOVE_SLOTS and not m.disabled and usable:
                actions.append(LegalAction(SemanticAction("move", move=m.slot), None))
                if screen.mega_available and not mega_used:
                    actions.append(LegalAction(SemanticAction("move", move=m.slot, mega=True), None))
    for slot in range(team_size):
        if slot not in fainted and slot != (leaving if screen.forced else active):
            actions.append(LegalAction(SemanticAction("switch", slot=slot), None))
    return actions


def blind_actions(
    screen: ScreenFacts, *, fainted: set[int], active: int | None, team_size: int = 3
) -> list[LegalAction]:
    """Nothing readable at all: every move row, every non-fainted switch."""
    actions = [] if screen.forced else [LegalAction(SemanticAction("move", move=k), None) for k in range(MOVE_SLOTS)]
    actions += [
        LegalAction(SemanticAction("switch", slot=s), None)
        for s in range(team_size)
        if s not in fainted and s != active
    ]
    return actions

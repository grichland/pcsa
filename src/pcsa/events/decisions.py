"""decision_request, the order answer, refusals and fallbacks from pcsa decision records (LIVE.md).

A record's `order` is pcsa's Showdown order ("/choose move earthquake", "/choose move X mega",
"/choose switch Garchomp") or, when there was no tracked Battle, a screen label ("move2", "mega+move0",
"switch1": a move row of the active Pokemon, a slot of the brought three).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pcsa.battle.ids import to_id
from pcsa.events.live_ids import VERSION, species_id

FALLBACK_RULE = "neural:fallback"  # the player gave up and its fallback chose


@dataclass
class OrderContext:
    """What turns a recorded order into a LIVE order: our sets and the brought three."""

    species_by_name: dict[str, str]  # to_id(display name) -> species id
    moves_by_species: dict[str, list[str]]  # species id -> move ids in set order
    brought: list[str]  # display names in pick order
    problems: list[str] = field(default_factory=list)

    def species(self, name: str) -> str:
        return self.species_by_name.get(to_id(name)) or species_id(name)


def _active_species(record: dict, active_hint: str | None, ctx: OrderContext) -> str | None:
    side = ((record.get("request") or {}).get("side") or {}).get("pokemon") or []
    for mon in side:
        if mon.get("active") and mon.get("ident"):
            return ctx.species(mon["ident"].split(":", 1)[-1].strip())
    return active_hint


def order_of(record: dict, ctx: OrderContext, active_hint: str | None = None) -> dict | None:
    """The record's order in LIVE shape; None (and a problem noted) when it cannot be named."""
    text = str(record.get("order") or "")
    if m := re.fullmatch(r"/choose move (\S+)( mega)?", text):
        return {"kind": "move", "move": m[1], "mega": bool(m[2])}
    if m := re.fullmatch(r"/choose switch (.+)", text):
        return {"kind": "switch", "mon": ctx.species(m[1])}
    if m := re.fullmatch(r"(mega\+)?move(\d)", text):
        species = _active_species(record, active_hint, ctx)
        moves = ctx.moves_by_species.get(species or "", [])
        slot = int(m[2])
        if slot < len(moves):
            return {"kind": "move", "move": moves[slot], "mega": bool(m[1])}
        ctx.problems.append(f"order {text}: active {species} has no move row {slot}")
        return None
    if m := re.fullmatch(r"switch(\d)", text):
        slot = int(m[1])
        if slot < len(ctx.brought):
            return {"kind": "switch", "mon": ctx.species(ctx.brought[slot])}
        ctx.problems.append(f"order {text}: no brought slot {slot}")
        return None
    ctx.problems.append(f"order {text!r} not understood")
    return None


def kind_of(record: dict) -> str:
    """`forced` records are the replacement screen: a faint or a pivot."""
    return "replace" if record.get("kind") == "forced" else "turn"


def refused_by(record: dict, previous: dict | None) -> bool:
    """True when this record's screen came back after `previous` (a refusal or a tap that did not register).

    Action: pcsa's `same_turn` (the action screen came back in the same log turn with no new rows).
    Replace: `attempt` > 1 (the replacement screen came back in the same turn)."""
    if previous is None or previous.get("kind") != record.get("kind"):
        return False
    if record.get("kind") == "forced":
        return int(record.get("attempt") or 1) > 1
    return bool(record.get("same_turn"))


def repeat_of(record: dict, previous: dict | None, previous_order: dict | None) -> str | None:
    """"refused", "retry" or None (CF-8). A refusal only where pcsa recorded the move row as refused; a repeated
    switch or replacement, or a repeat pcsa did not mark, is a tap that did not register."""
    if not refused_by(record, previous):
        return None
    if record.get("kind") == "action" and record.get("refused") and (previous_order or {}).get("kind") == "move":
        return "refused"
    return "retry"


def fallback_reason(record: dict) -> str | None:
    """Why the controller chose instead of the player, or None when the player answered.

    `deadline`: the deadline was nearly gone and pcsa took the first legal action without asking.
    `player_fallback`: the player had no tracked Battle (rung screen or legal_random) and its fallback chose.
    """
    notes = record.get("notes") or {}
    if record.get("degrade") == "emergency":
        return "deadline"
    if notes.get("rule") == FALLBACK_RULE or "fallback" in notes:
        return "player_fallback"
    return None


def fallback_of(record: dict, order: dict | None) -> dict | None:
    """Section 3's {"seq", "order", "reason"}; None when the player answered or the order cannot be named."""
    reason = fallback_reason(record)
    if reason is None or order is None or not isinstance(record.get("sequence"), int):
        return None
    return {"seq": record["sequence"], "order": order, "reason": reason}


def decision_request(tag: str, record: dict, refused: dict | None, retry: bool = False) -> dict:
    """No deadline_ms: pcsa read the move timer but did not record it (rule 2). No screen_offer: the
    screen's own rows were not recorded, only the tracked orders that passed the screen check."""
    out = {"v": VERSION, "battle": tag, "type": "decision_request", "seq": record.get("sequence"),
           "kind": kind_of(record)}
    if refused:
        out["refused"] = {"order": refused}
    elif retry:
        out["retry"] = True
    return out


def answer(record: dict, order: dict) -> dict:
    return {"seq": record.get("sequence"), "order": order}


def game_turn(record: dict) -> int | None:
    """The turn a `turn` decision was for: the tracker's turn, else the log turn read plus one."""
    if isinstance(record.get("turn"), int):
        return record["turn"]
    if isinstance(record.get("log_turn"), int):
        return record["log_turn"] + 1
    return None

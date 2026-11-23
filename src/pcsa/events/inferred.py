"""An opposing switch whose rows the log read lost: when a row shows an opposing Pokémon acting or being hit while
another one (or none) is recorded on the field, its `enter` is sent first. In the LIVE protocol an `enter` with no
`withdraw` is the switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pcsa.events.live_ids import base_id

ON_FIELD = frozenset({"move", "hp", "faint", "status", "boost", "cant", "protect", "prepare"})  # rows of an active


@dataclass
class OppActive:
    """Follows the opposing active through the events in the order they are sent."""

    preview: tuple[str | None, ...] = ()  # base species ids; None where the preview slot was not read
    active: str | None = None
    fainted: set[str] = field(default_factory=set)
    entered: set[str] = field(default_factory=set)  # base species ids that have come in
    inferred: int = 0  # `enter` events added

    def step(self, event: dict, late: bool = False) -> list[dict]:
        """The event, with an `enter` before it when it shows an opposing Pokémon that is not recorded as active.
        `late`: a row of an older turn, which says nothing about who is active now."""
        if event.get("side") != "opp" or not event.get("mon"):
            return [event]
        kind, mon = event["k"], event["mon"]
        out = [event]
        if not late and kind in ON_FIELD and mon != self.active and self._possible(mon):
            out = [{"k": "enter", "side": "opp", "mon": mon}, event]
            self.active, self.inferred = mon, self.inferred + 1
        if kind == "enter" or len(out) > 1:
            self.entered.add(base_id(mon))
        if kind == "enter":
            self.active = mon
        elif kind == "withdraw" and not late:
            self.active = None
        elif kind == "faint":
            self.fainted.add(mon)
            self.active = None if mon == self.active else self.active
        elif kind == "revive":
            self.fainted.discard(mon)
        return out

    def _possible(self, mon: str) -> bool:
        if mon in self.fainted:
            return False
        return not self.preview or None in self.preview or base_id(mon) in {base_id(p) for p in self.preview if p}


def _candidates(slot) -> list:
    """A preview slot's sprite matches as (base species id, score), best first."""
    top = slot.get("top") if isinstance(slot, dict) else getattr(slot, "top", None)
    return [(base_id(c[0]), float(c[1])) for c in top or () if c and c[0]]


def misread_slot(preview, ids: list, entered: set[str], mon: str) -> int | None:
    """The preview slot an entering Pokémon outside the six names was misread in: among the slots not seen yet,
    one whose other sprite matches name it, else the weakest match. None when every slot has been seen."""
    open_slots = [i for i, ident in enumerate(ids) if ident is None or base_id(ident) not in entered]
    if not open_slots:
        return None
    slots = list(preview or ())
    found = {i: _candidates(slots[i]) if i < len(slots) else [] for i in open_slots}
    for i in open_slots:
        if any(name == mon for name, _ in found[i]):
            return i
    return min(open_slots, key=lambda i: (found[i][0][1] if found[i] else 0.0, i))

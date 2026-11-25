"""Evidence from the controller's events for the build weights (builds.py): items seen on each opposing Pokémon
and turn orders.

Turn order: at equal priority, outside Trick Room and with no ability or item that changes order on either side,
a build that is faster than our Pokémon was is likely to have moved first: 1 - ORDER_EPS, else ORDER_EPS. Speed is
CG's stat_from_points (engine/src/sim/calc.rs) with the stage, Choice Scarf, paralysis and Tailwind.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import worlds

if TYPE_CHECKING:
    from .builds import Build, Dex

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
SIDES = ("own", "opp")
ORDER_EPS = 0.05  # a turn order that contradicts a build's speed still keeps this likelihood
# Items and abilities that change turn order or speed in ways the tracker does not follow: no order evidence.
ORDER_ITEMS = {"quickclaw"}
SPEED_ITEMS = {"choicescarf": 1.5, "ironball": 0.5}
ORDER_ABILITIES = {"swiftswim", "chlorophyll", "sandrush", "slushrush", "surgesurfer", "unburden", "quickfeet",
                   "slowstart", "stall", "prankster", "galewings", "triage", "quickdraw", "myceliummight",
                   "protosynthesis", "quarkdrive"}  # fmt: skip


def to_id(text: Any) -> str:
    return worlds.to_id(text or "")


def points_of(p: Any) -> tuple[int, ...]:
    if isinstance(p, dict):
        return tuple(int(p.get(s) or 0) for s in STATS)
    if isinstance(p, list | tuple) and len(p) == len(STATS):
        return tuple(int(x) for x in p)
    return (0,) * len(STATS)


def stat_from_points(base: int, points: int, stat: str, nature: tuple[str | None, str | None]) -> int:
    """CG's stat_from_points: hp = base + points + 75; else base + points + 20 with the nature's 10%."""
    if stat == "hp":
        return base + points + 75
    v = base + points + 20
    if nature[0] == stat:
        v = ((v * 110) & 0xFFFF) // 100
    elif nature[1] == stat:
        v = ((v * 90) & 0xFFFF) // 100
    return v


def apply_stage(stat: int, boost: int) -> int:
    b = max(-6, min(6, int(boost)))
    return stat * (2 + b) // 2 if b >= 0 else stat * 2 // (2 - b)


@dataclass
class Order:
    """One turn's order: their `mon` moved first (`first`) against our Pokémon of effective speed `ours`, with
    their stage, paralysis and Tailwind at that time."""

    mon: str
    form: str
    stage: int
    par: bool
    tailwind: bool
    ours: int
    first: bool


@dataclass
class Tracker:
    """One battle's evidence, fed with the controller's events: items seen on each opposing Pokémon and turn
    orders. `own` is our six sets from battle_start."""

    dex: Dex
    own: list[dict] = field(default_factory=list)
    had_item: dict[str, str] = field(default_factory=dict)
    orders: list[Order] = field(default_factory=list)
    active: dict[str, str] = field(default_factory=dict)
    form: dict[tuple, str] = field(default_factory=dict)
    stage: dict[str, int] = field(default_factory=dict)
    par: set[tuple] = field(default_factory=set)
    tailwind: set[str] = field(default_factory=set)
    trick_room: bool = False
    turn: list[dict] = field(default_factory=list)

    def feed(self, events: Sequence[dict]) -> None:
        for e in events:
            if not isinstance(e, dict):
                continue
            if e.get("k") == "turn":
                self._close_turn()
                continue
            self.turn.append(e)
            self._state(e)

    def _state(self, e: dict) -> None:
        k, side, mon = e.get("k"), e.get("side"), to_id(e.get("mon"))
        if k == "enter" and side:
            self.active[side], self.stage[side] = mon, 0
        elif k in ("item", "mega_stone") and side == "opp" and mon:
            item = to_id(e.get("item") or e.get("stone"))
            if item and e.get("what") not in ("received", "found"):
                self.had_item.setdefault(mon, item)
        elif k in ("mega", "form") and side and e.get("form"):
            self.form[(side, mon)] = to_id(e.get("form"))
        else:
            self._speed_state(e, k, side, mon)

    def _speed_state(self, e: dict, k: Any, side: Any, mon: str) -> None:
        if k == "boost" and side and "spe" in (e.get("stats") or []):
            self.stage[side] = max(-6, min(6, self.stage.get(side, 0) + int(e.get("by") or 0)))
        elif k == "boost_reset":
            for s in [side] if side else ["own", "opp"]:
                self.stage[s] = 0
        elif k == "status" and side and e.get("status") == "par":
            self.par.add((side, mon))
        elif k == "cure" and side and e.get("status") == "par":
            self.par.discard((side, mon))
        elif k == "side" and to_id(e.get("effect")) == "tailwind" and side:
            (self.tailwind.add if e.get("what") == "start" else self.tailwind.discard)(side)
        elif k == "field" and to_id(e.get("effect")) == "trickroom":
            self.trick_room = e.get("what") == "start"

    def _close_turn(self) -> None:
        events, self.turn = self.turn, []
        first: dict[str, dict] = {}
        order: list[str] = []
        for e in events:
            if e.get("k") == "move" and e.get("side") in ("own", "opp") and e["side"] not in first:
                first[e["side"]] = e
                order.append(e["side"])
        if len(order) < len(SIDES) or self.trick_room or any(first[s].get("called_by") for s in order):
            return
        prio = [int(first[s].get("priority", self.dex.priority.get(to_id(first[s].get("move")), 0))) for s in order]
        if prio[0] != prio[1] or any(_changes_order(e) for e in events):
            return
        ours = self._our_speed(to_id(first["own"].get("mon")))
        if ours is None:
            return
        mon = to_id(first["opp"].get("mon"))
        par, tailwind = ("opp", mon) in self.par, "opp" in self.tailwind
        form = self.form.get(("opp", mon), mon)
        self.orders.append(Order(mon, form, self.stage.get("opp", 0), par, tailwind, ours, order[0] == "opp"))

    def _our_speed(self, mon: str) -> int | None:
        own = next((s for s in self.own if _same(to_id(s.get("species")), mon)), None)
        if own is None or to_id(own.get("ability")) in ORDER_ABILITIES or to_id(own.get("item")) in ORDER_ITEMS:
            return None
        form = self.form.get(("own", mon))
        stats = own.get("stats") or {}
        if form is None and isinstance(stats.get("spe"), int):  # the game's own stat (forms named by base species)
            raw: int | None = int(stats["spe"])
        else:
            species = form or to_id(own.get("species"))
            raw = self.dex.speed(species, to_id(own.get("nature")), points_of(own.get("points")))
        if raw is None:
            return None
        return effective(raw, self.stage.get("own", 0), to_id(own.get("item")), ("own", mon) in self.par,
                         "own" in self.tailwind)  # fmt: skip

    def order_loglik(self, species: str, b: Build) -> float:
        """The log-likelihood of this battle's turn orders for their `species` under build `b`."""
        if b.ability in ORDER_ABILITIES or b.item in ORDER_ITEMS:
            return 0.0
        total = 0.0
        for o in self.orders:
            if not _same(o.mon, species):
                continue
            raw = self.dex.speed(o.form, b.nature, b.points) or self.dex.speed(species, b.nature, b.points)
            if raw is None:
                continue
            theirs = effective(raw, o.stage, b.item, o.par, o.tailwind)
            p = 0.5 if theirs == o.ours else (1.0 - ORDER_EPS if (theirs > o.ours) == o.first else ORDER_EPS)
            total += float(np.log(p))
        return total


def _same(a: str, b: str) -> bool:
    return bool(a and b) and (a == b or a.startswith(b) or b.startswith(a))


def _changes_order(e: dict) -> bool:
    if e.get("k") not in ("item", "ability"):
        return False
    return to_id(e.get("item")) in ORDER_ITEMS or to_id(e.get("ability")) in ORDER_ABILITIES


def effective(raw: int, stage: int, item: str, par: bool, tailwind: bool) -> int:
    s = apply_stage(raw, stage)
    s = int(s * SPEED_ITEMS.get(item, 1.0))
    if tailwind:
        s *= 2
    if par:
        s //= 2
    return s

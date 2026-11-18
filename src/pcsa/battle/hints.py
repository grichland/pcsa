"""Hints from the battle's LIVE events: the controller's reading of the log for the active member. Advice only;
nothing here narrows the Offer. Counts are against the team sheet's moves and the Champions max PP."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pcsa.battle.ids import to_id
from pcsa.battle.moves import max_pp
from pcsa.battle.teams import OwnMember
from pcsa.decide.api import Hints

CHOICE_SCARF = "choicescarf"
NOT_LOCKING = frozenset({"struggle", "recharge"})
TRAPS = frozenset(
    {"bind", "wrap", "snaptrap", "trapped", "firespin", "whirlpool", "magmastorm", "sandtomb", "infestation",
     "thundercage", "octolock", "jawlock", "noretreat", "ingrain"}
)  # fmt: skip


def _kinds(events: Iterable[dict]) -> list[dict]:
    """The event list without battle_start (or any message that is not an event)."""
    return [e for e in events if isinstance(e, dict) and "k" in e]


def _own(e: dict, mon: str | None) -> bool:
    return e.get("side") == "own" and (mon is None or e.get("mon") == mon)


def since_entry(events: Sequence[dict], mon: str) -> list[dict]:
    """Events after our `mon`'s last `enter` (all of them when it never entered in the log)."""
    for i in range(len(events) - 1, -1, -1):
        if events[i].get("k") == "enter" and _own(events[i], mon):
            return list(events[i + 1 :])
    return list(events)


def pp_counted(events: Sequence[dict], mon: str, moves: Sequence[str | None], offsets: dict[str, int]) -> dict:
    """move id -> (left, max): one use per own `move` event (two while an opposing Pressure is out), minus `pp`
    rows, plus what a look at the move list showed the count had missed (`offsets`, never positive)."""
    used: dict[str, int] = {}
    pressure: set[str] = set()
    opp_active: str | None = None
    for e in events:
        k = e.get("k")
        if k == "enter" and e.get("side") == "opp":
            opp_active = e.get("mon")
        elif k == "ability" and e.get("side") == "opp" and e.get("ability") == "pressure":
            pressure.add(e.get("mon") or "")
        elif k == "move" and _own(e, mon):
            used[e["move"]] = used.get(e["move"], 0) + (2 if opp_active in pressure else 1)
        elif k == "pp" and _own(e, mon):
            used[e["move"]] = used.get(e["move"], 0) + int(e.get("lost") or 0)
    out = {}
    for name in moves:
        if not name:
            continue
        ident = to_id(name)
        maximum = max_pp(ident)
        out[ident] = (max(0, maximum - used.get(ident, 0) + offsets.get(ident, 0)), maximum)
    return out


def mega_evolved(events: Iterable[dict]) -> bool:
    return any(e.get("k") == "mega" and e.get("side") == "own" for e in events)


def held_item(events: Sequence[dict], mon: str, sheet_item: str | None) -> str | None:
    """The sheet's item, unless the log shows it removed (Knock Off) or used up."""
    item = to_id(sheet_item or "") or None
    for e in events:
        if e.get("k") == "item" and _own(e, mon) and e.get("what") in ("removed", "used") and e.get("item") == item:
            item = None
    return item


def last_move(events: Sequence[dict], mon: str) -> str | None:
    for e in reversed(events):
        if e.get("k") == "move" and _own(e, mon) and e.get("move") not in NOT_LOCKING:
            return e["move"]
    return None


def under(events: Sequence[dict], mon: str, effect: str) -> bool:
    """A volatile `effect` started on our `mon` and has not ended (within the events given)."""
    on = False
    for e in events:
        if _own(e, mon) and e.get("effect") == effect:
            if e.get("k") == "volatile":
                on = True
            elif e.get("k") == "volatile_end":
                on = False
    return on


def build(
    events: Iterable[dict],
    mon: str | None,
    member: OwnMember | None,
    *,
    refused: tuple[int, ...] = (),
    offsets: dict[str, int] | None = None,
    mega_spent: bool = False,
) -> Hints:
    """Hints for our active `mon` (species id as the events name it); moves and item from its team sheet."""
    events = _kinds(events)
    mega = mega_spent or mega_evolved(events)
    if mon is None or member is None:
        return Hints(mega_used=mega, refused=refused)
    recent = since_entry(events, mon)
    last = last_move(recent, mon)
    scarf = held_item(events, mon, member.item) == CHOICE_SCARF
    return Hints(
        pp_counted=pp_counted(events, mon, member.moves, offsets or {}),
        mega_used=mega,
        choice_locked=last if scarf else None,
        encore=last if under(recent, mon, "encore") else None,
        taunted=under(recent, mon, "taunt"),
        first_turn_over=any(e.get("k") in ("move", "cant") and _own(e, mon) for e in recent),
        trapped=any(under(recent, mon, effect) for effect in TRAPS),
        refused=refused,
    )

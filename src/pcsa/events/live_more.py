"""Rows LIVE.md 0.5 and 0.6 added a place for (X2, X3, X5, X9, X10, X13, LV5Q-2). None of these wordings is in the
recordings: they are Showdown's sentences, kept until a phone recording shows the game's own."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pcsa.battle.log_patterns import APOS
from pcsa.battle.log_rows import item_name
from pcsa.events.live_ids import legal, move_id

if TYPE_CHECKING:
    from pcsa.events.live import Converter

SWAPS = {  # boost_reset `swap` rows: the stats swapped, None for all
    "switched stat changes with its target!": None,
    "switched all changes to its Attack and Sp. Atk with its target!": ["atk", "spa"],
    "switched all changes to its Defense and Sp. Def with its target!": ["def", "spd"],
}
ONCE = {  # one-off sentences naming the Pokemon first
    "switched Speed with its target!": "speedswap",
    "shared its power with the target!": "powersplit",
    "shared its guard with the target!": "guardsplit",
    "swapped Abilities with its target!": "skillswap",
    "switched items with its target!": "trick",  # Trick and Switcheroo; the `received` rows follow
    "foresaw an attack!": "futuresight_start",
    "chose Doom Desire as its destiny!": "futuresight_start",
}
TYPE_IDS = frozenset("normal fire water electric grass ice fighting poison ground flying psychic bug rock ghost "
                     "dragon dark steel fairy".split())  # fmt: skip


def who(text: str) -> tuple[str, str]:
    m = re.match(r"^[Tt]he opposing (.+)$", text.strip())
    return ("opp", m[1].strip()) if m else ("own", text.strip())


def special(conv: Converter, msg: str, text: str) -> dict | None:  # noqa: PLR0911
    """The event for one of these rows, or None."""
    for tail, stats in SWAPS.items():
        if msg.endswith(" " + tail):
            out = {"k": "boost_reset", **conv._mon(*who(msg[: -len(tail) - 1])), "what": "swap"}
            return {**out, "stats": stats} if stats else out
    for tail, effect in ONCE.items():
        if msg.endswith(" " + tail):
            return {"k": "once", **conv._mon(*who(msg[: -len(tail) - 1])), "effect": effect}
    if m := re.fullmatch(r"(.+?) took the (?:Future Sight|Doom Desire) attack!", msg):
        return {"k": "once", **conv._mon(*who(m[1])), "effect": "futuresight"}
    if m := re.fullmatch(rf"(.+?){APOS}s stat changes were all reversed!", msg):
        return {"k": "boost_reset", **conv._mon(*who(m[1])), "what": "invert"}
    if m := re.fullmatch(rf"(.+?){APOS}s stat changes were removed!", msg):  # Clear Smog
        return {"k": "boost_reset", **conv._mon(*who(m[1])), "what": "all"}
    if m := re.fullmatch(r"(\w+) type was added to (.+?)!", msg):  # Forest's Curse, Trick-or-Treat
        if m[1].lower() in TYPE_IDS:
            return {"k": "types", **conv._mon(*who(m[2])), "add": m[1].lower()}
    if msg == "But there was no PP left for the move!":  # about the `move` row before it; names nobody
        return {"k": "cant", "reason": "nopp"}
    if (m := re.fullmatch(r"(.+?) cannot use (.+?)!", msg)) and move_id(m[2]) in legal("moves"):
        return {"k": "cant", **conv._mon(*who(m[1])), "reason": move_id(m[2])}  # Damp and the like, after its row
    if m := re.fullmatch(r"(.+?) is loafing around!", msg):
        return {"k": "cant", **conv._mon(*who(m[1])), "reason": "truant"}
    if m := re.fullmatch(r"(.+?) became cloaked in mystical moonlight!", msg):  # Lunar Dance
        return {"k": "cause", "effect": "healingwish", **conv._mon(*who(m[1]))}
    if m := re.fullmatch(r"(.+?) is being salted!", msg):
        return {"k": "volatile", **conv._mon(*who(m[1])), "effect": "saltcure"}
    if m := re.fullmatch(r"(.+?) obtained (.+?)[.!]", msg):  # Trick and Switcheroo: one row per Pokemon
        return {"k": "item", **conv._mon(*who(m[1])), "item": conv._item(item_name(m[2]), text), "what": "received"}
    if m := re.fullmatch(rf"(.+?) stole (.+?){APOS}s (.+?)!", msg):  # Thief, Covet, Pickpocket, Magician
        if "stole and ate" not in msg:
            return {"k": "item", **conv._mon(*who(m[1])), "item": conv._item(m[3], text), "what": "received",
                    "of": conv._mon(*who(m[2]))}
    return None

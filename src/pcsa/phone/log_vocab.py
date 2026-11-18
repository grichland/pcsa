"""Legal names for the holes of a battle-log wording, and the unique-nearest-name snap for a misread name."""

from __future__ import annotations

import json
import re
from functools import cache

from pcsa.battle import game, names
from pcsa.battle.ids import to_id
from pcsa.battle.log_patterns import STAT_IDS

TYPES = frozenset(
    "Normal Fire Water Electric Grass Ice Fighting Poison Ground Flying Psychic Bug Rock Ghost Dragon Dark Steel "
    "Fairy Stellar".split()
)
SIDES = {"your side", "the opposing side"}
OPPOSING = re.compile(r"^([Tt]he opposing )(.+)$")
TITLE = re.compile(r"^[A-Z][A-Za-zé’-]*(?: [A-Za-zé’-]+){0,4}$")  # "Champion-Tiered", "Catch of the Day"
TITLE_CASE = re.compile(r"^[A-Z0-9][\w’.-]*(?: [A-Za-z0-9][\w’.-]*)*$")
STAT_SPLIT = re.compile(r", and |, | and ")
SNAP_DISTANCE = 2


def apostrophe(text: str) -> str:
    return text.replace("'", "’").replace("‘", "’")


@cache
def displays(kind: str) -> frozenset[str]:
    """Display names the game writes for one hole kind."""
    if kind == "MON":
        dex = json.loads(game.asset_path("species.json").read_text(encoding="utf-8"))["species"]
        base = {v["name"] for v in dex.values() if not v.get("forme")}
        unlettered = [re.sub(r" [XYZ]$", "", m) for m in names.MEGA]  # "Mega Raichu" for X and Y
        extra = [*names.MEGA, *unlettered, *names.AMBIGUOUS, *names.FORMS, *names.SPACED_SPECIES]
        return frozenset(apostrophe(n) for n in [*base, *extra])
    table = {"MOVE": "moves", "ITEM": "items", "ABILITY": "abilities"}[kind]
    rows = json.loads(game.asset_path(f"cg/{table}.json").read_text(encoding="utf-8"))[table]
    return frozenset(apostrophe(v["name"]) for v in rows.values())


@cache
def ids(kind: str) -> frozenset[str]:
    """Ids of every name of this kind, including the wider pinned vocab without a display spelling."""
    tables = json.loads(game.asset_path("vocab.json").read_text(encoding="utf-8"))["tables"]
    extra = tables.get({"MOVE": "move", "ITEM": "item", "ABILITY": "ability"}.get(kind, ""), [])
    return frozenset([*(to_id(n) for n in displays(kind)), *extra])


def valid_name(kind: str, text: str) -> bool:
    if text in displays(kind):
        return True
    return (
        kind != "MON" and bool(TITLE_CASE.match(text)) and to_id(text) in ids(kind) and not near_display(kind, text)
    )


def near_display(kind: str, text: str) -> bool:
    """A different display spelling has this id ("RockyHelmet"): not a clean reading of it."""
    return any(to_id(d) == to_id(text) for d in displays(kind))


def split_side(text: str) -> tuple[str, str]:
    """("The opposing ", "Garchomp") or ("", "Garchomp")."""
    m = OPPOSING.match(text)
    return (m[1], m[2]) if m else ("", text)


def valid_stats(text: str) -> bool:
    parts = STAT_SPLIT.split(text)
    return all(p in STAT_IDS for p in parts) and len(set(parts)) == len(parts)


def valid(kind: str, text: str) -> bool:
    """One hole's reading is a legal value (MON may carry the side words)."""
    if kind == "MON":
        return valid_name("MON", split_side(text)[1])
    if kind in ("MOVE", "ITEM", "ABILITY"):
        return valid_name(kind, text)
    checks = {
        "TYPE": lambda: text in TYPES,
        "STATS": lambda: valid_stats(text),
        "N": lambda: text.isdigit() and len(text) <= 3,
        "SIDE": lambda: text.lower() in SIDES and text[1:] == text.lower()[1:],
        "TITLE": lambda: bool(TITLE.match(text)),
        "TRAINER": lambda: True,
    }
    return checks[kind]()


def distance(a: str, b: str, limit: int) -> int:
    """Edit distance, or limit + 1 once it is certainly larger."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def nearest(pool, text: str) -> tuple[str | None, list[str]]:
    """The one legal name within the snap distance, else None with the candidates found."""
    found = [name for name in pool if distance(text, name, SNAP_DISTANCE) <= SNAP_DISTANCE]
    return (found[0] if len(found) == 1 else None), sorted(found)


def snap(kind: str, text: str) -> tuple[str | None, list[str]]:
    """A misread hole snapped to the unique legal value within edit distance 1-2 (names and stat words only)."""
    if kind == "MON":
        side, name = split_side(text)
        found, candidates = nearest(displays("MON"), name)
        return (side + found if found else None), candidates
    if kind in ("MOVE", "ITEM", "ABILITY"):
        return nearest(displays(kind), text)
    if kind == "STATS":
        words = [w for w in re.split(r"(, and |, | and )", text)]
        out = []
        for index, word in enumerate(words):
            if index % 2:
                out.append(word)
                continue
            found, candidates = (word, [word]) if word in STAT_IDS else nearest(STAT_IDS, word)
            if found is None:
                return None, candidates
            out.append(found)
        joined = "".join(out)
        return (joined, [joined]) if valid_stats(joined) else (None, [])
    return None, []

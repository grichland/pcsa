"""What the network's belief head thinks each opposing Pokémon holds (`net.belief`), as `pcsa watch` shows it.

Per member: the chance it was brought, the chance it leads (at the preview only), and its top items, abilities and
moves. A fact the battle has revealed replaces its belief: a revealed item or ability leaves that list empty and a
revealed move is never listed again. The head predicts no nature or stat spread.
"""

from __future__ import annotations

from math import isfinite

from pcsa.run.watch_names import pretty, to_id

NO_ITEM = "No item"  # the belief's "" item: holds nothing


def _p(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        return None
    return round(min(1.0, max(0.0, float(value))), 4)


def _pairs(value, names: dict[str, str], item: bool = False) -> list[dict]:
    """[[id, p], ...] as [{id, name, p}], highest first; malformed pairs are dropped."""
    out = []
    for pair in value if isinstance(value, list) else []:
        if not isinstance(pair, list | tuple) or len(pair) != 2 or not isinstance(pair[0], str):  # noqa: PLR2004
            continue
        p = _p(pair[1])
        if p is None:
            continue
        name = NO_ITEM if item and pair[0] == "" else pretty(pair[0], names)
        if name:
            out.append({"id": to_id(pair[0]), "name": name, "p": p})
    return sorted(out, key=lambda x: -x["p"])


def belief_index(entries) -> dict[str, dict]:
    """The net's belief entries by species id (an entry without a species belongs to no card)."""
    out: dict[str, dict] = {}
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("species"), str) and entry["species"]:
            out.setdefault(to_id(entry["species"]), entry)
    return out


def belief_view(entry: dict | None, member: dict | None, names: dict[str, str]) -> dict | None:
    """{brought, lead, item, ability, moves} for one opposing member, with what it has revealed left out; None
    without an entry."""
    if not isinstance(entry, dict):
        return None
    member = member if isinstance(member, dict) else {}
    revealed_moves = {
        to_id(m[0]) for m in member.get("moves") or [] if isinstance(m, list | tuple) and m and isinstance(m[0], str)
    }
    item_known = member.get("item") is not None  # "" is known too: used up or removed
    ability_known = isinstance(member.get("ability"), str) and bool(member.get("ability"))
    moves = [m for m in _pairs(entry.get("moves"), names) if m["id"] not in revealed_moves]
    return {
        "brought": _p(entry.get("brought")),
        "lead": _p(entry.get("lead")),
        "item": [] if item_known else _pairs(entry.get("item"), names, item=True),
        "ability": [] if ability_known else _pairs(entry.get("ability"), names),
        "moves": moves,
    }

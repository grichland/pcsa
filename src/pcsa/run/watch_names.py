"""Display names for `pcsa watch`: the team sheet and the record first, then CG's tables (assets/game/cg/*.json)."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from pcsa.common import paths

TABLES = ("species", "moves", "items", "abilities")
ORDER_PARTS = 2


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _table(kind: str) -> dict:
    try:
        data = json.loads((paths.assets_dir() / "game" / "cg" / f"{kind}.json").read_text(encoding="utf-8"))
        table = data[kind]
    except (OSError, ValueError, KeyError, TypeError):  # the page still works: ids are shown with a capital
        return {}
    return table if isinstance(table, dict) else {}


@lru_cache(maxsize=1)
def game_names() -> dict[str, str]:
    """id -> written name of every species, move, item and ability in CG's tables; {} when they cannot be read."""
    out: dict[str, str] = {}
    for kind in TABLES:
        for key, entry in _table(kind).items():
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                out.setdefault(str(key), entry["name"])
    return out


@lru_cache(maxsize=1)
def _moves_pp() -> dict[str, int]:
    out = {}
    for key, entry in _table("moves").items():
        pp = entry.get("pp") if isinstance(entry, dict) else None
        if isinstance(pp, int) and not isinstance(pp, bool):
            out[str(key)] = pp
    return out


def max_pp(move: str) -> int | None:
    return _moves_pp().get(to_id(move))


@lru_cache(maxsize=1)
def _species_types() -> dict[str, list[str]]:
    out = {}
    for key, entry in _table("species").items():
        types = entry.get("types") if isinstance(entry, dict) else None
        if isinstance(types, list):
            out[str(key)] = [str(t) for t in types]
    return out


@lru_cache(maxsize=1)
def _mega_bases() -> dict[str, str]:
    """Mega form id -> its base species' written name (Dragonite-Mega -> Dragonite)."""
    out = {}
    for key, entry in _table("species").items():
        if isinstance(entry, dict) and "mega" in str(entry.get("forme") or "").lower() and entry.get("baseSpecies"):
            out[str(key)] = str(entry["baseSpecies"])
    return out


def species_label(species: str, names: dict[str, str] | None = None) -> str:
    """The written name of a species, a Mega form named by its base species (the team cards' name)."""
    return _mega_bases().get(to_id(species)) or pretty(species, names)


def species_types(species: str) -> list[str]:
    return list(_species_types().get(to_id(species), []))


def own_sets(export: str | None) -> dict[str, dict]:
    """Display names from the team sheet: {name: {item, ability, moves}} (a nickname or the species)."""
    sets: dict[str, dict] = {}
    for block in re.split(r"\n\s*\n", export or ""):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        head, _, item = lines[0].partition(" @ ")
        name = re.sub(r"\s*\((?:M|F)\)\s*$", "", head).strip()
        name = re.sub(r"^.*\((.+)\)$", r"\1", name) if "(" in name else name
        ability = next((x.split(":", 1)[1].strip() for x in lines if x.startswith("Ability:")), "")
        moves = [x[1:].strip() for x in lines if x.startswith("-")]
        sets[name] = {"item": item.strip(), "ability": ability, "moves": moves}
    return sets


def names_of(data: dict) -> dict[str, str]:
    """id -> written name, for every species, move, item and ability the record's team sheet spells out."""
    found: list[str] = []
    export = data.get("own_export") if isinstance(data, dict) else None
    for name, sheet in own_sets(export if isinstance(export, str) else None).items():
        found += [name, sheet["item"], sheet["ability"], *sheet["moves"]]
    own = data.get("own_names") if isinstance(data, dict) else None
    found += [n for n in own if isinstance(n, str)] if isinstance(own, list) else []
    return {to_id(name): name.strip() for name in found if name and name.strip()}


def shown(text: str, names: dict[str, str]) -> str:
    """The written name of an id (the record's, else CG's), or the text unchanged when neither knows it."""
    key = to_id(text)
    return names.get(key) or game_names().get(key) or text


def pretty(ident: str, names: dict[str, str] | None = None) -> str:
    """The written name of an id; an id no table knows is shown with a capital; '' for an empty slot."""
    if not ident:
        return ""
    found = shown(ident, names or {})
    return found if found != ident else ident[:1].upper() + ident[1:]


def order_view(order: str, names: dict[str, str]) -> dict:
    """'move ironhead mega' -> Iron Head + Mega (a move); 'switch Kingambit' -> Kingambit (a switch)."""
    parts = order.split()
    if len(parts) < ORDER_PARTS or parts[0] not in ("move", "switch"):
        return {"order": order, "label": order, "kind": "other"}
    if parts[0] == "switch":
        return {"order": order, "label": shown(" ".join(parts[1:]), names), "kind": "switch"}
    extras = [word.title() for word in parts[2:]]
    return {"order": order, "label": " + ".join([shown(parts[1], names), *extras]), "kind": "move"}

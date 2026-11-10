"""Display names for `pcsa watch`, taken from the decision record itself.

The tracked-battle snapshot and the legal orders use ids (`ninetalesalola`, `swordsdance`). The same record
carries the written names: its protocol lines (`|move|p2a: Ceruledge|Swords Dance|...`), the request's move
list and our own team sheet. An id none of them names is shown as it is; nothing is guessed.
"""

from __future__ import annotations

import re

SPECIES_LINES = {"poke": 3, "switch": 3, "drag": 3, "replace": 3, "detailschange": 3, "-formechange": 3}
NAME_LINES = {"move": 3, "-ability": 3, "-item": 3, "-enditem": 3}
TAGGED = re.compile(r"^(?:move|ability|item): (.+)$")
ORDER_PARTS = 2


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


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


def _protocol_names(protocol) -> list[str]:
    out = []
    for line in protocol if isinstance(protocol, list) else []:
        parts = line.split("|") if isinstance(line, str) else []
        kind = parts[1] if len(parts) > 1 else ""
        if kind in SPECIES_LINES and len(parts) > SPECIES_LINES[kind]:
            out.append(parts[SPECIES_LINES[kind]].split(",")[0])
        elif kind in NAME_LINES and len(parts) > NAME_LINES[kind]:
            out.append(parts[NAME_LINES[kind]])
        out += [m.group(1) for part in parts[2:] if (m := TAGGED.match(part.strip()))]
    return out


def _request_names(request) -> list[str]:
    request = request if isinstance(request, dict) else {}
    out = []
    for active in request.get("active") or []:
        moves = active.get("moves") if isinstance(active, dict) else None
        out += [m["move"] for m in moves or [] if isinstance(m, dict) and isinstance(m.get("move"), str)]
    side = request.get("side")
    for member in (side.get("pokemon") if isinstance(side, dict) else None) or []:
        if isinstance(member, dict) and isinstance(member.get("details"), str):
            out.append(member["details"].split(",")[0])
    return out


def names_of(data: dict) -> dict[str, str]:
    """id -> written name, for every species, move, item and ability the record spells out."""
    found = _protocol_names(data.get("protocol")) + _request_names(data.get("request"))
    export = data.get("own_export")
    for name, sheet in own_sets(export if isinstance(export, str) else None).items():
        found += [name, sheet["item"], sheet["ability"], *sheet["moves"]]
    own = data.get("own_names")
    found += [n for n in own if isinstance(n, str)] if isinstance(own, list) else []
    return {to_id(name): name.strip() for name in found if name and name.strip()}


def shown(text: str, names: dict[str, str]) -> str:
    """The written name of an id, or the text unchanged when the record never spells it out."""
    return names.get(to_id(text), text)


def order_view(order: str, names: dict[str, str]) -> dict:
    """'move ironhead mega' -> Iron Head + Mega (a move); 'switch Kingambit' -> Kingambit (a switch)."""
    parts = order.split()
    if len(parts) < ORDER_PARTS or parts[0] not in ("move", "switch"):
        return {"order": order, "label": order, "kind": "other"}
    if parts[0] == "switch":
        return {"order": order, "label": shown(" ".join(parts[1:]), names), "kind": "switch"}
    extras = [word.title() for word in parts[2:]]
    return {"order": order, "label": " + ".join([shown(parts[1], names), *extras]), "kind": "move"}

"""The teams and field panels of `pcsa watch`, read from CG's Battle as the record keeps it (`notes.cg.view`).

HP is a percentage on both sides, as CG holds it. Our members' item, ability and moves come from the view, then the
team sheet; PP is shown for the active only. Turns left are estimates from the default durations; items such as
Heat Rock or Light Clay extend them, which the battle does not say until the effect ends.
"""

from __future__ import annotations

from math import isfinite

from pcsa.run.watch_belief import belief_index, belief_view
from pcsa.run.watch_names import max_pp, own_sets, pretty, species_types, to_id

WEATHER_TURNS, FIELD_TURNS = 5, 5
SIDE_TURNS = {
    "reflect": 5,
    "lightscreen": 5,
    "auroraveil": 5,
    "tailwind": 4,
    "safeguard": 5,
    "mist": 5,
    "luckychant": 5,
}
STACKING = ("spikes", "toxicspikes")  # shown with their layers; the other hazards stay until removed
HAZARDS = ("stealthrock", "stickyweb")
WEATHER_NAMES = {
    "sunnyday": "Sun",
    "desolateland": "Harsh sun",
    "raindance": "Rain",
    "primordialsea": "Heavy rain",
    "sandstorm": "Sand",
    "snowscape": "Snow",
    "snow": "Snow",
    "hail": "Hail",
    "deltastream": "Strong winds",
}
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
CARD_KEYS = ("name", "hp", "fraction", "status", "boosts")


def _int(value) -> int | None:
    return value if type(value) is int else None


def _fraction(member: dict) -> float | None:
    if member.get("fainted"):
        return 0.0
    hp = member.get("hp")
    if isinstance(hp, bool) or not isinstance(hp, int | float) or not isfinite(hp):
        return None
    return round(min(1.0, max(0.0, float(hp))), 4)


def _percent(fraction: float | None) -> str:
    if fraction is None:
        return ""
    whole = round(fraction * 100)
    return f"{1 if fraction > 0 and whole == 0 else whole}%"


def _boosts(member: dict) -> str:
    raw = member.get("boosts")
    raw = raw if isinstance(raw, dict) else {}
    return " ".join(f"{k}{n:+d}" for k in BOOSTS if (n := _int(raw.get(k))))


def _types(member: dict) -> str:
    types = [str(t) for t in member.get("types") or [] if isinstance(t, str)]
    types = types or species_types(member.get("form") or member.get("species") or "")
    return "/".join(t[:1].upper() + t[1:] for t in types)


def _moves(member: dict) -> list[tuple[str, int | None]]:
    out = []
    for move in member.get("moves") or []:
        if isinstance(move, list | tuple) and move and isinstance(move[0], str) and move[0]:
            out.append((move[0], _int(move[1]) if len(move) > 1 else None))
    return out


def _base(member: dict, names: dict[str, str]) -> dict:
    """What both sides' cards share: name, HP, status, stat stages and types."""
    fraction = _fraction(member)
    fainted = bool(member.get("fainted"))
    return {
        "name": pretty(str(member.get("species") or ""), names) or "?",
        "active": bool(member.get("active")),
        "fainted": fainted,
        "hp": _percent(fraction),
        "fraction": fraction,
        "status": "" if fainted else str(member.get("status") or ""),
        "boosts": "" if fainted else _boosts(member),
        "types": _types(member),
    }


def _sheet(sets: dict[str, dict], name: str, species: str) -> dict:
    if name in sets:
        return sets[name]
    key = to_id(species)
    return next((s for n, s in sets.items() if to_id(n) and key.startswith(to_id(n))), {})


def our_card(member: dict, sets: dict[str, dict], names: dict[str, str]) -> dict:
    card = _base(member, names)
    sheet = _sheet(sets, card["name"], str(member.get("species") or ""))
    item, ability = member.get("item"), member.get("ability")
    moves = _moves(member)
    shown_moves = [
        {
            "name": pretty(move, names),
            "pp": [pp, top] if card["active"] and pp is not None and (top := max_pp(move)) else None,
        }
        for move, pp in moves
    ] or [{"name": name, "pp": None} for name in sheet.get("moves") or []]
    if item is None:  # not in the view: the team sheet's
        item = to_id(sheet.get("item", ""))
    return {
        **card,
        "item": pretty(item, names) if isinstance(item, str) else "",
        "ability": pretty(ability, names) if isinstance(ability, str) and ability else sheet.get("ability", ""),
        "moves": shown_moves,
    }


def foe_card(member: dict, names: dict[str, str], beliefs: dict[str, dict]) -> dict:
    item, ability = member.get("item"), member.get("ability")
    return {
        **_base(member, names),
        "item": pretty(item, names) if isinstance(item, str) else "",
        "ability": pretty(ability, names) if isinstance(ability, str) else "",
        "moves": [pretty(move, names) for move, _ in _moves(member)],
        "belief": belief_view(beliefs.get(to_id(str(member.get("species") or ""))), member, names),
    }


def _members(value) -> list[dict]:
    return [m for m in value if isinstance(m, dict)] if isinstance(value, list) else []


def _left(turns: int, since) -> int | None:
    return max(1, turns - since) if type(since) is int else None


def _side_view(conditions, names: dict[str, str]) -> list[dict]:
    out = []
    for key, value in (conditions if isinstance(conditions, dict) else {}).items():
        name = pretty(str(key), names)
        if key in STACKING:
            out.append({"name": name, "layers": _int(value) or 1, "left": None})
        elif key in HAZARDS:
            out.append({"name": name, "layers": None, "left": None})
        else:
            turns = SIDE_TURNS.get(key)
            out.append({"name": name, "layers": None, "left": _left(turns, value) if turns else None})
    return out


def field_view(view: dict, names: dict[str, str]) -> dict:
    """Turn, weather, terrain and rooms with turns left, and each side's conditions."""
    weather, fields = view.get("weather"), view.get("fields")
    weather = weather if isinstance(weather, dict) else {}
    fields = fields if isinstance(fields, dict) else {}
    first = next(iter(weather.items()), None)
    shown_weather = None
    if first is not None:
        name = WEATHER_NAMES.get(first[0], pretty(str(first[0]), names))
        shown_weather = {"name": name, "left": _left(WEATHER_TURNS, first[1])}
    return {
        "turn": _int(view.get("turn")) or None,
        "weather": shown_weather,
        "fields": [{"name": pretty(str(k), names), "left": _left(FIELD_TURNS, v)} for k, v in fields.items()],
        "ours": _side_view(view.get("side"), names),
        "theirs": _side_view(view.get("opp_side"), names),
    }


def on_field(members: list[dict]) -> dict:
    member = next((m for m in members if m["active"] and not m["fainted"]), None)
    return {k: member[k] for k in CARD_KEYS} if member else dict.fromkeys(CARD_KEYS)


def _condition_text(c: dict) -> str:
    if c["layers"]:
        return f"{c['name']} x{c['layers']}"
    return f"{c['name']} ({c['left']} left)" if c["left"] else c["name"]


def _conditions(conditions: list[dict]) -> str:
    return ", ".join(_condition_text(c) for c in conditions) or "-"


def _move_text(move) -> str:
    if not isinstance(move, dict):
        return str(move)
    return f"{move['name']} {move['pp'][0]}/{move['pp'][1]}" if move.get("pp") else move["name"]


def _card_line(card: dict) -> str:
    state = "fainted" if card["fainted"] else "ACTIVE" if card["active"] else "bench"
    moves = [_move_text(m) for m in card["moves"]]
    return (
        f"  {card['name']:<14} {state:<7} HP {card['hp'] or '?':<5} {card['types'] or '-':<16} "
        f"status {card['status'] or '-':<4} boosts {card['boosts'] or '-':<12} ability {card['ability'] or '?'}   "
        f"item {card['item'] or '?'}   moves {', '.join(moves) or '?'}"
    )


def battle_text(panels: dict) -> str:
    """A plain-text summary of the battle at the decision: the field, then each side's members."""
    field = panels["field"]
    weather = f"{field['weather']['name']} ({field['weather']['left']} left)" if field["weather"] else "-"
    fields = ", ".join(f"{f['name']} ({f['left']} left)" for f in field["fields"]) or "-"
    lines = [f"TURN {field['turn'] or '?'}   weather: {weather}   fields: {fields}"]
    lines.append(f"OURS   side: {_conditions(field['ours'])}")
    lines += [_card_line(card) for card in panels["your_team"]]
    lines.append(f"THEIRS side: {_conditions(field['theirs'])}")
    lines += [_card_line(card) for card in panels["foes"]]
    if panels["unseen"]:
        lines.append(f"  preview, not yet seen: {', '.join(panels['unseen'])}")
    return "\n".join(lines) + "\n"


def view_panels(view: dict, export: str | None, names: dict[str, str], belief=None) -> dict:
    """your_team, foes, unseen (names) with unseen_belief beside it, field, active and opponent from one view;
    `belief` is the net's belief list (matched to their members by species)."""
    own, opp = _members(view.get("own")), _members(view.get("opp"))
    picks = [i for i in view.get("picks") or [] if type(i) is int and 0 <= i < len(own)]
    sets = own_sets(export)
    team = [our_card(own[i], sets, names) for i in picks] if picks else [our_card(m, sets, names) for m in own]
    beliefs = belief_index(belief)
    foes = [foe_card(m, names, beliefs) for m in opp if m.get("revealed")]
    hidden = [m for m in opp if not m.get("revealed") and isinstance(m.get("species"), str) and m["species"]]
    unseen = [
        {"name": pretty(m["species"], names), "belief": belief_view(beliefs.get(to_id(m["species"])), m, names)}
        for m in hidden
    ]
    return {
        "your_team": team,
        "foes": foes,
        "unseen": [u["name"] for u in unseen],
        "unseen_belief": unseen,
        "field": field_view(view, names),
        "active": on_field(team),
        "opponent": on_field(foes),
    }

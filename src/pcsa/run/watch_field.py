"""The battle-state box of `pcsa watch`: weather, terrain, rooms and each side's hazards and screens, read from a
decision record's protocol (p1 is us). Turns left are estimates from the default durations; items such as Heat
Rock or Light Clay extend them, which the protocol does not say until the effect ends.
"""

from __future__ import annotations

import re

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
STACKING = {"spikes": 3, "toxicspikes": 2}
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


def _id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.split(":", 1)[-1].lower())


def _pretty(text: str) -> str:
    return text.split(":", 1)[-1].strip()


def _left(start: int, turns: int, now: int) -> int:
    return max(1, start + turns - now)


class _Field:
    def __init__(self) -> None:
        self.turn = 0
        self.weather: tuple[str, int] | None = None
        self.fields: dict[str, tuple[str, int]] = {}
        self.sides: dict[str, dict[str, list]] = {"p1": {}, "p2": {}}

    def take(self, parts: list[str]) -> None:
        kind = parts[1] if len(parts) > 1 else ""
        if kind == "turn" and len(parts) > 2 and parts[2].isdigit():  # noqa: PLR2004
            self.turn = int(parts[2])
        elif kind == "-weather" and len(parts) > 2:  # noqa: PLR2004
            name = _id(parts[2])
            if name == "none":
                self.weather = None
            elif "[upkeep]" not in parts:
                self.weather = (name, self.turn)
        elif kind == "-fieldstart" and len(parts) > 2:  # noqa: PLR2004
            self.fields[_id(parts[2])] = (_pretty(parts[2]), self.turn)
        elif kind == "-fieldend" and len(parts) > 2:  # noqa: PLR2004
            self.fields.pop(_id(parts[2]), None)
        elif kind in ("-sidestart", "-sideend") and len(parts) > 3:  # noqa: PLR2004
            self._side(kind, parts[2][:2], parts[3])
        elif kind == "-swapsideconditions":
            self.sides = {"p1": self.sides["p2"], "p2": self.sides["p1"]}

    def _side(self, kind: str, side: str, condition: str) -> None:
        if side not in self.sides:
            return
        key, conditions = _id(condition), self.sides[side]
        if kind == "-sideend":
            conditions.pop(key, None)
        elif key in STACKING:
            entry = conditions.setdefault(key, [_pretty(condition), 0])
            entry[1] = min(STACKING[key], entry[1] + 1)
        elif key not in conditions:
            conditions[key] = [_pretty(condition), self.turn]


def _side_view(conditions: dict[str, list], now: int) -> list[dict]:
    out = []
    for key, (name, value) in conditions.items():
        if key in STACKING:
            out.append({"name": name, "layers": value, "left": None})
        else:
            turns = SIDE_TURNS.get(key)
            out.append({"name": name, "layers": None, "left": _left(value, turns, now) if turns else None})
    return out


def field_view(protocol, turn=None) -> dict:
    """Turn, weather, terrain and rooms with turns left, and each side's conditions (ours: p1)."""
    state = _Field()
    for line in protocol if isinstance(protocol, list) else []:
        if isinstance(line, str):
            state.take(line.split("|"))
    now = turn if isinstance(turn, int) else state.turn
    weather = None
    if state.weather is not None:
        name, start = state.weather
        weather = {"name": WEATHER_NAMES.get(name, name), "left": _left(start, WEATHER_TURNS, now)}
    fields = [{"name": name, "left": _left(start, FIELD_TURNS, now)} for name, start in state.fields.values()]
    return {
        "turn": now or None,
        "weather": weather,
        "fields": fields,
        "ours": _side_view(state.sides["p1"], now),
        "theirs": _side_view(state.sides["p2"], now),
    }

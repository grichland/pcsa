"""Showdown's battle-message templates from the pinned simulator, and lookup of one log message.

The snapshot assets/game/battle_text.json holds the English text tables. A template identifies which
effect and message a log row is; what the row does to the battle is decided in log_patterns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from pcsa.battle import game

SIDE_WORDS = "the opposing team|your team|the opposing side|your side"
PLACEHOLDERS = {
    "POKEMON": "(?P<{name}>.+?)",
    "NICKNAME": "(?P<{name}>.+?)",
    "TARGET": "(?P<{name}>.+?)",
    "SOURCE": "(?P<{name}>.+?)",
    "TEAM": f"(?P<{{name}}>{SIDE_WORDS})",
    "PARTY": "(?P<{name}>.+?)",
}
APOSTROPHES = "[’']"
MIN_LITERAL_LETTERS = 8  # "{NICKNAME}" alone would match any row
NAME_FORMATS = {"pokemon", "opposingPokemon", "fullName", "team", "opposingTeam", "party", "opposingParty"}
# Champions wordings that differ from Showdown's, as templates of the same effect and key.
CHAMPIONS_TEXT = [
    ("default", "par", "Paralysis", "cant", "{POKEMON} couldn't move because it's paralyzed!"),
    ("moves", "rest", "Rest", "start", "{POKEMON} slept and restored its HP!"),
    ("moves", "toxicspikes", "Toxic Spikes", "start", "Toxic spikes were scattered on the ground all around {TEAM}!"),
]


@dataclass(frozen=True)
class TextMatch:
    effect: str
    name: str
    key: str
    groups: dict


def pattern(template: str) -> re.Pattern | None:
    counts: dict[str, int] = {}
    parts, position = [], 0
    for m in re.finditer(r"\{([A-Z0-9]+)\}", template):
        parts.append(re.escape(template[position : m.start()]).replace("'", APOSTROPHES))
        base = re.sub(r"\d+$", "", m[1])
        counts[base] = counts.get(base, 0) + 1
        name = base if counts[base] == 1 else f"{base}{counts[base]}"
        parts.append(PLACEHOLDERS.get(base, "(?P<{name}>.+?)").format(name=name))
        position = m.end()
    parts.append(re.escape(template[position:]).replace("'", APOSTROPHES))
    try:
        return re.compile("".join(parts), re.IGNORECASE)
    except re.error:
        return None


@lru_cache(maxsize=1)
def compiled() -> tuple:
    entries = json.loads(game.asset_path("battle_text.json").read_text(encoding="utf-8"))["entries"]
    entries += [{"section": s, "effect": e, "name": n, "key": k, "template": t} for s, e, n, k, t in CHAMPIONS_TEXT]
    out = []
    for entry in entries:
        if entry["section"] == "default" and entry["key"] in NAME_FORMATS:
            continue
        literal = len(re.sub(r"[^A-Za-z]", "", re.sub(r"\{[A-Z0-9]+\}", "", entry["template"])))
        if literal >= MIN_LITERAL_LETTERS and (regex := pattern(entry["template"])) is not None:
            out.append((literal, regex, entry))
    return tuple(sorted(out, key=lambda item: -item[0]))


def lookup(message: str) -> tuple[TextMatch, ...]:
    """Every template of the most specific length that matches the message."""
    best, found = None, []
    for literal, regex, entry in compiled():
        if best is not None and literal < best:
            break
        if m := regex.fullmatch(message):
            best = literal
            found.append(
                TextMatch(
                    entry["effect"],
                    entry["name"],
                    entry["key"],
                    {k: v.strip() for k, v in m.groupdict().items() if v},
                )
            )
    return tuple(found)

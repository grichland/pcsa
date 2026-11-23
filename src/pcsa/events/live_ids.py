"""Ids and wording tables for LIVE protocol events: species, moves, items, abilities, effects."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache

from pcsa.battle import game
from pcsa.battle.ids import to_id
from pcsa.battle.names import showdown_species

VERSION = "0.7"

# Weather rows name the weather in both the start and the end wording.
WEATHER_ROWS = {
    "A sandstorm kicked up!": ("sandstorm", "start"),
    "The sandstorm subsided.": ("sandstorm", "end"),
    "It started to rain!": ("raindance", "start"),
    "The rain stopped.": ("raindance", "end"),
    "The sunlight turned harsh!": ("sunnyday", "start"),
    "The harsh sunlight faded.": ("sunnyday", "end"),
    "It started to snow!": ("snowscape", "start"),
    "The snow stopped.": ("snowscape", "end"),
}
# Cause rows: the parser's cause name -> the effect id sent. Ids that name the wording are used where one
# wording fits several mechanics.
CAUSES = {
    "Sandstorm": "sandstorm",
    "brn": "brn",
    "psn": "psn",
    "Stealth Rock": "stealthrock",
    "Spikes": "spikes",
    "Recoil": "recoil",
    "Leech Seed": "leechseed",
    "Curse": "curse",
    "Wish": "wish",
    "crash": "crashed",
    "drain": "energy_drained",
}
TRAILING_CAUSES = {
    "hp_restored": "hp_restored",
    "lifeorb": "lost_some_hp",
    "hurt": "was_hurt",
    "pain_split": "pain_split",
}
TRAILING_ONCE = {"struggle": "no_moves_left", "destiny_bond": "took_attacker_down", "weather_suppressed": "weather_effects_gone"}
# fail rows that say why: the reason sent; heal_failed only for "X's HP is full!"
TRAILING_FAIL = {"yawn_already": "yawn", "substitute_already": "substitute", "heal_failed": "hpfull"}
ALREADY = {"poisoned": "psn", "badly poisoned": "tox", "burned": "brn", "paralyzed": "par", "asleep": "slp"}
EFFECTIVENESS = {
    "super": "super",
    "extremely": "extreme",
    "not_very": "resisted",
    "mostly_ineffective": "mostly_ineffective",
    "immune": "immune",
}
FORM_IDS = {"Mimikyu-Busted": "mimikyubusted", "Palafin-Hero": "palafinhero"}


def effect_id(effect: str) -> str:
    """'move: Heal Block' -> 'healblock'; 'perish3' stays."""
    return to_id(effect.split(":", 1)[-1])


@dataclass
class Teams:
    """What battle_start tells the converter: our brought species and the opponent's preview."""

    own: tuple[str, ...]
    opp_preview: tuple[str | None, ...] = ()
    notes: list = field(default_factory=list)

    def own_ids(self) -> set[str]:
        return {base_id(s) for s in self.own}

    def opp_ids(self) -> set[str]:
        return {base_id(s) for s in self.opp_preview if s}


def species_id(name: str) -> str:
    """A shown name as a species id; Mega display names are left to `mega_form`. "’s Archaludon" is the
    borrowed-team wording with the owner's name unread."""
    name = re.sub(r"^[’']s\s*", "", name.strip())
    return to_id(showdown_species(name).species)


def base_id(name: str) -> str:
    ident = species_id(name)
    entry = game.species_entry(ident) if ident in game.dex() else {}
    return to_id(entry.get("baseSpecies") or ident) if entry else ident


def preview_id(name: str) -> str:
    """Species id as brought of a preview name: the form the sprite shows (Rotom Wash, Alolan Ninetales), which
    the log's rows leave out. A Mega or another form taken only in battle is its base species."""
    ident = species_id(name)
    entry = game.species_entry(ident) if ident in game.dex() else {}
    in_battle = str(entry.get("forme") or "").startswith("Mega") or entry.get("battleOnly")
    return ident if entry and not in_battle else base_id(name)


@lru_cache(maxsize=None)
def form_by_move(mon: str, move: str) -> str | None:
    """The one form of the species that learns a move `mon` does not (Hydro Pump: Rotom Wash); else None."""
    sets = game.learnsets()
    if mon not in sets or move in sets[mon]:
        return None
    base = base_id(mon)
    forms = [s for s in sets if s != mon and move in sets[s] and base_id(s) == base and preview_id(s) == s]
    return forms[0] if len(forms) == 1 else None


def mon_id(name: str, side: str | None, teams: Teams) -> str:
    """Species id as brought: an own row's name is matched to our brought species by base species, another
    row's to the form their preview showed. An OCR-damaged name is mapped to a uniquely close species of the
    battle's teams, then of the dex."""
    ident = species_id(name)
    if ident not in game.dex():
        ident = repair_species(ident, teams) or ident
    base = base_id(ident)
    if side == "own":
        for brought in teams.own:
            if base_id(brought) == base:
                return to_id(brought)
    elif side == "opp" or base not in teams.own_ids():
        shown = [preview_id(p) for p in teams.opp_preview if p and base_id(p) == base]
        if len(shown) == 1 and ident == base:
            return shown[0]
    return ident


def repair_species(ident: str, teams: Teams) -> str | None:
    pool = {base_id(s) for s in (*teams.own, *(p for p in teams.opp_preview if p))}
    for choices, floor in ((pool, 0.8), (set(game.dex()), 0.88)):
        ranked = sorted(((difflib.SequenceMatcher(None, ident, c).ratio(), c) for c in choices), reverse=True)
        if ranked and ranked[0][0] >= floor and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.05):
            return ranked[0][1]
    return None


def mega_form(shown: str, base: str) -> str | None:
    """The Mega's species id when the shown name fits one Mega of the species; None when it fits several
    (Mega Charizard X/Y, Mega Garchomp / Garchomp Z): the row alone does not say which."""
    forms = game.megas()[0].get(base_id(base), [])
    return forms[0] if len(forms) == 1 else None


@lru_cache(maxsize=None)
def legal(table: str) -> frozenset[str]:
    """Legal ids of one CG table (species, moves, items, abilities)."""
    return frozenset(game.dex()) if table == "species" else frozenset(game.cg_table(table))


def move_id(name: str) -> str:
    return to_id(name)


def item_id(name: str) -> tuple[str, bool]:
    """(id, repaired): OCR slips are mapped to a uniquely close legal item."""
    found = game.canonical_item(name)
    ident = to_id(found or name)
    return ident, found is not None and to_id(found) != to_id(name)


def ability_id(name: str, species: str | None = None) -> tuple[str, bool]:
    try:
        found = game.canonical_ability(name, species)
    except KeyError:
        found = game.canonical_ability(name)
    ident = to_id(found or name)
    return ident, found is not None and to_id(found) != to_id(name)


PRIORITY_LOOSE = re.compile(r"^Move\s*([+\-—–])\s*(\d+)")


def priority_of(label: str | None) -> int | None:
    """Signed priority from the row category ('Move +1 Priority'); None when absent or unreadable."""
    if not label or label == "Move":
        return None
    m = PRIORITY_LOOSE.match(label)
    if not m:
        return None
    return int(m[2]) * (-1 if m[1] in "-—–" else 1)

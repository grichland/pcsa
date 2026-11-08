"""What the scout already knows: species, moves, items and abilities from the game assets."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass

from pcsa.common import paths

NAME_MATCH = 0.80  # "Garchornp" vs "garchomp" scores 0.82; higher would report OCR slips as new names
MIN_CONFIDENCE = 0.85
NAME_TOKEN = re.compile(r"^[a-z][a-z'.\- ]{2,24}$")
ITEM_WORDS = (
    "ite z|ite|stone|berry|orb|scarf|band|specs|sash|helmet|seed|herb|glasses|belt|plate|gem|vest|lens|claw|fang|"
    "rock|policy|energy|button|pack|card|balloon"
)
ITEM_HINT = re.compile(rf".*({ITEM_WORDS})$")
UI_TEXT = """
back done ok close cancel next skip menu battle battles ranked single double casual friend recruit shop team
teams edit select fight pokemon view log switch in move moves info mega evolve continue quit battling rank tier
rating placement season regulation pass premium rewards reward vp earned streak win won lost lose turn hp pp
level lv ability item nature stats points base evolution roster filter sort search all type types moveset
summary begin matchmaking preparing standing by communicating loading data trainer icon clothing coupon ticket
tickets
"""
UI_WORDS = frozenset(UI_TEXT.split())


def letters(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalpha())


@dataclass
class Known:
    species: dict[str, str]  # letters -> display name
    other: set[str]  # letters of moves, items, abilities
    items: set[str]

    @classmethod
    def load(cls) -> Known:
        species = json.loads((paths.assets_dir() / "game/species.json").read_text(encoding="utf-8"))["species"]
        vocab = json.loads((paths.assets_dir() / "game/vocab.json").read_text(encoding="utf-8"))["tables"]
        names = {letters(e["name"]): e["name"] for e in species.values()}
        for e in species.values():
            if e.get("baseSpecies"):
                names.setdefault(letters(e["baseSpecies"]), e["baseSpecies"])
        other = {letters(v) for t in ("move", "item", "ability") for v in vocab.get(t, [])}
        return cls(names, other, {letters(v) for v in vocab.get("item", [])})

    def match_species(self, token: str) -> str | None:
        key = letters(token)
        if key in self.species:
            return self.species[key]
        close = difflib.get_close_matches(key, self.species.keys(), n=1, cutoff=NAME_MATCH)
        return self.species[close[0]] if close else None

    def is_known(self, token: str) -> bool:
        key = letters(token)
        if key in self.species or key in self.other or key in UI_WORDS:
            return True
        return bool(difflib.get_close_matches(key, self.other, n=1, cutoff=NAME_MATCH))


def name_candidates(boxes, known: Known) -> tuple[set[str], set[str]]:
    """(matched species, unmatched name-like tokens) from one roster or summary frame."""
    matched, unmatched = set(), set()
    for box in boxes:
        norm = box.norm
        if box.confidence < MIN_CONFIDENCE or not NAME_TOKEN.fullmatch(norm) or norm in UI_WORDS:
            continue
        if any(ch.isdigit() for ch in norm):
            continue
        species = known.match_species(norm)
        if species:
            matched.add(species)
        elif not known.is_known(norm):
            unmatched.add(box.text.strip())
    return matched, unmatched


def item_candidates(boxes, known: Known) -> set[str]:
    out = set()
    for box in boxes:
        if box.confidence < MIN_CONFIDENCE or not ITEM_HINT.fullmatch(box.norm):
            continue
        key = letters(box.norm)
        if key not in known.items and not difflib.get_close_matches(key, known.items, n=1, cutoff=NAME_MATCH):
            out.add(box.text.strip())
    return out

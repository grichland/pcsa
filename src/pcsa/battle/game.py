"""Static game knowledge: the Champions dex, Mega forms, abilities, items, learnsets and format rules.

Every table is a pinned asset under assets/game/, regenerated from the pinned Showdown commit outside
this repo and checked against tools.lock.json at load time.
"""

from __future__ import annotations

import difflib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.common import paths
from pcsa.common.errors import ConfigError

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
LEVEL = 50
ABILITY_MATCH_FLOOR = 0.75  # with the species known; one-letter OCR slips resolve
ABILITY_MATCH_FLOOR_ANY = 0.9
ITEM_MATCH_FLOOR = 0.9
MATCH_MARGIN = 0.08


def _asset(name: str) -> dict:
    path = paths.assets_dir() / "game" / name
    if not path.is_file():
        raise ConfigError(f"game asset missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def pinned_commit() -> str:
    return json.loads(paths.tools_lock().read_text(encoding="utf-8"))["showdown"]["commit"]


def _check_pin(data: dict, name: str) -> None:
    if data.get("engine_commit") not in (None, pinned_commit()):
        raise ConfigError(
            f"{name} was built from Showdown {data.get('engine_commit')!r}; tools.lock.json pins {pinned_commit()!r}"
        )


# Dex ---------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def dex() -> dict[str, dict]:
    """Species id -> entry: poke-env's Gen 9 dex overlaid with the pinned Champions species table."""
    out = dict(GenData.from_gen(9).pokedex)
    data = _asset("species.json")
    _check_pin(data, "species.json")
    out.update(data["species"])
    return out


def species_entry(key: str) -> dict:
    return dex().get(to_id_str(key)) or {}


def base_name(key: str) -> str:
    """Base species id (an id on base entries, a display name on formes)."""
    e = species_entry(key)
    return to_id_str(e.get("baseSpecies") or e.get("name") or key)


def is_mega(key: str) -> bool:
    return "Mega" in (species_entry(key).get("forme") or "")


@lru_cache(maxsize=1)
def megas() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Base species id -> its Mega entry keys; Mega stone id -> Mega entry key."""
    by_base: dict[str, list[str]] = {}
    by_stone: dict[str, str] = {}
    for key, e in dex().items():
        if "Mega" in (e.get("forme") or ""):
            by_base.setdefault(to_id_str(e.get("baseSpecies") or key), []).append(key)
            if e.get("requiredItem"):
                by_stone[to_id_str(e["requiredItem"])] = key
    return by_base, by_stone


@lru_cache(maxsize=512)
def mega_form_for_stone(species: str, stone: str) -> str | None:
    """The unique Mega form of `species` that `stone` produces, by display name."""
    if not stone:
        return None
    entry = species_entry(species)
    base = to_id_str(entry.get("baseSpecies") or species)
    matches = [
        v
        for v in dex().values()
        if to_id_str(v.get("baseSpecies", "")) == base
        and "Mega" in v.get("forme", "")
        and to_id_str(v.get("requiredItem", "")) == to_id_str(stone)
    ]
    if len(matches) > 1 and entry.get("gender") in ("M", "F"):
        matches = [v for v in matches if v.get("gender") == entry["gender"]]
    return matches[0]["name"] if len(matches) == 1 else None


def mega_form_for_ability(species: str, ability: str) -> tuple[str, str] | None:
    """(Mega form, stone) when a revealed ability identifies exactly one Mega form of the species."""
    base = to_id_str(species)
    ability_id = to_id_str(canonical_ability(ability) or ability)
    matches = []
    for key, entry in dex().items():
        if to_id_str(entry.get("baseSpecies", "")) != base or "Mega" not in entry.get("forme", ""):
            continue
        stone = entry.get("requiredItem")
        if ability_id in {to_id_str(v) for v in entry.get("abilities", {}).values()} and stone:
            matches.append((entry.get("name") or key, stone))
    return matches[0] if len(matches) == 1 else None


def mega_ability(entry: dict) -> str | None:
    """A Mega form's single ability id, or None for anything else."""
    abilities = set(entry.get("abilities", {}).values())
    if "Mega" in entry.get("forme", "") and len(abilities) == 1:
        return to_id_str(next(iter(abilities)))
    return None


# Abilities and items ---------------------------------------------------------------------------
def _closest(ident: str, choices: dict[str, str], floor: float) -> str | None:
    if ident in choices:
        return choices[ident]
    ranked = sorted(
        ((difflib.SequenceMatcher(None, ident, key).ratio(), value) for key, value in choices.items()), reverse=True
    )
    if ranked and ranked[0][0] >= floor and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= MATCH_MARGIN):
        return ranked[0][1]
    return None


def canonical_ability(value: str, species: str | None = None) -> str | None:
    """An exact or uniquely close ability name; with the species, only its forms' abilities compete."""
    mons = list(dex().values())
    if species is not None:
        base = to_id_str(species_entry(species).get("baseSpecies") or species)
        mons = [m for m in mons if to_id_str(m.get("baseSpecies") or m.get("name", "")) == base] or mons
    choices = {to_id_str(a): a for m in mons for a in m.get("abilities", {}).values()}
    return _closest(
        to_id_str(value), choices, ABILITY_MATCH_FLOOR if species is not None else ABILITY_MATCH_FLOOR_ANY
    )


@lru_cache(maxsize=1)
def item_ids() -> frozenset[str]:
    return frozenset(_asset("vocab.json")["tables"]["item"])


@lru_cache(maxsize=2048)
def canonical_item(value: str) -> str | None:
    """The item text itself when it is a known item, a uniquely close known id for OCR slips, else None."""
    ident = to_id_str(value)
    if not ident:
        return value
    if ident in item_ids():
        return value
    return _closest(ident, {i: i for i in item_ids()}, ITEM_MATCH_FLOOR)


# Learnsets ----------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def learnsets() -> dict[str, frozenset[str]]:
    data = _asset("learnsets.json")
    _check_pin(data, "learnsets.json")
    return {key: frozenset(moves) for key, moves in data["learnsets"].items()}


def can_learn(species: str, move: str) -> bool:
    """Whether the species can know the move in Champions; unknown species count as able to."""
    pool = learnsets().get(to_id_str(species))
    return pool is None or to_id_str(move) in pool


# Format rules ---------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def rules() -> dict:
    data = _asset("rules.json")
    if data.get("schema") != "pcsa-champions-rules/1":
        raise ConfigError(f"unsupported rules schema {data.get('schema')!r}")
    return data


def rule(path: str) -> Any:
    value: Any = rules()
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f"missing Champions rule {path!r}")
        value = value[key]
    return value


def asset_path(name: str) -> Path:
    return paths.assets_dir() / "game" / name

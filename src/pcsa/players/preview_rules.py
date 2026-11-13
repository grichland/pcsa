"""Team-preview rules over the neural player's critic (neural.yaml `preview_lead_avoid`, `preview_bring_max_one`).

The critic scores all 60 (lead, second, third) picks; these rules only remove picks, and the critic chooses among
the rest as before (its softmax / nucleus, or greedy). When no pick is left, the rules are dropped for that
battle (noted) and the critic chooses among all 60.

  preview_lead_avoid     {species: [types]}: do not lead `species` when their preview six holds a Pokemon of one
                         of those types (e.g. {golisopod: [Fire]})
  preview_bring_max_one  [{types: [...], species: [...]}]: when their six holds one of `types`, bring at most one
                         of `species` (e.g. Golisopod and Gholdengo into a Fire six)

Types come from poke-env's pokedex: a preview species (a form like Rotom-Heat or Basculegion-F has its own entry)
counts with its own types and those of its Mega forms (Charizard: Fire). Our members match by species or base
species (Golisopod-Mega is golisopod).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache

from poke_env.battle import PokemonType
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.common.errors import ConfigError

TYPES = frozenset(t.name.lower() for t in PokemonType)
Keep = Callable[[object, Sequence[tuple[int, ...]]], tuple[list[bool], dict]]


def _dex() -> dict:
    return GenData.from_gen(9).pokedex


@lru_cache(maxsize=1024)
def types_of(species: str) -> frozenset[str]:
    """The types (lower case) a preview species may have in battle: its own and its Mega forms'."""
    dex, key = _dex(), to_id_str(species)
    entry = dex.get(key)
    if entry is None:
        return frozenset()
    out = {t.lower() for t in entry.get("types", ())}
    for form in entry.get("otherFormes") or ():
        if "-mega" in form.lower() and (mega := dex.get(to_id_str(form))):
            out |= {t.lower() for t in mega.get("types", ())}
    return frozenset(out)


@lru_cache(maxsize=1024)
def names_of(species: str) -> frozenset[str]:
    """A member's species id and its base species id (golisopodmega -> {golisopodmega, golisopod})."""
    key = to_id_str(species)
    base = (_dex().get(key) or {}).get("baseSpecies")
    return frozenset({key, to_id_str(base)} if base else {key})


def check(lead_avoid: object, bring_max_one: object, where: str) -> None:
    """ConfigError on a malformed setting."""
    if not isinstance(lead_avoid, Mapping) or not all(
        isinstance(k, str) and _types_ok(v) for k, v in lead_avoid.items()
    ):
        raise ConfigError(
            f"{where}: preview_lead_avoid must map species to a list of types, e.g. {{golisopod: [Fire]}}"
        )
    rules = bring_max_one if isinstance(bring_max_one, list) else None
    if rules is None or not all(
        isinstance(r, Mapping)
        and set(r) == {"types", "species"}
        and _types_ok(r["types"])
        and isinstance(r["species"], list)
        and all(isinstance(x, str) for x in r["species"])
        for r in rules
    ):
        raise ConfigError(
            f"{where}: preview_bring_max_one must be a list of {{types: [...], species: [...]}}, "
            "e.g. [{types: [Fire], species: [golisopod, gholdengo]}]"
        )


def _types_ok(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(t, str) and t.lower() in TYPES for t in value)


def their_types(battle) -> dict[str, frozenset[str]]:
    """Their preview six: species -> types."""
    return {m.species: types_of(m.species) for m in battle.teampreview_opponent_team}


def _with(theirs: Mapping[str, frozenset[str]], types: Sequence[str]) -> list[str]:
    wanted = {t.lower() for t in types}
    return sorted(s for s, have in theirs.items() if have & wanted)


def keep_mask(
    battle, picks: Sequence[tuple[int, ...]], lead_avoid: Mapping, bring_max_one: Sequence[Mapping]
) -> tuple[list[bool], dict]:
    """(keep per pick, notes): False for a pick a rule forbids; notes name the rules that applied."""
    team = [names_of(m.species) for m in battle.team.values()]
    theirs = their_types(battle)
    banned_leads: set[int] = set()
    groups: list[set[int]] = []
    why: list[str] = []
    for species, types in lead_avoid.items():
        seen = _with(theirs, types)
        rows = {i for i, names in enumerate(team) if to_id_str(species) in names}
        if seen and rows:
            banned_leads |= rows
            why.append(f"no {to_id_str(species)} lead ({', '.join(seen)})")
    for rule in bring_max_one:
        seen = _with(theirs, rule["types"])
        wanted = {to_id_str(x) for x in rule["species"]}
        rows = {i for i, names in enumerate(team) if names & wanted}
        if seen and len(rows) > 1:
            groups.append(rows)
            why.append(f"at most one of {', '.join(sorted(wanted))} ({', '.join(seen)})")
    keep = [p[0] not in banned_leads and all(len(g & set(p)) <= 1 for g in groups) for p in picks]
    if not why:
        return keep, {}
    notes: dict = {"why": why, "kept": sum(keep), "of": len(picks)}
    if not any(keep):
        notes["unsatisfiable"] = True  # the critic chooses among every pick
    return keep, notes


def keeper(lead_avoid: Mapping, bring_max_one: Sequence[Mapping]) -> Keep | None:
    """The FormerPlayer's `preview_keep` for these settings, None when both are empty."""
    if not lead_avoid and not bring_max_one:
        return None
    return lambda battle, picks: keep_mask(battle, picks, lead_avoid, bring_max_one)

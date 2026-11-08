"""Visible-state tactical estimates: attack scores, accuracy, entry hazards. Not an engine calculation."""

from __future__ import annotations

from dataclasses import dataclass

from poke_env.battle import PokemonType
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle.game import mega_ability, mega_form_for_stone, species_entry
from pcsa.battle.stats import current_stats, form_of, known_ability

WEIGHT_POWER = ((10, 20), (25, 40), (50, 60), (100, 80), (200, 100))
NO_GUARD = "noguard"
IMMUNITIES = {
    "GROUND": {"levitate"},
    "ELECTRIC": {"voltabsorb", "lightningrod", "motordrive"},
    "WATER": {"waterabsorb", "stormdrain", "dryskin"},
}


@dataclass(frozen=True)
class Scoring:
    """How `attack_score` reads the attacker: STAB, exact own stats, and the form it fights as (None: the
    current one; a Mega form key when it Mega Evolves before the move)."""

    stab: float = 1.5
    exact: bool = True
    form: str | None = None


def mega_form_of(mon) -> str | None:
    """The Mega form the held stone turns `mon` into this turn, or None (no stone, no such form, already a Mega)."""
    if form_of(mon)[1]:
        return None
    item = to_id_str(mon.item or "")
    name = mega_form_for_stone(mon.species, item) if item and item != GenData.UNKNOWN_ITEM else None
    return to_id_str(name) if name else None


def attacker_profile(attacker, scoring: Scoring) -> tuple[dict[str, float], tuple, str | None]:
    """(stats, types, ability override) of the attacker in the form it fights as."""
    if scoring.form is None:
        return current_stats(attacker, scoring.exact), tuple(attacker.types), None
    entry = species_entry(scoring.form)
    types = tuple(PokemonType.from_name(t) for t in entry.get("types", []))
    return current_stats(attacker, scoring.exact, scoring.form), types, mega_ability(entry)


def stage_multiplier(mon, stat: str) -> float:
    value = mon.boosts.get(stat, 0)
    return (2 + value) / 2 if value >= 0 else 2 / (2 - value)


def effective_accuracy(battle, attacker, defender, move, attacker_ability: str | None = None) -> float:
    ours = known_ability(battle, attacker) if attacker_ability is None else attacker_ability
    if NO_GUARD in (ours, known_ability(battle, defender)):
        return 1.0
    weather = {getattr(v, "name", str(v)).lower() for v in getattr(battle, "weather", {})}
    if move.id in ("thunder", "hurricane") and weather & {"raindance", "primordialsea"}:
        return 1.0
    if move.id == "blizzard" and weather & {"snow", "hail"}:
        return 1.0
    accuracy = move.accuracy
    if move.id in ("thunder", "hurricane") and weather & {"sunnyday", "desolateland"}:
        accuracy = 0.5
    stage = max(-6, min(6, attacker.boosts.get("accuracy", 0) - defender.boosts.get("evasion", 0)))
    return min(1.0, accuracy * ((3 + stage) / 3 if stage >= 0 else 3 / (3 - stage)))


def effective_power(attacker, defender, move, attacker_stats=None, defender_stats=None, form=None) -> float:
    """Common variable-power moves resolved instead of counting as zero power (`form`: the attacker's form)."""
    if move.id in ("grassknot", "lowkick"):
        weight = species_entry(form_of(defender)[0]).get("weightkg", 0)
        return next((power for limit, power in WEIGHT_POWER if weight < limit), 120)
    if move.id in ("heavyslam", "heatcrash"):
        ratio = species_entry(form or form_of(attacker)[0]).get("weightkg", 1) / max(
            0.1, species_entry(form_of(defender)[0]).get("weightkg", 1)
        )
        return 120 if ratio >= 5 else 100 if ratio >= 4 else 80 if ratio >= 3 else 60 if ratio >= 2 else 40
    if move.id == "facade" and attacker.status:
        return move.base_power * 2
    if move.id in ("electroball", "gyroball"):
        ours = (attacker_stats or current_stats(attacker))["spe"]
        theirs = max(1, (defender_stats or current_stats(defender, False))["spe"])
        if move.id == "gyroball":
            return min(150, 25 * theirs / max(1, ours) + 1)
        ratio = ours / theirs
        return 150 if ratio >= 4 else 120 if ratio >= 3 else 80 if ratio >= 2 else 60 if ratio >= 1 else 40
    return move.base_power


def attack_score(battle, attacker, defender, move, scoring: Scoring | None = None) -> float:
    """A visible-state damage estimate of `move`; the attacker fights as `scoring.form` when one is given."""
    scoring = scoring or Scoring()
    ours, types, ability = attacker_profile(attacker, scoring)
    defending_own = any(member is defender for member in battle.team.values())
    theirs = current_stats(defender, scoring.exact and defending_own)
    physical = move.category.name == "PHYSICAL"
    attack_stat, defense_stat = ("atk", "def") if physical else ("spa", "spd")
    if move.id == "bodypress":
        attack_stat = "def"
    source = defender if move.id == "foulplay" else attacker
    attack = theirs["atk"] if move.id == "foulplay" else ours[attack_stat]
    ratio = (
        attack
        * stage_multiplier(source, attack_stat)
        / max(1, theirs[defense_stat] * stage_multiplier(defender, defense_stat))
    )
    multiplier = defender.damage_multiplier(move)
    ability = known_ability(battle, defender)
    if ability in IMMUNITIES.get(move.type.name, set()) or (
        move.type.name == "GROUND" and defender.item == "airballoon"
    ):
        multiplier = 0
    power = effective_power(attacker, defender, move, ours, theirs, scoring.form)
    return (
        power
        * (scoring.stab if move.type in types else 1)
        * ratio
        * effective_accuracy(battle, attacker, defender, move, ability)
        * move.expected_hits
        * multiplier
    )


def entry_hazard_fraction(battle, mon) -> float:
    conditions = {getattr(k, "name", str(k)).lower(): v for k, v in battle.side_conditions.items()}
    if to_id_str(mon.item or "") == "heavydutyboots" or known_ability(battle, mon) == "magicguard":
        return 0.0
    damage = 0.125 * mon.damage_multiplier(PokemonType.ROCK) if "stealth_rock" in conditions else 0.0
    airborne = PokemonType.FLYING in mon.types or known_ability(battle, mon) == "levitate" or mon.item == "airballoon"
    if not airborne and "spikes" in conditions:
        layers = min(3, max(1, conditions["spikes"]))
        damage += (1 / 8, 1 / 6, 1 / 4)[layers - 1]
    return damage

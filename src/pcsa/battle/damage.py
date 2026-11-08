"""Damage estimates: the Showdown damage formula at level 50 under Champions stat rules.

`damage` returns (low, high) as fractions of the defender's max HP: the lowest and highest random roll.
When the defender's spread is unknown, low is the minimum roll against a max-investment defender and high
the maximum roll against a zero-investment one. Weather, terrain, screens, burn, stat boosts, STAB, abilities
and items with simple multipliers are applied (the tables below: move-flag and type abilities such as Tough
Claws or Transistor, Technician, the -ate abilities, Filter-family and other defender abilities), as are Foul
Play and Body Press stats; anything else (HP-dependent abilities, Multiscale) is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from poke_env.battle import PokemonType
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.moves import base_data
from pcsa.battle.stats import NATURE_DOWN, NATURE_NEUTRAL, NATURE_UP, POINTS_MAX, hp_value, stage, stat_value

__all__ = ["CALM", "Conditions", "Fighter", "build_stats", "damage", "effectiveness_of", "nature_percent", "stage"]

ROLL_MIN = 85
ROLL_MAX = 100
STAB = 1.5
ADAPTABILITY_STAB = 2.0
SCREEN = 0.5
BURN = 0.5
TERRAIN_BOOST = 1.3
MISTY_DRAGON = 0.5
TYPE_ITEM_BOOST = 1.2
EXPERT_BELT = 1.2
NEUTRAL = 1.0
PERCENT = 100

ATTACK_ABILITY = {"hugepower": 2.0, "purepower": 2.0}
# Attacker ability -> (move flag, base-power multiplier); "recoil" and "secondary" are move-entry keys.
FLAG_ABILITY = {
    "toughclaws": ("contact", 1.3),
    "ironfist": ("punch", 1.2),
    "strongjaw": ("bite", 1.5),
    "sharpness": ("slicing", 1.5),
    "megalauncher": ("pulse", 1.5),
    "punkrock": ("sound", 1.3),
    "reckless": ("recoil", 1.2),
    "sheerforce": ("secondary", 1.3),
}
TYPE_ABILITY = {
    "steelworker": ("steel", 1.5),
    "transistor": ("electric", 1.3),
    "dragonsmaw": ("dragon", 1.5),
    "rockypayload": ("rock", 1.5),
    "waterbubble": ("water", 2.0),
}
ATE_ABILITY = {"aerilate": "flying", "pixilate": "fairy", "refrigerate": "ice", "galvanize": "electric"}
ATE_BOOST = 1.2
TECHNICIAN, TECHNICIAN_LIMIT = 1.5, 60
SAND_FORCE, SAND_FORCE_TYPES = 1.3, ("rock", "ground", "steel")
STAT_BOOST = 1.5  # Guts / Hustle (Attack), Solar Power (Sp. Atk in sun)
SOLID = ("filter", "solidrock", "prismarmor")
SOLID_MULT = 0.75
FUR_COAT = 2.0
ATTACK_ITEM = {"choiceband": ("atk", 1.5), "choicespecs": ("spa", 1.5)}
FINAL_ITEM = {"lifeorb": 1.3}
# Defender ability -> {move type: multiplier}; 0 is an immunity.
DEFENDER_ABILITY = {
    "thickfat": {"fire": 0.5, "ice": 0.5},
    "heatproof": {"fire": 0.5},
    "waterbubble": {"fire": 0.5},
    "purifyingsalt": {"ghost": 0.5},
    "levitate": {"ground": 0.0},
    "eartheater": {"ground": 0.0},
    "flashfire": {"fire": 0.0},
    "wellbakedbody": {"fire": 0.0},
    "waterabsorb": {"water": 0.0},
    "stormdrain": {"water": 0.0},
    "dryskin": {"water": 0.0},
    "voltabsorb": {"electric": 0.0},
    "lightningrod": {"electric": 0.0},
    "motordrive": {"electric": 0.0},
    "sapsipper": {"grass": 0.0},
    "fluffy": {"fire": 2.0},
}
TYPE_ITEMS = {
    "silkscarf": "normal",
    "charcoal": "fire",
    "mysticwater": "water",
    "magnet": "electric",
    "miracleseed": "grass",
    "nevermeltice": "ice",
    "blackbelt": "fighting",
    "poisonbarb": "poison",
    "softsand": "ground",
    "sharpbeak": "flying",
    "twistedspoon": "psychic",
    "silverpowder": "bug",
    "hardstone": "rock",
    "spelltag": "ghost",
    "dragonfang": "dragon",
    "blackglasses": "dark",
    "metalcoat": "steel",
    "fairyfeather": "fairy",
}
WEATHER = {
    ("sunnyday", "fire"): 1.5,
    ("sunnyday", "water"): 0.5,
    ("desolateland", "fire"): 1.5,
    ("raindance", "water"): 1.5,
    ("raindance", "fire"): 0.5,
    ("primordialsea", "water"): 1.5,
}
TERRAIN = {"electricterrain": "electric", "grassyterrain": "grass", "psychicterrain": "psychic"}
SCREENS = {"physical": ("reflect", "auroraveil"), "special": ("lightscreen", "auroraveil")}


@dataclass(frozen=True)
class Fighter:
    """One side of a damage calculation: current-form stats (no boosts applied) and what modifies them."""

    stats: dict[str, int]
    types: tuple[str, ...]
    ability: str = ""
    item: str = ""
    boosts: dict[str, int] = field(default_factory=dict)
    status: str = ""
    base: dict[str, int] | None = None  # base stats; with spread_known False they bound the defender
    spread_known: bool = True

    @property
    def grounded(self) -> bool:
        return "flying" not in self.types and self.ability != "levitate" and self.item != "airballoon"


@dataclass(frozen=True)
class Conditions:
    weather: str = ""
    terrain: str = ""
    defender_screens: frozenset[str] = frozenset()


CALM = Conditions()


def build_stats(base: dict[str, int], points: dict[str, int], nature: dict[str, int]) -> dict[str, int]:
    """Level-50 stats from base stats, stat points and nature percentages (90/100/110)."""
    out = {"hp": hp_value(base["hp"], points.get("hp", 0))}
    for s in game.STATS[1:]:
        out[s] = stat_value(base[s], points.get(s, 0), nature.get(s, NATURE_NEUTRAL))
    return out


def investment_bounds(base: dict[str, int]) -> tuple[dict[str, int], dict[str, int]]:
    """(zero-investment hindering-nature stats, max-investment boosting-nature stats)."""
    low = {"hp": hp_value(base["hp"], 0), **{s: stat_value(base[s], 0, NATURE_DOWN) for s in game.STATS[1:]}}
    high = {
        "hp": hp_value(base["hp"], POINTS_MAX),
        **{s: stat_value(base[s], POINTS_MAX, NATURE_UP) for s in game.STATS[1:]},
    }
    return low, high


def nature_percent(nature: str) -> dict[str, int]:
    """Nature multipliers as percentages per stat (110 raised, 90 lowered); an unknown nature is neutral."""
    entry = GenData.from_gen(9).natures.get(to_id_str(nature)) or {}
    return {s: round(float(entry.get(s, 1)) * PERCENT) for s in game.STATS[1:]}


def damage(
    attacker: Fighter, defender: Fighter, move_id: str, where: Conditions = CALM, power: int | None = None
) -> tuple[float, float]:
    """(low, high) damage of `move_id` as fractions of the defender's max HP; (0, 0) for status moves.
    `power` overrides the table's base power (a variable-power move the caller resolved)."""
    if not defender.spread_known and defender.base is not None:
        weakest, strongest = investment_bounds(defender.base)
        low = _rolls(attacker, replace(defender, stats=strongest), move_id, where, power)[0]
        high = _rolls(attacker, replace(defender, stats=weakest), move_id, where, power)[1]
        return low, high
    return _rolls(attacker, defender, move_id, where, power)


def _rolls(
    attacker: Fighter, defender: Fighter, move_id: str, where: Conditions, power: int | None
) -> tuple[float, float]:
    move_id = to_id_str(move_id)
    data, entry = base_data(move_id), GenData.from_gen(9).moves.get(move_id) or {}
    category, move_type = data.get("category"), data.get("type", "")
    ate = ATE_ABILITY.get(attacker.ability) if move_type == "normal" else None
    move_type = ate or move_type
    power = data.get("power") if power is None else power
    if category not in SCREENS or not power:
        return 0.0, 0.0
    effectiveness = effectiveness_of(move_type, defender)
    if effectiveness == 0:
        return 0.0, 0.0
    power = max(1, int(power * _power_modifier(attacker, entry, move_type, power, where) * (ATE_BOOST if ate else 1)))
    attack, defense = _attack_defense(attacker, defender, category, move_id, where)
    base = ((2 * game.LEVEL // 5 + 2) * power * attack // max(1, defense)) // 50 + 2
    modifier = effectiveness * _modifiers(attacker, defender, move_type, category, effectiveness, where)
    modifier *= _defender_modifier(defender, entry, category, effectiveness)
    hp = max(1, defender.stats["hp"])
    low = (base * ROLL_MIN // ROLL_MAX) * modifier
    return low / hp, base * modifier / hp


def effectiveness_of(move_type: str, defender: Fighter) -> float:
    """Type chart multiplier with the defender's ability immunities and halvings."""
    types = []
    for name in defender.types:
        try:
            types.append(PokemonType.from_name(name))
        except Exception:  # a type the chart does not know counts as neutral
            continue
    chart = _type_multiplier(move_type, types)
    multiplier = NEUTRAL if chart is None else chart
    multiplier *= DEFENDER_ABILITY.get(defender.ability, {}).get(move_type, NEUTRAL)
    if move_type == "ground" and not defender.grounded:
        multiplier = 0.0
    return multiplier


def _type_multiplier(move_type: str, types: list[PokemonType]) -> float | None:
    if not move_type or not types:
        return None
    try:
        attack = PokemonType.from_name(move_type)
    except Exception:  # a move type the chart does not know
        return None
    chart = GenData.from_gen(9).type_chart
    return attack.damage_multiplier(types[0], types[1] if len(types) > 1 else None, type_chart=chart)


def _power_modifier(attacker: Fighter, entry: dict, move_type: str, power: int, where: Conditions) -> float:
    """Base-power multiplier from the attacker's ability (move flags, move type, Technician, Sand Force)."""
    ability = attacker.ability
    flag, mult = FLAG_ABILITY.get(ability, ("", NEUTRAL))
    if flag and (entry.get("flags", {}).get(flag) or (flag in ("recoil", "secondary") and entry.get(flag))):
        return mult
    kind, mult = TYPE_ABILITY.get(ability, ("", NEUTRAL))
    if kind == move_type:
        return mult
    if ability == "technician" and power <= TECHNICIAN_LIMIT:
        return TECHNICIAN
    if ability == "sandforce" and where.weather == "sandstorm" and move_type in SAND_FORCE_TYPES:
        return SAND_FORCE
    return NEUTRAL


def _attack_defense(
    attacker: Fighter, defender: Fighter, category: str, move_id: str, where: Conditions
) -> tuple[int, int]:
    attack_stat, defense_stat = ("atk", "def") if category == "physical" else ("spa", "spd")
    if move_id == "foulplay":  # the target's Attack and stages
        attack = stage(defender.stats["atk"], defender.boosts.get("atk", 0))
    else:
        attack_stat = "def" if move_id == "bodypress" else attack_stat
        attack = int(
            stage(attacker.stats[attack_stat], attacker.boosts.get(attack_stat, 0))
            * _stat_mult(attacker, attack_stat, where)
        )
    defense = stage(defender.stats[defense_stat], defender.boosts.get(defense_stat, 0))
    if defender.ability == "furcoat" and defense_stat == "def":
        defense = int(defense * FUR_COAT)
    return attack, defense


def _stat_mult(attacker: Fighter, stat: str, where: Conditions) -> float:
    """The attacker's ability and item multipliers on its attacking stat."""
    item_stat, mult = ATTACK_ITEM.get(attacker.item, ("", NEUTRAL))
    mult = mult if item_stat == stat else NEUTRAL
    if stat == "atk":
        mult *= ATTACK_ABILITY.get(attacker.ability, NEUTRAL)
        if attacker.ability == "hustle" or (attacker.ability == "guts" and attacker.status):
            mult *= STAT_BOOST
    elif stat == "spa" and attacker.ability == "solarpower" and where.weather in ("sunnyday", "desolateland"):
        mult *= STAT_BOOST
    return mult


def _defender_modifier(defender: Fighter, entry: dict, category: str, effectiveness: float) -> float:
    """Damage multiplier from the defender's ability (Filter family, Ice Scales, Fluffy, Punk Rock)."""
    flags = entry.get("flags", {})
    mult = SOLID_MULT if defender.ability in SOLID and effectiveness > NEUTRAL else NEUTRAL
    if (defender.ability == "icescales" and category == "special") or (
        defender.ability == "fluffy" and flags.get("contact")
    ):
        mult *= SCREEN
    if defender.ability == "punkrock" and flags.get("sound"):
        mult *= SCREEN
    return mult


def _modifiers(
    attacker: Fighter, defender: Fighter, move_type: str, category: str, effectiveness: float, where: Conditions
) -> float:
    mod = WEATHER.get((where.weather, move_type), NEUTRAL)
    if TERRAIN.get(where.terrain) == move_type and attacker.grounded:
        mod *= TERRAIN_BOOST
    if where.terrain == "mistyterrain" and move_type == "dragon" and defender.grounded:
        mod *= MISTY_DRAGON
    if move_type in attacker.types:
        mod *= ADAPTABILITY_STAB if attacker.ability == "adaptability" else STAB
    if attacker.status == "brn" and category == "physical" and attacker.ability != "guts":
        mod *= BURN
    if any(screen in where.defender_screens for screen in SCREENS[category]):
        mod *= SCREEN
    mod *= FINAL_ITEM.get(attacker.item, NEUTRAL)
    if attacker.item == "expertbelt" and effectiveness > NEUTRAL:
        mod *= EXPERT_BELT
    if TYPE_ITEMS.get(attacker.item) == move_type:
        mod *= TYPE_ITEM_BOOST
    return mod

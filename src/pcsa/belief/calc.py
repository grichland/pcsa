# From research/pcsa-belief/belief/calc.py (sha256 11ad07543688, 2026-11-12 12:06); keep diffable.
# pcsa changes are marked "pcsa:".
"""pcsa's damage and speed formulas, vectorised over hypothesis rows, plus local fixes.

`rolls` reproduces pcsa.battle.damage._rolls exactly (tests/test_calc.py compares them on random cases) when
`fixes` is off: the scalar parts (power modifier, effectiveness, attacker stat multiplier, final modifiers) come
from pcsa's own helpers on a representative Fighter of each (item, ability) group, and only the stat arithmetic is
done in numpy. pcsa is imported read-only; nothing in it is changed.

Local fixes (on by default in the tracker; pcsa's calculator lacks them, stage 0 "Looks wrong" 5):
  - Rock types' Sp. Def x1.5 in sand, Ice types' Defence x1.5 in snow
  - Multiscale / Shadow Shield halve damage at full HP
  - Blaze / Torrent / Overgrow / Swarm: attack x1.5 at 1/3 HP or less with a move of that type
  - Unaware: the other side's stat stages are ignored
  - resist berries halve a super-effective hit of their type (Chilan: any Normal hit)
  - Muscle Band: physical power x1.1; Freeze-Dry is super effective on Water
  - Protean / Libero: STAB on every move
  - Iron Ball halves speed (speed only)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import damage as D
from pcsa.battle import game
from pcsa.battle.damage import Conditions, Fighter
from pcsa.battle.moves import base_data
from pcsa.battle.stats import multipliers
from pcsa.players import opponent as opp

PINCH = {"torrent": "water", "blaze": "fire", "overgrow": "grass", "swarm": "bug"}
FULL_HP_HALVE = ("multiscale", "shadowshield")
RESIST_BERRY = {
    "occaberry": "fire",
    "passhoberry": "water",
    "wacanberry": "electric",
    "rindoberry": "grass",
    "yacheberry": "ice",
    "chopleberry": "fighting",
    "kebiaberry": "poison",
    "shucaberry": "ground",
    "cobaberry": "flying",
    "payapaberry": "psychic",
    "tangaberry": "bug",
    "chartiberry": "rock",
    "kasibberry": "ghost",
    "habanberry": "dragon",
    "colburberry": "dark",
    "babiriberry": "steel",
    "roseliberry": "fairy",
    "chilanberry": "normal",
}
# Abilities whose effect on damage the calculator does not model (the row's likelihood is widened instead)
UNMODELED_ATTACK = frozenset(
    "sniper analytic tintedlens supremeoverlord electromorphosis protosynthesis quarkdrive flashfire toxicboost "
    "flareboost defeatist stakeout neuroforce rivalry parentalbond fairyaura darkaura aurabreak steelyspirit "
    "gorillatactics merciless normalize liquidvoice beadsofruin swordofruin orichalcumpulse hadronengine "
    "sharpness slowstart".split()
)
UNMODELED_DEFEND = frozenset("marvelscale dryskin tabletsofruin vesselofruin wonderguard iceface grasspelt".split())


@dataclass(frozen=True)
class Fixes:
    weather_stats: bool = True
    full_hp_halve: bool = True
    pinch: bool = True
    unaware: bool = True
    resist_berry: bool = True
    muscle_band: bool = True
    freeze_dry: bool = True
    iron_ball: bool = True
    protean: bool = True


NO_FIXES = Fixes(*(False,) * 9)
ALL_FIXES = Fixes()


def stage_arr(v: np.ndarray, boost: int) -> np.ndarray:
    """pcsa.battle.stats.stage on an integer array."""
    boost = int(boost)
    if boost >= 0:
        return v * (2 + boost) // 2
    return v * 2 // (2 - boost)


@dataclass
class Side:
    """One side of a vectorised calculation: a representative Fighter (types, item, ability, boosts, status) and,
    optionally, per-row stat columns overriding its stats (dict stat -> int array)."""

    fighter: Fighter
    stats: dict | None = None  # stat -> np.ndarray, or None: the fighter's own stats for every row
    hp_full: bool = False  # at full HP (Multiscale)
    hp_third: bool = False  # at 1/3 HP or less (pinch abilities)

    def stat(self, name: str, n: int) -> np.ndarray:
        if self.stats is not None and name in self.stats:
            return np.asarray(self.stats[name], dtype=np.int64)
        return np.full(n, int(self.fighter.stats[name]), dtype=np.int64)


def rolls(
    att: Side, dfn: Side, move_id: str, where: Conditions, power: int | None, n: int, fixes: Fixes = NO_FIXES
) -> tuple[np.ndarray, np.ndarray]:
    """(low, high) arrays of length n as fractions of the defender's max HP; zeros when the move does no damage.
    Mirrors pcsa.battle.damage._rolls line by line."""
    attacker, defender = att.fighter, dfn.fighter
    zeros = (np.zeros(n), np.zeros(n))
    move_id = to_id_str(move_id)
    data, entry = base_data(move_id), GenData.from_gen(9).moves.get(move_id) or {}
    category, move_type = data.get("category"), data.get("type", "")
    ate = D.ATE_ABILITY.get(attacker.ability) if move_type == "normal" else None
    move_type = ate or move_type
    power = data.get("power") if power is None else power
    if category not in D.SCREENS or not power:
        return zeros
    effectiveness = D.effectiveness_of(move_type, defender)
    if fixes.freeze_dry and move_id == "freezedry" and "water" in defender.types and effectiveness > 0:
        effectiveness *= 4.0
    if effectiveness == 0:
        return zeros
    if fixes.protean and attacker.ability in ("protean", "libero") and move_type:
        attacker = replace(attacker, types=(move_type,))  # the user becomes the move's type: STAB on every move
    power = max(
        1, int(power * D._power_modifier(attacker, entry, move_type, power, where) * (D.ATE_BOOST if ate else 1))
    )
    if fixes.muscle_band and attacker.item == "muscleband" and category == "physical":
        power = max(1, int(power * 1.1))
    attack_stat, defense_stat = ("atk", "def") if category == "physical" else ("spa", "spd")
    att_boosts = {} if (fixes.unaware and defender.ability == "unaware") else attacker.boosts
    def_boosts = {} if (fixes.unaware and attacker.ability == "unaware") else defender.boosts
    if move_id == "foulplay":
        attack = stage_arr(dfn.stat("atk", n), def_boosts.get("atk", 0))
    else:
        attack_stat = "def" if move_id == "bodypress" else attack_stat
        mult = D._stat_mult(attacker, attack_stat, where)
        attack = (stage_arr(att.stat(attack_stat, n), att_boosts.get(attack_stat, 0)) * mult).astype(np.int64)
        if fixes.pinch and att.hp_third and PINCH.get(attacker.ability) == move_type:
            attack = (attack * 1.5).astype(np.int64)
    defense = stage_arr(dfn.stat(defense_stat, n), def_boosts.get(defense_stat, 0))
    if defender.ability == "furcoat" and defense_stat == "def":
        defense = (defense * D.FUR_COAT).astype(np.int64)
    if fixes.weather_stats:
        if where.weather == "sandstorm" and "rock" in defender.types and defense_stat == "spd":
            defense = (defense * 1.5).astype(np.int64)
        if where.weather in ("snowscape", "snow", "hail") and "ice" in defender.types and defense_stat == "def":
            defense = (defense * 1.5).astype(np.int64)
    base = ((2 * game.LEVEL // 5 + 2) * power * attack // np.maximum(1, defense)) // 50 + 2
    modifier = effectiveness * D._modifiers(attacker, defender, move_type, category, effectiveness, where)
    modifier *= D._defender_modifier(defender, entry, category, effectiveness)
    if fixes.full_hp_halve and dfn.hp_full and defender.ability in FULL_HP_HALVE:
        modifier *= 0.5
    if fixes.resist_berry:
        berry = RESIST_BERRY.get(defender.item)
        if berry == move_type and (effectiveness > 1 or berry == "normal"):
            modifier *= 0.5
    hp = np.maximum(1, dfn.stat("hp", n))
    low = (base * D.ROLL_MIN // D.ROLL_MAX) * modifier
    return low / hp, base * modifier / hp


def scalar_damage(
    attacker: Fighter,
    defender: Fighter,
    move_id: str,
    where: Conditions = D.CALM,
    power: int | None = None,
    fixes: Fixes = NO_FIXES,
    att_hp_third: bool = False,
    dfn_hp_full: bool = False,
) -> tuple[float, float]:
    lo, hi = rolls(
        Side(attacker, hp_third=att_hp_third), Side(defender, hp_full=dfn_hp_full), move_id, where, power, 1, fixes
    )
    return float(lo[0]), float(hi[0])


def speed_rows(
    battle, mon, spe: np.ndarray, item: str, ability: str, boosts: dict, status: str, fixes: Fixes = NO_FIXES
) -> np.ndarray:
    """pcsa.players.opponent.speed for one (item, ability) group over a speed-stat array."""
    value = spe * multipliers(item, ability, mon.species).get("spe", 1.0)
    if fixes.iron_ball and item == "ironball":
        value = value * 0.5
    if opp._weather(battle) in opp.SPEED_WEATHER.get(ability, ()):
        value = value * opp.WEATHER_SPEED
    value = stage_arr(np.floor(value).astype(np.int64), int(boosts.get("spe", 0))).astype(np.float64)
    if ability == "quickfeet" and status:
        value = value * opp.QUICK_FEET
    elif status == "par":
        value = value * opp.PARALYSIS
    side = battle.side_conditions if opp.is_ours(battle, mon) else battle.opponent_side_conditions
    from poke_env.battle import SideCondition

    return value * (opp.TAILWIND if SideCondition.TAILWIND in side else 1.0)


def with_item_ability(f: Fighter, item: str, ability: str) -> Fighter:
    return replace(f, item=item, ability=ability)

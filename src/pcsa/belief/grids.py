# From research/pcsa-belief/belief/fast.py (sha256 dcd849036429, 2026-11-12 14:38), split into keys, grids, board,
# member_state, member_grids, member_out, member, fast (pcsa's 400-line file limit); pcsa changes are marked "pcsa:".
"""The vectorised calculator parts and the Grid of one calculation (overview: pcsa.belief.fast)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import damage as D
from pcsa.battle import game
from pcsa.battle.damage import Fighter
from pcsa.battle.moves import base_data
from pcsa.battle.stats import current_stats, form_of
from pcsa.belief import calc as K
from pcsa.belief.keys import (
    LEVEL_FACTOR,
    SNOW,
)
from pcsa.belief.space import STATS
from pcsa.players import opponent as opp

# ------------------------------------------------------------------------------------------------ calculations
_PARAMS: dict = {}


def _params(att: Fighter, dfn: Fighter, move_id: str, where, power, fixes: K.Fixes, att_third: bool, dfn_full: bool):
    """_params_now, memoised by value: it reads no stat (power comes resolved), only the move, both sides' types,
    abilities, items, stages and status, the conditions and the HP thresholds."""
    key = (
        move_id,
        power,
        att.ability,
        att.item,
        att.types,
        tuple(sorted(att.boosts.items())),
        att.status,
        dfn.ability,
        dfn.item,
        dfn.types,
        tuple(sorted(dfn.boosts.items())),
        dfn.status,
        where,
        fixes,
        att_third,
        dfn_full,
    )
    hit = _PARAMS.get(key, _PARAMS)
    if hit is _PARAMS:
        if len(_PARAMS) > 200_000:
            _PARAMS.clear()
        hit = _PARAMS[key] = _params_now(att, dfn, move_id, where, power, fixes, att_third, dfn_full)
    return hit


def _params_now(
    att: Fighter, dfn: Fighter, move_id: str, where, power, fixes: K.Fixes, att_third: bool, dfn_full: bool
):
    """calc.rolls's scalar parts for one pair of representative fighters; None when the move does no damage."""
    data, entry = base_data(move_id), GenData.from_gen(9).moves.get(move_id) or {}
    category, move_type = data.get("category"), data.get("type", "")
    ate = D.ATE_ABILITY.get(att.ability) if move_type == "normal" else None
    move_type = ate or move_type
    power = data.get("power") if power is None else power
    if category not in D.SCREENS or not power:
        return None
    eff = D.effectiveness_of(move_type, dfn)
    if fixes.freeze_dry and move_id == "freezedry" and "water" in dfn.types and eff > 0:
        eff *= 4.0
    if eff == 0:
        return None
    if fixes.protean and att.ability in ("protean", "libero") and move_type:
        att = replace(att, types=(move_type,))
    power = max(1, int(power * D._power_modifier(att, entry, move_type, power, where) * (D.ATE_BOOST if ate else 1)))
    if fixes.muscle_band and att.item == "muscleband" and category == "physical":
        power = max(1, int(power * 1.1))
    attack_stat, defense_stat = ("atk", "def") if category == "physical" else ("spa", "spd")
    att_boosts = {} if (fixes.unaware and dfn.ability == "unaware") else att.boosts
    def_boosts = {} if (fixes.unaware and att.ability == "unaware") else dfn.boosts
    if move_id == "foulplay":
        a_side, a_stat, a_boost, mult, pinch = "dfn", "atk", int(def_boosts.get("atk", 0)), None, False
    else:
        a_stat = "def" if move_id == "bodypress" else attack_stat
        a_side, a_boost = "att", int(att_boosts.get(a_stat, 0))
        mult = D._stat_mult(att, a_stat, where)
        pinch = bool(fixes.pinch and att_third and K.PINCH.get(att.ability) == move_type)
    d_boost = int(def_boosts.get(defense_stat, 0))
    fur = dfn.ability == "furcoat" and defense_stat == "def"
    wthr = bool(
        fixes.weather_stats
        and (
            (where.weather == "sandstorm" and "rock" in dfn.types and defense_stat == "spd")
            or (where.weather in SNOW and "ice" in dfn.types and defense_stat == "def")
        )
    )
    modifier = eff * D._modifiers(att, dfn, move_type, category, eff, where)
    modifier *= D._defender_modifier(dfn, entry, category, eff)
    if fixes.full_hp_halve and dfn_full and dfn.ability in K.FULL_HP_HALVE:
        modifier *= 0.5
    if fixes.resist_berry:
        berry = K.RESIST_BERRY.get(dfn.item)
        if berry == move_type and (eff > 1 or berry == "normal"):
            modifier *= 0.5
    return power, a_side, a_stat, a_boost, mult, pinch, defense_stat, d_boost, fur, wthr, modifier


def _stage(v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """calc.stage_arr with a per-row stage (b broadcast against v)."""
    if not b.any():
        return v
    num = np.where(b >= 0, 2 + b, 2)
    den = np.where(b >= 0, 2, 2 - b)
    return v * num // den


def _rolls_groups(params: list, their_attacking: bool, their_stats: np.ndarray, own: Fighter):
    """(low, high) [G, S] for G groups of params (none None) over the S spreads of `their_stats` [S, 6]."""
    col = {s: their_stats[:, j][None, :] for j, s in enumerate(STATS)}

    def stat(side: str, name: str):
        theirs = (side == "att") == their_attacking
        return col[name] if theirs else np.int64(own.stats[name])

    p = params
    a_side, a_stat, d_stat = p[0][1], p[0][2], p[0][6]
    ints = np.array([(x[0], x[3], x[7], x[5], x[8], x[9]) for x in p], np.int64)  # power, stages, flags
    power, a_boost, d_boost = ints[:, 0:1], ints[:, 1:2], ints[:, 2:3]
    pinch, fur, wthr = ints[:, 3:4] != 0, ints[:, 4:5] != 0, ints[:, 5:6] != 0
    attack = _stage(stat(a_side, a_stat), a_boost)  # pyright: ignore[reportArgumentType]
    if a_side == "att":
        mult = np.array([x[4] for x in p], np.float64)[:, None]
        attack = (attack * mult).astype(np.int64)
        if pinch.any():
            attack = np.where(pinch, (attack * 1.5).astype(np.int64), attack)
    defense = _stage(stat("dfn", d_stat), d_boost)  # pyright: ignore[reportArgumentType]
    if fur.any():
        defense = np.where(fur, (defense * D.FUR_COAT).astype(np.int64), defense)
    if wthr.any():
        defense = np.where(wthr, (defense * 1.5).astype(np.int64), defense)
    base = (LEVEL_FACTOR * power * attack // np.maximum(1, defense)) // 50 + 2
    modifier = np.array([x[10] for x in p], np.float64)[:, None]
    hp = np.maximum(1, stat("dfn", "hp"))
    low = (base * D.ROLL_MIN // D.ROLL_MAX) * modifier
    return low / hp, base * modifier / hp


def own_fighter(battle, mon, mega_form: str | None = None) -> Fighter:
    """One of our members (opp.own_fighter), with the Mega's own ability once it is a Mega (poke-env keeps the
    base ability on a p1 seat: stage 0, Looks wrong 1), or in `mega_form` (the coming turn's Mega Evolution)."""
    if mega_form is None:
        f = opp.own_fighter(battle, mon)
        form, mega = form_of(mon, battle)
        if mega:
            ab = game.mega_ability(game.species_entry(form))
            if ab and ab != f.ability:
                f = replace(f, ability=ab)
        return f
    entry = game.species_entry(mega_form)
    stats = {s: int(v) for s, v in current_stats(mon, True, mega_form).items()}
    types = tuple(t.lower() for t in entry.get("types", ())) or opp._types(mon)
    item = to_id_str(opp.revealed_item(mon)[1] or "")
    return Fighter(
        stats,
        types,
        to_id_str(game.mega_ability(entry) or mon.ability or ""),
        item,
        dict(mon.boosts),
        opp._status(mon),
        entry.get("baseStats") or opp._base(mon, mega_form),
    )


def our_mega_form(battle, mon) -> str | None:
    """The Mega form our member can become this turn (we have not used Mega Evolution, it holds its stone)."""
    if battle.used_mega_evolve or mon.fainted:
        return None
    from pcsa.players.tactics import mega_form_of

    return mega_form_of(mon)


# ------------------------------------------------------------------------------------------------ one member
@dataclass
class Grid:
    """One calculation over a member's classes: values [Ic, Ac, U] over the U distinct values of the stats it reads
    (`inv`: spread -> its value index), the class of every item / ability of the member's vocabulary, and per class
    pair whether the calculator models its ability (damage) or whether its ability can change the order (speed)."""

    lo: np.ndarray
    hi: np.ndarray
    imap: np.ndarray
    amap: np.ndarray
    unmod: np.ndarray | None = None
    zero: bool = False  # the move does no damage in any class
    inv: np.ndarray | None = None  # [S] -> [0, U); None: U = S
    inv_key: object = None


def _lik(Lo, Hi, d, ko, s, tau, rel_slack):
    """tracker.Member._lik."""
    s = s + rel_slack * Hi
    lo_e, hi_e = Lo - s, Hi + s
    if ko:
        width = np.maximum(hi_e - lo_e, 1e-9)
        p = np.clip((hi_e - d) / width, 0.0, 1.0)
        over = np.maximum(0.0, d - hi_e)
        return np.where(over > 0, np.exp(-((over / tau) ** 2)) * 0.5 / np.maximum(1.0, width / tau), p)
    dist = np.maximum(0.0, np.maximum(lo_e - d, d - hi_e))
    width = np.maximum(hi_e - lo_e, 0.0) + tau * math.sqrt(math.pi)
    return np.exp(-((dist / tau) ** 2)) / width

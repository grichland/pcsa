# From research/pcsa-rl/rl/features_moves.py (sha256 b6ea2aee07de, 2026-11-08 20:13); keep diffable.
"""Move blocks: 27 static facts, 4 state columns and 9 engine damage columns against the opposing active."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from poke_env.battle import Move, PokemonType
from poke_env.data import GenData

from pcsa.battle.damage import damage
from pcsa.battle.moves import base_data
from pcsa.players.opponent import conditions, move_power
from pcsa.rl_training.features_log import _is_setup

TYPES = tuple(t for t in PokemonType if t.name not in ("THREE_QUESTION_MARKS", "STELLAR"))
TYPE_INDEX = {t: i for i, t in enumerate(TYPES)}
CATEGORIES = ("PHYSICAL", "SPECIAL", "STATUS")
FACTS = 27
STATE = 4
DAMAGE = 9
MOVE_WIDTH = FACTS + STATE + DAMAGE
POWER_DIV = 150.0
PRIORITY_DIV = 5.0
DAMAGE_CLIP = 2.0
BOOSTED = 1.5  # a +1 attacking stage, applied to the damage directly
EFFECT_DIV = 4.0


@lru_cache(maxsize=2048)
def facts(move_id: str) -> np.ndarray:
    """type 18 | category 3 | power x hits /150 | accuracy | priority /5 | pivot | setup | heal or drain."""
    out = np.zeros(FACTS, dtype=np.float32)
    try:
        move = Move(move_id, gen=9)
        entry = move.entry
    except (KeyError, ValueError):
        return out
    if not entry:
        return out
    if move.type in TYPE_INDEX:
        out[TYPE_INDEX[move.type]] = 1.0
    out[18 + CATEGORIES.index(move.category.name)] = 1.0
    data = base_data(move_id)
    power = float(data.get("power") or 0) * float(move.expected_hits)
    out[21] = min(DAMAGE_CLIP, power / POWER_DIV)
    accuracy = data.get("accuracy", 100)
    out[22] = 1.0 if accuracy is True else min(1.0, float(accuracy) / 100.0)
    out[23] = max(-1.5, min(1.5, move.priority / PRIORITY_DIV))
    out[24] = float(bool(move.self_switch))
    out[25] = float(_is_setup(move))
    out[26] = float(move.heal > 0 or move.drain > 0)
    return out


NO_HIT = (0.0, 0.0, 0.0)


def hits(battle, attacker, defender, fa, fd, moves: list[Move | None]) -> list[tuple[float, float, float]]:
    """Per move: (effectiveness /4, expected fraction of the defender's max HP, max-roll fraction x expected hits
    without accuracy). Status moves and unknown moves give zeros."""
    cond = conditions(battle, defender)
    weather = frozenset(w.name.lower() for w in battle.weather)
    stage = max(-6, min(6, attacker.boosts.get("accuracy", 0) - defender.boosts.get("evasion", 0)))
    types = (defender.type_1, defender.type_2)
    out = []
    for move in moves:
        static = _static(move.id) if move is not None else None
        if static is None or move is None:
            out.append(NO_HIT)
            continue
        move_type, n_hits, base_accuracy = static
        try:
            power = move_power(attacker, defender, fa, fd, move)
            low, high = damage(fa, fd, move.id, cond, power)
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            out.append(NO_HIT)
            continue
        if not high or high <= 0:
            out.append(NO_HIT)
            continue
        accuracy = _accuracy(move.id, base_accuracy, weather, stage, fa.ability, fd.ability)
        mean, top = (low + high) / 2 * accuracy * n_hits, high * n_hits
        out.append((min(1.0, _effectiveness(move_type, types) / EFFECT_DIV), _finite(mean), _finite(top)))
    return out


@lru_cache(maxsize=4096)
def _static(move_id: str) -> tuple[PokemonType, float, float] | None:
    """(type, expected hits, accuracy) of a damaging move; None for status moves."""
    move = Move(move_id, gen=9)
    if move.category.name == "STATUS":
        return None
    return move.type, float(move.expected_hits), float(move.accuracy)


@lru_cache(maxsize=4096)
def _effectiveness(move_type: PokemonType, types: tuple) -> float:
    return float(move_type.damage_multiplier(*types, type_chart=GenData.from_gen(9).type_chart))


def _accuracy(move_id: str, accuracy: float, weather: frozenset, stage: int, ours: str, theirs: str) -> float:
    """pcsa.players.tactics.effective_accuracy with both abilities given (the defender's assumed one for theirs)."""
    if "noguard" in (ours, theirs):
        return 1.0
    if move_id in ("thunder", "hurricane") and weather & {"raindance", "primordialsea"}:
        return 1.0
    if move_id == "blizzard" and weather & {"snowscape", "hail"}:
        return 1.0
    if move_id in ("thunder", "hurricane") and weather & {"sunnyday", "desolateland"}:
        accuracy = 0.5
    return min(1.0, accuracy * ((3 + stage) / 3 if stage >= 0 else 3 / (3 - stage)))


def _finite(x: float) -> float:
    return x if 0 < x < math.inf else 0.0


def damage_columns(effect: float, mean: float, top: float, hp: float) -> list[float]:
    """effectiveness | mean | max | KO mean | KO max | boosted mean | boosted max | boosted KO mean | max."""
    if hp <= 0:
        hp = 1.0
    bm, bt = mean * BOOSTED, top * BOOSTED
    clip = lambda v: min(DAMAGE_CLIP, v)
    return [
        effect,
        clip(mean),
        clip(top),
        float(mean > 0 and mean >= hp),
        float(top > 0 and top >= hp),
        clip(bm),
        clip(bt),
        float(bm > 0 and bm >= hp),
        float(bt > 0 and bt >= hp),
    ]


def block(out: np.ndarray, move: Move | None, state: tuple[float, float, float, float], dmg: list[float]) -> None:
    """Write one 40-wide move block into `out` (a view)."""
    if move is None:
        return
    out[:FACTS] = facts(move.id)
    out[FACTS : FACTS + STATE] = state
    out[FACTS + STATE :] = dmg


def pp_fraction(move: Move) -> float:
    maximum = move.max_pp
    if not maximum:
        return 1.0
    return max(0.0, min(1.0, move.current_pp / maximum))

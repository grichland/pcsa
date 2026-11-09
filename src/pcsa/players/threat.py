"""HP-fraction threat estimates for rule-based players: how hard a member hits, KOs, speed order.

Every fraction is of the defender's MAX HP (and may exceed 1). Our members fight with their exact sets; the
opponent's with `opponent.assumed` (top build + revealed facts + the tracker's corrections). Damage is the
Showdown formula in `battle.damage`; `hit_fraction` and `best_hit` are expected values (mean roll, accuracy,
expected hits); `can_ko` and `priority_ko` ignore accuracy and use the max roll against us, the mean against
them.
"""

from __future__ import annotations

from poke_env.battle import Field, Move

from pcsa.battle.builds import Catalog
from pcsa.battle.damage import Fighter, damage
from pcsa.players.opponent import (
    ESCALATING_POWER,
    Tracker,
    assumed,
    conditions,
    is_ours,
    move_object,
    move_power,
    own_fighter,
    set_fighter,
    speed,
)
from pcsa.players.tactics import effective_accuracy

THREAT_SHARE = 0.4


def fighters(battle, mon, catalog: Catalog, tracker: Tracker | None = None) -> Fighter:
    """A damage-calculation side for `mon`: exact for ours, the assumed set for theirs."""
    if is_ours(battle, mon):
        return own_fighter(battle, mon)
    return set_fighter(battle, mon, assumed(battle, mon, catalog, tracker))


def known_moves(battle, mon, catalog: Catalog, tracker: Tracker | None = None) -> list[Move]:
    """Our member's own moves, or the opponent's assumed four."""
    if is_ours(battle, mon):
        return list(mon.moves.values())
    chosen = assumed(battle, mon, catalog, tracker)
    return [m for m in (move_object(i) for i in chosen.moves) if m is not None]


def _range(battle, attacker, defender, fa: Fighter, fd: Fighter, move: Move) -> tuple[float, float]:
    power = move_power(attacker, defender, fa, fd, move)
    return damage(fa, fd, move.id, conditions(battle, defender), power)


ESCALATING_HITS = 1 + 0.9 + 0.9**2  # Triple Axel / Kick: each later hit checks 90% accuracy again


def _hits(move: Move) -> float:
    """Expected hits. poke-env's expected_hits is power-weighted for the escalating moves, whose mean power per hit
    move_power already returns, so those count plain hits."""
    return ESCALATING_HITS if move.id in ESCALATING_POWER else move.expected_hits


def _expected(battle, attacker, defender, fa: Fighter, fd: Fighter, move: Move) -> float:
    low, high = _range(battle, attacker, defender, fa, fd, move)
    if high <= 0:
        return 0.0
    accuracy = effective_accuracy(battle, attacker, defender, move, fa.ability)
    return (low + high) / 2 * accuracy * _hits(move)


def _ko_share(battle, attacker, defender, fa: Fighter, fd: Fighter, move: Move) -> float:
    """Damage for a KO check: the max roll and most hits against us, the mean roll and expected hits on them."""
    low, high = _range(battle, attacker, defender, fa, fd, move)
    if is_ours(battle, defender):
        return high * move.n_hit[1]
    return (low + high) / 2 * _hits(move)


def _as_move(move) -> Move | None:
    return move if isinstance(move, Move) else move_object(str(move))


def hit_fraction(battle, attacker, defender, move, catalog: Catalog, tracker: Tracker | None = None) -> float:
    """Expected damage of `move` (a Move or move id) as a fraction of the defender's max HP; 0.0 when immune."""
    move = _as_move(move)
    if move is None:
        return 0.0
    fa, fd = fighters(battle, attacker, catalog, tracker), fighters(battle, defender, catalog, tracker)
    return _expected(battle, attacker, defender, fa, fd, move)


def best_hit(
    battle, attacker, defender, catalog: Catalog, tracker: Tracker | None = None
) -> tuple[float, str | None]:
    """(expected fraction of the defender's max HP, move id) of the attacker's strongest known move."""
    fa, fd = fighters(battle, attacker, catalog, tracker), fighters(battle, defender, catalog, tracker)
    best: tuple[float, str | None] = (0.0, None)
    for move in known_moves(battle, attacker, catalog, tracker):
        share = _expected(battle, attacker, defender, fa, fd, move)
        if share > best[0]:
            best = (share, move.id)
    return best


def _ko_moves(battle, attacker, defender, catalog: Catalog, tracker: Tracker | None) -> list[tuple[float, Move]]:
    """(KO-check damage, move) for each known move that KOs the defender at its current HP."""
    hp = defender.current_hp_fraction
    if hp <= 0:
        return []
    fa, fd = fighters(battle, attacker, catalog, tracker), fighters(battle, defender, catalog, tracker)
    shares = [
        (_ko_share(battle, attacker, defender, fa, fd, m), m) for m in known_moves(battle, attacker, catalog, tracker)
    ]
    return [(share, m) for share, m in shares if share > 0 and share >= hp]


def can_ko(battle, attacker, defender, catalog: Catalog, tracker: Tracker | None = None) -> bool:
    """Whether a known move of the attacker KOs the defender from its current HP."""
    return bool(_ko_moves(battle, attacker, defender, catalog, tracker))


def priority_ko(battle, attacker, defender, catalog: Catalog, tracker: Tracker | None = None) -> str | None:
    """The attacker's strongest positive-priority move that KOs the defender, else None."""
    kos = [(share, m) for share, m in _ko_moves(battle, attacker, defender, catalog, tracker) if m.priority > 0]
    return max(kos, key=lambda pair: pair[0])[1].id if kos else None


def outspeeds(battle, a, b, catalog: Catalog, tracker: Tracker | None = None) -> bool:
    """Whether `a` moves before `b` at equal priority (strictly; Trick Room reverses the order)."""
    speed_a = speed(battle, a, fighters(battle, a, catalog, tracker))
    speed_b = speed(battle, b, fighters(battle, b, catalog, tracker))
    return speed_a < speed_b if Field.TRICK_ROOM in battle.fields else speed_a > speed_b


def threatened(
    battle, mine, theirs, catalog: Catalog, tracker: Tracker | None = None, threshold: float = THREAT_SHARE
) -> bool:
    """Whether their best expected hit takes `threshold` of my max HP, or they can KO me."""
    return best_hit(battle, theirs, mine, catalog, tracker)[0] >= threshold or can_ko(
        battle, theirs, mine, catalog, tracker
    )

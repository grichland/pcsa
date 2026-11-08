"""Endgame chains: with at most two members left per side, the rest of the game is a run of 1v1 duels.

Pure arithmetic, no battle objects. Damage is matchup-specific, so attacks come from a `Matchups` matrix.

Simplifications (cheap and deterministic on purpose):
- Expected-value race: a hit deals `fraction * accuracy`. A hit whose max roll KOs (`ko_fraction >= hp`) and whose
  accuracy is at least 0.5 is taken to KO outright.
- Each side greedily picks its best attack every turn (a KO first, best priority then accuracy; else the most
  expected damage), or recovers by the rule in `_choose`. No status moves, setup, crits, secondary effects or
  mid-duel switches.
- Order is priority then speed. Speed ties go to `tie_goes_to` (default "b": pessimistic for side a, which is our side
  in `chain`). Recovery is priority 0.
- Every damaging hit counts as contact: the attacker loses the target's `passive_on_hit`, scaled by accuracy.
- `residual` applies to each living combatant at end of turn; HP stays within [0, 1].
- A duel that reaches `horizon` turns is a draw (winner None); in `chain` it ends the game as a draw.
- `chain` is minimax over who fights whom. The opponent answers after seeing our choice (pessimistic for us), except
  after one of their members faints, when they send the replacement first.
- A voluntary switch costs one turn: the side that stays uses the action it picked against the old target (same move
  name, damage against the newcomer). Replacements after a faint come in free.
- `can_switch_on_faint`: when this member KOs its opponent and survives, its side may also swap it for a reserve
  (free, after seeing the replacement). Set it False for formats without that option.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, NamedTuple

RECOVER = "recover"  # move name reported for the recovery action
_LIKELY = 0.5  # accuracy from which a KO-capable hit is taken to land
_RECOVER_BELOW = 0.5  # HP fraction under which recovery is considered


@dataclass(frozen=True)
class Attack:
    """One move against one specific target; fractions are of the target's max HP."""

    move: str
    fraction: float  # expected damage
    priority: int
    accuracy: float  # 0-1
    ko_fraction: float  # max-roll damage, for KO checks


@dataclass(frozen=True)
class Combatant:
    """One member: current HP fraction, speed, and attacks against the opponent it currently faces."""

    name: str
    hp: float
    speed: int
    attacks: tuple[Attack, ...] = ()
    recover: float = 0.0  # fraction healed by its recovery move, 0 if none
    residual: float = 0.0  # per-turn HP change (poison, burn, Leftovers...); negative is a loss
    passive_on_hit: float = 0.0  # fraction an attacker loses per hit on this member (Rough Skin, Rocky Helmet)
    can_switch_on_faint: bool = True


@dataclass(frozen=True)
class Matchups:
    """Attack matrix, both directions keyed (our_i, their_j); a missing pair means no damaging attacks."""

    ours: Mapping[tuple[int, int], tuple[Attack, ...]]  # our i's attacks against their j
    theirs: Mapping[tuple[int, int], tuple[Attack, ...]]  # their j's attacks against our i


@dataclass(frozen=True)
class DuelResult:
    winner: Literal["a", "b"] | None  # None: draw by horizon, or both fainted
    turns: int
    a_hp_left: float
    b_hp_left: float
    a_first_move: str  # "" when a had nothing to do


@dataclass(frozen=True)
class ChainResult:
    wins: bool  # against the opponent's best replies
    order: tuple[int, ...]  # our members in the order they enter; order[0] != active_ours means switch / send it
    first_move: str | None  # our first move (after a replacement: against their reply); None when we switch
    margin: float  # our HP left minus theirs at the end: positive HP left on a win, negative deficit on a loss


def facing(
    ours: Sequence[Combatant], theirs: Sequence[Combatant], attacks: Matchups, i: int, j: int
) -> tuple[Combatant, Combatant]:
    """Our i and their j with their attacks against each other filled in from the matrix."""
    return (
        replace(ours[i], attacks=attacks.ours.get((i, j), ())),
        replace(theirs[j], attacks=attacks.theirs.get((i, j), ())),
    )


def lands_ko(attack: Attack, target_hp: float) -> bool:
    """Whether the hit is taken to KO a target at `target_hp`."""
    return attack.fraction > 0 and attack.accuracy >= _LIKELY and attack.ko_fraction >= target_hp


def best_attack(attacks: Sequence[Attack], target_hp: float) -> Attack | None:
    """A KO if one exists (highest priority, then accuracy), else the most expected damage."""
    if not attacks:
        return None
    kos = [a for a in attacks if lands_ko(a, target_hp)]
    if kos:
        return max(kos, key=lambda a: (a.priority, a.accuracy, a.fraction))
    return max(attacks, key=lambda a: (a.fraction * a.accuracy, a.priority))


_RECOVER_ACTION = Attack(RECOVER, 0.0, 0, 1.0, 0.0)


@dataclass
class _Fighter:
    unit: Combatant
    attacks: tuple[Attack, ...]
    hp: float


def _fighter(unit: Combatant) -> _Fighter:
    return _Fighter(unit, unit.attacks, unit.hp)


def _choose(me: _Fighter, foe: _Fighter) -> Attack | None:
    """Attack, or recover when below half, the healed HP survives the foe's best hit and the heal outweighs it."""
    attack = best_attack(me.attacks, foe.hp)
    if attack is not None and lands_ko(attack, foe.hp):
        return attack
    if me.unit.recover <= 0 or me.hp >= _RECOVER_BELOW:
        return attack
    healed = min(1.0, me.hp + me.unit.recover)
    threat = best_attack(foe.attacks, healed)
    if threat is not None and (lands_ko(threat, healed) or threat.fraction * threat.accuracy >= healed - me.hp):
        return attack
    return _RECOVER_ACTION


def _act(actor: _Fighter, target: _Fighter, action: Attack) -> None:
    if actor.hp <= 0 or target.hp <= 0:
        return
    if action is _RECOVER_ACTION:
        actor.hp = min(1.0, actor.hp + actor.unit.recover)
        return
    target.hp = 0.0 if lands_ko(action, target.hp) else max(0.0, target.hp - action.fraction * action.accuracy)
    if action.fraction > 0:
        actor.hp = max(0.0, actor.hp - target.unit.passive_on_hit * action.accuracy)


def _x_first(x: _Fighter, ax: Attack, y: _Fighter, ay: Attack, x_wins_tie: bool) -> bool:
    kx, ky = (ax.priority, x.unit.speed), (ay.priority, y.unit.speed)
    return kx > ky or (kx == ky and x_wins_tie)


def _turn(x: _Fighter, y: _Fighter, ax: Attack | None, ay: Attack | None, x_wins_tie: bool) -> None:
    """One turn; a None action means that side is switching (or has nothing to do)."""
    x_first = ax is not None and (ay is None or _x_first(x, ax, y, ay, x_wins_tie))
    steps = [(x, y, ax), (y, x, ay)] if x_first else [(y, x, ay), (x, y, ax)]
    for actor, target, action in steps:
        if action is not None:
            _act(actor, target, action)
    for fighter in (x, y):
        if fighter.hp > 0:
            fighter.hp = min(1.0, max(0.0, fighter.hp + fighter.unit.residual))


def _winner(a_hp: float, b_hp: float) -> Literal["a", "b"] | None:
    if a_hp > 0 >= b_hp:
        return "a"
    if b_hp > 0 >= a_hp:
        return "b"
    return None


def duel(
    a: Combatant,
    b: Combatant,
    a_attacks: tuple[Attack, ...] | None = None,
    b_attacks: tuple[Attack, ...] | None = None,
    *,
    horizon: int = 30,
    tie_goes_to: Literal["a", "b"] = "b",
) -> DuelResult:
    """Race a against b until one faints or `horizon` turns pass; attacks default to each Combatant's own."""
    x = _Fighter(a, a.attacks if a_attacks is None else a_attacks, a.hp)
    y = _Fighter(b, b.attacks if b_attacks is None else b_attacks, b.hp)
    if x.hp <= 0 or y.hp <= 0:
        return DuelResult(_winner(x.hp, y.hp), 0, x.hp, y.hp, "")
    first_move = ""
    for turn in range(1, horizon + 1):
        ax, ay = _choose(x, y), _choose(y, x)
        if turn == 1 and ax is not None:
            first_move = ax.move
        _turn(x, y, ax, ay, tie_goes_to == "a")
        if x.hp <= 0 or y.hp <= 0:
            return DuelResult(_winner(x.hp, y.hp), turn, x.hp, y.hp, first_move)
    return DuelResult(None, horizon, x.hp, y.hp, first_move)


class _Hp(NamedTuple):
    ours: tuple[float, ...]
    theirs: tuple[float, ...]


@dataclass(frozen=True)
class _Line:
    score: tuple[int, float]  # (1 win / 0 draw / -1 loss, margin)
    order: tuple[int, ...]


@dataclass(frozen=True)
class _Game:
    ours: tuple[Combatant, ...]
    theirs: tuple[Combatant, ...]
    matchups: Matchups
    horizon: int

    def pair(self, hp: _Hp, i: int, j: int) -> tuple[Combatant, Combatant]:
        a, b = facing(self.ours, self.theirs, self.matchups, i, j)
        return replace(a, hp=hp.ours[i]), replace(b, hp=hp.theirs[j])


def _set(values: tuple[float, ...], index: int, value: float) -> tuple[float, ...]:
    return (*values[:index], value, *values[index + 1 :])


def _alive(values: tuple[float, ...]) -> list[int]:
    return [n for n, h in enumerate(values) if h > 0]


def _active_first(values: tuple[float, ...], active: int) -> list[int]:
    return [active, *(n for n in _alive(values) if n != active)]


def _next(units: tuple[Combatant, ...], values: tuple[float, ...], active: int) -> list[int]:
    """Who a side can field after a faint: anyone alive if its active fell, else the active (or a free swap)."""
    if values[active] <= 0:
        return _alive(values)
    return _active_first(values, active) if units[active].can_switch_on_faint else [active]


def _terminal(hp: _Hp) -> _Line:
    ours, theirs = bool(_alive(hp.ours)), bool(_alive(hp.theirs))
    status = 1 if ours and not theirs else -1 if theirs and not ours else 0
    return _Line((status, sum(hp.ours) - sum(hp.theirs)), ())


def _prepend(member: int, line: _Line) -> _Line:
    return line if line.order[:1] == (member,) else _Line(line.score, (member, *line.order))


def _highest(lines: list[_Line]) -> _Line:
    return max(lines, key=lambda line: line.score)


def _lowest(lines: list[_Line]) -> _Line:
    return min(lines, key=lambda line: line.score)


def _fight(game: _Game, hp: _Hp, i: int, j: int) -> _Line:
    """Duel our i against their j from `hp`, then play out the rest."""
    result = duel(*game.pair(hp, i, j), horizon=game.horizon)
    after = _Hp(_set(hp.ours, i, result.a_hp_left), _set(hp.theirs, j, result.b_hp_left))
    drawn = result.a_hp_left > 0 and result.b_hp_left > 0
    return _prepend(i, _terminal(after) if drawn else _after(game, after, i, j))


def _after(game: _Game, hp: _Hp, i: int, j: int) -> _Line:
    """After a faint between our i and their j: replacements (and free swaps), then the next duel."""
    if not _alive(hp.ours) or not _alive(hp.theirs):
        return _terminal(hp)
    ks, ms = _next(game.ours, hp.ours, i), _next(game.theirs, hp.theirs, j)
    if hp.ours[i] > 0:  # their member fell: they send first, we answer
        return _lowest([_highest([_fight(game, hp, k, m) for k in ks]) for m in ms])
    return _highest([_lowest([_fight(game, hp, k, m) for m in ms]) for k in ks])


def _remap(action: Attack | None, attacks: tuple[Attack, ...], target_hp: float) -> Attack | None:
    """The same move against a new target (its best attack if that move is missing)."""
    if action is None or action is _RECOVER_ACTION:
        return action
    same = [a for a in attacks if a.move == action.move]
    return same[0] if same else best_attack(attacks, target_hp)


def _opening(game: _Game, hp: _Hp, our_step: tuple[int, int], their_step: tuple[int, int]) -> _Line:
    """First turn with both actives on the field: each side stays or switches (old, new)."""
    (i, k), (j, m) = our_step, their_step
    if k == i and m == j:
        return _fight(game, hp, i, j)
    old_x, old_y = (_fighter(c) for c in game.pair(hp, i, j))
    x, y = (_fighter(c) for c in game.pair(hp, k, m))
    ax = _remap(_choose(old_x, old_y), x.attacks, y.hp) if k == i else None
    ay = _remap(_choose(old_y, old_x), y.attacks, x.hp) if m == j else None
    _turn(x, y, ax, ay, x_wins_tie=False)
    after = _Hp(_set(hp.ours, k, x.hp), _set(hp.theirs, m, y.hp))
    line = _fight(game, after, k, m) if x.hp > 0 and y.hp > 0 else _after(game, after, k, m)
    return _prepend(k, line)


def _first_move(game: _Game, hp: _Hp, i: int, j: int) -> str | None:
    action = _choose(*(_fighter(c) for c in game.pair(hp, i, j)))
    return None if action is None else action.move


def _choices(game: _Game, i: int, j: int) -> dict[int, tuple[_Line, str | None]]:
    """Our first member -> (line against their best reply, our first move)."""
    hp = _Hp(tuple(c.hp for c in game.ours), tuple(c.hp for c in game.theirs))
    if not _alive(hp.ours) or not _alive(hp.theirs):
        return {}
    out: dict[int, tuple[_Line, str | None]] = {}
    if hp.ours[i] > 0 and hp.theirs[j] > 0:
        for k in _active_first(hp.ours, i):
            line = _lowest([_opening(game, hp, (i, k), (j, m)) for m in _active_first(hp.theirs, j)])
            out[k] = (line, _first_move(game, hp, i, j) if k == i else None)
        return out
    for k in _next(game.ours, hp.ours, i):  # replacement: we commit first (pessimistic)
        fights = [(m, _fight(game, hp, k, m)) for m in _next(game.theirs, hp.theirs, j)]
        m, line = min(fights, key=lambda fight: fight[1].score)
        out[k] = (line, _first_move(game, hp, k, m))
    return out


def _result(line: _Line, first_move: str | None) -> ChainResult:
    return ChainResult(line.score[0] > 0, line.order, first_move, line.score[1])


def chain_options(
    ours: Sequence[Combatant],
    theirs: Sequence[Combatant],
    attacks: Matchups,
    *,
    active_ours: int,
    active_theirs: int,
    horizon: int = 30,
) -> dict[int, ChainResult]:
    """Each first member we could field (stay, switch, or replacement) -> its result against their best reply.

    A fainted (hp <= 0) active means a replacement choice for that side. Empty when either side has nobody left.
    """
    game = _Game(tuple(ours), tuple(theirs), attacks, horizon)
    return {k: _result(*choice) for k, choice in _choices(game, active_ours, active_theirs).items()}


def chain(
    ours: Sequence[Combatant],
    theirs: Sequence[Combatant],
    attacks: Matchups,
    *,
    active_ours: int,
    active_theirs: int,
    horizon: int = 30,
) -> ChainResult:
    """Our best line through the rest of the endgame, and whether it wins against their best replies."""
    game = _Game(tuple(ours), tuple(theirs), attacks, horizon)
    choices = _choices(game, active_ours, active_theirs)
    if not choices:
        hp = _Hp(tuple(c.hp for c in ours), tuple(c.hp for c in theirs))
        return _result(_terminal(hp), None)
    return _result(*max(choices.values(), key=lambda choice: choice[0].score))

# From research/pcsa-rl/rl2/matrix2.py (14 Nov 2026, research/pcsa-search/PLAN.md), split for pcsa's file, function
# and argument limits: this module holds the pieces (solver, stage changes, HP bins, successor lines, the turn-1
# branches, the stage-aware belief view and the patches); pcsa.rl_training.matrix2 holds the search. The arithmetic
# is research's, in the same order (tests/rl_training/test_matrix2.py compares the two on archived decisions).
"""Pieces of the two-turn matrix search (see pcsa.rl_training.matrix2)."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from pcsa.rl_training.matrix_cells import NO_HIT, Hit

GAMMA = 1.0  # weight of the turn-2 value
LAMBDA = 0.5  # their turn-2 strategy: (1 - LAMBDA) x the head's prior + LAMBDA x the equilibrium
HP_BIN = 0.1  # successor HP fractions are rounded to this
P_BRANCH = 0.02  # turn-1 branches below this probability are dropped (the rest renormalised)
REVEAL_P = 0.65  # an unrevealed move of a new active enters the turn-2 columns only above this belief probability
WIN = 1.0  # their last member fainted (+) / ours (-)
RM_ITERS = 200  # regret-matching iterations per turn-2 matrix
TIME_CAP_S = 1.5  # past this, the decision keeps the one-turn values
MAX_STAGE = 6
MAX_COLUMNS_NEW = 4  # move columns of a new active
CHOICE = frozenset({"choicescarf", "choiceband", "choicespecs"})


def hp_bin(h: float) -> float:
    if h <= 0:
        return 0.0
    return float(min(1.0, max(HP_BIN / 2, round(h / HP_BIN) * HP_BIN)))


def _pos(r: np.ndarray) -> np.ndarray:
    p = np.maximum(r, 0.0)
    s = p.sum()
    return p / s if s > 0 else np.full(len(r), 1.0 / len(r))


def regret_matching(V: np.ndarray, iters: int = RM_ITERS) -> tuple[np.ndarray, np.ndarray]:
    """Average strategies (rows x, columns y) of the zero-sum matrix game V (row player's payoff), by regret
    matching+ for both players with linear averaging."""
    n, m = V.shape
    rx, ry = np.zeros(n), np.zeros(m)
    sx, sy = np.zeros(n), np.zeros(m)
    for t in range(1, iters + 1):
        x, y = _pos(rx), _pos(ry)
        u_row = V @ y
        u_col = -(x @ V)
        rx = np.maximum(rx + u_row - x @ u_row, 0.0)
        ry = np.maximum(ry + u_col - y @ u_col, 0.0)
        sx += t * x
        sy += t * y
    return sx / sx.sum(), sy / sy.sum()


def self_setup(mo) -> bool:
    target = getattr(mo.target, "name", "") or str(mo.target or "")
    return target.lower() == "self"


def stage_change(mo, boosts: dict, setup_target_self: bool) -> dict:
    """Stages after using `mo`: a self-targeting status move's boosts, or a move's self_boost (Draco Meteor)."""
    out = dict(boosts)
    changes = {}
    if setup_target_self and mo.boosts:
        changes = mo.boosts
    elif getattr(mo, "self_boost", None):
        changes = mo.self_boost
    for k, v in (changes or {}).items():
        out[k] = max(-MAX_STAGE, min(MAX_STAGE, int(out.get(k, 0)) + int(v)))
    return out


def after(mo, boosts: dict) -> dict:
    """The user's stages after `mo` (unchanged when the move is unknown)."""
    return stage_change(mo, boosts, self_setup(mo)) if mo is not None else boosts


def no_ko_damage(hit: Hit, hp: float) -> float:
    """Damage on a target at `hp` when the hit does not KO (the mean, capped just under the HP)."""
    return max(0.0, min(hit.mean, hp - HP_BIN / 2)) if hit.ko < 1.0 else 0.0


def bt(boosts: dict) -> tuple:
    return tuple(sorted((k, int(v)) for k, v in boosts.items() if int(v)))


@dataclass(frozen=True)
class Line:
    """The board after turn 1 (what turn 2 starts from)."""

    ours: Any  # poke_env Pokemon
    our_hp: float
    our_boosts: tuple
    our_new: bool
    form: str | None
    lock: str | None
    theirs: Any
    their_hp: float
    their_boosts: tuple
    their_new: bool

    def key(self) -> tuple:
        return (
            id(self.ours),
            self.our_hp,
            self.our_boosts,
            self.our_new,
            self.form,
            self.lock,
            id(self.theirs),
            self.their_hp,
            self.their_boosts,
            self.their_new,
        )


@dataclass(frozen=True)
class Exchange:
    """One move-vs-move cell of turn 1: both actives, their HP and stages before and after their own moves."""

    ours: object
    theirs: object
    h_us: float
    h_them: float
    b_us: dict
    b_them: dict
    ours_after: dict
    theirs_after: dict
    form: str | None
    lock: str | None


def exchange_branches(x: Exchange, mine: Hit, th: Hit, pf: float) -> list:
    """(probability, Line, our KO'd mon, their KO'd mon) per branch of a move-vs-move cell, with
    pcsa.rl_training.matrix_cells.exchange's Protect / Sucker Punch rules."""
    if mine.protect or th.protect:
        line = Line(
            x.ours, hp_bin(x.h_us), bt(x.b_us), False, x.form, None, x.theirs, hp_bin(x.h_them), bt(x.b_them), False
        )
        return [(1.0, line, None, None)]
    if mine.sucker and not th.damaging:
        mine = NO_HIT
    if th.sucker and not mine.damaging:
        th = NO_HIT
    d_them, d_us = no_ko_damage(mine, x.h_them), no_ko_damage(th, x.h_us)
    both = Line(
        x.ours,
        hp_bin(x.h_us - d_us),
        bt(x.ours_after),
        False,
        x.form,
        x.lock,
        x.theirs,
        hp_bin(x.h_them - d_them),
        bt(x.theirs_after),
        False,
    )
    return [
        # we move first
        (pf * mine.ko, replace(both, our_hp=hp_bin(x.h_us), their_boosts=bt(x.b_them)), None, x.theirs),
        (pf * (1 - mine.ko) * th.ko, replace(both, their_hp=hp_bin(x.h_them - d_them)), x.ours, None),
        (pf * (1 - mine.ko) * (1 - th.ko), both, None, None),
        # they move first
        ((1 - pf) * th.ko, replace(both, their_hp=hp_bin(x.h_them), our_boosts=bt(x.b_us)), x.ours, None),
        ((1 - pf) * (1 - th.ko) * mine.ko, replace(both, our_hp=hp_bin(x.h_us - d_us)), None, x.theirs),
        ((1 - pf) * (1 - th.ko) * (1 - mine.ko), both, None, None),
    ]


class StageAware:
    """The belief seen through the turn-2 patches: own() and speed() cached with the stages in the key (the
    belief's own caches key on the mon only); everything else is the belief's."""

    def __init__(self, bb) -> None:
        object.__setattr__(self, "_bb", bb)
        object.__setattr__(self, "_c", {})
        object.__setattr__(self, "_matrix_across", getattr(bb, "_matrix_across", {}))

    def __getattr__(self, name):
        return getattr(self._bb, name)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def own(self, battle, mon, form=None):
        from pcsa.belief.grids import own_fighter

        key = ("own", id(battle), id(mon), form, bt(dict(mon.boosts)))
        if key not in self._c:
            self._c[key] = own_fighter(battle, mon, form)
        return self._c[key]

    def speed(self, battle, mon, our_speed: float, coming: bool = False):
        m = self._bb.member_of(battle, mon)
        if m is None:
            return None
        key = ("spe", id(mon), round(float(our_speed), 4), coming, bt(dict(mon.boosts)))
        if key not in self._c:
            self._c[key] = m.speed_vs(battle, mon, our_speed, coming=coming, mass=self._bb.MASS)
        return self._c[key]


@contextlib.contextmanager
def patched(*states):
    """Each state (mon, boosts dict, new): the mon's stages and active-turn counter set for the call, then reset."""
    saved = [(mon, dict(mon._boosts), mon._active_turns) for mon, _, _ in states]
    try:
        for mon, boosts, new in states:
            mon._boosts = {k: int(boosts.get(k, 0)) for k in (set(mon._boosts) | set(boosts))}
            mon._active_turns = 1 if new else 2
        yield
    finally:
        for mon, b, t in saved:
            mon._boosts, mon._active_turns = b, t

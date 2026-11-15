"""Speed-ups of the two-turn search (pcsa.rl_training.matrix2), from research rl2/matrix2.py (15 Nov 2026, plan 1920,
branch feat/search-teacher): the same values, about 4x faster.

  regret_matching_batch   matrix2_parts.regret_matching's column strategies for many games in one numpy pass per
                          iteration (both players' regrets as one vector per game, the payoffs as one block matrix)
  MemoLookahead           matrix.Lookahead with a per-search memo of hit() and p_first(); within one search only
                          the mons' stages and active-turn counters change (matrix2_parts.patched), so they are in
                          the key with every argument hit() reads
"""

from __future__ import annotations

import numpy as np

from pcsa.rl_training.matrix import Lookahead
from pcsa.rl_training.matrix2_parts import RM_ITERS


def regret_matching_batch(games: list[np.ndarray], iters: int = RM_ITERS) -> list[np.ndarray]:
    """The column strategy y of `regret_matching(V)` for every V in `games`. A game's regrets are one vector
    z = [x-regrets, y-regrets] and its payoffs W = [[0, V], [-V^T, 0]]; games are zero-padded and masked, so the
    padding adds exact zeros. Agrees with regret_matching to float rounding."""
    if not games:
        return []
    k = len(games)
    n = max(v.shape[0] for v in games)
    m = max(v.shape[1] for v in games)
    size = n + m
    W = np.zeros((k, size, size))
    mask, uniform = np.zeros((k, size)), np.zeros((k, size))
    for g, v in enumerate(games):
        r, c = v.shape
        W[g, :r, n : n + c] = v
        W[g, n : n + c, :r] = -v.T
        mask[g, :r], mask[g, n : n + c] = 1.0, 1.0
        uniform[g, :r], uniform[g, n : n + c] = 1.0 / r, 1.0 / c
    blocks = np.zeros((size, size))  # sums within each player's block
    blocks[:n, :n], blocks[n:, n:] = 1.0, 1.0
    WT = np.ascontiguousarray(W.transpose(0, 2, 1))
    z, avg = np.zeros((k, size)), np.zeros((k, size))
    for t in range(1, iters + 1):
        q = np.maximum(z, 0.0)
        tot = q @ blocks
        x = np.where(tot > 0, q / np.where(tot > 0, tot, 1.0), uniform)
        u = np.einsum("kj,kji->ki", x, WT)
        z = np.maximum(z + u - (x * u) @ blocks, 0.0) * mask
        avg += t * x
    sy = avg[:, n:]
    sy = sy / sy.sum(1, keepdims=True)
    return [sy[g, : v.shape[1]] for g, v in enumerate(games)]


def _mon_state(mon) -> tuple:
    return (tuple(sorted((mon._boosts or {}).items())), mon._active_turns)


class MemoLookahead(Lookahead):
    """Lookahead whose hit() and p_first() remember their results for one search (callers never mutate a Hit)."""

    def __init__(self, bb, ko_bonus: float, p_min: float) -> None:
        super().__init__(bb, ko_bonus, p_min)
        self._hits: dict = {}
        self._firsts: dict = {}

    def hit(  # noqa: PLR0913 (matrix.Lookahead.hit's signature)
        self,
        battle,
        our_mon,
        their_mon,
        move_id: str,
        theirs_attacking: bool,
        *,
        coming: bool,
        our_form: str | None = None,
        defender_hp: float | None = None,
    ):
        key = (
            id(our_mon),
            id(their_mon),
            move_id,
            theirs_attacking,
            coming,
            our_form,
            defender_hp,
            _mon_state(our_mon),
            _mon_state(their_mon),
        )
        got = self._hits.get(key)
        if got is None:
            got = super().hit(
                battle,
                our_mon,
                their_mon,
                move_id,
                theirs_attacking,
                coming=coming,
                our_form=our_form,
                defender_hp=defender_hp,
            )
            self._hits[key] = got
        return got

    def p_first(self, battle, our_mon, their_mon, our_form, our_priority: int, their_priority: int) -> float:
        key = (id(our_mon), id(their_mon), our_form, our_priority, their_priority)
        key += (_mon_state(our_mon), _mon_state(their_mon))
        got = self._firsts.get(key)
        if got is None:
            got = super().p_first(battle, our_mon, their_mon, our_form, our_priority, their_priority)
            self._firsts[key] = got
        return got

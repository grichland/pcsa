"""The bookkeeping of preview search (preview_search.py): the cell table, the race over our picks and the choice.

Pure numpy, no network or Champions Gym. Ported from pcsa-former's tools/preview_search_test.py (`table`, the
halving part of `search`, the H_guard choice of `main`); tests/network/test_preview_race.py checks it against that
file's cuts and choice.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import layout as L

P = L.PREVIEW_ACTIONS  # picks per side; cell = ours * P + theirs


def table(a_sum: np.ndarray, a_cnt: np.ndarray, wins: np.ndarray, games: np.ndarray,
          prior_games: float) -> np.ndarray:  # fmt: skip
    """[P, P] cell values: (wins + prior_games * A) / (games + prior_games), A the mean stage-A score (else 0.5)."""
    a = np.where(a_cnt > 0, a_sum / np.maximum(a_cnt, 1), 0.5)
    return ((wins + prior_games * a) / (games + prior_games)).reshape(P, P)


def score(T: np.ndarray, rows: np.ndarray, rule: str) -> np.ndarray:
    """Each row's value over their P picks: the mean, or the worst cell (maximin)."""
    return T[rows].min(1) if rule == "maximin" else T[rows].mean(1)


class Race:
    """Our picks still in the race and the cell each game slot plays. `head`: the preview head's pick, always kept.
    `sizes`: the rows kept at the cuts, made at 1/4, 1/2, 3/4 ... of the budget (len(sizes) + 1 parts).
    `rows`: our picks open at the start (default all P)."""

    def __init__(self, n_slots: int, head: int, sizes: Sequence[int], rows: Sequence[int] | None = None) -> None:
        self.head, self.sizes = int(head), [int(s) for s in sizes]
        self.rows = np.arange(P) if rows is None else np.array(sorted({*map(int, rows), self.head}))
        self.cuts: list[dict] = []
        self.ptr = 0
        self.cell = np.zeros(n_slots, np.int64)
        self.reassign(np.arange(n_slots))  # slot i starts on the i-th cell of the cycle
        self.ptr = 0

    def cut_due(self, elapsed: float, budget: float) -> bool:
        return len(self.cuts) < len(self.sizes) and elapsed > budget * (len(self.cuts) + 1) / (len(self.sizes) + 1)

    def cut(self, T: np.ndarray, elapsed: float) -> None:
        """Keep the best rows by mean over their picks, plus the head's pick."""
        size = self.sizes[len(self.cuts)]
        keep = list(self.rows[np.argsort(-T[self.rows].mean(1))][:size])
        self.rows = np.array(sorted(set(keep) | {self.head}))
        means = T[self.rows].mean(1)
        self.cuts.append({"after_s": round(elapsed, 2), "size": size, "rows": [int(r) for r in self.rows],
                          "means": [round(float(m), 4) for m in means]})  # fmt: skip

    def reassign(self, ended: np.ndarray) -> None:
        """Slots whose games ended take the next cells of a cycle over (rows in the race) x (all their picks)."""
        active = (self.rows[:, None] * P + np.arange(P)[None]).reshape(-1)
        self.cell[ended] = active[(self.ptr + np.arange(len(ended))) % len(active)]
        self.ptr += len(ended)


def choose(T: np.ndarray, rows: np.ndarray, head: int, rule: str, margin: float) -> tuple[int, dict]:
    """The best row in the race by `rule`; the head's pick unless the best beats it by at least `margin`."""
    values = score(T, rows, rule)
    best = int(rows[int(values.argmax())])
    best_v = float(values.max())
    head_v = float(score(T, np.array([head]), rule)[0])
    pick = best if best_v - head_v >= margin else int(head)
    return pick, {"best": best, "best_value": round(best_v, 4), "head_value": round(head_v, 4)}

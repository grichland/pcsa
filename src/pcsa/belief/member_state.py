# From research/pcsa-belief/belief/fast.py (sha256 dcd849036429, 2026-11-12 14:38), split into keys, grids, board,
# member_state, member_grids, member_out, member, fast (pcsa's 400-line file limit); pcsa changes are marked "pcsa:".
"""The state of one opposing member and its posterior weights (overview: pcsa.belief.fast)."""

from __future__ import annotations

import numpy as np

from pcsa.belief import calc as K
from pcsa.belief.keys import FastSpecies
from pcsa.belief.space import Priors


class MemberState:
    """pcsa: what FastMember holds, declared for its parts (MemberGrids, MemberOutputs, FastMember: one class split
    over three modules for pcsa's file limit), with the posterior weights every part reads."""

    key: str
    species: str
    space: FastSpecies
    priors: Priors
    fixes: K.Fixes
    items: list[str]
    item_index: dict[str, int]
    abilities: list[str]
    ability_index: dict[str, int]
    S: int
    F: np.ndarray
    F0: np.ndarray
    off: float
    r_comp: np.ndarray
    r_build: np.ndarray
    r_spread: np.ndarray
    r_item: np.ndarray
    r_ability: np.ndarray
    r_lp: np.ndarray
    r_logw: np.ndarray
    moves: list[str]
    version: int
    vocab: int
    _wv: int
    _W: np.ndarray
    _Fw: np.ndarray
    _rw: np.ndarray
    _agg: dict
    _maps: dict
    _tables: dict
    _out: dict
    _ctx_memo: dict
    _by_id: dict

    def _top(self) -> float:
        a = self.F.max() + self.off if self.F.size else -np.inf
        b = self.r_logw.max() if len(self.r_logw) else -np.inf
        return float(max(a, b))

    def _weights(self):
        """(dense posterior W [I, A, S] over cells, free part Fw, row weights rw), normalised together."""
        if self._wv != self.version:
            I, A, S = self.F.shape
            top = self._top()
            Fw = np.exp(self.F + (self.off - top)) if self.F.size else np.zeros((I, A, S))
            rw = np.exp(self.r_logw - top)
            Z = Fw.sum() + rw.sum()
            Fw /= Z
            rw /= Z
            W = Fw.copy()
            if len(rw):
                flat = (self.r_item * A + self.r_ability) * S + self.r_spread
                W += np.bincount(flat, weights=rw, minlength=I * A * S).reshape(I, A, S)
            self._W, self._Fw, self._rw, self._wv = W, Fw, rw, self.version
        return self._W, self._Fw, self._rw

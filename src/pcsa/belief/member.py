# From research/pcsa-belief/belief/fast.py (sha256 dcd849036429, 2026-11-12 14:38), split into keys, grids, board,
# member_state, member_grids, member_out, member, fast (pcsa's 400-line file limit); pcsa changes are marked "pcsa:".
"""One opposing member: weights, evidence and outputs (overview: pcsa.belief.fast)."""

from __future__ import annotations

import collections
import math

import numpy as np

from pcsa.battle.stats import form_of
from pcsa.belief import calc as K
from pcsa.belief.grids import (
    Grid,
    _lik,
)
from pcsa.belief.keys import (
    FastSpecies,
)
from pcsa.belief.member_out import MemberOutputs
from pcsa.belief.space import Priors


class FastMember(MemberOutputs):
    def __init__(self, key: str, species: str, space: FastSpecies, priors: Priors, fixes: K.Fixes) -> None:
        self.key, self.species, self.space, self.priors, self.fixes = key, species, space, priors, fixes
        self.items = list(space.items)
        self.item_index = {i: k for k, i in enumerate(self.items)}
        self.abilities = list(space.abilities)
        self.ability_index = {a: k for k, a in enumerate(self.abilities)}
        self.S = space.S
        self.F = space.free_lp.copy()  # free component log-weights [I, A, S] (without the moves offset)
        self.F0 = space.free_lp  # its log-prior (shrunk with F)
        self.off = 0.0  # log-likelihood of the used moves, the same for every free row
        self.r_comp, self.r_build = space.r_comp, space.r_build
        self.r_spread, self.r_item, self.r_ability = space.r_spread, space.r_item.copy(), space.r_ability.copy()
        self.r_lp, self.r_logw = space.r_lp, space.r_lp.copy()
        self.moves: list[str] = []
        self.item_shown: str | None = None
        self.ability_shown: str | None = None
        self.mega: str | None = None
        self.spoiled = False
        self.evidence: collections.Counter = collections.Counter()
        self.fallbacks: collections.Counter = collections.Counter()
        self.version = 0  # weights changed
        self.vocab = 0  # item / ability vocabulary changed
        self._wv = -1
        self._agg: dict = {}
        self._maps: dict = {}
        self._tables: dict = {}
        self._out: dict = {}
        self._ctx_memo: dict = {}
        self._by_id: dict = {}

    # bookkeeping ----------------------------------------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.F.size + len(self.r_logw))

    def _changed(self, vocab: bool = False) -> None:
        self.version += 1
        self._out.clear()
        self._agg.clear()
        if vocab:
            self.vocab += 1
            self._maps.clear()

    def _rows_keep(self, keep: np.ndarray) -> None:
        for name in ("r_comp", "r_build", "r_spread", "r_item", "r_ability", "r_lp", "r_logw"):
            setattr(self, name, getattr(self, name)[keep])

    def _apply(
        self,
        lik_grid: np.ndarray | None,
        imap=None,
        amap=None,
        free_scalar: float = 0.0,
        row_loglik: np.ndarray | None = None,
        inv: np.ndarray | None = None,
    ) -> None:
        """Add a log-likelihood: a grid [Ic, Ac, U] (with the vocabulary's class maps and the spreads' value index),
        or a scalar for the free part and per-row values for the rows (moves). Then drop rows more than `prune` nats
        below the best."""
        if lik_grid is not None:
            if inv is not None:
                lik_grid = lik_grid[:, :, inv]
            full = lik_grid[imap][:, amap]
            self.F = self.F + full
            if len(self.r_logw):
                self.r_logw = self.r_logw + lik_grid[imap[self.r_item], amap[self.r_ability], self.r_spread]  # pyright: ignore[reportOptionalSubscript]
        else:
            self.off += free_scalar
            if row_loglik is not None:
                self.r_logw = self.r_logw + row_loglik
        top = self._top()
        if len(self.r_logw):
            keep = self.r_logw > top - self.priors.prune
            if not keep.all():
                self._rows_keep(keep)
        self._changed()

    def weights(self) -> np.ndarray:
        return self._weights()[0]

    # certain evidence -------------------------------------------------------------------------------------------
    def _has_item(self, idx: int) -> bool:
        return bool(np.isfinite(self.F[idx]).any() or (self.r_item == idx).any())

    def _shrink_items(self, keep_idx: list[int]) -> None:
        remap = np.full(len(self.items), -1, np.int64)
        remap[keep_idx] = np.arange(len(keep_idx))
        self.F, self.F0 = self.F[keep_idx], self.F0[keep_idx]
        keep = remap[self.r_item] >= 0
        if not keep.all():
            self._rows_keep(keep)
        self.r_item = remap[self.r_item]
        self.items = [self.items[i] for i in keep_idx]
        self.item_index = {i: k for k, i in enumerate(self.items)}
        self._changed(vocab=True)

    def _shrink_abilities(self, keep_idx: list[int]) -> None:
        remap = np.full(len(self.abilities), -1, np.int64)
        remap[keep_idx] = np.arange(len(keep_idx))
        self.F, self.F0 = self.F[:, keep_idx], self.F0[:, keep_idx]
        keep = remap[self.r_ability] >= 0
        if not keep.all():
            self._rows_keep(keep)
        self.r_ability = remap[self.r_ability]
        self.abilities = [self.abilities[i] for i in keep_idx]
        self.ability_index = {a: k for k, a in enumerate(self.abilities)}
        self._changed(vocab=True)

    def reveal_item(self, item: str) -> None:
        if self.spoiled or self.item_shown is not None:
            return
        self.item_shown = item
        self.evidence["item"] += 1
        idx = self.item_index.get(item)
        if idx is None or not self._has_item(idx):
            # outside the space: every row now holds it (stage 1: "item_outside_space")
            self.fallbacks["item_outside_space"] += 1
            with np.errstate(divide="ignore"):
                self.F = np.logaddexp.reduce(self.F, axis=0, keepdims=True)
                self.F0 = np.logaddexp.reduce(self.F0, axis=0, keepdims=True)
            self.r_item = np.zeros_like(self.r_item)
            self.items, self.item_index = [item], {item: 0}
            self._changed(vocab=True)
            return
        self._shrink_items([idx])

    def forbid_item(self, item: str) -> None:
        """A teammate brought this item (Item Clause)."""
        if self.item_shown is not None:
            return
        idx = self.item_index.get(item)
        if idx is None or not self._has_item(idx):
            return
        rest = [k for k in range(len(self.items)) if k != idx]
        if not any(self._has_item(k) for k in rest):
            self.fallbacks["item_clause"] += 1
            return
        self.evidence["item_clause"] += 1
        self._shrink_items(rest)

    def reveal_ability(self, ability: str) -> None:
        if self.mega is not None or self.ability_shown is not None:
            return
        if ability not in self.space.ability_index:
            return  # not one the species can have (a copied or changed ability)
        self.ability_shown = ability
        self.evidence["ability"] += 1
        k = self.ability_index.get(ability)
        if k is None or not (np.isfinite(self.F[:, k]).any() or (self.r_ability == k).any()):
            self.fallbacks["ability"] += 1
            return
        self._shrink_abilities([k])

    def reveal_mega(self, stone: str, form: str | None = None) -> None:
        self.mega = form or stone
        self.reveal_item(stone)

    def reveal_move(self, move: str) -> None:
        if move in self.moves or len(self.moves) >= 4:
            return
        self.moves.append(move)
        self.evidence["move"] += 1
        q = self.space.q(move)
        has = np.array([move in lb.moves for lb in self.space.builds] + [False], dtype=bool)
        tied = self.r_build >= 0
        f = np.where(tied, np.where(has[np.where(tied, self.r_build, -1)], 1.0, self.priors.move_swap * q), q)
        self._apply(None, free_scalar=math.log(q), row_loglik=np.log(f))

    # soft evidence ----------------------------------------------------------------------------------------------
    def observe_order(self, battle, mon, our_speed: float, they_first: bool) -> None:
        g = self.speed_grid(battle, mon)
        sp = g.lo
        eps = self.priors.eps_order
        consistent = sp > our_speed if they_first else sp < our_speed
        f = np.where(consistent, 1 - eps, np.where(sp == our_speed, 0.5, eps))
        if not form_of(mon, battle)[1]:
            f = np.where(g.unmod[:, :, None], 1.0, f)  # an order-changing ability: nothing learnt  # pyright: ignore[reportOptionalSubscript]
        self.evidence["order"] += 1
        self._apply(np.log(f), g.imap, g.amap, inv=g.inv)

    def observe_hit(
        self, g: Grid, d: float, n_hits: int, ko: bool, scale: float, ours_attacking: bool, crit_possible: bool
    ) -> None:
        pr = self.priors
        if ours_attacking:
            s, tau, eps = pr.slack_dealt, pr.tau_dealt, pr.eps_dealt
        else:
            s, tau, eps = pr.slack_taken, max(2.0, pr.tau_taken * scale), pr.eps_taken
        Lo, Hi = g.lo * scale * n_hits, g.hi * scale * n_hits
        um = g.unmod[:, :, None]  # pyright: ignore[reportOptionalSubscript]
        Lo = np.where(um, Lo / 1.5, Lo)
        Hi = np.where(um, Hi * 1.5, Hi)
        p = _lik(Lo, Hi, d, ko, s, tau, pr.rel_slack)
        if crit_possible:
            p = (1 - pr.crit_unreported) * p + pr.crit_unreported * _lik(
                Lo * 1.5, Hi * 1.5, d, ko, s, tau, pr.rel_slack
            )
        bg = 1.0 if ko else 1.0 / scale
        f = (1 - eps) * p + eps * bg
        self.evidence["dealt" if ours_attacking else "taken"] += 1
        self._apply(np.log(f), g.imap, g.amap, inv=g.inv)

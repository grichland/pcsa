# From research/pcsa-belief/belief/fast.py (sha256 dcd849036429, 2026-11-12 14:38), split into keys, grids, board,
# member_state, member_grids, member_out, member, fast (pcsa's 400-line file limit); pcsa changes are marked "pcsa:".
"""One member's outputs: stat intervals, speed, damage ranges, move / item / ability probabilities (overview:
pcsa.belief.fast)."""

from __future__ import annotations

import collections

import numpy as np

from pcsa.battle import game
from pcsa.battle.damage import Fighter
from pcsa.battle.stats import form_of
from pcsa.belief.keys import (
    COMPONENTS,
    Interval,
    wquantiles,
)
from pcsa.belief.member_grids import MemberGrids
from pcsa.belief.space import STATS
from pcsa.players import opponent as opp


class MemberOutputs(MemberGrids):
    """FastMember's outputs (pcsa: a mixin, split from FastMember for the file limit)."""

    def _agg_w(
        self, imap: np.ndarray, amap: np.ndarray, Ic: int, Ac: int, inv: np.ndarray | None = None, inv_key=None
    ) -> np.ndarray:
        """The posterior summed over each class pair: [Ic, Ac, S], or [Ic, Ac, U] summed over the spreads of each
        value index `inv` (identified by `inv_key`)."""
        if inv is not None:
            key = (imap.tobytes(), amap.tobytes(), inv_key)
            hit = self._agg.get(key)
            if hit is None:
                Wc = self._agg_w(imap, amap, Ic, Ac)
                S = Wc.shape[2]
                U = int(inv.max()) + 1 if len(inv) else 1
                flat = (np.arange(Ic * Ac)[:, None] * U + inv[None, :]).ravel()
                hit = np.bincount(flat, weights=Wc.reshape(-1), minlength=Ic * Ac * U).reshape(Ic, Ac, U)
                self._agg[key] = hit
            return hit
        key = (imap.tobytes(), amap.tobytes())
        hit = self._agg.get(key)
        if hit is None:
            W = self._weights()[0]
            I, A, S = W.shape
            Mi = np.zeros((Ic, I))
            Mi[imap, np.arange(I)] = 1.0
            Wi = (Mi @ W.reshape(I, A * S)).reshape(Ic, A, S)
            if Ac == A and (amap == np.arange(A)).all():
                hit = Wi
            else:
                hit = np.zeros((Ic, Ac, S))
                for a in range(A):
                    hit[:, amap[a]] += Wi[:, a]
            self._agg[key] = hit
        return hit

    # outputs ----------------------------------------------------------------------------------------------------
    def stat_intervals(self, battle, mon, mass: float = 0.95, form: str | None = None) -> dict:
        key = ("stats", mass, form or form_of(mon, battle)[0])
        hit = self._out.get(key)
        if hit is None:
            base = game.species_entry(form).get("baseStats") if form else opp._base(mon, form_of(mon, battle)[0])
            table = self.table(base)  # pyright: ignore[reportArgumentType]
            order = self._table_order(base, table)  # pyright: ignore[reportArgumentType]
            w = self._weights()[0].sum(axis=(0, 1))
            a = (1 - mass) / 2
            hit = {}
            n = len(table) - 1
            for j, s in enumerate(STATS):
                c = np.cumsum(w[order[j]])
                c = c / c[-1]
                v = table[order[j], j]
                lo, hi, mid = (float(v[min(n, int(np.searchsorted(c, q - 1e-12)))]) for q in (a, 1 - a, 0.5))
                hit[s] = Interval(lo, hi, mid)
            self._out[key] = hit
        return hit

    def speed_vs(self, battle, mon, our_speed: float, coming: bool = False, mass: float = 0.95) -> dict:
        g = self.speed_grid(battle, mon, coming)
        Ic, Ac = g.lo.shape[0], g.lo.shape[1]
        w = self._agg_w(g.imap, g.amap, Ic, Ac, g.inv, g.inv_key)
        sp = g.lo
        a = (1 - mass) / 2
        lo, hi = wquantiles(sp.ravel(), w.ravel(), (a, 1 - a))
        tot = w.sum()
        return {
            "faster": float(w[sp > our_speed].sum() / tot),
            "tie": float(w[sp == our_speed].sum() / tot),
            "slower": float(w[sp < our_speed].sum() / tot),
            "lo": lo,
            "hi": hi,
        }

    def damage_range(
        self,
        battle,
        our_mon,
        their_mon,
        move_id: str,
        theirs_attacking: bool,
        masses=(0.95, 0.99),
        *,
        coming: bool = False,
        our_form: str | None = None,
        own: Fighter | None = None,
        ko: bool = True,
    ) -> dict:
        g = self.damage_grid(
            battle, our_mon, their_mon, move_id, theirs_attacking, coming=coming, our_form=our_form, own=own
        )
        if g is None:
            return {"lo": 0.0, "hi": 0.0, "point": 0.0}
        Ic, Ac = g.lo.shape[0], g.lo.shape[1]
        w = self._agg_w(g.imap, g.amap, Ic, Ac, g.inv, g.inv_key)
        tot = w.sum()
        if g.zero or tot <= 0:
            return {"lo": 0.0, "hi": 0.0, "point": 0.0}
        w = w / tot
        lo, hi = g.lo, g.hi
        out = {"point": float((w * (lo + hi)).sum() / 2)}
        qs = []
        for m in masses:
            qs += [(1 - m) / 2]
        lo_q = wquantiles(lo.ravel(), w.ravel(), qs)
        hi_q = wquantiles(hi.ravel(), w.ravel(), [1 - q for q in qs])
        for k, m in enumerate(masses):
            tag = "" if m == masses[0] else str(int(round(m * 100)))
            out["lo" + tag], out["hi" + tag] = lo_q[k], hi_q[k]
        defender = our_mon if theirs_attacking else their_mon
        hp = (
            defender.current_hp_fraction
            if not theirs_attacking
            else (defender.current_hp or 0) / max(1, defender.max_hp)
        )
        mo = opp.move_object(move_id)
        n_hits = float(mo.expected_hits) if mo else 1.0
        if hp > 0 and ko:
            L, Hh = lo * n_hits, hi * n_hits
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.where(L >= hp, 1.0, np.where(Hh < hp, 0.0, (Hh - hp) / np.maximum(Hh - L, 1e-9)))
            out["p_ko"] = float((w * p).sum())
        return out

    def move_probs(self) -> dict[str, float]:
        _, Fw, rw = self._weights()
        seen = set(self.moves)
        slots = max(0, 4 - len(seen))
        if not slots:
            return {}
        out: collections.Counter = collections.Counter()
        nb = len(self.space.builds)
        mass = np.bincount(np.where(self.r_build >= 0, self.r_build, nb), weights=rw, minlength=nb + 1)
        mass[-1] += Fw.sum()
        for k, lb in enumerate(self.space.builds):
            if mass[k] <= 0:
                continue
            rest = [m for m in lb.moves if m not in seen]
            if not rest:
                continue
            share = min(1.0, slots / len(rest))
            for m in rest:
                out[m] += mass[k] * share
        free = mass[-1]
        if free > 0:
            q = {m: v for m, v in self.space.q_move.items() if m not in seen}
            tot = sum(q.values())
            if tot > 0:
                for m, v in q.items():
                    out[m] += free * min(1.0, slots * v / tot)
        return dict(out)

    def item_probs(self) -> dict[str, float]:
        w = self._weights()[0].sum(axis=(1, 2))
        return {self.items[i]: float(w[i]) for i in np.nonzero(w)[0]}

    def ability_probs(self) -> dict[str, float]:
        w = self._weights()[0].sum(axis=(0, 2))
        return {self.abilities[i]: float(w[i]) for i in np.nonzero(w)[0]}

    def component_mass(self) -> dict[str, float]:
        _, Fw, rw = self._weights()
        w = np.bincount(self.r_comp.astype(np.int64), weights=rw, minlength=4)
        w[3] += Fw.sum()
        return {c: float(w[i]) for i, c in enumerate(COMPONENTS)}

    def truth(self, battle, mon, item: str, ability: str, stats: dict) -> dict:
        """tracker.Member.truth on the dense cells."""
        table = self.stats_table(battle, mon)
        tv = np.array([stats[s] for s in STATS])
        eq_spread = np.all(table == tv[None, :], axis=1)  # [S]
        W, _, rw = self._weights()
        I, A, S = W.shape
        it = self.item_index.get(item, -1)
        ab = self.ability_index.get(ability, -1)
        cell = np.zeros((I, A, S), bool)
        if it >= 0:
            if ab >= 0 or ability not in self.space.ability_index:
                if ab >= 0:
                    cell[it, ab] = eq_spread
                else:
                    cell[it] = eq_spread[None, :]
        # prior weights over the alive cells
        top0 = max(self.F0.max() if self.F0.size else -np.inf, self.r_lp.max() if len(self.r_lp) else -np.inf)
        P0 = np.exp(self.F0 - top0)
        r0 = np.exp(self.r_lp - top0)
        if len(r0):
            P0 = P0 + np.bincount(
                (self.r_item * A + self.r_ability) * S + self.r_spread, weights=r0, minlength=I * A * S
            ).reshape(I, A, S)
        P0 = P0 / P0.sum()
        alive = np.isfinite(self.F)
        if len(rw):
            alive.ravel()[(self.r_item * A + self.r_ability) * S + self.r_spread] = True
        out = {
            "p": float(W[cell].sum()),
            "p_prior_rows": float(P0[cell].sum()),
            "exists": bool((cell & alive).any()),
            "cells": int(alive.sum()),
            "spread_exists": bool(eq_spread.any()),
        }
        if (cell & alive).any():
            best = W[cell & alive].max()
            out["rank"] = int((W > best).sum()) + 1
        return out

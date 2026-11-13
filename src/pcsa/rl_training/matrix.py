# From research/pcsa-rl/rl2/matrix.py (Lookahead, action_values); the cell arithmetic is in matrix_cells. pcsa
# changes are marked "pcsa:": Lookahead.values is split into helpers (pcsa's 60-line function limit; the same steps
# in the same order), and the across-decision cache is dropped when the belief was rebuilt (pcsa's belief starts
# again on a live protocol that does not extend the last one, and member versions start again with it).
"""The matrix blend's lookahead over one decision (see matrix_cells for the cell values): Q per action index from
the battle's belief (pcsa.belief.fast.BattleBelief) and the opponent head's kept columns."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from poke_env.battle import Field

from pcsa.belief.grids import our_mega_form
from pcsa.players import opponent as opp
from pcsa.rl_training import features as f1
from pcsa.rl_training import features_log
from pcsa.rl_training.features_vocab import tables
from pcsa.rl_training.former_features import their_rows
from pcsa.rl_training.matrix_cells import (
    ACTIONS,
    FIRST_TURN,
    FIXED_50,
    HALF_HP,
    KO_BONUS,
    NO_HIT,
    P_MIN,
    PROTECT,
    SUCKER,
    Hit,
    dealt_value,
    exchange,
    first_prob,
    hazard,
    hp_of,
    q_values,
)
from pcsa.rl_training.spec import VOCAB

ACROSS_MAX = 20000  # across-decision hit cache entries before it is cleared


@lru_cache(maxsize=1)
def _vocab_rev() -> dict[int, str]:
    return {v: k for k, v in tables()["move"].items()}


def _vocab_move(idx: int) -> str | None:
    return _vocab_rev().get(int(idx))


class Lookahead:
    """The cells of one decision (one per call of `values`), reading a pcsa.belief.fast.BattleBelief."""

    def __init__(self, bb, ko_bonus: float = KO_BONUS, p_min: float = P_MIN) -> None:
        self.bb, self.ko_bonus, self.p_min = bb, ko_bonus, p_min
        self.counts = {"hits": 0, "reused": 0}
        rebuilds = getattr(bb, "rebuilds", 0)  # pcsa: a rebuilt belief restarts its member versions
        if not hasattr(bb, "_matrix_across") or getattr(bb, "_matrix_rebuilds", rebuilds) != rebuilds:
            bb._matrix_across = {}
        bb._matrix_rebuilds = rebuilds

    # one hit ---------------------------------------------------------------------------------------------------
    def hit(
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
    ) -> Hit:
        mo = opp.move_object(move_id)
        if mo is None:
            return NO_HIT
        mid = mo.id
        priority = int(mo.priority or 0)
        h = Hit(priority=priority, protect=mid in PROTECT, sucker=mid in SUCKER)
        attacker = their_mon if theirs_attacking else our_mon
        if mid in FIRST_TURN and not attacker.first_turn:
            return h
        defender = our_mon if theirs_attacking else their_mon
        hp = hp_of(defender) if defender_hp is None else defender_hp
        if hp <= 0:
            return h
        m = self.bb.member_of(battle, their_mon)
        if m is None:
            return h
        acc = mo.accuracy if isinstance(mo.accuracy, float) else 1.0
        fixed = self._fixed(battle, m, our_mon, their_mon, mid, theirs_attacking, hp)
        if fixed is not None:
            mean = fixed * acc
            h.mean, h.damaging = mean, mean > 0
            h.ko = acc if fixed >= hp else 0.0
            return h
        mean, ko = self._cached_mean_ko(
            battle, m, our_mon, their_mon, mid, theirs_attacking, coming, our_form, hp, mo
        )
        if mean <= 0:
            return h
        h.damaging = True
        h.mean, h.ko = mean * acc, ko * acc
        self._sash(h, m, our_mon, their_mon, theirs_attacking, hp, mo)
        return h

    def _cached_mean_ko(self, battle, m, our_mon, their_mon, mid, theirs_attacking, coming, our_form, hp, mo):
        """pcsa: research's hit() cache step, as a helper."""
        own = self.bb.own(battle, our_mon, our_form)
        key = (
            self.bb._value_key(battle, m, our_mon, their_mon, own, mid, theirs_attacking, coming, our_form),
            round(hp, 4),
        )
        cache = self.bb._matrix_across
        got = cache.get(key)
        if got is None:
            if len(cache) > ACROSS_MAX:
                cache.clear()
            got = cache[key] = self._mean_ko(
                m, battle, our_mon, their_mon, mid, theirs_attacking, coming, our_form, own, hp, mo
            )
            self.counts["hits"] += 1
        else:
            self.counts["reused"] += 1
        return got

    def _sash(self, h: Hit, m, our_mon, their_mon, theirs_attacking: bool, hp: float, mo) -> None:
        """Focus Sash / Sturdy at full HP: a single hit cannot KO (pcsa: research's hit() tail, as a helper)."""
        if hp >= 1.0 and float(getattr(mo, "expected_hits", 1) or 1) <= 1.0:
            if theirs_attacking:
                item = our_mon.item or ""
                if item == "focussash" or (our_mon.ability or "") == "sturdy":
                    h.ko = 0.0
            else:
                h.ko *= max(0.0, 1.0 - self._p_sash(m, their_mon))

    def _p_sash(self, m, their_mon) -> float:
        item = their_mon.item
        if item not in (None, "unknown_item"):
            p_item = 1.0 if item == "focussash" else 0.0
        else:
            p_item = m.item_probs().get("focussash", 0.0)
        ab = their_mon.ability
        p_ab = (1.0 if ab == "sturdy" else 0.0) if ab else m.ability_probs().get("sturdy", 0.0)
        return min(1.0, p_item + p_ab)

    def _their_max_hp(self, battle, m, their_mon) -> float:
        try:
            return float(m.stat_intervals(battle, their_mon, 0.99)["hp"].mid)
        except Exception:  # noqa: BLE001
            return 0.0

    def _fixed(self, battle, m, our_mon, their_mon, mid, theirs_attacking, hp) -> float | None:
        """Fixed-damage moves as a fraction of the defender's max HP, or None for an ordinary move."""
        if mid not in FIXED_50 and mid not in HALF_HP and mid != "endeavor":
            return None
        if mid in HALF_HP:
            return hp / 2.0
        if theirs_attacking:
            our_max = float(our_mon.max_hp or 1)
            if mid in FIXED_50:
                return 50.0 / our_max
            their_abs = hp_of(their_mon) * self._their_max_hp(battle, m, their_mon)
            return max(0.0, hp * our_max - their_abs) / our_max
        their_max = self._their_max_hp(battle, m, their_mon)
        if their_max <= 0:
            return 0.0
        if mid in FIXED_50:
            return 50.0 / their_max
        our_abs = float(our_mon.current_hp or 0)
        return max(0.0, hp * their_max - our_abs) / their_max

    @staticmethod
    def _mean_ko(m, battle, our_mon, their_mon, mid, theirs_attacking, coming, our_form, own, hp, mo):
        g = m.damage_grid(
            battle, our_mon, their_mon, mid, theirs_attacking, coming=coming, our_form=our_form, own=own
        )
        if g is None or g.zero:
            return 0.0, 0.0
        w = m._agg_w(g.imap, g.amap, g.lo.shape[0], g.lo.shape[1], g.inv, g.inv_key)
        tot = float(w.sum())
        if tot <= 0:
            return 0.0, 0.0
        w = w / tot
        n = float(getattr(mo, "expected_hits", 1) or 1)
        lo, hi = g.lo * n, g.hi * n
        mean = float((w * (lo + hi)).sum() / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(lo >= hp, 1.0, np.where(hi < hp, 0.0, (hi - hp) / np.maximum(hi - lo, 1e-9)))
        return mean, float((w * p).sum())

    # speed -----------------------------------------------------------------------------------------------------
    def p_first(self, battle, our_mon, their_mon, our_form, our_priority: int, their_priority: int) -> float:
        if our_priority != their_priority:
            return 1.0 if our_priority > their_priority else 0.0
        our_speed = opp.speed(battle, our_mon, self.bb.own(battle, our_mon, our_form))
        sp = self.bb.speed(battle, their_mon, our_speed, coming=True)
        if sp is None:
            return 0.5
        return first_prob(our_priority, their_priority, sp["faster"], sp["tie"], Field.TRICK_ROOM in battle.fields)

    # the decision ----------------------------------------------------------------------------------------------
    def _their_actions(self, battle, out: dict, theirs) -> tuple[dict, dict, dict]:
        """(column -> move id, column -> switching member, column -> p) of the kept columns at or above p_min."""
        rows = their_rows(battle, features_log.read(battle))
        kept_idx = [int(x) for x in out["kept_idx"].tolist()]
        kept_p = [float(x) for x in out["kept_p"].tolist()]
        k = len(kept_idx) - 5
        v = VOCAB["move"]
        their_moves: dict = {}
        their_switch: dict = {}
        p: dict = {}
        for col, (idx, pj) in enumerate(zip(kept_idx, kept_p, strict=True)):
            if pj < self.p_min:
                continue
            if col < k:
                mv = _vocab_move(idx)
                if mv:
                    their_moves[col], p[col] = mv, pj
            else:
                r = 1 + (idx - v)
                mon = rows[r].mon if 0 < r < len(rows) else None
                if mon is not None and not mon.fainted and mon is not theirs:
                    their_switch[col], p[col] = mon, pj
        return their_moves, their_switch, p

    def _taken(self, battle, theirs, their_moves: dict, cache: dict, col, mon, form, hp) -> Hit:
        kk = (col, id(mon), form, round(hp, 4))
        if kk not in cache:
            cache[kk] = self.hit(
                battle, mon, theirs, their_moves[col], True, coming=True, our_form=form, defender_hp=hp
            )
        return cache[kk]

    def _move_row(self, battle, a, ours, theirs, their_moves, their_switch, taken_cache) -> dict | None:
        """The cells of one of our moves (None: a Mega move without a Mega form, not computed)."""
        h_us, h_them = hp_of(ours), hp_of(theirs)
        row: dict = {}
        move_id = a.order.order.id if hasattr(a.order.order, "id") else str(a.order.order)
        form = our_mega_form(battle, ours) if a.action.mega else None
        if a.action.mega and form is None:
            return None
        mine = self.hit(battle, ours, theirs, move_id, False, coming=True, our_form=form)
        for col in their_moves:
            th = self._taken(battle, theirs, their_moves, taken_cache, col, ours, form, h_us)
            pf = (
                self.p_first(battle, ours, theirs, form, mine.priority, th.priority)
                if mine.priority == th.priority
                else (1.0 if mine.priority > th.priority else 0.0)
            )
            row[col] = exchange(mine, th, pf, h_them, h_us, self.ko_bonus)
        for col, mon in their_switch.items():
            hz = hazard(battle, mon, False)
            hin = max(0.0, hp_of(mon) - hz)
            if mine.protect or move_id in SUCKER:
                row[col] = hz
                continue
            hit = self.hit(battle, ours, mon, move_id, False, coming=False, our_form=form, defender_hp=hin)
            row[col] = hz + dealt_value(hit, hin, self.ko_bonus)
        return row

    def _switch_row(self, battle, a, theirs, their_moves, their_switch, taken_cache) -> dict:
        """The cells of one of our switches."""
        row: dict = {}
        target = a.order.order
        hz = hazard(battle, target, True)
        hin = max(0.0, hp_of(target) - hz)
        for col in their_moves:
            th = self._taken(battle, theirs, their_moves, taken_cache, col, target, None, hin)
            th_v = 0.0 if th.sucker else dealt_value(th, hin, self.ko_bonus)
            row[col] = -hz - th_v
        for col, mon in their_switch.items():
            row[col] = hazard(battle, mon, False) - hz
        return row

    def values(self, battle, legal, out: dict) -> tuple[np.ndarray, dict]:
        """(Q [11], details) for one decision; Q = 0 where nothing is computed."""
        q = np.zeros(ACTIONS, np.float32)
        info: dict = {}
        ours, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        if ours is None or theirs is None or ours.fainted or theirs.fainted:
            return q, info
        their_moves, their_switch, p = self._their_actions(battle, out, theirs)
        if not p:
            return q, info
        cells: dict = {}
        taken_cache: dict = {}
        for a in legal:
            i = f1.action_index(a, battle)
            if i < 0 or i >= ACTIONS:
                continue
            if a.action.kind == "move":
                row = self._move_row(battle, a, ours, theirs, their_moves, their_switch, taken_cache)
                if row is None:
                    continue
            else:
                row = self._switch_row(battle, a, theirs, their_moves, their_switch, taken_cache)
            cells[i] = row
        qs = q_values(cells, p)
        for i, val in qs.items():
            q[i] = val if math.isfinite(val) else 0.0
        info = {
            "p": p,
            "cells": cells,
            "moves": their_moves,
            "switches": {c: m.species for c, m in their_switch.items()},
            "counts": dict(self.counts),
        }
        return q, info


def action_values(battle, legal, out: dict, bb, ko_bonus: float = KO_BONUS, p_min: float = P_MIN):
    """Q per action index (np [11]) and the cell details, for one decision."""
    return Lookahead(bb, ko_bonus, p_min).values(battle, legal, out)

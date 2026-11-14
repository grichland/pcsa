# From research/pcsa-rl/rl2/matrix2.py (14 Nov 2026, research/pcsa-search/PLAN.md); pcsa changes: the pieces are in
# matrix2_parts, the long research functions are split into helpers (pcsa's 60-line / 6-argument limits; the same
# steps in the same order), time comes from pcsa.common.clock, and a time-capped decision is noted in the details
# (research also writes it to its fallbacks file).
"""Two-turn matrix search: the one-turn matrix (pcsa.rl_training.matrix) plus the value of the turn after, computed
with the same damage calculator; no battle engine. On with `matrix_depth: 2` in config/players/neural.yaml.

    Q2_i = Q1_i + gamma * C_i,   C_i = sum_j p_j sum_b P(b | i, j) Val2(line_b) / sum_j p_j

Q1 is the one-turn value, unchanged. Every turn-1 cell (our action i, their column j) splits into branches b (who
moves first, KO or not), each a Line: both actives, HP fractions (binned), stat stages (setup moves and self-boosts
from the move data), first-turn flags, our Choice lock and Mega form. A KO puts in a replacement: ours = the best for
us, theirs = the opponent head's switch weights over their known bench (an unknown replacement: 0; their last member
down: +WIN). Val2(line) is the turn-2 matrix game's value: our best response to their mix of (1 - LAMBDA) x the
head's prior and LAMBDA x the regret-matching equilibrium. Their columns for a new active: its revealed moves plus
the belief's predicted ones >= REVEAL_P. Not modelled: end-of-turn chip, status effects, crits, secondary effects,
their Choice lock, a new Mega at turn 2.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import replace

import numpy as np

from pcsa.belief.grids import our_mega_form
from pcsa.common.clock import SystemClock
from pcsa.players import opponent as opp
from pcsa.rl_training import features as f1
from pcsa.rl_training import matrix2_parts as P
from pcsa.rl_training.matrix import Lookahead
from pcsa.rl_training.matrix_cells import (
    ACTIONS,
    KO_BONUS,
    NO_HIT,
    P_MIN,
    PROTECT,
    SUCKER,
    dealt_value,
    exchange,
    hazard,
    hp_of,
)

Line = P.Line


class Timeout(Exception):
    """The two-turn search passed its time cap."""


def _move(move_id):
    return opp.move_object(move_id) if move_id else None


class TwoTurn:
    def __init__(
        self,
        bb,
        ko_bonus: float = KO_BONUS,
        p_min: float = P_MIN,
        gamma: float = P.GAMMA,
        lam: float = P.LAMBDA,
        time_cap_s: float = P.TIME_CAP_S,
    ) -> None:
        self.la = Lookahead(bb, ko_bonus, p_min)  # depth 1, exactly pcsa.rl_training.matrix
        self.la2 = Lookahead(P.StageAware(bb), ko_bonus, p_min)  # turn 2: stage-aware reads
        self.bb, self.kb, self.gamma, self.lam, self.cap = bb, ko_bonus, gamma, lam, time_cap_s
        self.clock = SystemClock()
        self.memo: dict = {}
        self.counts = {"lines": 0, "memo": 0, "matrices": 0}

    # turn-2 value of a line -------------------------------------------------------------------------------------
    def value2(self, battle, line: Line) -> float:
        k = line.key()
        if k in self.memo:
            self.counts["memo"] += 1
            return self.memo[k]
        self.counts["lines"] += 1
        if self.clock.now() > self.deadline:
            raise Timeout
        v = self._value2(battle, line)
        self.memo[k] = v
        return v

    def _value2(self, battle, line: Line) -> float:
        rows = self._our_rows(battle, line)
        cols, prior = self._their_cols(battle, line)
        if not rows or not cols:
            return 0.0
        V = np.zeros((len(rows), len(cols)))
        with P.patched(
            (line.ours, dict(line.our_boosts), line.our_new), (line.theirs, dict(line.their_boosts), line.their_new)
        ):
            for r, (kind, x) in enumerate(rows):
                for c, (ckind, y) in enumerate(cols):
                    V[r, c] = self._cell2(battle, line, (kind, x), (ckind, y))
        self.counts["matrices"] += 1
        if not np.isfinite(V).all():
            V = np.nan_to_num(V)
        _, y_eq = P.regret_matching(V)
        y = (1.0 - self.lam) * prior + self.lam * y_eq
        return float((V @ y).max())

    @staticmethod
    def _our_rows(battle, line: Line) -> list:
        out = []
        for mv in line.ours.moves.values():
            if getattr(mv, "current_pp", 1) == 0 or (line.lock and mv.id != line.lock):
                continue
            out.append(("move", mv.id))
        for mon in battle.team.values():
            if mon is not line.ours and not mon.fainted and hp_of(mon) > 0:
                out.append(("switch", mon))
        return out

    def _new_active_moves(self, battle, mon) -> list:
        moves = list(dict.fromkeys(list(mon.moves.keys())))
        m = self.bb.member_of(battle, mon)
        if m is not None:
            with contextlib.suppress(Exception):  # a belief read never stops the search
                moves += [mv for mv, pr in m.move_probs().items() if pr >= P.REVEAL_P and mv not in moves]
        return moves[: P.MAX_COLUMNS_NEW]

    def _their_cols(self, battle, line: Line):
        cols, w = [], []
        move_mass = sum(self.p[c] for c in self.their_moves) / self.p_tot
        if not line.their_new and line.theirs is self.theirs0:
            for c, mv in self.their_moves.items():
                cols.append(("move", mv))
                w.append(self.p[c] / self.p_tot)
        else:
            moves = self._new_active_moves(battle, line.theirs)
            for mv in moves:
                cols.append(("move", mv))
                w.append(move_mass / max(1, len(moves)))
        for c, mon in self.their_switch.items():
            if mon is not line.theirs and not mon.fainted:
                cols.append(("switch", mon))
                w.append(self.p[c] / self.p_tot)
        if not cols:
            return [], None
        w = np.asarray(w, float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        return cols, w

    def _cell2(self, battle, line: Line, row: tuple, col: tuple) -> float:
        (kind, x), (ckind, y) = row, col
        la, kb, form = self.la2, self.kb, line.form
        if kind == "move" and ckind == "move":
            mine = la.hit(
                battle, line.ours, line.theirs, x, False, coming=True, our_form=form, defender_hp=line.their_hp
            )
            th = la.hit(battle, line.ours, line.theirs, y, True, coming=True, our_form=form, defender_hp=line.our_hp)
            pf = la.p_first(battle, line.ours, line.theirs, form, mine.priority, th.priority)
            return exchange(mine, th, pf, line.their_hp, line.our_hp, kb)
        if kind == "move":  # their switch to y
            hz = hazard(battle, y, False)
            hin = P.hp_bin(max(0.0, hp_of(y) - hz))
            if x in PROTECT or x in SUCKER:
                return hz
            hit = la.hit(battle, line.ours, y, x, False, coming=False, our_form=form, defender_hp=hin)
            return hz + dealt_value(hit, hin, kb)
        hz = hazard(battle, x, True)
        hin = P.hp_bin(max(0.0, hp_of(x) - hz))
        if ckind == "move":
            th = la.hit(battle, x, line.theirs, y, True, coming=True, our_form=None, defender_hp=hin)
            return -hz - (0.0 if th.sucker else dealt_value(th, hin, kb))
        return hazard(battle, y, False) - hz

    # replacements ---------------------------------------------------------------------------------------------
    def _ours_replaced(self, battle, line: Line, gone) -> float:
        best = None
        for mon in battle.team.values():
            if mon is gone or mon.fainted or hp_of(mon) <= 0:
                continue
            h = P.hp_bin(max(0.0, hp_of(mon) - hazard(battle, mon, True)))
            if h <= 0:
                continue
            v = self.value2(
                battle, replace(line, ours=mon, our_hp=h, our_boosts=(), our_new=True, form=None, lock=None)
            )
            best = v if best is None else max(best, v)
        return -P.WIN if best is None else best

    def _theirs_replaced(self, battle, line: Line, gone) -> float:
        cands = [(c, mon) for c, mon in self.their_switch.items() if mon is not gone and not mon.fainted]
        if not cands:
            return P.WIN if self._their_left(battle, gone) == 0 else 0.0
        w = np.asarray([self.p[c] for c, _ in cands], float)
        w = w / w.sum()
        tot = 0.0
        for (_, mon), wi in zip(cands, w, strict=True):
            h = P.hp_bin(max(0.0, hp_of(mon) - hazard(battle, mon, False)))
            tot += wi * self.value2(battle, replace(line, theirs=mon, their_hp=h, their_boosts=(), their_new=True))
        return tot

    @staticmethod
    def _their_left(battle, gone) -> int:
        sizes = getattr(battle, "_team_size", None)
        size = int(sizes.get(battle.opponent_role or "p2", 3) or 3) if isinstance(sizes, dict) else 3
        fainted = sum(1 for m in battle.opponent_team.values() if m.fainted) + (0 if gone.fainted else 1)
        return max(0, size - fainted)

    def leaf(self, battle, line: Line, ko_ours, ko_theirs) -> float:
        """Val2 of a branch, with the KO'd side's replacement."""
        if ko_ours is not None:
            return self._ours_replaced(battle, line, ko_ours)
        if ko_theirs is not None:
            return self._theirs_replaced(battle, line, ko_theirs)
        return self.value2(battle, line)

    def _mix(self, battle, branches) -> float:
        kept = [b for b in branches if b[0] >= P.P_BRANCH]
        tot = sum(b[0] for b in kept)
        if tot <= 0:
            return 0.0
        return sum(pb * self.leaf(battle, line, ko_o, ko_t) for pb, line, ko_o, ko_t in kept) / tot

    # turn-1 branches ------------------------------------------------------------------------------------------
    def _move_branches(self, battle, col, move_id, form) -> list:
        """Our move `move_id` against their column `col`."""
        la, ours, theirs = self.la, self.ours0, self.theirs0
        h_us, h_them = hp_of(ours), hp_of(theirs)
        b_us, b_them = dict(ours.boosts), dict(theirs.boosts)
        mo = _move(move_id)
        ours_after = P.after(mo, b_us)
        lock = move_id if (ours.item or "") in P.CHOICE else None
        if col in self.their_moves:
            theirs_after = P.after(_move(self.their_moves[col]), b_them)
            mine = la.hit(battle, ours, theirs, move_id, False, coming=True, our_form=form)
            th = la.hit(
                battle, ours, theirs, self.their_moves[col], True, coming=True, our_form=form, defender_hp=h_us
            )
            pf = la.p_first(battle, ours, theirs, form, mine.priority, th.priority)
            x = P.Exchange(ours, theirs, h_us, h_them, b_us, b_them, ours_after, theirs_after, form, lock)
            return P.exchange_branches(x, mine, th, pf)
        mon = self.their_switch[col]
        hin = max(0.0, hp_of(mon) - hazard(battle, mon, False))
        if (mo is not None and mo.id in PROTECT) or move_id in SUCKER:
            hit = NO_HIT
        else:
            hit = la.hit(battle, ours, mon, move_id, False, coming=False, our_form=form, defender_hp=hin)
        base = Line(ours, P.hp_bin(h_us), P.bt(ours_after), False, form, lock, mon, P.hp_bin(hin), (), True)
        return [
            (hit.ko, base, None, mon),
            (1 - hit.ko, replace(base, their_hp=P.hp_bin(hin - P.no_ko_damage(hit, hin))), None, None),
        ]

    def _switch_branches(self, battle, col, target) -> list:
        """Our switch to `target` against their column `col`."""
        theirs = self.theirs0
        hin = max(0.0, hp_of(target) - hazard(battle, target, True))
        if col in self.their_moves:
            theirs_after = P.after(_move(self.their_moves[col]), dict(theirs.boosts))
            th = self.la.hit(
                battle, target, theirs, self.their_moves[col], True, coming=True, our_form=None, defender_hp=hin
            )
            if th.sucker:
                th = NO_HIT
            base = Line(
                target,
                P.hp_bin(hin),
                (),
                True,
                None,
                None,
                theirs,
                P.hp_bin(hp_of(theirs)),
                P.bt(theirs_after),
                False,
            )
            return [
                (th.ko, base, target, None),
                (1 - th.ko, replace(base, our_hp=P.hp_bin(hin - P.no_ko_damage(th, hin))), None, None),
            ]
        mon = self.their_switch[col]
        base = Line(
            target,
            P.hp_bin(hin),
            (),
            True,
            None,
            None,
            mon,
            P.hp_bin(max(0.0, hp_of(mon) - hazard(battle, mon, False))),
            (),
            True,
        )
        return [(1.0, base, None, None)]

    def continuation(self, battle, a, form, move_id) -> float:
        """C_i: sum_j p_j sum_b P(b) Val2(b) / sum_j p_j for our turn-1 action `a`."""
        tot = 0.0
        for col, pj in self.p.items():
            if a.action.kind == "move":
                branches = self._move_branches(battle, col, move_id, form)
            else:
                branches = self._switch_branches(battle, col, a.order.order)
            tot += pj * self._mix(battle, branches)
        return tot / self.p_tot

    # the decision ----------------------------------------------------------------------------------------------
    def _setup(self, battle, out: dict, info: dict) -> None:
        self.ours0, self.theirs0 = battle.active_pokemon, battle.opponent_active_pokemon
        self.their_moves, self.their_switch, self.p = self.la._their_actions(battle, out, self.theirs0)
        self.p_tot = sum(self.p.values())

    def values(self, battle, legal, out: dict) -> tuple[np.ndarray, dict]:
        t0 = self.clock.now()
        q1, info = self.la.values(battle, legal, out)
        if not info:
            return q1, info
        self.deadline = t0 + self.cap
        self._setup(battle, out, info)
        cont = np.zeros(ACTIONS, np.float32)
        try:
            for a in legal:
                i = f1.action_index(a, battle)
                if i < 0 or i >= ACTIONS or i not in info["cells"]:
                    continue
                move_id, form = None, None
                if a.action.kind == "move":
                    move_id = a.order.order.id if hasattr(a.order.order, "id") else str(a.order.order)
                    form = our_mega_form(battle, self.ours0) if a.action.mega else None
                c = self.continuation(battle, a, form, move_id)
                cont[i] = c if math.isfinite(c) else 0.0
        except Timeout:
            ms = round(1000 * (self.clock.now() - t0), 1)
            return q1, {**info, "depth": 1, "depth2": "time cap", "ms2": ms}
        q2 = (q1 + self.gamma * cont).astype(np.float32)
        ms = round(1000 * (self.clock.now() - t0), 1)
        return q2, {
            **info,
            "depth": 2,
            "q1": [round(float(x), 4) for x in q1],
            "cont": [round(float(x), 4) for x in cont],
            "counts2": dict(self.counts),
            "ms2": ms,
        }


def action_values(battle, legal, out: dict, bb, ko_bonus: float = KO_BONUS, p_min: float = P_MIN):
    """Q2 per action index (np [11]) and the details, for one decision (the one-turn Q past the time cap)."""
    return TwoTurn(bb, ko_bonus, p_min).values(battle, legal, out)

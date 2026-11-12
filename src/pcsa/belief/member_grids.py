# From research/pcsa-belief/belief/fast.py (sha256 dcd849036429, 2026-11-12 14:38), split into keys, grids, board,
# member_state, member_grids, member_out, member, fast (pcsa's 400-line file limit); pcsa changes are marked "pcsa:".
"""One member's forms, item / ability classes and calculation grids (overview: pcsa.belief.fast)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from poke_env.battle import SideCondition
from poke_env.data.normalize import to_id_str

from pcsa.battle import damage as D
from pcsa.battle import game
from pcsa.battle.damage import Fighter
from pcsa.battle.moves import base_data
from pcsa.battle.stats import form_of, multipliers
from pcsa.belief import calc as K
from pcsa.belief import evidence as E
from pcsa.belief.grids import (
    Grid,
    _params,
    _rolls_groups,
    own_fighter,
)
from pcsa.belief.keys import (
    MEGA_WEATHER,
    att_ability_key,
    att_item_key,
    def_ability_key,
    def_item_key,
    speed_ability_key,
    speed_item_key,
)
from pcsa.belief.member_state import MemberState
from pcsa.belief.space import STATS, stats_of
from pcsa.players import opponent as opp


class MemberGrids(MemberState):
    """FastMember's forms, classes and grids (pcsa: a mixin, split from FastMember for the file limit)."""

    # forms and classes --------------------------------------------------------------------------------------------
    def table(self, base: dict) -> np.ndarray:
        """[S, 6] stats of the member's spreads on these base stats (looked up by the dict's identity first: the
        dex's base-stat dicts live for the whole process)."""
        hit = self._by_id.get(id(base))
        if hit is not None and hit[0] is base:
            return hit[1]
        key = tuple(base[s] for s in STATS)
        t = self._tables.get(key)
        if t is None:
            tb = self.space.table
            t = self._tables[key] = stats_of(tb.points[: self.S], tb.effect[: self.S], base)
        self._by_id[id(base)] = (base, t)
        return t

    def _table_order(self, base: dict, table: np.ndarray) -> list[np.ndarray]:
        """Per stat, the spreads sorted by that stat (fixed per form)."""
        key = ("order",) + tuple(base[s] for s in STATS)
        hit = self._tables.get(key)
        if hit is None:
            hit = self._tables[key] = [np.argsort(table[:, j], kind="stable") for j in range(6)]
        return hit

    def _reduce(self, bases: list[dict], cols: tuple[str, ...]):
        """(key, inv [S], reps [U]): the spreads grouped by the values of `cols` in every form of `bases` (a
        calculation that reads only those stats is the same within a group); reps = one spread per group."""
        fast_key = ("reduce-id", cols) + tuple(id(b) for b in bases)
        hit = self._by_id.get(fast_key)
        if hit is not None and all(x is y for x, y in zip(hit[0], bases, strict=True)) and len(hit[0]) == len(bases):
            return hit[1]
        key = ("reduce", cols) + tuple(tuple(b[s] for s in STATS) for b in bases)
        hit = self._tables.get(key)
        if hit is not None:
            self._by_id[fast_key] = (list(bases), hit)
        if hit is None:
            if not cols:
                inv, reps = np.zeros(self.S, np.int64), np.zeros(1, np.int64)
            else:
                j = [STATS.index(c) for c in cols]
                keymat = np.hstack([self.table(b)[:, j] for b in bases])
                _, reps, inv = np.unique(keymat, axis=0, return_index=True, return_inverse=True)
                inv = inv.reshape(-1).astype(np.int64)
            hit = self._tables[key] = (key, inv, reps)
        return hit

    def stats_table(self, battle, mon) -> np.ndarray:
        return self.table(opp._base(mon, form_of(mon, battle)[0]))

    def _map(self, name: object, keyfun, vocab: list[str]) -> tuple[np.ndarray, list[str]]:
        """(class of every vocabulary entry, a representative per class) for one key function; cached (shared by
        every member of the species while its vocabulary is still the species' own)."""
        maps = self._maps if self.vocab else self.space.__dict__.setdefault("_shared_maps", {})
        hit = maps.get(name)
        if hit is None:
            keys, reps, out = {}, [], np.zeros(len(vocab), np.int64)
            for k, v in enumerate(vocab):
                c = keyfun(v)
                if c not in keys:
                    keys[c] = len(reps)
                    reps.append(v)
                out[k] = keys[c]
            hit = maps[name] = (out, reps)
        return hit

    def _context(self, battle, mon, coming: bool):
        """What the calculator sees now: (item mode, ability mode, Mega plan, current form).
        item mode: None (each row its own item) or the shown item ('' once gone); ability mode: None or the Mega's
        ability; plan: {item index: Mega form} for the rows that Mega Evolve in the coming turn. Memoised per board
        state (the same Battle object at the same protocol length)."""
        key = (
            id(battle),
            id(mon),
            len(battle._replay_data),
            coming,
            self.vocab,
            mon.item,
            battle.opponent_used_mega_evolve,
            mon.fainted,
        )
        hit = self._ctx_memo.get(key)
        if hit is None:
            if len(self._ctx_memo) > 64:
                self._ctx_memo.clear()
            hit = self._ctx_memo[key] = self._context_now(battle, mon, coming)
        return hit

    def _context_now(self, battle, mon, coming: bool):
        known, item = opp.revealed_item(mon)
        form, mega = form_of(mon, battle)
        cur_item = to_id_str(item or "") if known else None
        cur_ab = to_id_str(game.mega_ability(game.species_entry(form)) or "") if mega else None
        plan: dict[int, str] = {}
        if coming and not mega and not battle.opponent_used_mega_evolve and not mon.fainted:
            stones = game.megas()[1]
            base = to_id_str(game.base_name(form))
            for k, it in enumerate(self.items):
                f = stones.get(it)
                if f and to_id_str(game.base_name(f)) == base and (cur_item is None or cur_item == it):
                    plan[k] = f
        return cur_item, cur_ab, plan, form

    def _classes(self, role: str, keyfun_item, keyfun_ab, cur_item, cur_ab, plan, tag) -> tuple:
        """(imap, item reps, amap, ability reps, per item class: Mega form or None)."""
        if cur_item is not None and not plan:
            imap, ireps = np.zeros(len(self.items), np.int64), [cur_item]
        else:
            base_key = keyfun_item

            def key(it, base_key=base_key):
                k = self.item_index[it]
                return ("mega", it) if k in plan else base_key(it if cur_item is None else cur_item)

            imap, ireps = self._map(("i", role, tag, cur_item, tuple(sorted(plan))), key, self.items)
            if cur_item is not None:
                ireps = [cur_item if self.item_index[r] not in plan else r for r in ireps]
        if cur_ab is not None:
            amap, areps = np.zeros(len(self.abilities), np.int64), [cur_ab]
        else:
            amap, areps = self._map(("a", role, tag), keyfun_ab, self.abilities)
        forms = [plan.get(self.item_index.get(r, -1)) if plan else None for r in ireps]
        return imap, ireps, amap, areps, forms

    def _rep_fighter(self, mon, stats_row, item, ability, form: str | None, types=None, base=None) -> Fighter:
        return Fighter(
            {s: int(v) for s, v in zip(STATS, stats_row, strict=True)},
            types or opp._types(mon),
            ability,
            item,
            dict(mon.boosts),
            opp._status(mon),
            base,
        )

    # calculations ------------------------------------------------------------------------------------------------
    def damage_grid(
        self,
        battle,
        our_mon,
        their_mon,
        move_id: str,
        theirs_attacking: bool,
        *,
        coming: bool = False,
        our_form: str | None = None,
        their_full: bool | None = None,
        their_third: bool | None = None,
        our_full: bool | None = None,
        our_third: bool | None = None,
        own: Fighter | None = None,
    ) -> Grid | None:
        """Per-hit fractions of the defender's max HP over [item class, ability class, spread]. `own`: our
        member's Fighter when the caller has it (own_fighter(battle, our_mon, our_form)). pcsa: split in
        _damage_keys / _segments / _fill_segment for pcsa's function limit (same steps, same order)."""
        move = opp.move_object(move_id)
        if move is None:
            return None
        move_id = move.id
        own = own if own is not None else own_fighter(battle, our_mon, our_form)
        their_full, their_third, our_full, our_third = _thresholds(
            their_mon, our_mon, their_full, their_third, our_full, our_third
        )
        defender_mon = our_mon if theirs_attacking else their_mon
        wkey = (id(battle), id(defender_mon), len(battle._replay_data))
        where = self._ctx_memo.get(wkey)
        if where is None:
            where = self._ctx_memo[wkey] = opp.conditions(battle, defender_mon)
        cur_item, cur_ab, plan, form = self._context(battle, their_mon, coming)
        data = base_data(move_id)
        category, mtype = data.get("category"), data.get("type", "")
        ik, ak, tag, unmodeled = _damage_keys(theirs_attacking, category, mtype, move_id, own)
        imap, ireps, amap, areps, forms = self._classes("dmg", ik, ak, cur_item, cur_ab, plan, tag)
        Ic, Ac = len(ireps), len(areps)
        unmod = np.zeros((Ic, Ac), bool)
        cur_base = opp._base(their_mon, form)
        if coming and theirs_attacking and form == "aegislash" and category in D.SCREENS:
            cur_base = game.species_entry("aegislashblade").get("baseStats") or cur_base  # Stance Change
        cur_table = self.table(cur_base)
        segments = _segments(ireps, forms, areps)
        cols = _read_cols(move_id, category, theirs_attacking)
        bases = [cur_base] + [game.species_entry(mf).get("baseStats") or cur_base for mf in segments if mf]
        rkey, inv, reps = self._reduce(bases, cols)
        lo, hi = np.zeros((Ic, Ac, len(reps))), np.zeros((Ic, Ac, len(reps)))
        thirds = (their_third if theirs_attacking else our_third, our_full if theirs_attacking else their_full)
        pair, tables = (
            (their_mon, our_mon, own, move, move_id, theirs_attacking),
            (cur_base, cur_table, reps, unmodeled),
        )
        for mf, groups in segments.items():
            self._fill_segment(mf, groups, pair, where, thirds, tables, (lo, hi, unmod))
        return Grid(lo, hi, imap, amap, unmod, zero=not hi.any(), inv=inv, inv_key=rkey)

    def _fill_segment(self, mf, groups, pair, where, thirds, tables, out) -> None:
        """damage_grid's loop body for one form segment (pcsa: split out, unchanged)."""
        their_mon, our_mon, own, move, move_id, theirs_attacking = pair
        cur_base, cur_table, reps, unmodeled = tables
        lo, hi, unmod = out
        if mf is None:
            base, table, types = cur_base, cur_table, None
        else:
            entry = game.species_entry(mf)
            base = entry.get("baseStats") or cur_base
            table = self.table(base)
            types = tuple(t.lower() for t in entry.get("types", ())) or None
        seg_where = where
        if mf is not None:  # a Mega whose ability sets the weather as it evolves (Charizard Y's Drought)
            weather = MEGA_WEATHER.get(to_id_str(game.mega_ability(game.species_entry(mf)) or ""))
            if weather and where.weather != weather:
                seg_where = replace(where, weather=weather)
        rep_stats = table[self.space.rep_spread] if self.space.rep_spread < len(table) else table[0]
        params, where_to = [], []
        for ic, ac, it, ab in groups:
            rep = self._rep_fighter(their_mon, rep_stats, it, ab, mf, types, base)
            att, dfn = (rep, own) if theirs_attacking else (own, rep)
            att_mon, dfn_mon = (their_mon, our_mon) if theirs_attacking else (our_mon, their_mon)
            try:
                power = opp.move_power(att_mon, dfn_mon, att, dfn, move)
                pr = _params(att, dfn, move_id, seg_where, power, self.fixes, thirds[0], thirds[1])
            except (KeyError, ValueError, TypeError, ZeroDivisionError):
                pr = None
            if pr is None:
                continue
            params.append(pr)
            where_to.append((ic, ac, ab in unmodeled))
        if not params:
            return
        a, b = _rolls_groups(params, theirs_attacking, table[reps], own)
        for g, (ic, ac, um) in enumerate(where_to):
            if ac < 0:
                lo[ic, :], hi[ic, :], unmod[ic, :] = a[g], b[g], um
            else:
                lo[ic, ac], hi[ic, ac], unmod[ic, ac] = a[g], b[g], um

    def speed_grid(self, battle, mon, coming: bool = False) -> Grid:
        """Engine speed (opp.speed / calc.speed_rows) over [item class, ability class, spread]."""
        cur_item, cur_ab, plan, form = self._context(battle, mon, coming)
        sp = to_id_str(mon.species)
        imap, ireps, amap, areps, forms = self._classes(
            "spe",
            lambda it: speed_item_key(it, sp),
            lambda a: speed_ability_key(a, sp),
            cur_item,
            cur_ab,
            plan,
            ("spe", sp),
        )
        Ic, Ac = len(ireps), len(areps)
        cur_base = opp._base(mon, form)
        bases = [cur_base] + [game.species_entry(mf).get("baseStats") for mf in dict.fromkeys(forms) if mf]
        rkey, inv, reps = self._reduce(bases, ("spe",))  # pyright: ignore[reportArgumentType]
        out = np.zeros((Ic, Ac, len(reps)))
        order_free = np.zeros((Ic, Ac), bool)
        boost = int(dict(mon.boosts).get("spe", 0))
        status = opp._status(mon)
        weather = opp._weather(battle)
        side = battle.side_conditions if opp.is_ours(battle, mon) else battle.opponent_side_conditions
        tail = opp.TAILWIND if SideCondition.TAILWIND in side else 1.0
        cur_spe = self.table(cur_base)[reps, 5].astype(np.float64)
        for ic, (it, mf) in enumerate(zip(ireps, forms, strict=True)):
            if mf is None:
                spe, abilities = cur_spe, list(enumerate(areps))
            else:
                entry = game.species_entry(mf)
                spe = self.table(entry.get("baseStats"))[reps, 5].astype(np.float64)  # pyright: ignore[reportArgumentType]
                abilities = [(-1, to_id_str(game.mega_ability(entry) or ""))]
            for ac, ab in abilities:
                v = spe * multipliers(it, ab, mon.species).get("spe", 1.0)
                if self.fixes.iron_ball and it == "ironball":
                    v = v * 0.5
                if weather in opp.SPEED_WEATHER.get(ab, ()):
                    v = v * opp.WEATHER_SPEED
                v = K.stage_arr(np.floor(v).astype(np.int64), boost).astype(np.float64)
                if ab == "quickfeet" and status:
                    v = v * opp.QUICK_FEET
                elif status == "par":
                    v = v * opp.PARALYSIS
                v = v * tail
                if ac < 0:
                    out[ic, :] = v
                else:
                    out[ic, ac] = v
                    order_free[ic, ac] = ab in E.ORDER_ABILITIES
        return Grid(out, out, imap, amap, order_free, inv=inv, inv_key=rkey)


def _damage_keys(theirs_attacking: bool, category, mtype: str, move_id: str, own: Fighter):
    """damage_grid's class keys: (item key, ability key, cache tag, unmodeled abilities) (pcsa: split out)."""
    if theirs_attacking:
        types = frozenset({mtype} | (set(D.ATE_ABILITY.values()) if mtype == "normal" else set()))
        return (
            lambda it: att_item_key(it, category, move_id, types),
            att_ability_key,
            ("att", category, move_id, types),
            K.UNMODELED_ATTACK,
        )
    final = (D.ATE_ABILITY.get(own.ability) if mtype == "normal" else None) or mtype
    return lambda it: def_item_key(it, move_id, final), def_ability_key, ("def", move_id, final), K.UNMODELED_DEFEND


def _segments(ireps, forms, areps) -> dict:
    """{Mega form or None: [(item class, ability class or -1, item, ability)]} (pcsa: split out of damage_grid)."""
    segments: dict = {}
    for ic, (it, mf) in enumerate(zip(ireps, forms, strict=True)):
        if mf is None:
            for ac, ab in enumerate(areps):
                segments.setdefault(None, []).append((ic, ac, it, ab))
        else:
            segments.setdefault(mf, []).append(
                (ic, -1, it, to_id_str(game.mega_ability(game.species_entry(mf)) or ""))
            )
    return segments


def _read_cols(move_id: str, category, theirs_attacking: bool) -> tuple:
    """The stats this move reads on their side: the grid runs over their distinct values only."""
    foul = move_id == "foulplay"
    a_stat, d_stat = ("def" if move_id == "bodypress" else "atk", "def") if category == "physical" else ("spa", "spd")
    if category not in D.SCREENS:
        return ()
    if theirs_attacking:
        return () if foul else (a_stat,)
    return (d_stat, "hp") + (("atk",) if foul else ())


def _thresholds(their_mon, our_mon, their_full, their_third, our_full, our_third) -> tuple:
    """damage_grid's HP thresholds, each from the members' HP unless given (pcsa: split out, unchanged)."""
    frac = their_mon.current_hp_fraction if their_mon.current_hp is not None else 1.0
    their_full = frac >= 1.0 if their_full is None else their_full
    their_third = frac * 3 <= 1.0 if their_third is None else their_third
    our_frac = (our_mon.current_hp or 0) / max(1, our_mon.max_hp or 1)
    our_full = our_frac >= 1.0 if our_full is None else our_full
    our_third = our_frac * 3 <= 1.0 if our_third is None else our_third
    return their_full, their_third, our_full, our_third

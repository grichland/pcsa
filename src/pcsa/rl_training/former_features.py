# From research/pcsa-rl/rl2/features.py (sha256 6b9f6eeb6cb2, 2026-11-11 12:28); keep diffable.
"""v2 decision observations: revealed-only opposing rows, damage ranges, a last-turn token.

The network never sees a guessed build. Opposing rows hold what the battle revealed (moves, item, ability), facts
true for every set of the species (base stats, stat bounds, possible abilities, whether it could Mega Evolve), and
damage *ranges* over every spread, nature and unrevealed item (pcsa.battle.damage with spread_known=False; the upper
bound of their hits on us assumes a boosting item while theirs is unrevealed). Our rows are S1's (exact).

Layout: S1's 318-float row (rl.features_rows; the field encoder reads it) + EXTRA floats; ids = S1's 7 + 3 possible
abilities. The last-turn token is the previous turn's record (rl2.turns) from our point of view.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.damage import Fighter, investment_bounds
from pcsa.battle.stats import form_of, known_ability
from pcsa.players.base import LegalAction
from pcsa.players.opponent import _base, _status, _types, move_object, own_fighter, revealed_item, revealed_moves
from pcsa.rl_training import features as f1
from pcsa.rl_training import features_field, features_log, spec
from pcsa.rl_training import features_moves as fm
from pcsa.rl_training import features_rows as fr
from pcsa.rl_training import turns as tt
from pcsa.rl_training.features_rows import Row
from pcsa.rl_training.features_vocab import index

ROWS = 12
ROW = fr.ROW
EXTRA = 20
WIDTH = ROW + EXTRA
IDS = 10  # species, item, ability, move 1-4 (S1's 7), possible abilities 1-3
LAST = 20
LAST_IDS = 4  # our move, our switch-in species, their move, their switch-in species
ACTIONS = spec.ACTIONS
ITEM_UNKNOWN_UPPER = 1.5  # the upper bound of an unrevealed item on their damage (Choice Band / Specs)
STAT_DIV = fr.STAT_DIV
STATS = fr.STATS


@dataclass
class Obs:
    rows: np.ndarray  # [ROWS, WIDTH] float32
    ids: np.ndarray  # [ROWS, IDS] int64
    field: np.ndarray  # [126] float32
    last: np.ndarray  # [LAST] float32
    last_ids: np.ndarray  # [LAST_IDS] int64
    mask: np.ndarray  # [ACTIONS] bool
    turn: int = 0
    forced: bool = False  # a forced switch (no move legal): no opponent label for this decision


# Fighters without a guessed build -----------------------------------------------------------------------------
def their_fighter(battle, mon, stats: dict | None = None, *, spread_known: bool = False) -> Fighter:
    """An opposing member with only revealed item and ability; `stats` (bounds) when attacking."""
    form = form_of(mon, battle)[0]
    base = _base(mon, form)
    known, item = revealed_item(mon)
    ability = known_ability(battle, mon) or ""
    stats = stats or {s: int(v) for s, v in investment_bounds(base)[1].items()}
    item_id = to_id_str(item or "") if known else ""
    return Fighter(
        stats, _types(mon), to_id_str(ability), item_id, dict(mon.boosts), _status(mon), base, spread_known
    )


class Ctx(f1._Ctx):
    """S1's per-observation cache, with damage from ranges instead of assumed sets. Fighters (the stats, types,
    item and ability of one side of a damage calculation) are built once per member per observation."""

    def __init__(self, battle, log, legal) -> None:
        super().__init__(battle, log, legal)
        self._fighters: dict[tuple[int, str], Fighter] = {}

    def own(self, mon) -> Fighter:
        k = (id(mon), "own")
        if k not in self._fighters:
            self._fighters[k] = own_fighter(self.battle, mon)
        return self._fighters[k]

    def their(self, mon, bound: str = "") -> Fighter:
        """An opposing member: bound "" (defending, spread unknown), "low" or "high" (attacking at that bound)."""
        k = (id(mon), bound)
        if k not in self._fighters:
            if bound:
                base = _base(mon, form_of(mon, self.battle)[0])
                low, high = investment_bounds(base)
                stats = low if bound == "low" else high
                self._fighters[k] = their_fighter(self.battle, mon, stats, spread_known=True)
            else:
                self._fighters[k] = their_fighter(self.battle, mon)
        return self._fighters[k]

    def hits(self, att: Row, dfn: Row | None) -> list:
        if dfn is None or att.mon is None or dfn.mon is None or att.mon.fainted or dfn.mon.fainted:
            return []
        k = (id(att), id(dfn))
        if k not in self._hits:
            self._hits[k] = self._range_hits(att, dfn)
        return self._hits[k]

    def _range_hits(self, att: Row, dfn: Row) -> list:
        b, moves = self.battle, att.moves[:4]
        if att.ours:
            fd = self.own(dfn.mon) if dfn.ours else self.their(dfn.mon)
            return fm.hits(b, att.mon, dfn.mon, self.own(att.mon), fd, moves)
        fd = self.own(dfn.mon)
        weak = fm.hits(b, att.mon, dfn.mon, self.their(att.mon, "low"), fd, moves)
        strong = fm.hits(b, att.mon, dfn.mon, self.their(att.mon, "high"), fd, moves)
        boost = 1.0 if revealed_item(att.mon)[0] else ITEM_UNKNOWN_UPPER
        return [(w[0], w[1], s[2] * boost) for w, s in zip(weak, strong, strict=True)]


# Rows ----------------------------------------------------------------------------------------------------------
def their_rows(battle, log) -> list[Row]:
    members = f1._their_members(battle, log)
    revealed = sum(1 for k, _ in members if k is not None)
    rows = []
    for k, mon in members:
        shown = revealed_moves(mon)[:4] if k is not None else []
        active = k is not None and mon is battle.opponent_active_pokemon and not mon.fainted
        rows.append(
            Row(
                mon,
                False,
                key=k,
                brought=float(k is not None),
                not_brought=float(k is None and revealed >= 3),
                active=active,
                moves=[mon.moves.get(i) or move_object(i) for i in shown],
                known=[True] * len(shown),
            ).resolve(battle)
        )
    rows += [Row(None, False) for _ in range(6 - len(rows))]
    return rows


@lru_cache(maxsize=4096)
def _possible_abilities(form: str) -> tuple[str, ...]:
    return tuple(to_id_str(a) for a in (game.species_entry(form).get("abilities") or {}).values())[:3]


def _could_mega(battle, row: Row) -> float:
    mon = row.mon
    if mon.fainted or row.mega or battle.opponent_used_mega_evolve:
        return 0.0
    base = to_id_str(game.base_name(row.form))
    forms = game.megas()[0].get(base, [])
    if not forms:
        return 0.0
    known, item = revealed_item(mon)
    return float(not known or game.megas()[1].get(to_id_str(item or "")) in forms)


def extra(battle, row: Row, ctx: Ctx | None = None) -> tuple[np.ndarray, list[int]]:
    """(EXTRA floats, 3 possible-ability ids) beyond S1's row."""
    f = np.zeros(EXTRA, dtype=np.float32)
    if row.mon is None:
        return f, [0, 0, 0]
    mon = row.mon
    base = _base(mon, row.form)
    if row.ours:
        exact = (ctx.own(mon) if ctx is not None else own_fighter(battle, mon)).stats
        low = high = exact
    else:
        low, high = investment_bounds(base)
    f[0:6] = [min(2.0, float(low.get(s, 0)) / STAT_DIV) for s in STATS]
    f[6:12] = [min(2.0, float(high.get(s, 0)) / STAT_DIV) for s in STATS]
    f[12] = (
        float(not battle.used_mega_evolve and fr.mega_form_of(mon) is not None)
        if row.ours
        else _could_mega(battle, row)
    )
    f[13] = 0.0 if row.ours else float(not revealed_item(mon)[0])
    f[14] = 0.0 if row.ours else float(not row.ability)
    possible = (to_id_str(mon.ability or ""),) if row.ours else _possible_abilities(row.form)
    f[15] = len(possible) / 3.0
    f[16] = len([m for m in row.moves if m is not None]) / 4.0
    f[17] = float(row.ours)
    f[18] = min(1.0, len(game.learnsets().get(to_id_str(game.base_name(row.form)), ())) / 150.0)
    ids = [index("ability", a) for a in possible] + [0] * (3 - len(possible))
    return f, ids[:3]


def _encode(
    ctx: Ctx, mine: list[Row], theirs: list[Row], legal, cache: dict | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, ids, field); `cache` reuses row encodings across team-preview picks (keyed by row and opponent row)."""
    rows = np.zeros((ROWS, WIDTH), dtype=np.float32)
    ids = np.zeros((ROWS, IDS), dtype=np.int64)
    my_opp = theirs[0] if theirs[0].mon is not None and theirs[0].key is not None else None
    their_opp = mine[0] if mine[0].mon is not None else None
    for i, row in enumerate(mine + theirs):
        opp = my_opp if i < 6 else their_opp
        k = (id(row), id(opp))
        if cache is None or k not in cache:
            f, d = fr.encode(ctx.battle, row, opp, ctx)
            e, ed = extra(ctx.battle, row, ctx)
            entry = (np.concatenate([f, e]), np.concatenate([d, np.asarray(ed, np.int64)]))
            if cache is None:
                rows[i], ids[i] = entry
                continue
            cache[k] = entry
        rows[i], ids[i] = cache[k]
    lead = ("field", id(mine[0]))  # at team preview the field vector depends on the pick only through its lead
    if cache is not None and lead in cache:
        return _clean(rows), ids, cache[lead]
    field = _clean(features_field.encode(ctx.battle, mine, theirs, legal, ctx.log, rows[:, :ROW]))
    if cache is not None:
        cache[lead] = field
    return _clean(rows), ids, field


def _clean(x: np.ndarray) -> np.ndarray:
    """f1._clean, skipped when every value is finite already (the usual case)."""
    return x if np.isfinite(x).all() else f1._clean(x)


# The last turn -------------------------------------------------------------------------------------------------
def last_turn(battle, records: dict[int, tt.TurnRecord] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(LAST floats, LAST_IDS ids): the previous complete turn from our side, then theirs."""
    records = records if records is not None else tt.read(battle._replay_data)
    me = battle.player_role or "p1"
    them = "p2" if me == "p1" else "p1"
    rec = records.get(int(battle.turn or 0) - 1)
    f = np.zeros(LAST, dtype=np.float32)
    ids = np.zeros(LAST_IDS, dtype=np.int64)
    f[18] = min(1.0, (battle.turn or 0) / 30.0)
    f[19] = float((battle.turn or 0) <= 1)
    if rec is None:
        return f, ids
    for j, side in enumerate((me, them)):
        s = rec.sides[side]
        f[j * 9 : j * 9 + 9] = [
            float(s.action == "move"),
            float(s.action == "switch"),
            float(not s.action),
            float(s.mega),
            float(s.moved_first),
            min(2.0, s.damage_taken),
            min(1.0, s.fainted / 3.0),
            float(s.revealed_item),
            float(s.revealed_ability),
        ]
        ids[j * 2] = index("move", s.move or None)
        ids[j * 2 + 1] = index("species", s.switch_to or None)
    return f, ids


# Observations --------------------------------------------------------------------------------------------------
def observe(battle, legal: Sequence[LegalAction]) -> Obs:
    """The v2 observation of one decision (no catalog, no tracker)."""
    log = features_log.read(battle)
    mine, theirs = f1.my_rows(battle, legal), their_rows(battle, log)
    rows, ids, field = _encode(Ctx(battle, log, legal), mine, theirs, legal)
    f1._remember_bench(battle)
    mask = np.zeros(ACTIONS, dtype=bool)
    for a in legal:
        mask[f1.action_index(a, battle)] = True
    last, last_ids = last_turn(battle)
    forced = bool((battle.last_request or {}).get("forceSwitch")) or not mask[: f1.SWITCH_AT].any()
    return Obs(rows, ids, field, last, last_ids, mask, int(battle.turn or 0), forced)


def preview_observations(battle, picks: Sequence[tuple[int, int, int]]) -> list[Obs]:
    """Turn-0 observations, one per (lead, second, third) pick: our trio with the lead active, their six."""
    team = list(battle.team.values())
    log = features_log.read(battle)
    ctx, cache = Ctx(battle, log, ()), {}
    theirs = their_rows(battle, log)
    last, last_ids = last_turn(battle)
    leads = {i: f1.my_rows(battle, (), [team[i]], active=0)[0] for i in {p[0] for p in picks}}
    benches = {i: f1.my_rows(battle, (), [team[i]], active=None)[0] for p in picks for i in p[1:]}
    out = []
    for lead, *rest in picks:
        mine = [leads[lead], *(benches[i] for i in rest)]
        mine += [Row(None, True) for _ in range(6 - len(mine))]
        rows, ids, field = _encode(ctx, mine, theirs, (), cache)
        rows[0, 147] = 1.0  # the lead
        mask = np.zeros(ACTIONS, dtype=bool)
        out.append(Obs(rows, ids, field, last.copy(), last_ids.copy(), mask, 0, True))
    return out

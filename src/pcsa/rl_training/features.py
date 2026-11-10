# From research/pcsa-rl/rl/features.py (sha256 61303b72c917, 2026-11-08 20:15); keep diffable.
"""Decision observations for the policy: 12 Pokémon rows [12, 318] + ids [12, 7], a field vector [126] and the
legal-action mask [11], from the poke-env Battle a pcsa player sees (always the p1 view).

Canonical rows: 0 my active (or the member that just fainted), 1-2 my bench in request order (= switch actions
8-9), 3-5 my unbrought (always empty: the Battle holds only the request's members); 6 their active (or the one
that just fainted), 7.. their other members seen in battle, then the preview-only ones in preview order.
Row layout in features_rows, field layout in features_field.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from poke_env.battle import Move
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.players.base import LegalAction
from pcsa.players.opponent import assumed, move_object, own_fighter, revealed_moves, set_fighter
from pcsa.rl_training import features_field, features_log, features_rows
from pcsa.rl_training.features_moves import hits, pp_fraction
from pcsa.rl_training.features_rows import Row
from pcsa.rl_training.spec import ACTIONS, POKEMON_FLOATS, POKEMON_IDS, ROWS

MOVE_ACTIONS, MEGA_AT, SWITCH_AT = 4, 4, 8


@dataclass
class Obs:
    rows: np.ndarray  # [ROWS, POKEMON_FLOATS] float32
    ids: np.ndarray  # [ROWS, POKEMON_IDS] int64
    field: np.ndarray  # [FIELD_FLOATS] float32
    mask: np.ndarray  # [ACTIONS] bool


class _Ctx:
    """Per-observation caches: damage per (attacker row, defender row), speed per row, legal move ids."""

    def __init__(self, battle, log, legal) -> None:
        self.battle, self.log = battle, log
        self._hits: dict[tuple[int, int], list] = {}
        self._speed: dict[int, tuple[float, float]] = {}
        self.legal_moves = {
            a.order.order.id
            for a in legal
            if a.action.kind == "move" and a.order is not None and isinstance(a.order.order, Move)
        }

    def hits(self, att: Row, dfn: Row | None) -> list:
        if dfn is None or att.mon is None or dfn.mon is None or att.fighter is None or dfn.fighter is None:
            return []
        if att.mon.fainted or dfn.mon.fainted:
            return []
        k = (id(att), id(dfn))
        if k not in self._hits:
            self._hits[k] = hits(self.battle, att.mon, dfn.mon, att.fighter, dfn.fighter, att.moves[:4])
        return self._hits[k]

    def speed(self, row: Row) -> tuple[float, float]:
        if id(row) not in self._speed:
            self._speed[id(row)] = features_rows.speed_bounds(self.battle, row)
        return self._speed[id(row)]

    def move_state(self, row: Row, j: int, move: Move) -> tuple[float, float, float, float]:
        """(PP fraction, disabled, locked in, known)."""
        known = row.known[j] if j < len(row.known) else False
        if not row.ours:
            return (pp_fraction(move) if known else 1.0), 0.0, 0.0, float(known)
        if not row.active or not self.legal_moves:
            return pp_fraction(move), 0.0, 0.0, 1.0
        legal = move.id in self.legal_moves
        locked = legal and len(self.legal_moves) == 1 and len(row.moves) > 1
        return pp_fraction(move), float(not legal), float(locked), 1.0


def _slot_moves(mon, legal: Sequence[LegalAction]) -> list[Move | None]:
    """Our active's moves with move j = the move legal action slot j orders (else the Battle's move order)."""
    own = list(mon.moves.values())[:4]
    placed: list[Move | None] = [None] * 4
    for a in legal:
        if a.action.kind == "move" and a.order is not None and isinstance(a.order.order, Move):
            slot = a.action.move or 0
            move = next((m for m in own if m.id == a.order.order.id), None)
            if move is not None and 0 <= slot < 4 and placed[slot] is None and move not in placed:
                placed[slot] = move
    rest = iter(m for m in own if m not in placed)
    return [m if m is not None else next(rest, None) for m in placed]


def _request_members(battle) -> list:
    side = ((battle.last_request or {}).get("side") or {}).get("pokemon") or []
    members = [battle.team[p["ident"]] for p in side if p.get("ident") in battle.team]
    if members:
        return members
    team = list(battle.team.values())
    return sorted(team, key=lambda m: not m.active)


def my_rows(battle, legal: Sequence[LegalAction], members: list | None = None, active: int | None = 0) -> list[Row]:
    members = (members if members is not None else _request_members(battle))[:3]
    rows = []
    for i, mon in enumerate(members):
        moves = _slot_moves(mon, legal) if i == active else list(mon.moves.values())[:4]
        rows.append(
            Row(
                mon,
                True,
                key=f"p1: {mon.name}",
                brought=1.0,
                active=i == active and not mon.fainted,
                moves=moves,
                known=[True] * len(moves),
                fighter=own_fighter(battle, mon),
            ).resolve(battle)
        )
    rows += [Row(None, True) for _ in range(6 - len(rows))]
    return rows


def _their_members(battle, log) -> list[tuple[str | None, Any]]:
    seen = [(k, m) for k, m in battle.opponent_team.items() if ": " in k]  # preview keys are bare species
    active = battle.opponent_active_pokemon
    if active is None and seen:
        entered = [(log.members.get(k).entered_turn if k in log.members else -1, i) for i, (k, _) in enumerate(seen)]
        active = seen[max(entered)[1]][1]
    seen.sort(key=lambda km: km[1] is not active)
    bases = {game.base_name(m.species) for _, m in seen} | {to_id_str(m.species) for _, m in seen}
    preview = [
        (None, m)
        for m in battle.teampreview_opponent_team
        if to_id_str(m.species) not in bases and game.base_name(m.species) not in bases
    ]
    return (seen + preview)[:6]


def their_rows(battle, log, catalog, tracker) -> list[Row]:
    members = _their_members(battle, log)
    revealed = sum(1 for k, _ in members if k is not None)
    rows = []
    for k, mon in members:
        chosen = assumed(battle, mon, catalog, tracker)
        shown = revealed_moves(mon)[:4] if k is not None else []
        ids = (shown + [m for m in chosen.moves if m not in shown])[:4]
        moves = [mon.moves.get(i) or move_object(i) for i in ids]
        active = k is not None and mon is battle.opponent_active_pokemon and not mon.fainted
        rows.append(
            Row(
                mon,
                False,
                key=k,
                chosen=chosen,
                brought=float(k is not None),
                not_brought=float(k is None and revealed >= 3),
                active=active,
                moves=moves,
                known=[i in shown for i in ids],
                fighter=set_fighter(battle, mon, chosen),
            ).resolve(battle)
        )
    rows += [Row(None, False) for _ in range(6 - len(rows))]
    return rows


def _encode(ctx: _Ctx, mine: list[Row], theirs: list[Row], legal, cache: dict | None = None) -> tuple:
    """(rows, ids, field). My rows face their active once it has been seen in battle; theirs face my active (at
    preview: the lead). `cache` reuses row encodings across preview picks."""
    rows = np.zeros((ROWS, POKEMON_FLOATS), dtype=np.float32)
    ids = np.zeros((ROWS, POKEMON_IDS), dtype=np.int64)
    my_opp = theirs[0] if theirs[0].mon is not None and theirs[0].key is not None else None
    their_opp = mine[0] if mine[0].mon is not None else None
    for i, row in enumerate(mine + theirs):
        opp = my_opp if i < 6 else their_opp
        k = (id(row), id(opp))
        if cache is None or k not in cache:
            encoded = features_rows.encode(ctx.battle, row, opp, ctx)
            if cache is None:
                rows[i], ids[i] = encoded
                continue
            cache[k] = encoded
        rows[i], ids[i] = cache[k]
    field = features_field.encode(ctx.battle, mine, theirs, legal, ctx.log, rows)
    return _clean(rows), ids, _clean(field)


def _clean(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-2.0).astype(np.float32, copy=False)


def observe(battle, legal: Sequence[LegalAction], *, catalog, tracker) -> Obs:
    """The observation of one decision."""
    if tracker is not None:
        tracker.observe(battle)
    log = features_log.read(battle)
    mine, theirs = my_rows(battle, legal), their_rows(battle, log, catalog, tracker)
    rows, ids, field = _encode(_Ctx(battle, log, legal), mine, theirs, legal)
    _remember_bench(battle)
    mask = np.zeros(ACTIONS, dtype=bool)
    for a in legal:
        mask[action_index(a, battle)] = True
    return Obs(rows, ids, field, mask)


_BENCH: dict[int, int] = {}  # id(Pokemon) -> bench slot, from the battle of the last observe()


def _remember_bench(battle) -> None:
    _BENCH.clear()
    _BENCH.update({id(m): i for i, m in enumerate(_request_members(battle)[1:])})


def action_index(legal_action: LegalAction, battle=None) -> int:
    """0-3 move slot, 4-7 move slot + Mega, 8-10 switch to bench slot (request order minus the active). A switch
    is resolved against `battle`, else against the battle of the last observe()."""
    a = legal_action.action
    if a.kind == "move":
        return (MEGA_AT if a.mega else 0) + min(3, max(0, a.move or 0))
    target = getattr(legal_action.order, "order", None)
    if battle is not None:
        bench = _request_members(battle)[1:]
        slot = next((i for i, m in enumerate(bench) if m is target), -1)
    else:
        slot = _BENCH.get(id(target), -1)
    if slot < 0:
        slot = a.slot or 0
    return SWITCH_AT + min(2, max(0, slot))


def legal_action(index: int, legal: Sequence[LegalAction], battle=None) -> LegalAction:
    """The legal action with this index; ValueError when none."""
    for a in legal:
        if action_index(a, battle) == index:
            return a
    raise ValueError(f"action {index} is not legal here")


def preview_observations(battle, picks: Sequence[tuple[int, int, int]], *, catalog, tracker) -> list[Obs]:
    """Turn-0 observations, one per (lead, second, third) pick: our trio with the lead active, their six."""
    team = list(battle.team.values())
    log = features_log.read(battle)
    ctx, cache = _Ctx(battle, log, ()), {}
    theirs = their_rows(battle, log, catalog, tracker)
    lead_rows = {i: my_rows(battle, (), [team[i]], active=0)[0] for i in {p[0] for p in picks}}
    bench_rows = {i: my_rows(battle, (), [team[i]], active=None)[0] for p in picks for i in p[1:]}
    out = []
    for lead, *rest in picks:
        mine = [lead_rows[lead], *(bench_rows[i] for i in rest)]
        mine += [Row(None, True) for _ in range(6 - len(mine))]
        rows, ids, field = _encode(ctx, mine, theirs, (), cache)
        rows[0, 147] = 1.0  # the lead
        out.append(Obs(rows, ids, field, np.zeros(ACTIONS, dtype=bool)))
    return out

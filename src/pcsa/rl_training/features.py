# From research/pcsa-rl/rl/features.py (sha256 61303b72c917, 2026-11-08 20:15); keep diffable.
# pcsa: the first network's own observation (their_rows with assumed sets, observe, preview_observations) is
# removed with that network; what is left is what former_features builds on.
"""Shared parts of a decision observation: our rows, their members in canonical order, the per-observation damage
cache and the action mapping, from the poke-env Battle a pcsa player sees (always the p1 view).

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
from pcsa.players.opponent import own_fighter
from pcsa.rl_training import features_rows
from pcsa.rl_training.features_moves import hits, pp_fraction
from pcsa.rl_training.features_rows import Row

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


def _clean(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=2.0, neginf=-2.0).astype(np.float32, copy=False)


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

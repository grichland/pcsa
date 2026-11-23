"""Rule masks used while training: what a seat may not choose because of what it has just done.

  - the same move of the same Pokémon at most `max_same_move` turns in a row: then that move is closed while another
    move is open (a Pokémon with one usable move, as when locked into it, is left alone)
  - at most `max_switches` switches by choice in a row: then switching is closed while a move is open

A replacement after a faint is not a choice: it neither counts nor ends a run of switches. The rules are applied to
the mask before a net chooses, and the closed mask is the one stored, so the update sees what the net saw. The live
player needs the same rules to play as trained.

Copied from pcsa-former (pcsa_former/rules.py); behaviour is kept identical to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import layout as L


@dataclass(frozen=True)
class RuleConfig:
    max_same_move: int = 4  # 0 = no such rule
    max_switches: int = 3  # 0 = no such rule


class Rules:
    """What each seat of each game has just done, and the masks that follow from it."""

    def __init__(self, cfg: RuleConfig, layout: L.Layout, games: int) -> None:
        self.cfg = cfg
        self.active_col, self.species_col = layout.col(L.F_IS_ACTIVE), layout.col(L.F_SPECIES)
        self.move_cols = layout[L.F_MOVE_ID].columns()
        self.key = np.zeros((games, 2, 2), np.int64)  # (species, move) of the run of one move
        self.same = np.zeros((games, 2), np.int64)  # its length
        self.switches = np.zeros((games, 2), np.int64)  # switches by choice in a row
        self.closed_moves = self.closed_switches = 0  # decisions a rule closed something for

    def _active(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """(species [N], move ids [N, 4]) of the asked seat's Pokémon on the field."""
        e = np.arange(len(obs["kind"]))
        row = np.asarray(obs["rows"])[:, : L.OWN_PICKS, self.active_col].argmax(-1)
        ids = np.asarray(obs["row_ids"])[e, row]
        return ids[:, self.species_col].astype(np.int64), ids[:, self.move_cols].astype(np.int64)

    def mask(self, obs: dict) -> np.ndarray:
        """obs["mask"] [N, 11] with the rules applied."""
        m = np.array(obs["mask"], copy=True)
        e, s = np.arange(len(m)), np.asarray(obs["seat"]).astype(np.int64)
        turn = np.asarray(obs["kind"]) == L.KIND_TURN
        moves_open = m[:, :8] > 0
        if self.cfg.max_same_move:
            species, moves = self._active(obs)
            due = turn & (self.same[e, s] >= self.cfg.max_same_move) & (self.key[e, s, 0] == species)
            slot = (moves == self.key[e, s, 1][:, None]) & (moves != 0) & due[:, None]
            close = np.concatenate([slot, slot], 1)  # with and without Mega
            close &= (moves_open & ~close).any(1)[:, None]  # only while another move is open
            self.closed_moves += int((close & moves_open).any(1).sum())
            m[:, :8][close] = 0
            moves_open = m[:, :8] > 0
        if self.cfg.max_switches:
            due = turn & (self.switches[e, s] >= self.cfg.max_switches) & moves_open.any(1)
            self.closed_switches += int((due & (m[:, 8:] > 0).any(1)).sum())
            m[due, 8:] = 0
        return m

    def record(self, obs: dict, action: np.ndarray) -> None:
        """The actions chosen for `obs` (every game's asked seat)."""
        action = np.asarray(action).astype(np.int64)
        e, s = np.arange(len(action)), np.asarray(obs["seat"]).astype(np.int64)
        turn = np.asarray(obs["kind"]) == L.KIND_TURN
        move, switch = turn & (action < 8), turn & (action >= 8)
        species, moves = self._active(obs)
        used = moves[e, np.clip(action, 0, 7) % L.MOVE_SLOTS]
        again = move & (self.key[e, s, 0] == species) & (self.key[e, s, 1] == used)
        self.same[e, s] = np.where(move, np.where(again, self.same[e, s] + 1, 1), np.where(switch, 0, self.same[e, s]))
        self.key[e, s, 0] = np.where(move, species, self.key[e, s, 0])
        self.key[e, s, 1] = np.where(move, used, self.key[e, s, 1])
        self.switches[e, s] = np.where(switch, self.switches[e, s] + 1, np.where(move, 0, self.switches[e, s]))

    def reset(self, games: np.ndarray) -> None:
        """Games that ended: nothing done yet in the next ones."""
        self.key[games], self.same[games], self.switches[games] = 0, 0, 0

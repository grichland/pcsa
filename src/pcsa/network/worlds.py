"""The search's guesses of the opponent's hidden team ("worlds"): what the Battle showed is kept, the rest is drawn
from the belief head's outputs inside the species' learnset, with the commonest known set of the species for
nature and stat points. Each guess becomes an engine Game (Champions Gym 0.5.0 or later); a guess the Game still
changes after one repair is dropped.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import layout as L
from . import vocab

TOP = 3  # items and abilities are drawn from the belief head's likeliest few
BROUGHT, MOVES_FIELD = "brought", "moves"


class NoSearch(Exception):
    """Search cannot answer this request; the message says why."""


def to_id(text: str) -> str:
    return "".join(c for c in str(text).lower() if c.isalnum())


def common_sets(builds: dict | None) -> dict[str, dict]:
    """species id -> the commonest set of a top-builds file (ids): moves, item, ability, nature, points."""
    out: dict[str, dict] = {}
    for name, entry in ((builds or {}).get("species") or {}).items():
        sets = entry.get("builds") or []
        if sets:
            s = sets[0]
            out[to_id(name)] = {
                "moves": [to_id(m) for m in s.get("moves") or []],
                "item": to_id(s.get("item") or ""),
                "ability": to_id(s.get("ability") or ""),
                "nature": to_id(s.get("nature") or ""),
                "points": dict(s.get("points") or {}),
            }
    return out


@dataclass
class Belief:
    """The belief head's outputs for one decision, per observation row 6-11: probabilities and the row's species."""

    species: list[str | None]
    brought: np.ndarray  # [6]
    moves: np.ndarray  # [6, moves]
    item: np.ndarray  # [6, items]
    ability: np.ndarray  # [6, abilities]

    def row(self, names: Sequence[str | None]) -> int | None:
        wanted = [n for n in names if n]
        for j, s in enumerate(self.species):
            if s and s in wanted:
                return j
        for j, s in enumerate(self.species):
            if s and any(s.startswith(n) or n.startswith(s) for n in wanted):
                return j
        return None


@dataclass
class Tables:
    """Vocabulary look-ups and the learnset, shared by every decision."""

    learnset: np.ndarray  # bool [species, moves]
    sets: dict[str, dict]
    species_id: dict[str, int] = field(default_factory=dict)
    move_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = vocab.tables()
        self.species_id = {n: i for i, n in enumerate(t["species"]) if n}
        self.move_id = {n: i for i, n in enumerate(t["moves"]) if n}
        self.moves, self.items, self.abilities = t["moves"], t["items"], t["abilities"]

    def learnable(self, names: Sequence[str | None]) -> np.ndarray | None:
        for n in names:
            i = self.species_id.get(n or "")
            if i is not None and i < len(self.learnset) and self.learnset[i].any():
                return self.learnset[i]
        return None


def _draw(rng: np.random.Generator, p: np.ndarray, k: int) -> list[int]:
    """Up to k different indices drawn in proportion to p."""
    p = np.clip(np.asarray(p, np.float64), 0.0, None)
    live = int((p > 0).sum())
    if k <= 0 or live == 0:
        return []
    return [int(i) for i in rng.choice(len(p), size=min(k, live), replace=False, p=p / p.sum())]


def _top(p: np.ndarray, k: int) -> np.ndarray:
    """p with all but its k largest entries (and id 0, unknown) at zero."""
    q = np.array(p, np.float64)
    q[0] = 0.0
    if len(q) > k:
        q[np.argsort(q)[:-k]] = 0.0
    return q


def guess_set(p: Any, slot: int, species: str, belief: Belief, tables: Tables, rng: np.random.Generator) -> dict:
    """One opposing Pokémon's set: what the Battle showed, the rest drawn from the belief head inside the learnset;
    nature and stat points (and anything the belief cannot give) from the species' commonest set."""
    names = [p.form, p.species, species]
    row = belief.row(names)
    common = next((tables.sets[n] for n in names if n and n in tables.sets), {})
    known = [str(m) for m, _ in (p.moves or ())][: L.MOVE_SLOTS]
    need = L.MOVE_SLOTS - len(known)
    can = tables.learnable(names)
    moves = list(known)
    if need > 0 and row is not None and can is not None:
        prob = np.where(can[: belief.moves.shape[1]], belief.moves[row][: len(can)], 0.0)
        prob[0] = 0.0
        for m in known:
            if m in tables.move_id and tables.move_id[m] < len(prob):
                prob[tables.move_id[m]] = 0.0
        moves += [tables.moves[i] for i in _draw(rng, prob, need)]  # type: ignore[misc]
    ids = tables.move_id
    fits = [m for m in common.get("moves", []) if can is None or (m in ids and ids[m] < len(can) and can[ids[m]])]
    moves += [m for m in fits if m not in moves][: L.MOVE_SLOTS - len(moves)]
    out: dict[str, Any] = {"slot": slot, "moves": moves}
    if p.item is None:
        drawn = _draw(rng, _top(belief.item[row], TOP), 1) if row is not None else []
        out["item"] = (tables.items[drawn[0]] or "") if drawn else common.get("item", "")
    if p.ability is None:
        drawn = _draw(rng, _top(belief.ability[row], TOP), 1) if row is not None else []
        ability = tables.abilities[drawn[0]] if drawn else common.get("ability")
        if ability:
            out["ability"] = ability
    if common.get("nature"):
        out["nature"] = common["nature"]
    if common.get("points"):
        out["points"] = common["points"]
    return out


def guess_team(
    battle: Any, belief: Belief, tables: Tables, rng: np.random.Generator, lead: str | None
) -> tuple[list[dict], list[str]]:
    """A guess at their three, lead first: (the sets for Session.game, each one's species)."""
    preview = [str(s) for s in battle.opponent_preview]
    team = list(battle.opponent_team)
    seen = [i for i, p in enumerate(team) if p.revealed][: L.OWN_PICKS]
    hidden = [i for i, p in enumerate(team) if not p.revealed and i < len(preview)]
    if len(seen) < L.OWN_PICKS and hidden:
        weight = []
        for i in hidden:
            row = belief.row([preview[i]])
            weight.append(float(belief.brought[row]) if row is not None else 0.5)
        picked = _draw(rng, np.maximum(np.array(weight), 1e-6), L.OWN_PICKS - len(seen))
        seen = seen + [hidden[k] for k in picked]
    if len(seen) != L.OWN_PICKS:
        raise NoSearch(f"only {len(seen)} of their three can be named")

    def first(i: int) -> tuple:
        p = team[i]
        led = lead is not None and lead in (p.species, p.form, preview[i])
        return (not led, not p.active, not p.revealed)

    order = sorted(seen, key=first)
    return [guess_set(team[i], i, preview[i], belief, tables, rng) for i in order], [preview[i] for i in order]


def repair(guess: list[dict], species: list[str], rejected: Sequence, shown: dict[int, set] | None = None) -> None:
    """Take out of the guess what the Game refused, so the guess is the set the Game then holds. A move the
    Battle showed (`shown`, by preview slot) is never taken out: a Game that refuses one is not this battle."""
    for name, part, reason in rejected:
        at = [k for k, s in enumerate(species) if s == name or s.startswith(str(name)) or str(name).startswith(s)]
        if not at or part == BROUGHT:
            continue
        g = guess[at[0]]
        if part == MOVES_FIELD:
            words = set(str(reason).replace(";", " ").replace(",", " ").split())
            keep = (shown or {}).get(g["slot"], set())
            g["moves"] = [m for m in g["moves"] if m not in words or m in keep]
        else:
            g.pop(part, None)


@dataclass
class World:
    guess: list[dict]
    species: list[str]  # by the Game's team slot
    game: Any
    root: Any = None
    sims: int = 0

    def moves(self, slot: int) -> list[str]:
        return self.guess[slot]["moves"] if 0 <= slot < len(self.guess) else []


def build_worlds(battle: Any, belief: Belief, tables: Tables, make: Callable, count: int,
                 rng: np.random.Generator, lead: str | None) -> tuple[list[World], list[str]]:  # fmt: skip
    """Up to `count` different guesses as Games. A guess the Game changes (a set part refused) is repaired and
    tried once more, and dropped if the Game still changes it."""
    worlds, dropped, seen = [], [], set()
    for _ in range(count):
        guess, species = guess_team(battle, belief, tables, rng, lead)
        key = json.dumps(guess, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        try:
            game, rejected = make(battle, guess, int(rng.integers(1 << 62)))
            if any(r[1] != BROUGHT for r in rejected):  # once more without the parts the Game refused
                team = list(battle.opponent_team)
                shown = {g["slot"]: {str(m) for m, _ in team[g["slot"]].moves or ()} for g in guess}
                repair(guess, species, rejected, shown)
                game, rejected = make(battle, guess, int(rng.integers(1 << 62)))
        except Exception as error:
            dropped.append(f"{type(error).__name__}: {error}")
            continue
        changed = [r for r in rejected if r[1] != BROUGHT]
        if changed:
            dropped.append("; ".join(f"{r[0]} {r[1]}: {r[2]}" for r in changed))
            continue
        for r in rejected:  # the one that led is moved first in the State
            at = [k for k, s in enumerate(species) if s == r[0] or s.startswith(r[0]) or str(r[0]).startswith(s)]
            if at:
                guess.insert(0, guess.pop(at[0]))
                species.insert(0, species.pop(at[0]))
        worlds.append(World(guess, species, game))
    return worlds, dropped

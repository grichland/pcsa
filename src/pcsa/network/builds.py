"""Whole-build guesses of the opponent's hidden sets (`worlds_mode: builds`): each hidden Pokémon's set is one
build from the builds we know (assets/sets/seen_builds.json and top_builds.json), so item, ability, moves, nature
and stat points come together, plus one "other" candidate drawn part by part from the belief head.

- candidates are filtered by what the battle showed (moves, the item even once consumed or knocked off, the
  ability, Mega Evolution) and by the species' legal abilities; when none is left, only "other" stays.
- weight = prior (sightings, plus one sighting's worth for the top builds together) x belief (the head's
  probabilities of the build's item, ability and hidden moves) x evidence (turn order seen this battle, floored).
- a team is drawn build by build under Item Clause, never repeating a teammate's shown item. Several Mega stones
  on one team are allowed (77% of the live teams in pcsa-former's pool hold two or more; only one can Mega Evolve,
  which the engine enforces).

Turn-order evidence is evidence.py's. Damage is not used: Champions Gym exposes no damage calculation to Python.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pcsa.common.paths import assets_dir

from . import vocab, worlds
from .evidence import STATS, Tracker, points_of, stat_from_points, to_id

BELIEF_FLOOR = 1e-4  # the belief head's probability of a build part never counts as less than this
OTHER = "other"


@dataclass
class Build:
    item: str  # "" holds nothing
    ability: str
    nature: str
    points: tuple[int, ...]  # by STATS
    moves: tuple[str, ...]
    prior: float
    label: str

    def as_set(self, slot: int) -> dict:
        return {"slot": slot, "moves": list(self.moves), "item": self.item, "ability": self.ability,
                "nature": self.nature, "points": dict(zip(STATS, self.points, strict=True))}  # fmt: skip


@dataclass
class Dex:
    """CG's species, nature and move tables (assets/game/cg)."""

    base: dict[str, dict[str, int]] = field(default_factory=dict)
    abilities: dict[str, set[str]] = field(default_factory=dict)
    natures: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)
    nature_names: dict[str, str] = field(default_factory=dict)
    priority: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls) -> Dex:
        root = assets_dir() / "game" / "cg"
        read = lambda name: json.loads((root / f"{name}.json").read_text(encoding="utf-8"))[name]  # noqa: E731
        dex = cls()
        for sid, s in read("species").items():
            dex.base[sid] = dict(s.get("baseStats") or {})
            dex.abilities[sid] = {to_id(a) for a in (s.get("abilities") or {}).values()}
        for n in read("natures"):
            dex.natures[n["id"]] = (n.get("plus"), n.get("minus"))
            dex.nature_names[n["id"]] = n["name"]
        dex.priority = {mid: int(m.get("priority") or 0) for mid, m in read("moves").items()}
        return dex

    def speed(self, species: str, build_nature: str, points: Sequence[int]) -> int | None:
        base = self.base.get(species)
        if not base or "spe" not in base:
            return None
        return stat_from_points(int(base["spe"]), int(points[5]), "spe", self.natures.get(build_nature, (None, None)))


def _read_sets(name: str) -> dict:
    path = assets_dir() / "sets" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_catalog(seen: dict | None = None, top: dict | None = None) -> dict[str, list[Build]]:
    """species id -> its known builds, merged: a seen build's prior is its sightings, the top builds of a species
    share one sighting's worth by their ladder score."""
    seen = _read_sets("seen_builds.json") if seen is None else seen
    top = _read_sets("top_builds.json") if top is None else top
    merged: dict[str, dict[tuple, Build]] = {}

    def add(species: str, b: dict, prior: float, label: str) -> None:
        moves = tuple(sorted({to_id(m) for m in b.get("moves") or [] if m}))
        if not moves:
            return
        key = (to_id(b.get("item")), to_id(b.get("ability")), to_id(b.get("nature")), points_of(b.get("points")),
               moves)
        found = merged.setdefault(species, {}).get(key)
        if found is None:
            merged[species][key] = Build(key[0], key[1], key[2], key[3], moves, prior, label)
        else:
            found.prior += prior

    for sid, builds in (seen.get("species") or {}).items():
        for b in builds:
            add(to_id(sid), b, float(b.get("count") or 1), "seen")
    for name, entry in (top.get("species") or {}).items():
        builds = entry.get("builds") or []
        shares = np.array([10.0 ** float(b.get("score") or 0.0) for b in builds])
        shares = shares / shares.sum() if shares.sum() > 0 else shares
        for b, share in zip(builds, shares, strict=True):
            add(to_id(name), b, float(share), "top")
    return {s: list(v.values()) for s, v in merged.items()}


@dataclass
class Facts:
    """What the battle showed of one opposing Pokémon (the Battle's, plus the items the events named)."""

    moves: set[str]
    item: str | None  # None: not known (an item that is gone counts as not known unless an event named it)
    ability: str | None
    mega: bool  # it Mega Evolved, so it holds a Mega stone


def facts_of(p: Any, had_item: str | None) -> Facts:
    item = had_item or (to_id(p.item) if p.item else None)
    mega = "mega" in to_id(getattr(p, "form", "") or "")
    return Facts({to_id(m) for m, _ in (p.moves or ())}, item, to_id(p.ability) or None, mega)


def fits(b: Build, f: Facts, legal: set[str] | None) -> bool:
    """Whether build `b` agrees with every shown fact and with the species' legal abilities."""
    if not f.moves <= set(b.moves) or (f.item is not None and b.item != f.item):
        return False
    if f.mega and b.item not in set(vocab.mega_stones()):
        return False
    if f.ability and b.ability != f.ability:
        return False
    return not (legal and b.ability and b.ability not in legal)


def describe(b: Build | None) -> str:
    if b is None:
        return OTHER
    return f"{b.item or '-'}/{b.ability}/{b.nature}/spe{b.points[5]}/{'+'.join(b.moves)}"


class Guesser:
    """Weights and team draws over the catalog, for one policy."""

    def __init__(self, tables: worlds.Tables, dex: Dex | None = None, catalog: dict | None = None) -> None:
        self.tables, self.dex = tables, dex or Dex.load()
        self.catalog = catalog if catalog is not None else load_catalog()
        self.item_ix = {n: i for i, n in enumerate(tables.items) if n}
        self.ability_ix = {n: i for i, n in enumerate(tables.abilities) if n}

    def _p(self, probs: np.ndarray, ix: dict[str, int], key: str) -> float:
        i = ix.get(key)
        return max(float(probs[i]), BELIEF_FLOOR) if i is not None and i < len(probs) else BELIEF_FLOOR

    def belief_loglik(self, b: Build, belief: worlds.Belief, row: int | None, shown: set[str]) -> float:
        if row is None:
            return 0.0
        total = np.log(self._p(belief.item[row], self.item_ix, b.item or "nothing"))
        total += np.log(self._p(belief.ability[row], self.ability_ix, b.ability))
        for m in b.moves:
            if m not in shown:
                total += np.log(self._p(belief.moves[row], self.tables.move_id, m))
        return float(total)

    def other_loglik(self, belief: worlds.Belief, row: int | None, shown: set[str]) -> float:
        """'other' is drawn from the head part by part: weigh it as the head's likeliest such build."""
        if row is None:
            return 0.0
        hidden = max(0, 4 - len(shown))
        probs = belief.moves[row]
        drop = [i for m in shown if (i := self.tables.move_id.get(m, len(probs))) < len(probs)]
        top = np.sort(np.delete(probs, drop))[::-1][:hidden]
        return float(np.log(max(belief.item[row][1:].max(), BELIEF_FLOOR))
                     + np.log(max(belief.ability[row][1:].max(), BELIEF_FLOOR))
                     + np.log(np.maximum(top, BELIEF_FLOOR)).sum())  # fmt: skip

    def weigh(self, p: Any, name: str, belief: worlds.Belief, tracker: Tracker | None
              ) -> tuple[list[Build | None], np.ndarray, int]:  # fmt: skip
        """Candidates (None = other) and their normalised weights; the number of known builds that fit."""
        species = to_id(p.species) or name
        mon = to_id(getattr(p, "species", "") or name)
        facts = facts_of(p, tracker.had_item.get(mon) if tracker else None)
        legal = self.dex.abilities.get(species) or self.dex.abilities.get(name)
        builds = self.catalog.get(species) or self.catalog.get(name) or []
        kept = [b for b in builds if fits(b, facts, legal)]
        row = belief.row([getattr(p, "form", None), p.species, name])
        logw = [np.log(b.prior) + self.belief_loglik(b, belief, row, facts.moves)
                + (tracker.order_loglik(species, b) if tracker else 0.0) for b in kept]  # fmt: skip
        logw.append(self.other_loglik(belief, row, facts.moves))  # one sighting's worth
        w = np.exp(np.array(logw) - max(logw))
        return [*kept, None], w / w.sum(), len(kept)

    def guess_team(self, team: Sequence[tuple[int, Any, str]], belief: worlds.Belief, tracker: Tracker | None,
                   rng: np.random.Generator, shown_items: Sequence[str] = ()) -> tuple[list[dict], list[dict]]:
        """One set per (slot, Pokémon, preview name), in order, under Item Clause, never a teammate's shown item:
        (the sets, a note per Pokémon)."""
        used, sets, notes = set(shown_items), [], []
        for slot, p, name in team:
            cands, w, known = self.weigh(p, name, belief, tracker)
            own_item = facts_of(p, tracker.had_item.get(to_id(p.species)) if tracker else None).item
            ok = np.array([c is None or c.item == own_item or not self._clash(c.item, used) for c in cands])
            pick_w = np.where(ok, w, 0.0)
            k = int(rng.choice(len(cands), p=pick_w / pick_w.sum())) if pick_w.sum() > 0 else len(cands) - 1
            chosen = cands[k]
            if chosen is None:
                out = worlds.guess_set(p, slot, name, belief, self.tables, rng)
                self._legal_ability(out, p, name, belief)
                if out.get("item") and out["item"] != own_item and self._clash(out["item"], used):
                    out["item"] = ""
            else:
                out = chosen.as_set(slot)
            if out.get("item"):
                used.add(out["item"])
            sets.append(out)
            top = np.argsort(-w)[:3]
            ranked = [[describe(cands[i]), round(float(w[i]), 4)] for i in top]
            notes.append({"species": name, "n": known, "top": ranked, "drawn": describe(chosen)})
        return sets, notes

    @staticmethod
    def _clash(item: str, used: set[str]) -> bool:
        return bool(item) and item in used

    def _legal_ability(self, out: dict, p: Any, name: str, belief: worlds.Belief) -> None:
        legal = self.dex.abilities.get(to_id(p.species) or name) or self.dex.abilities.get(name)
        if not legal or p.ability or out.get("ability") in legal:
            return
        row = belief.row([getattr(p, "form", None), p.species, name])
        ranked = sorted(legal, key=lambda a: -(self._p(belief.ability[row], self.ability_ix, a) if row is not None
                                               else 0.0))  # fmt: skip
        out["ability"] = ranked[0]

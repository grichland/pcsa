# From research/pcsa-belief/belief/space.py (sha256 027bb2a0ac64, 2026-11-12 12:19); keep diffable.
# pcsa changes are marked "pcsa:".
"""The hypothesis space of one opposing species: weighted rows (spread, item, ability, moves source).

A row is one complete stat-relevant build: a spread (stat points + nature), the item it brought, its base ability,
and where its moves come from (a listed build, or the species' generic move frequencies). Rows come from four
components, each with a prior mass (Priors):

  A  "list"    the listed builds (seen on the ladder first, then the usage catalog), exactly
  B  "spread"  a listed build's item, ability and moves with any spread of the spread table
  C  "item"    a listed build's spread, ability and moves with any item the species can hold
  D  "free"    any spread x any item x any ability, generic moves (the factored component)

The spread table is species-independent: every spread of the catalog and the seen builds, the simulator's
perturbation spreads, and a grid of "standard" patterns (each stat at 0 or 32 points, at most two maxed) under each
of the 21 nature effects (neutral, or + one stat and - another). Stats of a row in a form are computed from that
form's base stats (pcsa.battle.damage.build_stats arithmetic, vectorised).
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.builds import Build, Catalog
from pcsa.battle.damage import nature_percent
from pcsa.battle.stats import IV, NATURE_DOWN, NATURE_NEUTRAL, NATURE_UP, POINTS_MAX
from pcsa.battle.teams import STAT_ORDER

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
assert tuple(STAT_ORDER) == STATS
NAT_STATS = STATS[1:]
# nature effects: index 0 neutral, then (plus, minus) pairs over the five non-HP stats
EFFECTS: list[tuple[str | None, str | None]] = [(None, None), *itertools.permutations(NAT_STATS, 2)]
# the simulator's perturbation spreads (research/pcsa-rl/rl2/teams.py SPREADS), copied
PERTURB_SPREADS = (
    ({"hp": 2, "atk": 32, "spe": 32}, "Jolly"),
    ({"hp": 2, "spa": 32, "spe": 32}, "Timid"),
    ({"hp": 32, "def": 16, "spd": 18}, "Bold"),
    ({"hp": 32, "atk": 32, "def": 2}, "Adamant"),
)
COMPONENTS = ("list", "spread", "item", "free")
GENERIC = -1  # moves source of a free row


def effect_of(nature: str) -> int:
    pct = nature_percent(nature)
    plus = next((s for s in NAT_STATS if pct[s] > NATURE_NEUTRAL), None)
    minus = next((s for s in NAT_STATS if pct[s] < NATURE_NEUTRAL), None)
    if plus is None or minus is None:
        return 0
    return EFFECTS.index((plus, minus))


@lru_cache(maxsize=32)
def effect_name(i: int) -> str:
    from pcsa.battle.stats import nature_name

    plus, minus = EFFECTS[i]
    if plus is None:
        return "Hardy"
    pct = {s: NATURE_UP if s == plus else NATURE_DOWN if s == minus else NATURE_NEUTRAL for s in NAT_STATS}
    return nature_name(pct) or "Hardy"


EFFECT_PCT = np.array(
    [[NATURE_UP if s == p else NATURE_DOWN if s == m else NATURE_NEUTRAL for s in NAT_STATS] for p, m in EFFECTS],
    dtype=np.int64,
)  # [21, 5]


@dataclass(frozen=True)
class Priors:
    """Prior masses and evidence tolerances (README.md explains each)."""

    list_mass: float = 0.45
    spread_mass: float = 0.20
    item_mass: float = 0.15
    free_mass: float = 0.20
    seen_share: float = 0.6  # of the list mass, when the species has both seen and catalog builds
    grid_share: float = 0.4  # of the spread prior: the standard grid (rest: listed spreads by frequency)
    item_species_share: float = 0.6  # of the item prior: the species' listed items (rest: global usage)
    ability_list_share: float = 0.7
    move_list_share: float = 0.8  # generic move frequencies: listed builds (rest: uniform over the learnset)
    move_swap: float = 0.15  # a build-tied row showing a move its build lacks: x move_swap x P(move)
    eps_order: float = 0.01
    eps_dealt: float = 0.02
    eps_taken: float = 0.04
    slack_dealt: float = 1.0  # percent points (their HP is read in whole percent)
    slack_taken: float = 1.0  # HP
    rel_slack: float = 0.01
    tau_dealt: float = 2.0  # percent points
    tau_taken: float = 0.02  # fraction of our max HP
    crit_unreported: float = 1 / 24  # live: no -crit lines, so any hit may be a critical hit
    prune: float = 40.0  # rows more than this many nats below the best are dropped (numerically zero)
    role_power: float = 0.0  # off-list spreads weighted by the species' listed investment pattern (0: off)
    role_floor: float = 0.15


# ------------------------------------------------------------------------------------------------ spread table
@dataclass
class SpreadTable:
    points: np.ndarray  # [S, 6] int
    effect: np.ndarray  # [S] int (index into EFFECTS)
    prior: np.ndarray  # [S] float
    index: dict  # (points tuple, effect) -> row

    def add(self, points: tuple, effect: int) -> int:
        key = (tuple(int(p) for p in points), int(effect))
        if key in self.index:
            return self.index[key]
        self.index[key] = len(self.effect)
        self.points = np.vstack([self.points, np.asarray(key[0], dtype=np.int64)[None]])
        self.effect = np.append(self.effect, key[1])
        self.prior = np.append(self.prior, 0.0)
        return self.index[key]


def _builds_everywhere(catalog: Catalog) -> list[Build]:
    out = [b for builds in catalog.by_id.values() for b in builds]
    if catalog.seen is not None:
        out += [s.build for entries in catalog.seen.by_species.values() for s in entries]
    return out


def spread_table(catalog: Catalog, priors: Priors) -> SpreadTable:
    emp: collections.Counter = collections.Counter()
    nat: collections.Counter = collections.Counter()
    for b in _builds_everywhere(catalog):
        e = effect_of(b.nature)
        emp[(tuple(int(p) for p in b.points), e)] += 1.0  # pyright: ignore[reportArgumentType]
        nat[e] += 1.0  # pyright: ignore[reportArgumentType]
    for pts, nature in PERTURB_SPREADS:
        emp[(tuple(pts.get(s, 0) for s in STATS), effect_of(nature))] += 1.0  # pyright: ignore[reportArgumentType]
    total_nat = sum(nat.values()) or 1.0
    p_nat = np.array([0.7 * nat[e] / total_nat + 0.3 / len(EFFECTS) for e in range(len(EFFECTS))])
    grid: dict = {}
    patterns = [()] + [(s,) for s in STATS] + list(itertools.combinations(STATS, 2))
    for maxed in patterns:
        pts = tuple(POINTS_MAX if s in maxed else 0 for s in STATS)
        for e in range(len(EFFECTS)):
            grid[(pts, e)] = p_nat[e] / len(patterns)
    keys = list(dict.fromkeys([*emp.keys(), *grid.keys()]))
    total_emp = sum(emp.values())
    prior = np.array(
        [(1 - priors.grid_share) * emp.get(k, 0.0) / total_emp + priors.grid_share * grid.get(k, 0.0) for k in keys]
    )
    prior = prior / prior.sum()
    return SpreadTable(
        np.array([k[0] for k in keys], dtype=np.int64),
        np.array([k[1] for k in keys], dtype=np.int64),
        prior,
        {k: i for i, k in enumerate(keys)},
    )


def stats_of(points: np.ndarray, effect: np.ndarray, base: dict) -> np.ndarray:
    """[S, 6] level-50 stats (Champions rules: pcsa.battle.stats.stat_value / hp_value, vectorised)."""
    level = game.LEVEL
    out = np.empty((len(effect), 6), dtype=np.int64)
    out[:, 0] = ((2 * base["hp"] + IV + 2 * points[:, 0]) * level) // 100 + level + 10
    pct = EFFECT_PCT[effect]
    for j, s in enumerate(NAT_STATS, start=1):
        raw = ((2 * base[s] + IV + 2 * points[:, j]) * level) // 100 + 5
        out[:, j] = raw * pct[:, j - 1] // 100
    return out


# ------------------------------------------------------------------------------------------------ species space
@lru_cache(maxsize=1)
def legal_items() -> tuple[str, ...]:
    """Items some catalog build holds (all legal in Champions), the most used first; Mega Stones left out
    (as research/pcsa-rl/rl2/teams.py legal_items, on the plain catalog)."""
    cat = Catalog.load()
    count = collections.Counter(to_id_str(b.item) for builds in cat.by_id.values() for b in builds if b.item)
    return tuple(i for i, _ in count.most_common() if not game.megas()[1].get(i))


@lru_cache(maxsize=1)
def global_item_freq() -> dict[str, float]:
    cat = Catalog.load()
    count = collections.Counter(to_id_str(b.item) for builds in cat.by_id.values() for b in builds if b.item)
    total = sum(count.values())
    return {i: c / total for i, c in count.items()}


def species_abilities(species: str) -> list[str]:
    entry = game.species_entry(species) or game.species_entry(game.base_name(species))
    return [to_id_str(a) for a in (entry.get("abilities") or {}).values()]


def species_stones(species: str) -> list[str]:
    base = to_id_str(game.base_name(species))
    out = []
    for key in game.megas()[0].get(base, []):
        stone = game.species_entry(key).get("requiredItem")
        if stone:
            out.append(to_id_str(stone))
    return out


@dataclass
class ListBuild:
    build: Build
    item: str
    ability: str
    moves: frozenset
    spread: int  # row in the spread table
    prior: float  # within the list
    seen: bool


class SpeciesSpace:
    """The rows of one species (base form) and their prior log-weights."""

    def __init__(self, species: str, catalog: Catalog, table: SpreadTable, priors: Priors) -> None:
        self.species = to_id_str(species)
        self.priors = priors
        self.table = table
        raw = catalog.builds(species)
        seen_counts: dict[int, int] = {}
        if catalog.seen is not None:
            ranked = catalog.seen.ranked(self.species) or catalog.seen.ranked(game.base_name(self.species))
            seen_counts = {id(s.build): max(1, s.count) for s in ranked}
        seen_w = [seen_counts[id(b)] for b in raw if id(b) in seen_counts]
        cat_w = [10**b.score for b in raw if id(b) not in seen_counts]
        share = priors.seen_share if seen_w and cat_w else (1.0 if seen_w else 0.0)
        self.builds: list[ListBuild] = []
        for b in raw:
            is_seen = id(b) in seen_counts
            if is_seen:
                p = share * seen_counts[id(b)] / sum(seen_w)
            else:
                p = (1 - share) * (10**b.score) / sum(cat_w)
            s = table.add(tuple(int(x) for x in b.points), effect_of(b.nature))
            self.builds.append(
                ListBuild(
                    b,
                    to_id_str(b.item),
                    to_id_str(b.ability),
                    frozenset(to_id_str(m) for m in b.moves),
                    s,
                    p,
                    is_seen,
                )
            )
        items = list(legal_items())
        unknown = bool(getattr(priors, "empty_unknown", False))  # pcsa: an empty item is unknown, not "no item"
        for i in [*species_stones(self.species), *(lb.item for lb in self.builds)]:
            if i not in items and not (unknown and not i):  # '' (no item) too, when a listed build has none
                items.append(i)
        self.items = items
        self.item_index = {i: k for k, i in enumerate(items)}
        abilities = species_abilities(self.species)
        for lb in self.builds:
            if lb.ability and lb.ability not in abilities:
                abilities.append(lb.ability)
        self.abilities = abilities or [""]
        self.ability_index = {a: k for k, a in enumerate(self.abilities)}
        self.p_item = self._item_prior()
        self.p_ability = self._ability_prior()
        self.q_move = self._move_freq()
        self._rows()

    # priors ---------------------------------------------------------------------------------------------------
    def _item_prior(self) -> np.ndarray:
        pr = self.priors
        glob = global_item_freq()
        own: collections.Counter = collections.Counter()
        for lb in self.builds:
            own[lb.item] += lb.prior  # pyright: ignore[reportArgumentType]
        stones = species_stones(self.species)
        g = np.array([glob.get(i, 0.0) + 1e-3 for i in self.items])
        g = g / g.sum()
        if stones and not any(own[s] for s in stones):  # a Mega species with no listed stone
            for s in stones:
                own[s] += 0.3 / len(stones)  # pyright: ignore[reportArgumentType]
        o = np.array([own.get(i, 0.0) for i in self.items])
        p = pr.item_species_share * o / o.sum() + (1 - pr.item_species_share) * g if o.sum() > 0 else g
        return p / p.sum()

    def _ability_prior(self) -> np.ndarray:
        own: collections.Counter = collections.Counter()
        for lb in self.builds:
            own[lb.ability] += lb.prior  # pyright: ignore[reportArgumentType]
        o = np.array([own.get(a, 0.0) for a in self.abilities])
        u = np.full(len(self.abilities), 1 / len(self.abilities))
        s = self.priors.ability_list_share
        p = s * o / o.sum() + (1 - s) * u if o.sum() > 0 else u
        return p / p.sum()

    def _move_freq(self) -> dict[str, float]:
        """P(move in the set) for a build not in the list: listed builds' frequencies, smoothed by the learnset."""
        learn = (
            game.learnsets().get(self.species) or game.learnsets().get(to_id_str(game.base_name(self.species))) or ()
        )
        own: collections.Counter = collections.Counter()
        tot = sum(lb.prior for lb in self.builds) or 1.0
        for lb in self.builds:
            for m in lb.moves:
                own[m] += lb.prior / tot  # pyright: ignore[reportArgumentType]
        s = self.priors.move_list_share if self.builds else 0.0
        uni = 4.0 / max(1, len(learn))
        return {m: s * own.get(m, 0.0) + (1 - s) * uni for m in set(learn) | set(own)}

    def _role(self, S: int) -> np.ndarray:
        """Per spread: how well its investment (stats with 16+ points) and nature fit the listed builds of the
        species (frequency of each stat being invested, of each stat being raised), floored, to role_power."""
        pr = self.priors
        if not self.builds or pr.role_power <= 0:
            return np.ones(S)
        pts = self.table.points[:S]
        eff = self.table.effect[:S]
        tot = sum(lb.prior for lb in self.builds)
        inv = np.zeros(6)
        up = np.zeros(6)
        for lb in self.builds:
            inv += lb.prior / tot * (np.asarray(lb.build.points) >= 16)
            plus = EFFECTS[effect_of(lb.build.nature)][0]
            if plus is not None:
                up[STATS.index(plus)] += lb.prior / tot
        f = pr.role_floor
        r = np.ones(S)
        for j in range(6):
            has = pts[:, j] >= 16
            r *= np.where(has, f + inv[j], f + 1 - inv[j])
        plus_idx = np.array([STATS.index(EFFECTS[e][0]) if EFFECTS[e][0] else -1 for e in range(len(EFFECTS))])
        pi = plus_idx[eff]
        r *= np.where(pi >= 0, f + up[np.maximum(pi, 0)], f + max(0.0, 1 - up.sum()))
        return r**pr.role_power

    def q(self, move: str) -> float:
        return min(1.0, self.q_move.get(move, 1e-4))

    # rows -----------------------------------------------------------------------------------------------------
    def _rows(self) -> None:
        """pcsa: stage 1's rows (every free row listed: ~110k) are not ported; pcsa.belief.keys.FastSpecies builds
        the rows with the free component dense."""
        raise NotImplementedError("use pcsa.belief.fast.FastSpaces")

    def stats(self, base: dict) -> np.ndarray:
        """[S, 6] stats of every spread of the table on `base` stats."""
        return stats_of(self.table.points, self.table.effect, base)

    def add_item(self, item: str) -> int:
        """An item outside the space (a reveal nothing predicted): appended to the vocabulary."""
        if item not in self.item_index:
            self.item_index[item] = len(self.items)
            self.items.append(item)
        return self.item_index[item]

    def add_ability(self, ability: str) -> int:
        if ability not in self.ability_index:
            self.ability_index[ability] = len(self.abilities)
            self.abilities.append(ability)
        return self.ability_index[ability]


class Spaces:
    """The spread table and the species spaces of one catalog (cached per species and seen-builds version)."""

    def __init__(self, catalog: Catalog, priors: Priors | None = None) -> None:
        self.catalog = catalog
        self.priors = priors or Priors()
        self.table = spread_table(catalog, self.priors)
        self._by: dict[str, SpeciesSpace] = {}
        self._version = getattr(catalog.seen, "version", None)

    def get(self, species: str) -> SpeciesSpace:
        version = getattr(self.catalog.seen, "version", None)
        if version != self._version:
            self._by.clear()
            self._version = version
        key = to_id_str(species)
        if key not in self._by:
            self._by[key] = SpeciesSpace(key, self.catalog, self.table, self.priors)
        return self._by[key]

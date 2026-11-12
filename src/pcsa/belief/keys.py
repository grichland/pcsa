# From research/pcsa-belief/belief/fast.py (sha256 dcd849036429, 2026-11-12 14:38), split into keys, grids, board,
# member_state, member_grids, member_out, member, fast (pcsa's 400-line file limit); pcsa changes are marked "pcsa:".
"""Class keys, priors and species spaces of the belief tracker (overview: pcsa.belief.fast)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from poke_env.data.normalize import to_id_str

from pcsa.battle import damage as D
from pcsa.battle import game
from pcsa.battle.stats import multipliers
from pcsa.belief import calc as K
from pcsa.belief import evidence as E
from pcsa.belief.space import Priors, Spaces, SpeciesSpace
from pcsa.players import opponent as opp

SKIP_POWER = opp.CONDITIONAL_POWER | frozenset(opp.ESCALATING_POWER)
SPEED_POWER = frozenset({"gyroball", "electroball"})
HP_POWER = frozenset("eruption waterspout dragonenergy flail reversal crushgrip wringout hardpress".split())
COMPONENTS = ("list", "spread", "item", "free")
LEVEL_FACTOR = 2 * game.LEVEL // 5 + 2
SNOW = ("snowscape", "snow", "hail")
MEGA_WEATHER = {"drought": "sunnyday", "drizzle": "raindance", "sandstream": "sandstorm", "snowwarning": "snowscape"}


# ------------------------------------------------------------------------------------------------ class keys
# Abilities the calculator reads on the attacking side, on the defending side, and for speed (anything else behaves
# as no ability there). The unmodeled ones (calc.UNMODELED_*) are kept apart: evidence widens their likelihood.
ATT_ABILITIES = (
    frozenset(D.ATTACK_ABILITY)
    | frozenset(D.FLAG_ABILITY)
    | frozenset(D.TYPE_ABILITY)
    | frozenset(D.ATE_ABILITY)
    | frozenset(K.PINCH)
    | K.UNMODELED_ATTACK
    | {
        "technician",
        "sandforce",
        "hustle",
        "guts",
        "solarpower",
        "adaptability",
        "protean",
        "libero",
        "unaware",
        "levitate",
    }
)
DEF_ABILITIES = (
    frozenset(D.DEFENDER_ABILITY)
    | frozenset(D.SOLID)
    | frozenset(K.FULL_HP_HALVE)
    | K.UNMODELED_DEFEND
    | {"furcoat", "icescales", "fluffy", "punkrock", "levitate", "unaware"}
)


def att_item_key(item: str, category: str, move_id: str, types: frozenset) -> str:
    """The attacker's item as rolls() reads it for this move (`types`: the move's type and, for a Normal move,
    every -ate type)."""
    if item in D.ATTACK_ITEM:
        used = "def" if move_id == "bodypress" else ("atk" if category == "physical" else "spa")
        return item if move_id != "foulplay" and D.ATTACK_ITEM[item][0] == used else "*"
    if item in D.FINAL_ITEM or item in ("expertbelt", "airballoon"):
        return item
    if item == "muscleband":
        return item if category == "physical" else "*"
    t = D.TYPE_ITEMS.get(item)
    if t is not None:
        return item if t in types else "*"
    return "*"


def def_item_key(item: str, move_id: str, move_type: str) -> str:
    """The defender's item as rolls() and move_power read it (`move_type`: after the attacker's -ate)."""
    berry = K.RESIST_BERRY.get(item)
    if berry is not None:
        return item if berry == move_type else ("*" if move_id != "knockoff" else "+")
    if item == "airballoon":
        return item
    if move_id == "knockoff":
        return "+" if item else ""  # Knock Off's power reads only whether an item is held
    return "*"


def speed_item_key(item: str, species: str) -> tuple:
    return (multipliers(item, None, species).get("spe", 1.0), item == "ironball")


def speed_ability_key(ability: str, species: str) -> tuple:
    return (
        multipliers(None, ability, species).get("spe", 1.0),
        opp.SPEED_WEATHER.get(ability),
        ability == "quickfeet",
        ability in E.ORDER_ABILITIES,
    )


def att_ability_key(a: str) -> str:
    return a if a in ATT_ABILITIES else "*"


def def_ability_key(a: str) -> str:
    return a if a in DEF_ABILITIES else "*"


# ------------------------------------------------------------------------------------------------ priors, spaces
@dataclass
class Interval:
    """pcsa: belief/tracker.py's Interval (stage 1's tracker is not ported)."""

    lo: float
    hi: float
    point: float


@dataclass(frozen=True)
class FastPriors(Priors):
    """Stage 1's priors plus a floor on Choice Scarf's share of the item prior (0: stage 1's prior).

    pcsa: `empty_unknown`: a listed build with an empty item or ability (seen_builds.json has a dozen: the ladder
    never showed it) holds an unknown one (the species' item / ability prior) instead of "no item" / the species'
    first ability. False = research's behaviour (stage 2, the equivalence tests); the live player sets True."""

    scarf_floor: float = 0.0
    empty_unknown: bool = False


def wquantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """tracker.wquantile (same convention)."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    c = c / c[-1]
    return float(v[min(len(v) - 1, int(np.searchsorted(c, q - 1e-12)))])


def wquantiles(values: np.ndarray, weights: np.ndarray, qs) -> list[float]:
    keep = weights > 0
    if not keep.any():
        return [0.0 for _ in qs]
    v, w = values[keep], weights[keep]
    order = np.argsort(v)  # any order of equal values gives the same quantile value
    v, c = v[order], np.cumsum(w[order])
    c = c / c[-1]
    n = len(v) - 1
    return [float(v[min(n, int(np.searchsorted(c, q - 1e-12)))]) for q in qs]


EMPTY_ROWS = {
    "comp": np.int8,
    "build": np.int16,
    "spread": np.int64,
    "item": np.int64,
    "ability": np.int64,
    "lp": np.float64,
}  # pcsa: FastSpecies._rows' `empty`, moved out for the function limit


class FastSpecies(SpeciesSpace):
    """SpeciesSpace with the free component kept dense ([items, abilities, spreads] log-prior) and the item prior
    floored for Choice Scarf."""

    S: int
    r_comp: np.ndarray
    r_build: np.ndarray
    r_spread: np.ndarray
    r_item: np.ndarray
    r_ability: np.ndarray
    r_lp: np.ndarray
    free_lp: np.ndarray
    rep_spread: int

    def _item_prior(self) -> np.ndarray:
        p = super()._item_prior()
        floor = getattr(self.priors, "scarf_floor", 0.0)
        k = self.item_index.get("choicescarf")
        if floor > 0 and k is not None and p[k] < floor:
            rest = 1.0 - p[k]
            p = p * ((1.0 - floor) / rest) if rest > 0 else p
            p[k] = floor
        return p / p.sum()

    def _rows(self) -> None:
        pr = self.priors
        S = len(self.table.effect)
        self.S = S
        p_s = self.table.prior[:S] * self._role(S)
        p_s = p_s / p_s.sum()
        I, A = len(self.items), len(self.abilities)
        cols: dict[str, list] = {"comp": [], "build": [], "spread": [], "item": [], "ability": [], "lp": []}

        def put(comp, build, spread, item, ability, lp):
            n = len(lp)
            cols["comp"].append(np.full(n, comp, np.int8))
            cols["build"].append(np.full(n, build, np.int16))
            cols["spread"].append(np.broadcast_to(spread, (n,)).astype(np.int64))
            cols["item"].append(np.broadcast_to(item, (n,)).astype(np.int64))
            cols["ability"].append(np.broadcast_to(ability, (n,)).astype(np.int64))
            cols["lp"].append(np.asarray(lp, dtype=np.float64))

        masses = {"list": pr.list_mass, "spread": pr.spread_mass, "item": pr.item_mass, "free": pr.free_mass}
        if not self.builds:
            masses = {"list": 0.0, "spread": 0.0, "item": 0.0, "free": 1.0}
        total = sum(masses.values())
        lm = {k: math.log(v / total) if v > 0 else -np.inf for k, v in masses.items()}
        all_s, all_i = np.arange(S), np.arange(I)
        log_ps = np.log(np.maximum(p_s, 1e-300))
        unknown = bool(getattr(pr, "empty_unknown", False))
        for k, lb in enumerate(self.builds):
            lpb = math.log(max(lb.prior, 1e-12))
            if unknown and (not lb.item or not lb.ability):  # pcsa: an unknown item / ability (FastPriors)
                self._put_unknown(put, k, lb, lm, lpb, masses, all_s, all_i, log_ps)
                continue
            it, ab = self.item_index[lb.item], self.ability_index.get(lb.ability, 0)
            if masses["list"] > 0:
                put(0, k, lb.spread, it, ab, [lm["list"] + lpb])
            if masses["spread"] > 0:
                put(1, k, all_s, it, ab, lm["spread"] + lpb + log_ps)
            if masses["item"] > 0:
                put(2, k, lb.spread, all_i, ab, lm["item"] + lpb + np.log(self.p_item))
        for name, dt in EMPTY_ROWS.items():
            setattr(self, "r_" + name, np.concatenate(cols[name]) if cols[name] else np.zeros(0, dt))
        if masses["free"] > 0:
            self.free_lp = (
                lm["free"]
                + np.log(self.p_item)[:, None, None]
                + np.log(self.p_ability)[None, :, None]
                + log_ps[None, None, :]
            )
        else:
            self.free_lp = np.full((I, A, S), -np.inf)
        # stage 1 attributes some callers read
        self.comp = self.r_comp
        self.build = self.r_build
        self.spread = self.r_spread
        self.item = self.r_item
        self.ability = self.r_ability
        self.logprior = self.r_lp
        self.rep_spread = int(np.argmax(p_s))

    def _put_unknown(self, put, k, lb, lm, lpb, masses, all_s, all_i, log_ps) -> None:
        """pcsa: the rows of a listed build whose item and / or ability is empty (unknown): every item (by the item
        prior) and / or every ability (by the ability prior) instead of one. Components keep their masses: list =
        the build's spread x the unknown parts, spread = every spread x the unknown parts, item = the build's spread x
        every item (x the unknown abilities)."""
        log_pi, log_pa = np.log(self.p_item), np.log(self.p_ability)
        if lb.item:
            items, l_items = np.array([self.item_index[lb.item]]), np.zeros(1)
        else:
            items, l_items = all_i, log_pi
        if lb.ability and lb.ability in self.ability_index:
            abilities, l_abilities = np.array([self.ability_index[lb.ability]]), np.zeros(1)
        else:
            abilities, l_abilities = np.arange(len(self.abilities)), log_pa

        def grid(spreads, l_spreads, its, l_its):
            s, i, a = np.meshgrid(
                np.arange(len(spreads)), np.arange(len(its)), np.arange(len(abilities)), indexing="ij"
            )
            s, i, a = s.ravel(), i.ravel(), a.ravel()
            return spreads[s], its[i], abilities[a], l_spreads[s] + l_its[i] + l_abilities[a]

        one = np.array([lb.spread])
        if masses["list"] > 0:
            sp, it, ab, lp = grid(one, np.zeros(1), items, l_items)
            put(0, k, sp, it, ab, lm["list"] + lpb + lp)
        if masses["spread"] > 0:
            sp, it, ab, lp = grid(all_s, log_ps, items, l_items)
            put(1, k, sp, it, ab, lm["spread"] + lpb + lp)
        if masses["item"] > 0:
            sp, it, ab, lp = grid(one, np.zeros(1), all_i, log_pi)
            put(2, k, sp, it, ab, lm["item"] + lpb + lp)


class FastSpaces(Spaces):
    def get(self, species: str) -> FastSpecies:
        version = getattr(self.catalog.seen, "version", None)
        if version != self._version:
            self._by.clear()
            self._version = version
        key = to_id_str(species)
        if key not in self._by:
            self._by[key] = FastSpecies(key, self.catalog, self.table, self.priors)
        return cast(FastSpecies, self._by[key])

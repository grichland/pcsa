"""Opposing lineups for the playout team preview (pcsa.players.preview_playout).

Their lead: how often each species led when it was in a six on the live ladder (assets/sets/lead_prior.json,
built by scripts/build_lead_prior.py; 29% top-1 under cross-validation against 17% by chance, 14 Nov 2026), as a
smoothed rate. Their trio: Gumbel top-3 over the log rates (a Plackett-Luce draw without replacement), the first
one leads. Their sets: the build catalog with the builds seen on the ladder first (Catalog.load(seen=True)); a
member without any build is left out of the draw (counted) and replaced on the bench, never a failure. One of each
item (Item Clause). The export holds six (the format's team size): their trio first, the lead first.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.builds import Build, Catalog
from pcsa.common import paths
from pcsa.sim.teams import team_export

SMOOTH = 3.0  # pseudo-games at the chance rate added to every species' count
CHANCE = 1.0 / 6.0
TRIO = 3
BUILD_WEIGHTS = (0.6, 0.25, 0.15)  # the species' first three builds (seen ones first), how often each is drawn
FILLER_ITEMS = ("Leftovers", "Sitrus Berry", "Lum Berry", "Focus Sash", "Life Orb", "Rocky Helmet")


def prior_path() -> Path:
    return paths.assets_dir() / "sets" / "lead_prior.json"


def species_key(name: str) -> str:
    """Rotom-Heat, Rotom and rotomheat count as one species."""
    return to_id_str(game.base_name(to_id_str(name)))


@dataclass(frozen=True)
class LeadPrior:
    """species key -> (times it led, times it was in a six)."""

    counts: dict
    games: int = 0

    @classmethod
    def load(cls, path: Path | None = None) -> LeadPrior:
        """The file's counts; an empty prior (every lead equally likely) when there is no readable file."""
        path = path or prior_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            counts = {k: (int(v["lead"]), int(v["six"])) for k, v in data["species"].items()}
            return cls(counts, int(data.get("games", 0)))
        except (OSError, ValueError, KeyError, TypeError):
            return cls({})

    def rate(self, species: str) -> float:
        lead, six = self.counts.get(species_key(species), (0, 0))
        return (lead + CHANCE * SMOOTH) / (six + SMOOTH)

    def scores(self, six: list[str]) -> np.ndarray:
        """Log lead rates of their six (the Plackett-Luce scores)."""
        return np.array([math.log(self.rate(s)) for s in six])

    def probabilities(self, six: list[str]) -> np.ndarray:
        rates = np.array([self.rate(s) for s in six])
        return rates / rates.sum()


def gumbel_top(scores: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """k distinct indices drawn without replacement in proportion to exp(scores), in draw order."""
    noisy = np.asarray(scores, float) + rng.gumbel(size=len(scores))
    return [int(i) for i in np.argsort(-noisy, kind="stable")[:k]]


@dataclass(frozen=True)
class Lineup:
    species: tuple[str, ...]  # their trio, the lead first
    export: str  # a full team export (the format wants six): the trio first, in that order, then the rest
    no_build: int  # members of their six left out of the draw because no build is known


def builds_of(catalog: Catalog, name: str) -> list[Build]:
    """The species' builds; for a name the catalog does not hold (the preview shows "Rotom", "Samurott",
    "Indeedee Female"): the name without its last words, then the builds of its formes, taken in turn."""
    words = str(name).split()
    for end in range(len(words), 0, -1):
        found = catalog.builds(" ".join(words[:end]))
        if found:
            return found
    key = species_key(words[0] if words else "")
    known = set(catalog.by_id) | set(catalog.seen.by_species if catalog.seen else ())
    formes = [catalog.builds(k) for k in sorted(known) if k != key and to_id_str(game.base_name(k)) == key]
    formes.sort(key=len, reverse=True)  # the forme seen most first
    longest = max((len(f) for f in formes), default=0)
    return [f[i] for i in range(longest) for f in formes if i < len(f)]


def _draw_build(builds: list[Build], rng: np.random.Generator) -> int:
    weights = np.array(BUILD_WEIGHTS[: len(builds)])
    return int(rng.choice(len(weights), p=weights / weights.sum()))


def pick_builds(species: list[str], catalog: Catalog, rng: np.random.Generator) -> list[Build]:
    """One build per member, no item twice: another build of the member first, else a filler item."""
    out: list[Build] = []
    taken: set[str] = set()
    for name in species:
        builds = builds_of(catalog, name)
        first = _draw_build(builds, rng)
        order = [first] + [i for i in range(len(builds)) if i != first]
        build = next((builds[i] for i in order if to_id_str(builds[i].item) not in taken), None)
        if build is None:
            free = [item for item in FILLER_ITEMS if to_id_str(item) not in taken]
            build = replace(builds[first], item=free[0] if free else builds[first].item)
        taken.add(to_id_str(build.item))
        out.append(build)
    return out


def their_lineup(six: list[str], prior: LeadPrior, catalog: Catalog, rng: np.random.Generator) -> Lineup | None:
    """A sampled trio of theirs with sets, the lead first; None when fewer than three of their six have a build."""
    usable = [i for i, s in enumerate(six) if builds_of(catalog, s)]
    if len(usable) < TRIO:
        return None
    names = [six[i] for i in usable]
    trio = [names[i] for i in gumbel_top(prior.scores(names), TRIO, rng)]
    rest = [n for n in names if n not in trio]
    taken = {species_key(n) for n in names}
    for filler in catalog.species():  # a member without a build is replaced on the bench (it never plays)
        if len(trio) + len(rest) >= len(six):
            break
        if species_key(filler) not in taken and catalog.builds(filler):
            rest.append(filler)
            taken.add(species_key(filler))
    builds = pick_builds(trio + rest, catalog, rng)
    return Lineup(tuple(trio), team_export(builds), len(six) - len(usable))

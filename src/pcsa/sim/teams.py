"""Legal six-member teams drawn from the build catalog, for simulator games.

The six species are drawn by ladder usage. Each member is its species' top build, or with `noisy` any of its
catalog builds weighted by score. Item Clause is respected while drawing; Showdown's validator has the last word.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.builds import Build, Catalog
from pcsa.battle.teams import export_text
from pcsa.sim.engine import Engine
from pcsa.sim.validate import validate

TEAM_SIZE = 6
MAX_ATTEMPTS = 40


def _pick_build(builds: list[Build], used_items: set[str], rng: random.Random, noisy: bool) -> Build | None:
    allowed = [b for b in builds if to_id_str(b.item) not in used_items or not b.item]
    if not allowed:
        return None
    if not noisy:
        return allowed[0]
    weights = [10.0 ** (b.score - allowed[0].score) for b in allowed]
    return rng.choices(allowed, weights=weights, k=1)[0]


@dataclass
class _Draft:
    members: list[Build] = field(default_factory=list)
    species: set[str] = field(default_factory=set)
    items: set[str] = field(default_factory=set)

    def add(self, build: Build) -> None:
        self.members.append(build)
        self.species.add(game.base_name(to_id_str(build.species)))
        if build.item:
            self.items.add(to_id_str(build.item))


def _usage(catalog: Catalog, names: list[str]) -> list[int]:
    return [max(1, catalog.usage.get(to_id_str(n), 0)) for n in names]


def _fill(draft: _Draft, catalog: Catalog, rng: random.Random, noisy: bool) -> bool:
    """Usage-drawn members until the six is complete; False when the draws run out first."""
    names = [n for n in catalog.species() if catalog.builds(n)]
    weights = _usage(catalog, names)
    for _ in range(MAX_ATTEMPTS * TEAM_SIZE):
        if len(draft.members) == TEAM_SIZE:
            break
        name = rng.choices(names, weights=weights, k=1)[0]
        if game.base_name(to_id_str(name)) in draft.species:
            continue
        build = _pick_build(catalog.builds(name), draft.items, rng, noisy)
        if build is not None:
            draft.add(build)
    return len(draft.members) == TEAM_SIZE


def sample_team(catalog: Catalog, rng: random.Random, *, noisy: bool = False) -> list[Build]:
    """Six members by usage."""
    draft = _Draft()
    if not _fill(draft, catalog, rng, noisy):
        raise ValueError("could not draw six distinct legal members from the catalog")
    return draft.members


def team_export(members: list[Build]) -> str:
    return export_text([b.member() for b in members])


def sample_teams(
    catalog: Catalog, engine: Engine, rng: random.Random, count: int, *, noisy: bool = False
) -> list[str]:
    """`count` validated team exports; drawing continues until the validator accepts enough."""
    return validated_teams(engine, count, lambda: sample_team(catalog, rng, noisy=noisy))


def validated_teams(engine: Engine, count: int, draw: Callable[[], list[Build]]) -> list[str]:
    """`count` exports of `draw()` teams that Showdown's validator accepts."""
    accepted: list[str] = []
    rejected: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        if len(accepted) >= count:
            break
        batch = [team_export(draw()) for _ in range(count - len(accepted))]
        for text, result in zip(batch, validate(engine, batch), strict=True):
            (accepted if result["ok"] else rejected).append(text if result["ok"] else "; ".join(result["problems"]))
    if len(accepted) < count:
        raise ValueError(f"only {len(accepted)} of {count} teams validated; last problems: {rejected[-3:]}")
    return accepted[:count]

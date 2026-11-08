"""Legal six-member teams drawn from the build catalog, for simulator games.

The six species are drawn by ladder usage. Each member is its species' top build, or with `noisy` any of its
catalog builds weighted by score. Item Clause is respected while drawing; Showdown's validator has the last word.
"""

from __future__ import annotations

import random

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


def sample_team(catalog: Catalog, rng: random.Random, *, noisy: bool = False) -> list[Build]:
    """Six members by usage."""
    names = [n for n in catalog.species() if catalog.builds(n)]
    weights = [max(1, catalog.usage.get(to_id_str(n), 0)) for n in names]
    members: list[Build] = []
    used_species: set[str] = set()
    used_items: set[str] = set()
    for _ in range(MAX_ATTEMPTS * TEAM_SIZE):
        if len(members) == TEAM_SIZE:
            break
        name = rng.choices(names, weights=weights, k=1)[0]
        base = game.base_name(to_id_str(name))
        if base in used_species:
            continue
        build = _pick_build(catalog.builds(name), used_items, rng, noisy)
        if build is None:
            continue
        members.append(build)
        used_species.add(base)
        if build.item:
            used_items.add(to_id_str(build.item))
    if len(members) != TEAM_SIZE:
        raise ValueError("could not draw six distinct legal members from the catalog")
    return members


def team_export(members: list[Build]) -> str:
    return export_text([b.member() for b in members])


def sample_teams(
    catalog: Catalog, engine: Engine, rng: random.Random, count: int, *, noisy: bool = False
) -> list[str]:
    """`count` validated team exports; drawing continues until the validator accepts enough."""
    return _validated(engine, count, lambda: team_export(sample_team(catalog, rng, noisy=noisy)))


def _validated(engine: Engine, count: int, draw) -> list[str]:
    accepted: list[str] = []
    rejected: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        if len(accepted) >= count:
            break
        batch = [draw() for _ in range(count - len(accepted))]
        for text, result in zip(batch, validate(engine, batch), strict=True):
            (accepted if result["ok"] else rejected).append(text if result["ok"] else "; ".join(result["problems"]))
    if len(accepted) < count:
        raise ValueError(f"only {len(accepted)} of {count} teams validated; last problems: {rejected[-3:]}")
    return accepted[:count]

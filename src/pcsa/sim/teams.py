"""Legal six-member teams drawn from the build catalog, for simulator games.

The six species are drawn by ladder usage. Each member is its species' top build, or with `noisy` any of its
catalog builds weighted by score. Item Clause is respected while drawing; Showdown's validator has the last word.
A style player's teams (`sample_styled_team`) start from members that fill its style's roles
(pcsa.players.styles) and must fit it.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

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


# Style teams ---------------------------------------------------------------------------------------
class RoleDraw(Protocol):
    """One role of a style plan: `count` members whose build `accepts` (pcsa.players.styles.RoleDraw)."""

    @property
    def count(self) -> int: ...
    @property
    def label(self) -> str: ...
    def accepts(self, build: Build) -> bool: ...


class StyleRules(Protocol):
    """pcsa.players.styles.Styles, as the sampler sees it."""

    def plans(self, style: str) -> Sequence[Sequence[RoleDraw]]: ...
    def fit(self, style: str, members: Sequence[Build]) -> object | None: ...


def role_builds(catalog: Catalog, draw: RoleDraw) -> dict[str, list[Build]]:
    """Species -> its catalog builds that fill the role, best first."""
    out = {name: [b for b in catalog.builds(name) if draw.accepts(b)] for name in catalog.species()}
    return {name: builds for name, builds in out.items() if builds}


def role_species(catalog: Catalog, styles: StyleRules, style: str) -> dict[str, list[str]]:
    """Role label (with the weather, for weather styles) -> the catalog species that can fill it."""
    return {d.label: list(role_builds(catalog, d)) for plan in styles.plans(style) for d in plan}


def _draw_role(draft: _Draft, catalog: Catalog, rng: random.Random, draw: RoleDraw, noisy: bool) -> bool:
    pool = role_builds(catalog, draw)
    for _ in range(draw.count):
        names = [n for n in pool if game.base_name(to_id_str(n)) not in draft.species]
        names = [n for n in names if _pick_build(pool[n], draft.items, rng, False) is not None]
        if not names:
            return False
        name = rng.choices(names, weights=_usage(catalog, names), k=1)[0]
        build = _pick_build(pool[name], draft.items, rng, noisy)
        assert build is not None
        draft.add(build)
    return True


def sample_styled_team(
    catalog: Catalog, rng: random.Random, style: str, styles: StyleRules, *, noisy: bool = False
) -> list[Build]:
    """Six that fit `style`: its role members first (one plan, so one weather, drawn at random), then usage
    draws, until `styles.fit` accepts; shuffled so the role members sit anywhere."""
    plans = styles.plans(style)
    weights = [_plan_usage(catalog, plan) for plan in plans]
    for _ in range(MAX_ATTEMPTS):
        draft = _Draft()
        plan = rng.choices(plans, weights=weights, k=1)[0] if plans else ()
        if not all(_draw_role(draft, catalog, rng, d, noisy) for d in plan) or not _fill(draft, catalog, rng, noisy):
            continue
        if styles.fit(style, draft.members) is not None:
            rng.shuffle(draft.members)
            return draft.members
    raise ValueError(f"could not draw a six that fits the {style} style")


def _plan_usage(catalog: Catalog, plan: Sequence[RoleDraw]) -> int:
    """Ladder usage of the species filling the plan's first role (its setter), so weathers come up as played."""
    return sum(_usage(catalog, list(role_builds(catalog, plan[0])))) if plan else 1


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

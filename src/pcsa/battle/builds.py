"""The build catalog: usage-ranked full builds per species (assets/sets/top_builds.json), best first."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from poke_env.data.normalize import to_id_str
from poke_env.teambuilder import TeambuilderPokemon

from pcsa.battle import game
from pcsa.battle.teams import LEVEL, STAT_ORDER
from pcsa.common import paths

DEFAULT_PATH = "sets/top_builds.json"
SCHEMA = "pcsa-top-builds/1"


@dataclass(frozen=True)
class Build:
    species: str
    item: str
    ability: str
    nature: str
    points: tuple[int, int, int, int, int, int]
    moves: tuple[str, ...]
    score: float = 0.0

    def member(self, name: str | None = None) -> TeambuilderPokemon:
        return TeambuilderPokemon(
            nickname=name if name and name != self.species else None,
            species=self.species,
            item=self.item,
            ability=self.ability,
            moves=list(self.moves),
            nature=self.nature,
            evs=list(self.points),
            level=LEVEL,
        )


class Catalog:
    """The usage-ranked build catalog. With `seen` (pcsa.battle.seen_builds; the players' catalog) a species' builds
    seen on the live ladder come first, the most often seen first, then the rest of the catalog in usage order."""

    def __init__(self, data: dict, seen=None) -> None:
        self.seen = seen
        self._merged: dict[str, tuple[int, list[Build]]] = {}
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported build catalog schema {data.get('schema')!r}")
        self.by_id: dict[str, list[Build]] = {}
        self.usage: dict[str, int] = {}
        for name, entry in data["species"].items():
            builds = [
                Build(
                    name,
                    b["item"],
                    b["ability"],
                    b["nature"],
                    tuple(int(b["points"][s]) for s in STAT_ORDER),  # type: ignore[arg-type]
                    tuple(b["moves"]),
                    float(b.get("score", 0.0)),
                )
                for b in entry["builds"]
            ]
            self.by_id[to_id_str(name)] = builds
            self.usage[to_id_str(name)] = int(entry.get("usage", 0))

    @classmethod
    def load(cls, path: Path | None = None, *, seen: bool = False) -> Catalog:
        """The catalog; `seen=True` puts the builds seen on the live ladder first (the players' catalog)."""
        path = path or paths.assets_dir() / DEFAULT_PATH
        store = None
        if seen:
            from pcsa.battle import seen_builds

            store = seen_builds.shared()
        return cls(json.loads(path.read_text(encoding="utf-8")), seen=store)

    def species(self) -> list[str]:
        return [builds[0].species for builds in self.by_id.values() if builds]

    def builds(self, species: str) -> list[Build]:
        """The species' builds, best first (seen ones first when the catalog has `seen`); a form without builds
        falls back to its base species."""
        key = to_id_str(species)
        if key not in self.by_id and key not in (self.seen.by_species if self.seen else {}):
            key = to_id_str(game.base_name(key))
        plain = self.by_id.get(key, [])
        if self.seen is None:
            return plain
        cached = self._merged.get(key)
        if cached is not None and cached[0] == self.seen.version:
            return cached[1]
        first = [s.build for s in self.seen.ranked(key)]
        taken = {signature(b) for b in first}
        merged = first + [b for b in plain if signature(b) not in taken]
        self._merged[key] = (self.seen.version, merged)
        return merged


def signature(b: Build) -> tuple:
    """A build's identity without its usage score."""
    return (to_id_str(b.item), to_id_str(b.ability), b.nature, b.points, tuple(sorted(to_id_str(m) for m in b.moves)))


def fits(build: Build, moves, item: str | None = None, ability: str | None = None) -> bool:
    """Whether `build` has every revealed move and the revealed item and ability (None: not revealed)."""
    if not {to_id_str(m) for m in moves} <= {to_id_str(m) for m in build.moves}:
        return False
    if item is not None and to_id_str(build.item) != to_id_str(item):
        return False
    return ability is None or to_id_str(build.ability) == to_id_str(ability)


def first_fit(builds: list[Build], *tries) -> Build:
    """The first build (in the catalog's order: seen builds first) meeting the first satisfiable try."""
    for keep in tries:
        found = next((b for b in builds if keep(b)), None)
        if found is not None:
            return found
    return builds[0]

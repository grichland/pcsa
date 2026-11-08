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
    def __init__(self, data: dict) -> None:
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
    def load(cls, path: Path | None = None) -> Catalog:
        path = path or paths.assets_dir() / DEFAULT_PATH
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def species(self) -> list[str]:
        return [builds[0].species for builds in self.by_id.values() if builds]

    def builds(self, species: str) -> list[Build]:
        """The species' builds, best first; a form without builds falls back to its base species."""
        key = to_id_str(species)
        if key in self.by_id:
            return self.by_id[key]
        return self.by_id.get(to_id_str(game.base_name(key)), [])

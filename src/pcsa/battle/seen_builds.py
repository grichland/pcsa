"""Builds seen on the live ladder: every opposing set a battle revealed is either covered by the build catalog or
added to it, and seen builds rank first when an opponent's set is assumed.

After each live battle (and with `pcsa sets learn` for recorded runs) each opposing Pokémon's revealed set (moves,
item, ability; a Mega Evolution reveals its stone) is checked in this order:
1. a seen build already fits it: that build's count goes up (an added build also learns the newly revealed parts);
2. a catalog build covers it (every revealed move in the build, same item and ability when revealed): that catalog
   build becomes a seen build;
3. otherwise a new build is added: the revealed parts, the rest (other moves, nature, spread, unrevealed item or
   ability) from the closest catalog build of the species.
`Catalog.load(seen=True)` (the players' catalog) lists a species' seen builds first, the most often seen first, then
the rest of the catalog in its usage order. The simulator's team draws and research use the plain catalog.

The file (assets/sets/seen_builds.json) also records which battles were learned, so a battle is never counted twice.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.builds import Build, Catalog
from pcsa.battle.teams import STAT_ORDER
from pcsa.common import paths

SCHEMA = "pcsa-seen-builds/1"
DEFAULT_PATH = "sets/seen_builds.json"
ENV_PATH = "PCSA_SEEN_BUILDS"  # another file (tests point it at one that does not exist)
MOVE_SLOTS = 4
FALLBACK_NATURE = "Hardy"
FLAT_POINTS = 11
NOT_MOVES = frozenset({"struggle", "recharge"})


@dataclass(frozen=True)
class Observed:
    """One opposing Pokémon's revealed set (ids), its species the base form."""

    species: str
    moves: tuple[str, ...] = ()
    item: str | None = None
    ability: str | None = None

    def empty(self) -> bool:
        return not (self.moves or self.item or self.ability)


@dataclass
class Seen:
    build: Build
    source: str  # "catalog" (a catalog build that covered what was seen) or "added"
    count: int = 0
    first: str = ""
    last: str = ""
    known_moves: tuple[str, ...] = ()  # added builds: the moves actually seen (ids); the rest is filled in
    known_item: bool = False
    known_ability: bool = False


@dataclass
class SeenBuilds:
    path: Path
    by_species: dict[str, list[Seen]] = field(default_factory=dict)
    battles: list[str] = field(default_factory=list)
    version: int = 0

    # Files ---------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | None = None) -> SeenBuilds:
        """The seen builds at `path` (default: $PCSA_SEEN_BUILDS, else assets/sets/seen_builds.json)."""
        path = path or Path(os.environ.get(ENV_PATH) or paths.assets_dir() / DEFAULT_PATH)
        out = cls(path)
        if not path.is_file():
            return out
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported seen-builds schema {data.get('schema')!r} in {path}")
        out.battles = list(data.get("battles", []))
        for key, entries in data.get("species", {}).items():
            out.by_species[key] = [_seen_from_json(e) for e in entries]
        return out

    def save(self) -> None:
        data = {
            "schema": SCHEMA,
            "note": "opposing builds seen on the live ladder (pcsa.battle.seen_builds); seen builds rank first",
            "battles": self.battles,
            "species": {k: [_seen_to_json(s) for s in self.ranked(k)] for k in sorted(self.by_species)},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def ranked(self, species: str) -> list[Seen]:
        """A species' seen builds, the most recently seen first (then the most often seen): what the ladder brings now
        comes before what it brought earlier in the season."""
        return sorted(self.by_species.get(to_id_str(species), []), key=lambda s: (s.last, s.count), reverse=True)

    # Learning ------------------------------------------------------------------------------------------------
    def learn(
        self, battle_id: str, sets: Iterable[Observed], catalog: Catalog, when: str | None = None
    ) -> list[dict]:
        """Record one battle's revealed sets; a battle already learned is skipped. Returns what happened per set."""
        if battle_id in self.battles:
            return [{"battle": battle_id, "result": "already learned"}]
        when = when or datetime.now(UTC).strftime("%Y-%m-%d")
        report = [self.observe(o, catalog, when) for o in sets if not o.empty()]
        self.battles.append(battle_id)
        self.version += 1
        return report

    def observe(self, o: Observed, catalog: Catalog, when: str) -> dict:
        if len(o.moves) > MOVE_SLOTS:
            return {"species": o.species, "result": "skipped", "why": f"{len(o.moves)} moves revealed"}
        key = to_id_str(o.species)
        entries = self.by_species.setdefault(key, [])
        ranked = self.ranked(key)
        # 1. a seen build that already has all of it; 2. a catalog build that covers it
        for s in ranked:
            if covers(s.build, o):
                _extend(s, o, catalog)
                _seen_again(s, when)
                return _row(o, "seen again" if s.source == "catalog" else "added build seen again", s)
        cover = next((b for b in catalog.builds(o.species) if covers(b, o)), None)
        if cover is not None:
            s = Seen(cover, "catalog", 1, when, when)
            entries.append(s)
            return _row(o, "covered by the catalog", s)
        # 3. nothing covers it: an added build it is consistent with learns the new parts; 4. else a new build
        for s in ranked:
            if s.source == "added" and _fits_seen(s, o):
                _extend(s, o, catalog)
                _seen_again(s, when)
                return _row(o, "added build extended", s)
        s = Seen(_new_build(o, catalog), "added", 1, when, when, o.moves, o.item is not None, o.ability is not None)
        entries.append(s)
        return _row(o, "added", s)


def _seen_again(s: Seen, when: str) -> None:
    """One more sighting on `when`; battles learned out of order (an older run learned later) keep the dates right."""
    s.count += 1
    s.first = min(s.first, when) if s.first else when
    s.last = max(s.last, when)


# Coverage --------------------------------------------------------------------------------------------------------
def covers(build: Build, o: Observed) -> bool:
    """Every revealed move is in the build, and the item and ability match when revealed."""
    moves = {to_id_str(m) for m in build.moves}
    if not set(o.moves) <= moves:
        return False
    if o.item is not None and to_id_str(build.item) != o.item:
        return False
    return o.ability is None or to_id_str(build.ability) == o.ability


def _fits_seen(s: Seen, o: Observed) -> bool:
    if s.source == "catalog":
        return covers(s.build, o)
    if o.item is not None and s.known_item and to_id_str(s.build.item) != o.item:
        return False
    if o.ability is not None and s.known_ability and to_id_str(s.build.ability) != o.ability:
        return False
    return len(set(s.known_moves) | set(o.moves)) <= MOVE_SLOTS


def species_key(form: str, catalog: Catalog) -> str:
    """The catalog species a battle form counts as: its own builds when the catalog has them (regional forms), else
    its base species (Megas, Aegislash-Blade and other in-battle forms)."""
    form = to_id_str(form)
    return form if form in catalog.by_id else to_id_str(game.base_name(form))


def _extend(s: Seen, o: Observed, catalog: Catalog) -> None:
    """An added build learns what this sighting revealed beyond what it knew."""
    if s.source != "added":
        return
    known = tuple(dict.fromkeys([*s.known_moves, *o.moves]))
    item_known, ability_known = s.known_item or o.item is not None, s.known_ability or o.ability is not None
    if known == s.known_moves and item_known == s.known_item and ability_known == s.known_ability:
        return
    merged = Observed(
        o.species,
        known,
        o.item if o.item is not None else (to_id_str(s.build.item) if s.known_item else None),
        o.ability if o.ability is not None else (to_id_str(s.build.ability) if s.known_ability else None),
    )
    s.build, s.known_moves, s.known_item, s.known_ability = (
        _new_build(merged, catalog),
        known,
        item_known,
        ability_known,
    )


def _new_build(o: Observed, catalog: Catalog) -> Build:
    """The revealed parts, the rest from the closest catalog build of the species."""
    builds = catalog.builds(o.species)

    def closeness(b: Build) -> tuple:
        ids = {to_id_str(m) for m in b.moves}
        return (
            o.item is not None and to_id_str(b.item) == o.item,
            len(ids & set(o.moves)),
            o.ability is not None and to_id_str(b.ability) == o.ability,
            b.score,
        )

    base = max(builds, key=closeness) if builds else None
    moves = [move_name(m) for m in o.moves]
    for m in base.moves if base else ():
        if len(moves) >= MOVE_SLOTS:
            break
        if to_id_str(m) not in o.moves:
            moves.append(m)
    item = item_name(o.item, o.species, catalog) if o.item is not None else (base.item if base else "")
    ability = ability_name(o.ability, o.species) if o.ability is not None else (base.ability if base else "")
    nature = base.nature if base else FALLBACK_NATURE
    points = base.points if base else (FLAT_POINTS,) * len(STAT_ORDER)
    species = game.species_entry(o.species).get("name", o.species)
    return Build(species, item, ability, nature, points, tuple(moves))  # type: ignore[arg-type]


# Names -----------------------------------------------------------------------------------------------------------
def is_mega(species: str) -> bool:
    return str(game.species_entry(species).get("forme", "")).startswith("Mega")


def move_name(move_id: str) -> str:
    return GenData.from_gen(9).moves.get(move_id, {}).get("name", move_id)


def ability_name(ability_id: str, species: str) -> str:
    names = game.species_entry(species).get("abilities", {}).values()
    return next((n for n in names if to_id_str(n) == ability_id), ability_id)


def item_name(item_id: str, species: str, catalog: Catalog) -> str:
    for builds in catalog.by_id.values():
        for b in builds:
            if to_id_str(b.item) == item_id:
                return b.item
    for entry in game.dex().values():
        if to_id_str(entry.get("requiredItem", "")) == item_id:
            return entry["requiredItem"]
    return item_id


# Reading a battle ------------------------------------------------------------------------------------------------
def observed_from_battle(
    battle, abilities: dict[str, str] | None = None, catalog: Catalog | None = None
) -> list[Observed]:
    """The opposing sets a (finished) tracked Battle revealed. `abilities`: the tracker's revealed abilities by
    'p2: Name' (the phone's log shows abilities the Battle may not have parsed)."""
    catalog = catalog or Catalog.load()
    out = []
    side = "p2" if (battle.player_role or "p1") == "p1" else "p1"
    for key, mon in battle.opponent_team.items():
        form = to_id_str(mon.species)
        species = species_key(form, catalog)
        moves = tuple(m for m in mon.moves if m not in NOT_MOVES)
        item = to_id_str(mon.item) if mon.item and mon.item != GenData.UNKNOWN_ITEM else None
        mega_form = form if is_mega(form) else None
        if mega_form is None and item:  # the Battle may keep the base species after Mega Evolution
            mega_form = to_id_str(game.mega_form_for_stone(species, item) or "") or None
        if mega_form and item is None:
            stone = game.species_entry(mega_form).get("requiredItem")
            item = to_id_str(stone) if stone else None
        name = key.split(": ", 1)[-1]
        raw = (abilities or {}).get(f"{side}: {name}") or (abilities or {}).get(key) or mon.ability
        ability = to_id_str(raw) if raw else None
        if ability and mega_form and ability == to_id_str(game.mega_ability(game.species_entry(mega_form)) or ""):
            ability = None  # the Mega's own ability, not the base build's
        if ability and ability not in {
            to_id_str(a) for a in game.species_entry(species).get("abilities", {}).values()
        }:
            ability = None  # not one the base species can have (a form's or a copied ability)
        out.append(Observed(species, moves, item, ability))
    return out


def _row(o: Observed, result: str, s: Seen) -> dict:
    b = s.build
    return {
        "species": o.species,
        "result": result,
        "seen": list(o.moves),
        "item": o.item,
        "ability": o.ability,
        "build": {"item": b.item, "ability": b.ability, "moves": list(b.moves)},
        "count": s.count,
    }


def _seen_to_json(s: Seen) -> dict:
    b = s.build
    out = {
        "source": s.source,
        "count": s.count,
        "first": s.first,
        "last": s.last,
        "species": b.species,
        "item": b.item,
        "ability": b.ability,
        "nature": b.nature,
        "points": dict(zip(STAT_ORDER, b.points, strict=True)),
        "moves": list(b.moves),
    }
    if s.source == "added":
        out |= {"known_moves": list(s.known_moves), "known_item": s.known_item, "known_ability": s.known_ability}
    return out


def _seen_from_json(e: dict) -> Seen:
    build = Build(
        e["species"],
        e["item"],
        e["ability"],
        e["nature"],
        tuple(int(e["points"][s]) for s in STAT_ORDER),  # type: ignore[arg-type]
        tuple(e["moves"]),
    )
    return Seen(
        build,
        e["source"],
        int(e["count"]),
        e.get("first", ""),
        e.get("last", ""),
        tuple(e.get("known_moves", ())),
        bool(e.get("known_item")),
        bool(e.get("known_ability")),
    )


_SHARED: SeenBuilds | None = None


def shared() -> SeenBuilds:
    """The one SeenBuilds of this process (players' catalogs and the live learner share it)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = SeenBuilds.load()
    return _SHARED


# Recorded runs ---------------------------------------------------------------------------------------------------
def learn_battle(
    store: SeenBuilds, battle_id: str, battle, abilities: dict[str, str] | None, when: str | None = None
):
    """Learn one finished battle against the plain catalog (coverage is judged on the catalog itself)."""
    catalog = Catalog.load()
    return store.learn(battle_id, observed_from_battle(battle, abilities, catalog), catalog, when)

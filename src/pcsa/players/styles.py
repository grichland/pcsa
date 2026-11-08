"""Team styles (config/players/styles.yaml): which members of a six fill a style's roles.

The same rules serve the live preview (our battle.team, via `fit_mons`) and the simulator, which draws sixes
that fit (`plans` names the role members to draw first, `fit` checks the result).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle import moves as move_data
from pcsa.battle.builds import Build
from pcsa.common import paths
from pcsa.common.errors import ConfigError

DEFAULT_PATH = "players/styles.yaml"
ATTACKING = ("physical", "special")
BULK_STATS = ("hp", "def", "spd")


@dataclass(frozen=True)
class Member:
    """What the rules read from one team member: ids of its species, item, ability and moves."""

    species: str
    item: str
    ability: str
    moves: tuple[str, ...]

    @classmethod
    def of_build(cls, build: Build) -> Member:
        moves = tuple(to_id_str(m) for m in build.moves)
        return cls(to_id_str(build.species), to_id_str(build.item), to_id_str(build.ability), moves)

    @classmethod
    def of_mon(cls, mon: Any) -> Member:
        """A poke-env Pokemon of our own team."""
        moves = tuple(to_id_str(m) for m in mon.moves)
        return cls(to_id_str(mon.species), to_id_str(mon.item or ""), to_id_str(mon.ability or ""), moves)


@dataclass(frozen=True)
class Facts:
    abilities: frozenset[str]
    types: frozenset[str]
    bulk: int  # base HP + Def + SpD, of the Mega form when that is bulkier
    attacks: tuple[tuple[str, str, int], ...]  # (move id, type id, priority) of each attacking move
    moves: frozenset[str]
    item: str = ""


@lru_cache(maxsize=8192)
def facts(member: Member) -> Facts:
    """The member's abilities and types, its Mega form's included when it holds the stone."""
    entries = [game.species_entry(member.species)]
    mega = game.mega_form_for_stone(member.species, member.item) if member.item else None
    if mega:
        entries.append(game.species_entry(mega))
    abilities = {member.ability, *(a for a in map(game.mega_ability, entries[1:]) if a)} - {""}
    types = {to_id_str(t) for e in entries for t in e.get("types", [])}
    bulk = max(sum(e.get("baseStats", {}).get(s, 0) for s in BULK_STATS) for e in entries)
    attacks = []
    for move in member.moves:
        data = move_data.base_data(move)
        if data["category"] in ATTACKING:
            attacks.append((move, data["type"], int(GenData.from_gen(9).moves.get(move, {}).get("priority", 0))))
    return Facts(frozenset(abilities), frozenset(types), bulk, tuple(attacks), frozenset(member.moves), member.item)


@lru_cache(maxsize=8192)
def build_facts(build: Build) -> Facts:
    return facts(Member.of_build(build))


@dataclass(frozen=True)
class Weather:
    name: str
    setters: frozenset[str]
    abilities: frozenset[str]
    moves: frozenset[str]
    stab: frozenset[str]
    rivals: frozenset[str] = frozenset()  # abilities that set another weather: never an abuser of this one

    def set_by(self, f: Facts) -> bool:
        return bool(f.abilities & self.setters)

    def abused_by(self, f: Facts) -> bool:
        if f.abilities & self.rivals:
            return False
        stab = f.types & self.stab
        return bool(f.abilities & self.abilities or f.moves & self.moves or any(t in stab for _, t, _ in f.attacks))


Test = Callable[[Facts, Weather | None], bool]


def _ids(names: Iterable[str]) -> frozenset[str]:
    return frozenset(to_id_str(n) for n in names)


def _any_ability(value: list[str]) -> Test:
    ids = _ids(value)
    return lambda f, w: bool(f.abilities & ids)


def _any_move(value: list[str]) -> Test:
    ids = _ids(value)
    return lambda f, w: bool(f.moves & ids)


def _priority_move(value: bool | dict) -> Test:
    excluded = _ids(value.get("except", [])) if isinstance(value, dict) else frozenset()
    return lambda f, w: any(p > 0 and m not in excluded for m, _, p in f.attacks)


def _base_bulk_min(value: int) -> Test:
    return lambda f, w: f.bulk >= int(value)


def _min_attacking_moves(value: int) -> Test:
    return lambda f, w: len(f.attacks) >= int(value)


def _not_item(value: list[str]) -> Test:
    ids = _ids(value)
    return lambda f, w: f.item not in ids


def _weather(value: str) -> Test:
    if value == "setter":
        return lambda f, w: w is not None and w.set_by(f)
    if value == "abuser":
        return lambda f, w: w is not None and w.abused_by(f)
    raise ConfigError(f"styles: weather must be setter or abuser, not {value!r}")


def _any_of(value: list[dict]) -> Test:
    tests = [_compile(spec) for spec in value]
    return lambda f, w: any(t(f, w) for t in tests)


CONDITIONS: dict[str, Callable[[Any], Test]] = {
    "any_ability": _any_ability,
    "any_move": _any_move,
    "priority_move": _priority_move,
    "base_bulk_min": _base_bulk_min,
    "min_attacking_moves": _min_attacking_moves,
    "not_item": _not_item,
    "weather": _weather,
    "any_of": _any_of,
}


def _compile(spec: dict) -> Test:
    unknown = set(spec) - set(CONDITIONS)
    if unknown:
        raise ConfigError(f"styles: unknown conditions {sorted(unknown)} (known: {', '.join(CONDITIONS)})")
    tests = [CONDITIONS[key](value) for key, value in spec.items()]
    return lambda f, w: all(t(f, w) for t in tests)


def _mentions_weather(spec: Any) -> bool:
    if isinstance(spec, dict):
        return "weather" in spec or any(_mentions_weather(v) for v in spec.values())
    return isinstance(spec, list) and any(_mentions_weather(v) for v in spec)


@dataclass(frozen=True)
class Role:
    name: str
    min: int
    test: Test | None  # None: any member


@dataclass(frozen=True)
class Style:
    name: str
    roles: tuple[Role, ...]  # tested roles first, so untested ones take what is left
    lead: str
    back: frozenset[str]  # base species ids that never lead
    uses_weather: bool


@dataclass(frozen=True)
class Fit:
    """How a six fits a style; indices are into the six."""

    style: str
    weather: str | None
    roles: dict[str, tuple[int, ...]]  # the three to bring, by the role each fills
    candidates: dict[str, tuple[int, ...]]  # every member that could fill each role
    lead: int

    @property
    def bring(self) -> tuple[int, ...]:
        """The three to bring, lead first."""
        rest = [i for picks in self.roles.values() for i in picks if i != self.lead]
        return (self.lead, *rest)


@dataclass(frozen=True)
class RoleDraw:
    """A role member to draw first when building a six for a style (one weather's, for weather styles)."""

    role: str
    count: int
    weather: Weather | None
    test: Test

    @property
    def label(self) -> str:
        return f"{self.weather.name} {self.role}" if self.weather else self.role

    def accepts(self, build: Build) -> bool:
        return self.test(build_facts(build), self.weather)


def _assign(roles: Sequence[Role], candidates: dict[str, tuple[int, ...]], used: tuple[int, ...] = ()):
    """Distinct members for each role's minimum, the first workable choice in candidate order."""
    if not roles:
        return {}
    role = roles[0]
    free = [i for i in candidates[role.name] if i not in used]
    for chosen in itertools.combinations(free, role.min):
        tail = _assign(roles[1:], candidates, (*used, *chosen))
        if tail is not None:
            return {role.name: chosen, **tail}
    return None


class Styles:
    def __init__(self, data: dict) -> None:
        specs = data.get("weathers") or {}
        setters = {name: _ids(w.get("setters", [])) for name, w in specs.items()}
        every = frozenset().union(*setters.values()) if setters else frozenset()
        self.weathers = {
            name: Weather(
                name,
                setters[name],
                _ids(w.get("abilities", [])),
                _ids(w.get("moves", [])),
                _ids(w.get("stab", [])),
                every - setters[name],
            )
            for name, w in specs.items()
        }
        self.styles = {name: _style(name, spec) for name, spec in (data.get("styles") or {}).items()}

    def names(self) -> list[str]:
        return list(self.styles)

    def style(self, name: str) -> Style:
        if name not in self.styles:
            raise ConfigError(f"unknown team style {name!r} (one of {', '.join(self.styles)})")
        return self.styles[name]

    def fit(self, style: str, members: Sequence[Build]) -> Fit | None:
        return self.fit_members(style, [Member.of_build(b) for b in members])

    def fit_mons(self, style: str, mons: Iterable[Any]) -> Fit | None:
        """`fit` for poke-env Pokemon of our battle.team (in the order given)."""
        return self.fit_members(style, [Member.of_mon(m) for m in mons])

    def fit_members(self, style: str, members: Sequence[Member]) -> Fit | None:
        """The fit with the most role candidates (over the weathers, for weather styles), or None."""
        spec = self.style(style)
        found = [facts(m) for m in members]
        back = [game.base_name(m.species) in spec.back for m in members]
        best: Fit | None = None
        for weather in self.weathers.values() if spec.uses_weather else [None]:
            candidates = {r.name: _candidates(r, found, back, weather) for r in spec.roles}
            roles = _assign(spec.roles, candidates)
            if roles is None or (best is not None and _breadth(candidates) <= _breadth(best.candidates)):
                continue
            brought = [i for picks in roles.values() for i in picks]
            lead = next((i for i in roles[spec.lead] if not back[i]), next((i for i in brought if not back[i]), 0))
            best = Fit(style, weather.name if weather else None, roles, candidates, lead)
        return best

    def plans(self, style: str) -> list[tuple[RoleDraw, ...]]:
        """The role members to draw first, one plan per weather for weather styles; untested roles are left out."""
        spec = self.style(style)
        weathers: list[Weather | None] = list(self.weathers.values()) if spec.uses_weather else [None]
        return [tuple(RoleDraw(r.name, r.min, w, r.test) for r in spec.roles if r.test is not None) for w in weathers]


def _candidates(role: Role, found: list[Facts], back: list[bool], weather: Weather | None) -> tuple[int, ...]:
    """Members that can fill the role; those that never lead come last."""
    fits = [i for i, f in enumerate(found) if role.test is None or role.test(f, weather)]
    return tuple(sorted(fits, key=back.__getitem__))


def _breadth(candidates: dict[str, tuple[int, ...]]) -> int:
    return sum(len(c) for c in candidates.values())


def _style(name: str, spec: dict) -> Style:
    roles = []
    for role_name, role in (spec.get("roles") or {}).items():
        conditions = {k: v for k, v in role.items() if k != "min"}
        roles.append(Role(role_name, int(role.get("min", 1)), _compile(conditions) if conditions else None))
    roles.sort(key=lambda r: r.test is None)
    lead = spec.get("lead") or (roles[0].name if roles else "")
    if lead not in {r.name for r in roles}:
        raise ConfigError(f"styles: {name} leads with {lead!r}, which is not one of its roles")
    back = frozenset(game.base_name(to_id_str(s)) for s in spec.get("back", []))
    return Style(name, tuple(roles), lead, back, _mentions_weather(spec.get("roles")))


def load(path: Path | None = None) -> Styles:
    path = path or paths.config_dir() / DEFAULT_PATH
    if not path.is_file():
        raise ConfigError(f"team styles missing: {path}")
    return Styles(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

"""The assumed set of an opposing member, and the damage-calculation sides both this module and `threat` use.

An assumed set is the species' top catalog build under what the battle revealed (item, ability, moves), then
corrected by what the opponent did: moving first when the uncorrected set says it is slower (Choice Scarf), or
hitting harder than the uncorrected set's max roll (a power item). `Tracker` records those observations.
Move ids are poke-env ids; items keep the catalog's display name or the revealed id (compare with to_id_str).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from poke_env.battle import Field, Move, MoveCategory, SideCondition
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.battle.builds import Build, Catalog
from pcsa.battle.damage import Conditions, Fighter, build_stats, damage, nature_percent
from pcsa.battle.moves import base_data
from pcsa.battle.stats import current_stats, form_of, known_ability, multipliers, stage
from pcsa.battle.teams import STAT_ORDER
from pcsa.players.tactics import effective_power

SCARF, LIFE_ORB = "Choice Scarf", "Life Orb"
POWER_ITEMS = ("choiceband", "choicespecs")
FALLBACK_NATURE = "Hardy"
FLAT_POINTS = 11  # 66 stat points spread evenly
MOVE_SLOTS = 4
STRONG_MARGIN = 1.15  # beyond the max roll by more than a nature (1.1) explains
PARALYSIS = 0.5
TAILWIND = 2.0
WEATHER_SPEED = 2.0
QUICK_FEET = 1.5
KNOCK_OFF = 1.5
SPEED_WEATHER = {
    "swiftswim": ("raindance", "primordialsea"),
    "chlorophyll": ("sunnyday", "desolateland"),
    "sandrush": ("sandstorm",),
    "slushrush": ("snowscape", "hail"),
}
VARIABLE_POWER = frozenset({"grassknot", "lowkick", "heavyslam", "heatcrash", "facade", "electroball", "gyroball"})
# Power that depends on state the strong check does not model: such hits are no evidence of a power item.
CONDITIONAL_POWER = frozenset(
    "acrobatics hex venoshock brine payback avalanche revenge assurance storedpower powertrip punishment "
    "boltbeak fishiousrend lastrespects ragefist weatherball terrainpulse risingvoltage expandingforce "
    "collisioncourse electrodrift knockoff facade retaliate stompingtantrum temperflare tripleaxel triplekick "
    "beatup present magnitude doubleshock burnup".split()  # the last two drop the user's type after the hit
)
ESCALATING_POWER = {"tripleaxel": 40, "triplekick": 20}  # mean power per hit of 20/40/60 and 10/20/30
NOT_MOVES = frozenset({"struggle", "recharge"})
EARLY_MOVERS = ("Quick Claw", "Custap Berry", "Quick Draw")


@dataclass(frozen=True)
class AssumedSet:
    species: str
    item: str | None
    ability: str | None
    nature: str
    points: dict[str, int]
    moves: tuple[str, ...]
    source: str  # "revealed+top" / "scarf-corrected" / "power-corrected" / "fallback"


@lru_cache(maxsize=1)
def default_catalog() -> Catalog:
    """The shipped build catalog, loaded once."""
    return Catalog.load()


@lru_cache(maxsize=1024)
def move_object(move_id: str) -> Move | None:
    """A shared read-only poke-env Move, or None for an id poke-env does not know."""
    move_id = to_id_str(move_id)
    return Move(move_id, gen=9) if move_id in GenData.from_gen(9).moves else None


def is_ours(battle, mon) -> bool:
    return any(member is mon for member in battle.team.values())


def revealed_item(mon) -> tuple[bool, str | None]:
    """(known, item); (True, None) once the item is known to be gone."""
    item = mon.item
    if item is None or item == GenData.UNKNOWN_ITEM:
        return False, None
    return True, item or None


def revealed_moves(mon) -> list[str]:
    return [m for m in mon.moves if m not in NOT_MOVES and base_data(m).get("type")]


# Assumed sets ---------------------------------------------------------------------------------------------------
def assumed(battle, mon, catalog: Catalog, tracker: Tracker | None = None) -> AssumedSet:
    """The set `mon` (an opposing member) is assumed to run; cached in the tracker per revealed signature."""
    if tracker is None:
        return _assume(battle, mon, catalog, None)
    species = to_id_str(mon.species)
    key = (
        species,
        form_of(mon, battle)[0],
        mon.item,
        known_ability(battle, mon),
        tuple(sorted(mon.moves)),
        species in tracker.fast,
        species in tracker.strong,
    )
    if key not in tracker.cache:
        tracker.cache[key] = _assume(battle, mon, catalog, tracker)
    return tracker.cache[key]


def _assume(battle, mon, catalog: Catalog, tracker: Tracker | None) -> AssumedSet:
    builds = catalog.builds(mon.species)
    form, mega = form_of(mon, battle)
    ability = game.mega_ability(game.species_entry(form)) if mega else (known_ability(battle, mon) or None)
    moves = revealed_moves(mon)
    if not builds:
        item = revealed_item(mon)[1]
        flat = {s: FLAT_POINTS for s in STAT_ORDER}
        return AssumedSet(mon.species, item, ability, FALLBACK_NATURE, flat, tuple(moves), "fallback")
    build, item, source = _base_build(builds, mon, form, mega, tracker)
    fill = [m for m in (to_id_str(x) for x in build.moves) if m not in moves]
    moves = (moves + fill)[: max(MOVE_SLOTS, len(moves))]
    points: dict[str, int] = {s: int(p) for s, p in zip(STAT_ORDER, build.points, strict=True)}
    return AssumedSet(mon.species, item, ability or build.ability, build.nature, points, tuple(moves), source)


def _base_build(builds: list[Build], mon, form: str, mega: bool, tracker: Tracker | None) -> tuple:
    """(build, item, source): the build the set starts from, the item it holds and why."""
    top = builds[0]
    if mega:
        stone = next(
            (b for b in builds if to_id_str(game.mega_form_for_stone(mon.species, b.item) or "") == form), None
        )
        item = stone.item if stone else game.species_entry(form).get("requiredItem")
        return stone or top, item, "revealed+top"
    known, item = revealed_item(mon)
    if known:
        return top, item, "revealed+top"
    species = to_id_str(mon.species)
    if tracker is not None and species in tracker.fast and to_id_str(top.item) != "choicescarf":
        scarf = next((b for b in builds if to_id_str(b.item) == "choicescarf"), None)
        return scarf or top, SCARF, "scarf-corrected"
    if tracker is not None and species in tracker.strong and to_id_str(top.item) not in POWER_ITEMS:
        power = next((b for b in builds if to_id_str(b.item) in POWER_ITEMS), None)
        return power or top, power.item if power else LIFE_ORB, "power-corrected"
    return top, top.item, "revealed+top"


# Damage-calculation sides ----------------------------------------------------------------------------------------
def _types(mon) -> tuple[str, ...]:
    return tuple(t.name.lower() for t in mon.types if t is not None)


def _status(mon) -> str:
    return mon.status.name.lower() if mon.status is not None else ""


def _base(mon, form: str) -> dict[str, int]:
    return game.species_entry(form).get("baseStats") or mon.base_stats


def own_fighter(battle, mon) -> Fighter:
    """One of our members: exact stats in its current form, its own item and ability."""
    form = form_of(mon, battle)[0]
    stats = {s: int(v) for s, v in current_stats(mon, True, form).items()}
    item = to_id_str(revealed_item(mon)[1] or "")
    return Fighter(
        stats, _types(mon), to_id_str(mon.ability or ""), item, dict(mon.boosts), _status(mon), _base(mon, form)
    )


def set_fighter(battle, mon, chosen: AssumedSet) -> Fighter:
    """An opposing member running `chosen`, in its current form."""
    base = _base(mon, form_of(mon, battle)[0])
    stats = build_stats(base, chosen.points, nature_percent(chosen.nature))
    item, ability = to_id_str(chosen.item or ""), to_id_str(chosen.ability or "")
    return Fighter(stats, _types(mon), ability, item, dict(mon.boosts), _status(mon), base)


def _weather(battle) -> str:
    return next((to_id_str(w.name) for w in battle.weather), "")


def conditions(battle, defender) -> Conditions:
    """Weather, terrain and the screens on the defender's side."""
    terrain = next((to_id_str(f.name) for f in battle.fields if f.name.endswith("TERRAIN")), "")
    side = battle.side_conditions if is_ours(battle, defender) else battle.opponent_side_conditions
    return Conditions(_weather(battle), terrain, frozenset(to_id_str(s.name) for s in side))


def move_power(attacker, defender, fa: Fighter, fd: Fighter, move: Move) -> int | None:
    """The resolved power of a variable-power move (None: the table's power)."""
    if move.id in VARIABLE_POWER:
        return int(effective_power(attacker, defender, move, fa.stats, fd.stats))
    if move.id in ESCALATING_POWER:
        return ESCALATING_POWER[move.id]
    if move.id == "knockoff" and fd.item:
        return int(base_data(move.id)["power"] * KNOCK_OFF)
    return None


def speed(battle, mon, fighter: Fighter) -> float:
    """Speed as the engine uses it: item and ability multipliers, weather abilities, stage, paralysis, Tailwind."""
    value = fighter.stats["spe"] * multipliers(fighter.item, fighter.ability, mon.species).get("spe", 1.0)
    if _weather(battle) in SPEED_WEATHER.get(fighter.ability, ()):
        value *= WEATHER_SPEED
    value = float(stage(int(value), int(fighter.boosts.get("spe", 0))))
    if fighter.ability == "quickfeet" and fighter.status:
        value *= QUICK_FEET
    elif fighter.status == "par":
        value *= PARALYSIS
    side = battle.side_conditions if is_ours(battle, mon) else battle.opponent_side_conditions
    return value * (TAILWIND if SideCondition.TAILWIND in side else 1.0)


# Observations ----------------------------------------------------------------------------------------------------
@dataclass
class _Hit:
    """One opposing move aimed at our member, and the HP it took."""

    attacker: str
    move: str
    target: str  # team key, "p1: Name"
    dealt: int = 0
    hits: int = 0
    crit: bool = False


def _key(ident: str) -> str:
    """'p1a: Name' -> 'p1: Name' (the team dict key)."""
    side, _, name = ident.partition(": ")
    return f"{side[:2]}: {name}"


def _hp(text: str) -> int | None:
    token = text.split(" ", 1)[0]
    if "/" in token:
        return int(token.split("/", 1)[0])
    return 0 if token == "0" else None


def _at(parts: list[str], index: int) -> str:
    """A protocol field, '' past the end."""
    return parts[index] if index < len(parts) else ""


def _called(parts: list[str]) -> bool:
    return any(p.startswith("[from]") for p in parts[4:])


def _spe_gained(chunk: list[list[str]], ident: str) -> int:
    """Speed stages `ident` gained during the turn (to undo them when judging who should have moved first)."""
    total = 0
    for parts in chunk:
        if _at(parts, 1) in ("-boost", "-unboost") and _at(parts, 2) == ident and _at(parts, 3) == "spe":
            total += int(parts[4]) * (1 if parts[1] == "-boost" else -1)
    return total


def _no_boosts(fighter: Fighter, keep) -> Fighter:
    return replace(fighter, boosts={k: keep(0, v) for k, v in fighter.boosts.items()})


class Tracker:
    """Per-battle evidence about the opponent's sets: species that moved first when the uncorrected set is slower
    (`fast`) and species that hit harder than the uncorrected set's max roll (`strong`). One per battle:
    `reset()` at team preview, `observe(battle)` every decision (a new battle tag also resets)."""

    def __init__(self, catalog: Catalog | None = None) -> None:
        self.catalog = catalog or default_catalog()
        self.reset()

    def reset(self) -> None:
        self.fast: set[str] = set()
        self.strong: set[str] = set()
        self.cache: dict[tuple, AssumedSet] = {}
        self._tag: str | None = None
        self._seen = 0
        self._hp: dict[str, int] = {}

    def observe(self, battle) -> None:
        """Read the turns finished since the last call."""
        if battle.battle_tag != self._tag:
            self.reset()
            self._tag = battle.battle_tag
        lines = battle._replay_data
        if len(lines) < self._seen:
            self._seen, self._hp = 0, {}
        start = self._seen
        for index in range(self._seen, len(lines)):
            if lines[index][1:2] == ["turn"]:
                self._turn(battle, lines[start:index])
                start = index + 1
        self._seen = start

    def _turn(self, battle, chunk: list[list[str]]) -> None:
        me = battle.player_role or "p1"
        order: list[tuple[str, str]] = []
        hit: _Hit | None = None
        for parts in chunk:
            kind, actor = _at(parts, 1), _at(parts, 2)
            if kind == "move":
                self._judge_hit(battle, hit)
                hit = None
                if not _called(parts):
                    order.append((actor, to_id_str(_at(parts, 3))))
                    if not actor.startswith(me) and _at(parts, 4).startswith(me):
                        hit = _Hit(actor, to_id_str(_at(parts, 3)), _key(_at(parts, 4)))
            elif kind == "-crit" and hit is not None and _key(actor) == hit.target:
                hit.crit = True
            elif actor.startswith(me):
                self._hp_line(parts, hit)
        self._judge_hit(battle, hit)
        self._judge_order(battle, chunk, order, me)

    def _hp_line(self, parts: list[str], hit: _Hit | None) -> None:
        kind, key = parts[1], _key(parts[2])
        if kind in ("switch", "drag", "replace"):
            value = _hp(_at(parts, 4))
        elif kind in ("-damage", "-heal", "-sethp"):
            value = _hp(_at(parts, 3))
        else:
            return
        if value is None:
            return
        before = self._hp.get(key)
        if kind == "-damage" and hit is not None and hit.target == key and before is not None and not _called(parts):
            hit.dealt += before - value
            hit.hits += 1
        self._hp[key] = value

    def _judge_hit(self, battle, hit: _Hit | None) -> None:
        """Mark the attacker strong when the hit beat the uncorrected set's max roll."""
        if hit is None or hit.crit or not hit.hits or hit.move in CONDITIONAL_POWER:
            return
        attacker, defender = battle.opponent_team.get(_key(hit.attacker)), battle.team.get(hit.target)
        move = move_object(hit.move)
        if attacker is None or defender is None or move is None or not defender.max_hp:
            return
        if attacker is not battle.opponent_active_pokemon or attacker.fainted:  # its stages are gone
            return
        fa = _no_boosts(set_fighter(battle, attacker, assumed(battle, attacker, self.catalog)), max)
        fd = _no_boosts(own_fighter(battle, defender), min)
        high = damage(fa, fd, move.id, conditions(battle, defender), move_power(attacker, defender, fa, fd, move))[1]
        if high > 0 and hit.dealt / defender.max_hp > high * hit.hits * STRONG_MARGIN + 1 / defender.max_hp:
            self.strong.add(to_id_str(attacker.species))

    def _judge_order(self, battle, chunk: list[list[str]], order: list[tuple[str, str]], me: str) -> None:
        """Mark their active fast when it moved first though our speed beat its uncorrected, scarf-less set's."""
        pair = _exchange(battle, chunk, order, me)
        if pair is None:
            return
        theirs, ours, our_ident = pair
        plain = set_fighter(battle, theirs, assumed(battle, theirs, self.catalog))
        if to_id_str(plain.item) == "choicescarf":
            plain = replace(plain, item="")
        mine = own_fighter(battle, ours)
        spe = int(mine.boosts.get("spe", 0)) - _spe_gained(chunk, our_ident)
        mine = replace(mine, boosts={**mine.boosts, "spe": spe})
        if speed(battle, ours, mine) > speed(battle, theirs, plain):
            self.fast.add(to_id_str(theirs.species))


def _exchange(battle, chunk: list[list[str]], order: list[tuple[str, str]], me: str) -> tuple | None:
    """(their mon, our mon, our ident) when their active moved first with a same-priority attack on a plain turn
    and both are still in; else None."""
    ours = next(((i, a, m) for i, (a, m) in enumerate(order) if a.startswith(me)), None)
    theirs = next(((i, a, m) for i, (a, m) in enumerate(order) if not a.startswith(me)), None)
    if ours is None or theirs is None or theirs[0] > ours[0] or Field.TRICK_ROOM in battle.fields:
        return None
    if any(early in part for parts in chunk for part in parts for early in EARLY_MOVERS):
        return None
    their_mon, our_mon = battle.opponent_team.get(_key(theirs[1])), battle.team.get(_key(ours[1]))
    if their_mon is None or their_mon is not battle.opponent_active_pokemon or our_mon is not battle.active_pokemon:
        return None
    first, second = move_object(theirs[2]), move_object(ours[2])
    if first is None or second is None or first.priority != second.priority or first.category == MoveCategory.STATUS:
        return None
    if revealed_item(their_mon)[0]:  # a shown item explains (or rules out) the speed
        return None
    return their_mon, our_mon, ours[1]

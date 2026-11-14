"""Switch away from a faster Encore user: the neural player's `encore_switch` guard rule (pcsa.players.neural_rules).

The user's rule (14 Nov 2026, live battles 14 and 35 of run 20261113-215202: a faster opposing Primarina Encored our
Primarina's Calm Mind every three turns, and on each free turn our Moonblast came after its next Encore): when their
active is the Pokemon that used Encore on us during its current stay on the field, and we know from earlier turns
that it moves before our active, we switch (the network picks which switch). It fires whether or not we are
encored right now; with no legal switch it does nothing.

`why(battle)` names the reason to switch now, or None. It reads the battle's protocol (`battle._replay_data`):

Encore user   their active used Encore (a `|move|` line of its own, not a called move) on our side since it last
              came in; a failed Encore counts too (it shows the move and the intent)
move order    a past turn where our active and their active (the same two Pokemon as now) each started the turn on
              the field and each used a move of its own, of the same priority, without an early-mover effect seen
              (Quick Claw, Custap Berry, Quick Draw) and without a move either side's possible ability may reorder
              (Prankster on a status move, Gale Wings on a Flying move, Triage on a healing move, Stall /
              Mycelium Might, a Lagging Tail or Full Incense); whoever moved first is that turn's observation
invalidation an observation counts only while the speed state is the one it was made in: both actives' Speed
              stages, statuses (paralysis, Quick Feet), Tailwind on either side, Trick Room, weather and terrain
              (speed abilities) are part of a signature compared with the signature now. Events that change speed
              in ways the signature cannot hold start a new epoch for that Pokemon and drop its earlier
              observations: a Mega Evolution or other form change, an item lost or swapped (Knock Off, Trick,
              a berry eaten: Unburden), a Choice Scarf revealed, Protosynthesis / Quark Drive / Slow Start, an
              odd boost operation (swap, copy, invert), a Baton Pass / Shed Tail switch
certainty     `known_first` needs no observation of ours moving first in that state, and at least
              MIN_OBSERVATIONS of theirs first; one observation is enough only when a speed tie and Quick Claw are
              both ruled out (their item known and not an early mover; plain state and our Speed outside their
              species' Speed range at level 50, Choice Scarf included unless their item is known). The live log
              does not show Quick Claw (the translator drops "can act faster than normal"), and a mirror match
              can tie, so in practice two observations are needed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from poke_env.data.normalize import to_id_str

from pcsa.battle import stats as stat_calc
from pcsa.players.opponent import EARLY_MOVERS, move_object, revealed_item
from pcsa.players.sucker_punch import ident

MIN_OBSERVATIONS = 2  # theirs-first turns needed when a tie or a hidden Quick Claw cannot be ruled out
WHO, MOVE, TARGET = 2, 3, 4  # |move|<who>|<move>|<target>
SCARF = 1.5
PAD = 6  # empty fields appended to a split line
MAX_STAGE = 6
EARLY_ITEMS = frozenset({"quickclaw", "custapberry", "laggingtail", "fullincense"})
LAST_ABILITIES = frozenset({"stall", "myceliummight"})
SWITCHES = frozenset({"switch", "drag", "replace"})
BOOSTS = frozenset({"-boost", "-unboost"})
CLEARS = frozenset({"-clearboost", "-clearallboost", "-clearnegativeboost"})
ODD_BOOSTS = frozenset({"-swapboost", "-copyboost", "-invertboost", "-setboost"})
FORM_CHANGES = frozenset({"detailschange", "-mega", "-primal", "-burst", "-formechange", "transform", "-transform"})
ITEM_MOVES = ("trick", "switcheroo", "thief", "covet", "bestow", "pickpocket", "magician", "symbiosis")
SPEED_STARTS = ("protosynthesis", "quarkdrive", "slowstart", "unburden")
STATUS_LINES = frozenset({"-status", "-curestatus", "-cureteam", "faint"})
PASSING = frozenset({"batonpass", "shedtail"})


@dataclass
class _Side:
    active: str = ""  # team key of the active, "p2: Name"
    spe: int = 0  # its Speed stage
    tailwind: bool = False
    passing: bool = False  # its last move passes stages (Baton Pass): the next switch-in's stages are unknown


@dataclass
class Facts:
    """What the protocol shows about the current pair."""

    ours: str = ""  # our active's key
    theirs: str = ""  # their active's key
    encored_us: bool = False  # their active used Encore on us since it came in
    theirs_first: int = 0  # observations in the current speed state with their active first
    ours_first: int = 0
    turns: list[int] = field(default_factory=list)  # the turns of the theirs-first observations
    plain: bool = True  # the current speed state has no stages, statuses, Tailwind, weather or terrain


class _Walk:
    """One pass over the protocol, keeping the speed state and the move-order observations per state."""

    def __init__(self, ours: str, theirs: str) -> None:
        self.me, self.foe = ours, theirs
        self.side = {ours: _Side(), theirs: _Side()}
        self.status: dict[str, str] = {}
        self.epoch: dict[str, int] = {}
        self.items: dict[str, str] = {}
        self.trick_room, self.weather, self.terrain = False, "", ""
        self.observed: dict[tuple, list[tuple[int, str]]] = {}  # signature -> [(turn, first side)]
        self.encorer: str | None = None  # their key that used Encore on us since it came in
        self.turn = 0
        self.start: dict[str, str] = {}  # the actives at the start of the turn
        self.movers: list[tuple[str, str, str, tuple]] = []  # (side, key, move id, signature at its move)
        self.spoiled = False  # an early mover or an unclear action this turn

    def bump(self, key: str) -> None:
        self.epoch[key] = self.epoch.get(key, 0) + 1

    def signature(self) -> tuple:
        a, b = self.side[self.me], self.side[self.foe]
        return (
            a.active,
            b.active,
            a.spe,
            b.spe,
            self.status.get(a.active, ""),
            self.status.get(b.active, ""),
            a.tailwind,
            b.tailwind,
            self.trick_room,
            self.weather,
            self.terrain,
            self.epoch.get(a.active, 0),
            self.epoch.get(b.active, 0),
        )

    def end_turn(self, battle) -> None:
        sides = [m[0] for m in self.movers]
        if not self.spoiled and sorted(sides) == sorted([self.me, self.foe]):
            first = self.movers[0]
            if all(self._same_bracket(battle, m) for m in self.movers) and self._equal_priority(battle):
                self.observed.setdefault(first[3], []).append((self.turn, first[0]))
        self.movers, self.spoiled = [], False
        self.start = {s: self.side[s].active for s in self.side}

    def _priority(self, battle, mover: tuple) -> int | None:
        """The move's priority, None when the mover's possible ability or item may change it or its order."""
        side, key, move_id, _ = mover
        move = move_object(move_id)
        if move is None:
            return None
        mon = (battle.team if side == self.me else battle.opponent_team).get(key)
        abilities = _abilities(mon)
        if abilities & LAST_ABILITIES:
            return None
        if move.category.name == "STATUS" and "prankster" in abilities:
            return None
        if move.type.name == "FLYING" and "galewings" in abilities:
            return None
        if "heal" in move.flags and "triage" in abilities:
            return None
        known, item = revealed_item(mon) if mon is not None else (False, None)
        if known and to_id_str(item or "") in EARLY_ITEMS:
            return None
        priority = int(move.priority)
        if move.id == "grassyglide" and self.terrain == "grassyterrain":
            priority += 1
        return priority

    def _same_bracket(self, battle, mover: tuple) -> bool:
        return self._priority(battle, mover) is not None

    def _equal_priority(self, battle) -> bool:
        a, b = (self._priority(battle, m) for m in self.movers)
        return a == b

    def line(self, battle, parts: list[str]) -> None:
        p = [*parts, *[""] * PAD]  # |kind|a|b|c|d ...: every field readable
        kind, who = p[1], p[WHO]
        side = who[:2]
        key = ident(who) if side in self.side else ""
        if any(early in part for part in parts for early in EARLY_MOVERS):
            self.spoiled = True  # Quick Claw / Custap Berry / Quick Draw: this turn's order says nothing
        if kind == "turn":
            self.end_turn(battle)
            self.turn = int(p[2]) if p[2].isdigit() else self.turn + 1
        elif kind in SWITCHES and key:
            state = self.side[side]
            if state.passing:
                self.bump(key)  # Baton Pass / Shed Tail: stages carried over, not tracked
            state.active, state.spe, state.passing = key, 0, False
            if side == self.foe:
                self.encorer = None  # a new stay on the field
        elif kind == "move" and key:
            self._move(parts, side, key)
        elif kind == "cant" and key:
            self.spoiled = True  # a turn lost (paralysis, sleep, flinch): no order seen
        elif kind in BOOSTS | CLEARS | ODD_BOOSTS:
            self._boost(kind, p, key)
        else:
            self._condition(kind, p, side, key)

    def _boost(self, kind: str, p: list[str], key: str) -> None:
        if kind in BOOSTS and key and p[3] == "spe" and self.side[key[:2]].active == key:
            step = int(p[4]) if p[4].lstrip("-").isdigit() else 0
            state = self.side[key[:2]]
            state.spe = max(-MAX_STAGE, min(MAX_STAGE, state.spe + (step if kind == "-boost" else -step)))
        elif kind in CLEARS:
            self._clear(kind, key)
        elif kind in ODD_BOOSTS:
            for k in (key, ident(p[3])):
                if k[:2] in self.side:
                    self.bump(k)

    def _condition(self, kind: str, p: list[str], side: str, key: str) -> None:
        """Statuses, Tailwind, Trick Room, weather, terrain, items, forms and speed volatiles."""
        if kind in FORM_CHANGES and key:
            self.bump(key)
        elif kind in STATUS_LINES and key:
            self._status(kind, p, side, key)
        elif kind in ("-sidestart", "-sideend") and "tailwind" in to_id_str(p[3]) and p[2][:2] in self.side:
            self.side[p[2][:2]].tailwind = kind == "-sidestart"
        elif kind in ("-fieldstart", "-fieldend"):
            self._field(kind, to_id_str(p[2].removeprefix("move: ")))
        elif kind == "-weather":
            self.weather = "" if p[2] in ("none", "") else to_id_str(p[2])
        elif kind in ("-item", "-enditem") and key:
            self._item(kind, p, key)
        elif kind in ("-start", "-end") and key and any(s in to_id_str(p[3]) for s in SPEED_STARTS):
            self.bump(key)

    def _status(self, kind: str, p: list[str], side: str, key: str) -> None:
        if kind == "-status":
            self.status[key] = p[3]
        elif kind == "-cureteam":
            for k in [k for k in self.status if k.startswith(side)]:
                self.status.pop(k)
        else:
            self.status.pop(key, None)

    def _move(self, parts: list[str], side: str, key: str) -> None:
        called = any(p.startswith("[from]") for p in parts[TARGET + 1 :])
        move_id = to_id_str(parts[MOVE]) if len(parts) > MOVE else ""
        if not called:
            self.side[side].passing = move_id in PASSING
        if called or key != self.start.get(side) or any(m[0] == side for m in self.movers):
            return
        self.movers.append((side, key, move_id, self.signature()))
        target = parts[TARGET] if len(parts) > TARGET else ""
        if side == self.foe and move_id == "encore" and (not target or target.startswith(self.me)):
            self.encorer = key

    def _clear(self, kind: str, key: str) -> None:
        if kind == "-clearallboost":
            for state in self.side.values():
                state.spe = 0
        elif key and self.side[key[:2]].active == key:
            state = self.side[key[:2]]
            state.spe = 0 if kind == "-clearboost" else max(state.spe, 0)  # -clearnegativeboost keeps a rise

    def _field(self, kind: str, name: str) -> None:
        on = kind == "-fieldstart"
        if name == "trickroom":
            self.trick_room = on
        elif name.endswith("terrain"):
            self.terrain = name if on else ""

    def _item(self, kind: str, parts: list[str], key: str) -> None:
        item = to_id_str(parts[3])
        source = " ".join(parts[4:]).lower().replace(" ", "")
        if kind == "-enditem" or any(m in source for m in ITEM_MOVES):
            self.bump(key)  # lost, eaten or swapped: Unburden, a Scarf gone or gained
        elif item == "choicescarf" and self.items.get(key) != item:
            self.bump(key)  # a Choice Scarf revealed (the user's list)
        self.items[key] = "" if kind == "-enditem" else item


def _abilities(mon) -> set[str]:
    if mon is None:
        return set()
    if mon.ability:
        return {to_id_str(mon.ability)}
    return {to_id_str(a) for a in (mon.possible_abilities or [])}


def _sides(battle) -> tuple[str, str]:
    ours = battle.player_role or "p1"
    return ours, "p2" if ours == "p1" else "p1"


def _key(battle, mon, team: dict) -> str:
    return next((k for k, v in team.items() if v is mon), "")


def facts(battle, protocol: Iterable[list[str]] | None = None) -> Facts:
    """The Encore user and the move-order observations for the current pair of actives."""
    ours, theirs = _sides(battle)
    walk = _Walk(ours, theirs)
    for parts in protocol if protocol is not None else getattr(battle, "_replay_data", []):
        walk.line(battle, parts)
    walk.end_turn(battle)  # an open turn where both already moved counts too
    now = walk.signature()  # the speed state at the decision (the open turn has no moves of both yet)
    out = Facts(
        ours=_key(battle, battle.active_pokemon, battle.team),
        theirs=_key(battle, battle.opponent_active_pokemon, battle.opponent_team),
    )
    out.encored_us = bool(out.theirs) and walk.encorer == out.theirs and walk.side[theirs].active == out.theirs
    if (now[0], now[1]) != (out.ours, out.theirs):
        return out  # the tracked actives disagree with the battle's: nothing is known
    for turn, first in walk.observed.get(now, []):
        if first == theirs:
            out.theirs_first += 1
            out.turns.append(turn)
        else:
            out.ours_first += 1
    out.plain = now[2:8] == (0, 0, "", "", False, False) and not now[9] and not now[10]
    return out


def tie_impossible(battle, plain: bool) -> bool:
    """A plain speed state, their item known and not an early mover, and our Speed outside their species' range."""
    mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
    known, item = revealed_item(theirs)
    if not plain or not known or to_id_str(item or "") in EARLY_ITEMS:
        return False
    our_speed = (mine.stats or {}).get("spe")
    if not our_speed or not theirs.base_stats:
        return False
    our_speed *= SCARF if to_id_str(mine.item or "") == "choicescarf" else 1.0
    low, high = stat_calc.stat_ranges(theirs.base_stats)["spe"]
    high *= SCARF if to_id_str(item or "") == "choicescarf" else 1.0
    return not low <= our_speed <= high


def known_first(battle, f: Facts | None = None) -> bool:
    """Whether the protocol shows for sure that their active moves before ours in the current speed state."""
    f = f or facts(battle)
    if f.ours_first or not f.theirs_first:
        return False
    return f.theirs_first >= MIN_OBSERVATIONS or tie_impossible(battle, f.plain)


def why(battle) -> str | None:
    """Why we should switch now (their active Encored us and surely moves first), else None."""
    if battle.active_pokemon is None or battle.opponent_active_pokemon is None:
        return None
    f = facts(battle)
    if not f.encored_us or not known_first(battle, f):
        return None
    turns = ", ".join(str(t) for t in f.turns)
    return f"{battle.opponent_active_pokemon.species} used Encore and moved first (turns {turns})"

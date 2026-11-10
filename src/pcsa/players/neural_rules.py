"""Override rules over the neural player's choice: clear-cut spots where the network is known to slip.

The network proposes an action and ranks every legal action (its capped probabilities). Each enabled rule looks at
the position and may replace the proposal; the first rule that fires decides, and the decision's notes carry
`override: <rule>`, the reason and the network's own choice. Everything else is left to the network.

General rules (any team):
  priority_ko      a positive-priority move KOs the foe and the network's pick does not KO it first: take the KO
  no_full_heal     a recovery move (or Rest) at full HP: the network's best other action
  locked_futile    Encore or a Choice lock leaves only moves that do nothing: the network's best switch
  yawn_switch      we are drowsy (Yawn), healthy, and cannot KO the foe now: the network's best switch
  no_setup_into_ko a setup move while the foe can KO us this turn: the network's best non-setup action
  useless_status   Stealth Rock / Spikes / Toxic Spikes already maxed, Yawn or a status move on a statused or
                   drowsy foe: the network's best other action
  sucker_read      Sucker Punch when the foe's last action was not an attack: the network's best other action
Team rules (only with that team, config/players/neural_rules.yaml):
  scarf_lock       a Choice Scarf member locked into a hit under `scarf_lock_below` of the foe's HP switches out
  rocks_first      a member with Stealth Rock sets it when it is not up, the foe cannot KO it and it cannot KO them

Damage and speed are pcsa.players.threat's estimates on the sets pcsa.players.opponent assumes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml
from poke_env.battle import Effect, Move, SideCondition
from poke_env.data.normalize import to_id_str

from pcsa.common.errors import ConfigError
from pcsa.players import threat
from pcsa.players.base import LegalAction
from pcsa.players.disciplined import RECOVERY, is_setup
from pcsa.players.opponent import move_object

HEALS = (RECOVERY - {"wish"}) | {"rest"}  # Wish heals the next member in: not a wasted heal at full HP
FULL_HP = 0.99
HAZARDS = {
    "stealthrock": (SideCondition.STEALTH_ROCK, 1),
    "spikes": (SideCondition.SPIKES, 3),
    "toxicspikes": (SideCondition.TOXIC_SPIKES, 2),
}
STATUS_MOVES = frozenset(
    "yawn thunderwave willowisp toxic poisonpowder sleeppowder spore hypnosis glare stunspore".split()
)
GENERAL = (
    "priority_ko",
    "no_full_heal",
    "locked_futile",
    "yawn_switch",
    "no_setup_into_ko",
    "useless_status",
    "sucker_read",
)
TEAM = ("scarf_lock", "rocks_first")
MODES = ("auto", "general", "none")
MIN_HP = 1e-3
LOOKBACK_TURNS = 1  # foe_attacked reads this turn so far and the one before
MOVE_FIELD = 3  # |move|<who>|<move>|...: the move's index in the split line


@dataclass
class RuleParams:
    futile_hit: float = 0.05  # locked_futile: every legal move hits under this share of the foe's max HP
    yawn_min_hp: float = 0.3  # yawn_switch: below this HP falling asleep costs little; stay in
    scarf_lock_below: float = 0.25  # scarf_lock: a Choice-locked hit under this share of the foe's current HP


def _params(data: dict | None, where: str, base: RuleParams | None = None) -> RuleParams:
    data = data or {}
    unknown = set(data) - {f.name for f in fields(RuleParams)}
    if unknown:
        raise ConfigError(f"unknown neural rule settings in {where}: {sorted(unknown)}")
    return RuleParams(**{**(vars(base) if base else {}), **data})


@dataclass
class Profile:
    name: str
    roster: tuple[str, ...]  # base species ids
    rules: tuple[str, ...]
    params: RuleParams = field(default_factory=RuleParams)


@dataclass
class RulesConfig:
    general: tuple[str, ...] = GENERAL
    profiles: tuple[Profile, ...] = ()
    params: RuleParams = field(default_factory=RuleParams)

    @classmethod
    def load(cls, path: Path) -> RulesConfig:
        data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}
        general = tuple(data.get("general", GENERAL))
        params = _params(data.get("params"), str(path))
        profiles = tuple(
            Profile(
                name,
                tuple(base(s) for s in spec.get("roster", ())),
                tuple(spec.get("rules", ())),
                _params(spec.get("params"), f"{path} ({name})", params),
            )
            for name, spec in (data.get("teams") or {}).items()
        )
        unknown = (set(general) - set(GENERAL)) | {r for p in profiles for r in p.rules if r not in GENERAL + TEAM}
        if unknown:
            raise ConfigError(f"unknown neural rules in {path}: {sorted(unknown)}")
        return cls(general, profiles, params)

    def match(self, species: Sequence[str]) -> Profile | None:
        """The profile whose whole roster is in our team (Mega forms by their base species)."""
        ours = {base(s) for s in species}
        return next((p for p in self.profiles if p.roster and set(p.roster) <= ours), None)

    def select(self, mode: str, species: Sequence[str]) -> tuple[tuple[str, ...], Profile | None]:
        """(enabled rules, the team profile): `none`; `general`; `auto` (general, plus the team's own when the
        roster matches); or a profile name (general plus that team's rules whatever the roster)."""
        if mode == "none":
            return (), None
        if mode == "general":
            return self.general, None
        if mode == "auto":
            profile = self.match(species)
        else:
            profile = next((p for p in self.profiles if p.name == mode), None)
            if profile is None:
                raise ConfigError(f"no neural rules profile {mode!r} (have {[p.name for p in self.profiles]})")
        return (self.general + profile.rules if profile else self.general), profile


def base(species: str) -> str:
    """The base species id: 'Charizard-Mega-X' -> 'charizard'."""
    return to_id_str(str(species).split("-Mega")[0])


def _move(action: LegalAction) -> Move | None:
    order = action.order.order if action.order is not None else None
    return order if isinstance(order, Move) else None


def _is_switch(action: LegalAction) -> bool:
    return action.order is not None and _move(action) is None


@dataclass
class Spot:
    """One decision as the rules see it."""

    battle: object
    legal: tuple[LegalAction, ...]
    chosen: LegalAction
    ranked: list[LegalAction]  # every legal action, the network's most likely first
    catalog: object
    tracker: object
    params: RuleParams

    @property
    def mine(self):
        return self.battle.active_pokemon  # type: ignore[attr-defined]

    @property
    def theirs(self):
        return self.battle.opponent_active_pokemon  # type: ignore[attr-defined]

    def moves(self) -> list[LegalAction]:
        return [a for a in self.legal if _move(a) is not None]

    def best(self, keep: Callable[[LegalAction], bool]) -> LegalAction | None:
        return next((a for a in self.ranked if keep(a)), None)

    def best_switch(self) -> LegalAction | None:
        return self.best(_is_switch)

    def hit(self, attacker, defender, move) -> float:
        return threat.hit_fraction(self.battle, attacker, defender, move, self.catalog, self.tracker)  # type: ignore[arg-type]

    def can_ko(self, attacker, defender) -> bool:
        return threat.can_ko(self.battle, attacker, defender, self.catalog, tracker=self.tracker)  # type: ignore[arg-type]

    def outspeeds(self) -> bool:
        return threat.outspeeds(self.battle, self.mine, self.theirs, self.catalog, tracker=self.tracker)  # type: ignore[arg-type]


Outcome = tuple[LegalAction, str] | None


def priority_ko(s: Spot) -> Outcome:
    pk = threat.priority_ko(s.battle, s.mine, s.theirs, s.catalog, s.tracker)  # type: ignore[arg-type]
    if pk is None or (pk == "suckerpunch" and not foe_attacked(s.battle)):
        return None
    action = s.best(lambda a: (m := _move(a)) is not None and m.id == to_id_str(pk))
    if action is None:
        return None
    move = _move(s.chosen)
    if move is not None and move.id != action.order.order.id:  # type: ignore[union-attr]
        kos = s.hit(s.mine, s.theirs, move) >= s.theirs.current_hp_fraction
        if kos and (move.priority > 0 or (move.priority == 0 and s.outspeeds())):
            return None  # the network's pick KOs first anyway
    return action, f"{pk} KOs {s.theirs.species} first"


def no_full_heal(s: Spot) -> Outcome:
    move = _move(s.chosen)
    if move is None or move.id not in HEALS or s.mine.current_hp_fraction < FULL_HP:
        return None
    other = s.best(lambda a: (m := _move(a)) is None or m.id not in HEALS)
    return (other, f"{move.id} at full HP") if other else None


def locked_futile(s: Spot) -> Outcome:
    moves = s.moves()
    if _is_switch(s.chosen) or not moves:
        return None
    ids = {_move(a).id for a in moves}  # type: ignore[union-attr]
    if Effect.ENCORE not in s.mine.effects and len(ids) != 1:
        return None
    if max(s.hit(s.mine, s.theirs, _move(a)) for a in moves) >= s.params.futile_hit:
        return None
    if any(_move(a).category.name == "STATUS" and not useless(s, _move(a)) for a in moves):  # type: ignore[union-attr,arg-type]
        return None  # a locked status move that still does something
    switch = s.best_switch()
    return (switch, f"locked into {sorted(ids)[0]}, which does nothing") if switch else None


def yawn_switch(s: Spot) -> Outcome:
    mine = s.mine
    if _is_switch(s.chosen) or Effect.YAWN not in mine.effects or mine.status is not None:
        return None
    if mine.current_hp_fraction < s.params.yawn_min_hp or s.can_ko(mine, s.theirs):
        return None
    switch = s.best_switch()
    return (switch, "drowsy: out before it falls asleep") if switch else None


def no_setup_into_ko(s: Spot) -> Outcome:
    move = _move(s.chosen)
    if move is None or not is_setup(move) or not s.can_ko(s.theirs, s.mine):
        return None
    other = s.best(lambda a: (m := _move(a)) is None or not is_setup(m))
    return (other, f"{move.id} while {s.theirs.species} can KO") if other else None


def useless_status(s: Spot) -> Outcome:
    move = _move(s.chosen)
    if move is None or not useless(s, move):
        return None
    other = s.best(lambda a: (m := _move(a)) is None or not useless(s, m))
    return (other, f"{move.id} does nothing here") if other else None


def sucker_read(s: Spot) -> Outcome:
    move = _move(s.chosen)
    if move is None or move.id != "suckerpunch" or foe_attacked(s.battle):
        return None
    other = s.best(lambda a: (m := _move(a)) is None or m.id != "suckerpunch")
    return (other, "Sucker Punch, but their last action was not an attack") if other else None


def scarf_lock(s: Spot) -> Outcome:
    moves = s.moves()
    ids = {_move(a).id for a in moves}  # type: ignore[union-attr]
    if _is_switch(s.chosen) or len(ids) != 1 or to_id_str(s.mine.item or "") != "choicescarf":
        return None
    hit = s.hit(s.mine, s.theirs, _move(moves[0]))
    if hit >= s.params.scarf_lock_below * max(s.theirs.current_hp_fraction, MIN_HP):
        return None
    switch = s.best_switch()
    return (switch, f"Choice-locked into {sorted(ids)[0]} for {hit:.2f}") if switch else None


def rocks_first(s: Spot) -> Outcome:
    rocks = s.best(lambda a: (m := _move(a)) is not None and m.id == "stealthrock")
    if rocks is None or rocks == s.chosen or useless(s, _move(rocks)):  # type: ignore[arg-type]
        return None
    if s.can_ko(s.theirs, s.mine) or s.can_ko(s.mine, s.theirs):
        return None
    return rocks, "Stealth Rock is not up and the turn is safe"


RULES: dict[str, Callable[[Spot], Outcome]] = {
    "priority_ko": priority_ko,
    "no_full_heal": no_full_heal,
    "locked_futile": locked_futile,
    "yawn_switch": yawn_switch,
    "no_setup_into_ko": no_setup_into_ko,
    "useless_status": useless_status,
    "sucker_read": sucker_read,
    "scarf_lock": scarf_lock,
    "rocks_first": rocks_first,
}


def apply(rules: Sequence[str], spot: Spot) -> tuple[LegalAction, str, str] | None:
    """(action, rule, reason) of the first rule that replaces the network's choice, else None."""
    if spot.mine is None or spot.theirs is None or not spot.moves():
        return None  # a forced switch or no tracked actives: the network's pick
    for name in rules:
        out = RULES[name](spot)
        if out is not None and out[0] != spot.chosen:
            return out[0], name, out[1]
    return None


def their_side(battle) -> str:
    return "p2" if (battle.player_role or "p1") == "p1" else "p1"


def foe_attacked(battle) -> bool:
    """Whether the foe's latest action (this turn so far, else last turn) was an attack; True when unknown."""
    side, turns = their_side(battle), 0
    for parts in reversed(getattr(battle, "_replay_data", [])):
        kind = parts[1] if len(parts) > 1 else ""
        if kind == "turn":
            turns += 1
            if turns == LOOKBACK_TURNS + 1:
                break
            continue
        if kind not in ("move", "switch", "drag") or len(parts) <= MOVE_FIELD or not parts[2].startswith(side):
            continue
        if any(p.startswith("[from]") for p in parts[4:]):
            continue
        if kind != "move":
            return False  # they switched
        move = move_object(to_id_str(parts[3]))
        return move is None or move.category.name != "STATUS"
    return True


def useless(s: Spot, move: Move) -> bool:
    """A hazard already at its maximum, or a status move on a foe that is statused (Yawn: or drowsy)."""
    if move.id in HAZARDS:
        condition, most = HAZARDS[move.id]
        conditions = s.battle.opponent_side_conditions  # type: ignore[attr-defined]
        if condition not in conditions:
            return False
        return most == 1 or conditions[condition] >= most  # Stealth Rock's value is the turn it was set
    if move.id in STATUS_MOVES:
        theirs = s.theirs
        return theirs.status is not None or (move.id == "yawn" and Effect.YAWN in theirs.effects)
    return False

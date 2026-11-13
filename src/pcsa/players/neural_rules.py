"""Guard rules over the neural player's choice: they stop plainly wasted actions and nothing else.

The network proposes an action and ranks every legal action (its capped probabilities). Each enabled rule looks at
the position and may replace the proposal with the network's best action that is not the mistake; the first rule
that fires decides, and the decision's notes carry `override: <rule>`, the reason and the network's own choice.

  no_effect       a move that surely does nothing to the foe: an immune type or ability (Earthquake into Levitate,
                  Electric into Volt Absorb), priority into Psychic Terrain, a status it cannot get, Fake Out after
                  the first turn ... (pcsa.players.no_effect)
  no_full_heal    Roost / Recover / Slack Off / Rest ... at full HP
  useless_status  Stealth Rock already up, Spikes or Toxic Spikes already at their maximum, or a status move
                  (Yawn, Toxic, Will-O-Wisp, Thunder Wave, sleep moves ...) on a foe that is already statused (Yawn:
                  or already drowsy)
  locked_futile   Encore or a Choice lock leaves only moves that do nothing: the network's best switch
  priority_ko     a priority move surely KOs the foe and the network's pick does not KO it first

Damage and speed are pcsa.players.threat's estimates on the sets pcsa.players.opponent assumes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from poke_env.battle import Effect, Move, SideCondition
from poke_env.data.normalize import to_id_str

from pcsa.common.errors import ConfigError
from pcsa.players import no_effect as nothing
from pcsa.players import threat
from pcsa.players.base import LegalAction
from pcsa.players.disciplined import RECOVERY
from pcsa.players.opponent import move_object

HEALS = (RECOVERY - {"wish"}) | {"rest"}  # Wish heals the next member in: not a wasted heal at full HP
FULL_HP = 0.99
FUTILE_HIT = 0.05  # locked_futile: every legal move hits under this share of the foe's max HP
HAZARDS = {
    "stealthrock": (SideCondition.STEALTH_ROCK, 1),
    "spikes": (SideCondition.SPIKES, 3),
    "toxicspikes": (SideCondition.TOXIC_SPIKES, 2),
}
STATUS_MOVES = frozenset(
    "yawn thunderwave willowisp toxic poisonpowder sleeppowder spore hypnosis glare stunspore".split()
)
LOOKBACK_TURNS = 1  # foe_attacked reads this turn so far and the one before
MOVE_FIELD = 3  # |move|<who>|<move>|...: the move's index in the split line


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

    def hit(self, move) -> float:
        return threat.hit_fraction(self.battle, self.mine, self.theirs, move, self.catalog, self.tracker)  # type: ignore[arg-type]

    def outspeeds(self) -> bool:
        return threat.outspeeds(self.battle, self.mine, self.theirs, self.catalog, tracker=self.tracker)  # type: ignore[arg-type]


Outcome = tuple[LegalAction, str] | None


def no_effect(s: Spot) -> Outcome:
    move = _move(s.chosen)
    reason = nothing.why(s.battle, s.mine, s.theirs, move) if move is not None else None
    if reason is None:
        return None
    other = s.best(lambda a: (m := _move(a)) is None or not useless(s, m))
    return (other, f"{move.id}: {reason}") if other else None  # type: ignore[union-attr]


def no_full_heal(s: Spot) -> Outcome:
    move = _move(s.chosen)
    if move is None or move.id not in HEALS or s.mine.current_hp_fraction < FULL_HP:
        return None
    other = s.best(lambda a: (m := _move(a)) is None or m.id not in HEALS)
    return (other, f"{move.id} at full HP") if other else None


def useless_status(s: Spot) -> Outcome:
    move = _move(s.chosen)
    if move is None or not useless(s, move):
        return None
    other = s.best(lambda a: (m := _move(a)) is None or not useless(s, m))
    return (other, f"{move.id} does nothing here") if other else None


def locked_futile(s: Spot) -> Outcome:
    moves = s.moves()
    if _is_switch(s.chosen) or not moves:
        return None
    ids = {_move(a).id for a in moves}  # type: ignore[union-attr]
    if Effect.ENCORE not in s.mine.effects and len(ids) != 1:
        return None
    if max(s.hit(_move(a)) for a in moves) >= FUTILE_HIT:
        return None
    if any(_move(a).category.name == "STATUS" and not useless(s, _move(a)) for a in moves):  # type: ignore[union-attr,arg-type]
        return None  # a locked status move that still does something
    switch = s.best(_is_switch)
    return (switch, f"locked into {sorted(ids)[0]}, which does nothing") if switch else None


def priority_ko(s: Spot) -> Outcome:
    pk = threat.priority_ko(s.battle, s.mine, s.theirs, s.catalog, s.tracker)  # type: ignore[arg-type]
    if pk is None or (pk == "suckerpunch" and not foe_attacked(s.battle)):
        return None  # Sucker Punch fails unless they attack
    action = s.best(lambda a: (m := _move(a)) is not None and m.id == to_id_str(pk))
    if action is None or useless(s, _move(action)):  # type: ignore[arg-type]
        return None  # e.g. the priority move cannot land (Psychic Terrain)
    move = _move(s.chosen)
    if move is not None and move.id != _move(action).id:  # type: ignore[union-attr]
        kos = s.hit(move) >= s.theirs.current_hp_fraction
        if kos and (move.priority > 0 or (move.priority == 0 and s.outspeeds())):
            return None  # the network's pick KOs first anyway
    return action, f"{pk} KOs {s.theirs.species} first"


RULES: dict[str, Callable[[Spot], Outcome]] = {
    "no_effect": no_effect,
    "no_full_heal": no_full_heal,
    "useless_status": useless_status,
    "locked_futile": locked_futile,
    "priority_ko": priority_ko,
}


def check(rules: Sequence[str], where: str) -> None:
    unknown = sorted(set(rules) - set(RULES))
    if unknown:
        raise ConfigError(f"{where}: unknown neural rules {unknown} (have {sorted(RULES)})")


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
        move = move_object(to_id_str(parts[MOVE_FIELD]))
        return move is None or move.category.name != "STATUS"
    return True


def useless(s: Spot, move: Move) -> bool:
    """A hazard already at its maximum, a status move on a foe that is statused (Yawn: or drowsy), or a move that
    surely does nothing to the foe (pcsa.players.no_effect)."""
    if nothing.why(s.battle, s.mine, s.theirs, move) is not None:
        return True
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

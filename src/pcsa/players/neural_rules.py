"""Guard rules over the neural player's choice: they stop plainly wasted actions and nothing else.

neural.yaml `rule_mode`: override (a rule replaces the pick with the network's best other action) or mask (forbidden
actions leave the softmax). Rules: no_effect (move does nothing), no_full_heal, useless_status (hazard or status
already set), locked_futile (Encore/lock leaves only useless moves), sucker_punch, priority_ko, encore_switch,
mega_weakness (Mega form 4x weak), weak4x_switch, yawn_switch. Damage and speed: pcsa.players.threat."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from poke_env.battle import Effect, Field, Move, SideCondition
from poke_env.data.normalize import to_id_str

from pcsa.common.errors import ConfigError
from pcsa.players import encore_switch as encore
from pcsa.players import mega_weakness as weak
from pcsa.players import no_effect as nothing
from pcsa.players import sucker_punch as sucker
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
    their: tuple[sucker.TheirAction, ...] | None = None  # the opponent head's kept actions; None: unknown
    passive_max: float = sucker.PASSIVE_MAX
    enabled: frozenset[str] = field(default_factory=frozenset)
    # this decision's facts that do not depend on the chosen action (shared by `replace` copies: mask mode asks
    memo: dict = field(default_factory=dict)

    def once(self, key, work: Callable[[], object]):
        if key not in self.memo:
            self.memo[key] = work()
        return self.memo[key]

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
        return self.once(
            ("hit", getattr(move, "id", move)),
            lambda: threat.hit_fraction(self.battle, self.mine, self.theirs, move, self.catalog, self.tracker),  # type: ignore[arg-type]
        )

    def taken(self, move) -> float:
        """Their `move` on our active, as a share of our max HP."""
        return self.once(
            ("taken", getattr(move, "id", move)),
            lambda: threat.hit_fraction(self.battle, self.theirs, self.mine, move, self.catalog, self.tracker),  # type: ignore[arg-type]
        )

    def outspeeds(self) -> bool:
        return self.once(
            "outspeeds",
            lambda: threat.outspeeds(self.battle, self.mine, self.theirs, self.catalog, tracker=self.tracker),  # type: ignore[arg-type]
        )


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


def sucker_blocked(s: Spot, move: Move | None) -> str | None:
    """Why `move` (one that fails unless the target attacks) is blocked here, else None."""
    if move is None or move.id not in sucker.FAIL_UNLESS_ATTACKED:
        return None
    return sucker.why(s.battle, s.theirs, s.their, s.passive_max)


def sucker_punch(s: Spot) -> Outcome:
    move = _move(s.chosen)
    reason = sucker_blocked(s, move)
    if reason is None:
        return None

    def ok(a: LegalAction, strict: bool) -> bool:
        m = _move(a)
        return m is None or (m.id not in sucker.FAIL_UNLESS_ATTACKED and not (strict and useless(s, m)))

    other = s.best(lambda a: ok(a, True)) or s.best(lambda a: ok(a, False))
    return (other, f"{move.id}: {reason}") if other else None  # type: ignore[union-attr]


def switch_in_top3(their) -> bool:
    """Whether a switch is among the opponent head's three likeliest actions (then the target may not be there)."""
    return bool(their) and any(kind == "switch" for kind, _, _ in sorted(their, key=lambda t: -t[2])[:3])


def priority_ko(s: Spot) -> Outcome:
    if switch_in_top3(s.their):
        return None  # a likely switch: the KO may hit the switch-in instead, so the network decides
    pk = s.once("priority_ko", lambda: threat.priority_ko(s.battle, s.mine, s.theirs, s.catalog, s.tracker))  # type: ignore[arg-type]
    if pk is None or (pk == "suckerpunch" and not foe_attacked(s.battle)):
        return None  # Sucker Punch fails unless they attack
    if "sucker_punch" in s.enabled and sucker_blocked(s, move_object(to_id_str(pk))) is not None:
        return None  # the sucker_punch rule would stop it
    action = s.best(lambda a: (m := _move(a)) is not None and m.id == to_id_str(pk))
    if action is None or useless(s, _move(action)):  # type: ignore[arg-type]
        return None  # e.g. the priority move cannot land (Psychic Terrain)
    move = _move(s.chosen)
    if move is not None and move.id != _move(action).id:  # type: ignore[union-attr]
        kos = s.hit(move) >= s.theirs.current_hp_fraction
        if kos and (move.priority > 0 or (move.priority == 0 and s.outspeeds())):
            return None  # the network's pick KOs first anyway
    return action, f"{pk} KOs {s.theirs.species} first"


def encore_switch(s: Spot) -> Outcome:
    if _is_switch(s.chosen):
        return None
    reason = s.once("encore_why", lambda: encore.why(s.battle))
    switch = s.best(_is_switch) if reason is not None else None
    return (switch, reason) if switch is not None and reason is not None else None


def mega_weakness(s: Spot) -> Outcome:
    move = _move(s.chosen)
    if move is None or not s.chosen.action.mega:
        return None
    reason = weak.mega_reason(s, move)
    if reason is None:
        return None
    plain = s.best(lambda a: (m := _move(a)) is not None and m.id == move.id and not a.action.mega)
    other = plain or s.best(lambda a: not a.action.mega)
    return (other, reason) if other else None


def weak4x_switch(s: Spot) -> Outcome:
    if _is_switch(s.chosen):
        return None
    found = weak.doomed_by(s)
    if found is None:
        return None
    attack, p = found
    switch = s.best(lambda a: _is_switch(a) and weak.safe_against(a.order.order, attack))  # type: ignore[union-attr]
    said = f"{p:.0%} likely" if p is not None else "shown"
    return (switch, f"{attack.id} ({said}) hits {s.mine.species} 4x first and KOs it") if switch else None


NO_SLEEP_ABILITIES = frozenset({"insomnia", "vitalspirit", "sweetveil", "comatose"})


def can_fall_asleep(s: Spot) -> bool:
    mine = s.mine
    if mine.status is not None or to_id_str(mine.ability or "") in NO_SLEEP_ABILITIES:
        return False
    terrain = Field.ELECTRIC_TERRAIN in s.battle.fields or Field.MISTY_TERRAIN in s.battle.fields  # type: ignore[attr-defined]
    return not (terrain and s.battle.is_grounded(mine))  # type: ignore[attr-defined]


def yawn_switch(s: Spot) -> Outcome:
    if _is_switch(s.chosen) or Effect.YAWN not in s.mine.effects or not can_fall_asleep(s):
        return None
    if s.mine.first_turn:
        return None  # Yawned as it came in: switching again only restarts the loop; it attacks instead
    hp = s.theirs.current_hp_fraction
    if any(s.hit(_move(a)) >= hp for a in s.moves()):
        return None  # a KO this turn is worth the sleep
    switch = s.best(_is_switch)
    reason = f"{s.mine.species} is drowsy and no move KOs {s.theirs.species}: switch before it falls asleep"
    return (switch, reason) if switch else None


RULES: dict[str, Callable[[Spot], Outcome]] = {
    "no_effect": no_effect,
    "no_full_heal": no_full_heal,
    "useless_status": useless_status,
    "locked_futile": locked_futile,
    "sucker_punch": sucker_punch,
    "priority_ko": priority_ko,
    "encore_switch": encore_switch,
    "mega_weakness": mega_weakness,
    "weak4x_switch": weak4x_switch,
    "yawn_switch": yawn_switch,
}


def check(rules: Sequence[str], where: str) -> None:
    unknown = sorted(set(rules) - set(RULES))
    if unknown:
        raise ConfigError(f"{where}: unknown neural rules {unknown} (have {sorted(RULES)})")


def apply(rules: Sequence[str], spot: Spot) -> tuple[LegalAction, str, str] | None:
    """(action, rule, reason) of the first rule that replaces the network's choice, else None."""
    if spot.mine is None or spot.theirs is None or not spot.moves():
        return None  # a forced switch or no tracked actives: the network's pick
    spot.enabled = frozenset(rules)
    for name in rules:
        out = RULES[name](spot)
        if out is not None and out[0] != spot.chosen:
            return out[0], name, out[1]
    return None


SAME_MOVE_OK = frozenset({"priority_ko"})  # mask mode: the rule's move with or without Mega Evolution is allowed


def masks(rules: Sequence[str], spot: Spot) -> dict[str, tuple[str, str]]:
    """Mask mode: {legal action's message: (rule, reason)} for every legal action an enabled rule would replace
    were it the network's choice (the first such rule); {} when the rules do not apply (as `apply`)."""
    if spot.mine is None or spot.theirs is None or not spot.moves():
        return {}
    spot.enabled = frozenset(rules)
    out: dict[str, tuple[str, str]] = {}
    for action in spot.legal:
        trial = replace(spot, chosen=action)
        for name in rules:
            res = RULES[name](trial)
            if res is None or res[0] == action:
                continue
            if name in SAME_MOVE_OK and _same_move(res[0], action):
                continue
            out[action.message] = (name, res[1])
            break
    return out


def _same_move(a: LegalAction, b: LegalAction) -> bool:
    ma, mb = _move(a), _move(b)
    return ma is not None and mb is not None and ma.id == mb.id


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

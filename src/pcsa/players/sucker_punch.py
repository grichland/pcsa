"""Moves that fail unless the target attacks (Sucker Punch, Thunderclap): the neural player's `sucker_punch` guard
rule (pcsa.players.neural_rules).

`why(battle, theirs, their_actions, passive_max)` names the reason such a move should not be played now, or None:

  failed   the move already failed this battle against the same opposing Pokemon (our `|move|` line of it, then
           our `|-fail|`): that foe has shown it plays around it (Calm Mind, Recover, Protect ...). A fail into a
           Pokemon that switched in that turn does not count against it: the switch made it fail, not its choice
  passive  the network's opponent head gives their active's non-attacking actions (status moves and switches)
           more than `passive_max` of its probability; the distribution is the one the matrix weighs (the
           decision notes' matrix.p_cols over matrix.cols), status vs damaging from poke-env's move data

`their_actions` holds (kind, move id, probability) per kept column: kind "move" or "switch"; a move poke-env does
not know counts as neither (it is left out of the passive share).
"""

from __future__ import annotations

from collections.abc import Sequence

from poke_env.data.normalize import to_id_str

from pcsa.players.opponent import move_object

FAIL_UNLESS_ATTACKED = frozenset({"suckerpunch", "thunderclap"})
PASSIVE_MAX = 0.5  # the default share of their non-attacking actions above which the move is blocked
WHO_FIELD, MOVE_FIELD = 2, 3  # |move|<who>|<move>|<target>
ENDS = frozenset({"move", "switch", "drag", "replace", "turn", "upkeep", "faint", "cant"})

TheirAction = tuple[str, str, float]  # (kind "move" | "switch", move id or "", probability)


def _sides(battle) -> tuple[str, str]:
    ours = battle.player_role or "p1"
    return ours, "p2" if ours == "p1" else "p1"


def ident(position: str) -> str:
    """'p2a: Charizard' -> 'p2: Charizard' (the key of the team dicts)."""
    side, rest = position[:3], position[3:]
    if len(side) == len("p2a") and side[2] in "abc" and rest.startswith(":"):
        return side[:2] + rest
    return position


def failed_against(battle, moves: frozenset[str] = FAIL_UNLESS_ATTACKED) -> set[str]:
    """The opposing Pokemon (team keys, 'p2: Name') against which one of our `moves` failed this battle."""
    ours, theirs = _sides(battle)
    out: set[str] = set()
    pending: str | None = None
    switched: set[str] = set()  # their Pokemon that came in this turn
    for parts in getattr(battle, "_replay_data", []):
        kind = parts[1] if len(parts) > 1 else ""
        who = parts[2] if len(parts) > WHO_FIELD else ""
        if kind == "turn":
            switched = set()
        elif kind in ("switch", "drag", "replace") and who.startswith(theirs):
            switched.add(ident(who))
        if kind == "-fail" and pending and who.startswith(ours):
            out.add(pending)
            pending = None
            continue
        if kind in ENDS:
            pending = None
        if kind != "move" or len(parts) <= MOVE_FIELD + 1 or not who.startswith(ours):
            continue
        target = ident(parts[MOVE_FIELD + 1])
        if to_id_str(parts[MOVE_FIELD]) in moves and target.startswith(theirs) and target not in switched:
            pending = target
    return out


def key_of(battle, mon) -> str | None:
    """`mon`'s key in the opposing team ('p2: Name'), or None."""
    return next((k for k, v in battle.opponent_team.items() if v is mon), None)


def passive_share(their_actions: Sequence[TheirAction]) -> float:
    """The probability their active does not attack this turn: their status moves plus their switches."""
    total = 0.0
    for kind, move_id, p in their_actions:
        move = move_object(move_id) if kind == "move" else None
        if kind == "switch" or (move is not None and move.category.name == "STATUS"):
            total += p
    return total


def why(battle, theirs, their_actions: Sequence[TheirAction] | None, passive_max: float = PASSIVE_MAX) -> str | None:
    """Why a move that fails unless the target attacks should not be played into `theirs` now, or None."""
    key = key_of(battle, theirs)
    if key is not None and key in failed_against(battle):
        return f"already failed against {theirs.species} this battle"
    if their_actions:
        share = passive_share(their_actions)
        if share > passive_max:
            return f"P(they do not attack) {share:.2f} > {passive_max:g}"
    return None

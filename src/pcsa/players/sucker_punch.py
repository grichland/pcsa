"""Moves that fail unless the target attacks (Sucker Punch, Thunderclap): the neural player's `sucker_punch` guard
rule (pcsa.players.neural_rules).

`why(battle, theirs, their_actions, passive_max)` names the reason such a move should not be played now, or None:

  passive  the network's opponent head gives their active's non-attacking actions (status moves and switches)
           more than `passive_max` of its probability; the distribution is the one the matrix weighs (the
           decision notes' matrix.p_cols over matrix.cols), status vs damaging from poke-env's move data

Switches that cannot happen are left out of that share (15 Nov 2026, user): once as many distinct opposing Pokemon
have been seen as they brought (|teamsize|, 3 here), a preview-only Pokemon is not in their team, so a switch to it
is not one of their options. Its columns are masked for the rule only (the share is renormalised over the rest);
the network's input features are unchanged. Fainted ones are already closed by the opponent head.
The rule used to also block a move that "already failed against this foe this battle"; that clause was wrong in 4
of 5 labelled live cases and was dropped (15 Nov 2026, user).

`their_actions` holds (kind, id, probability) per kept column: kind "move" (id: the move id) or "switch" (id: the
switch target's species id, "" when unknown; an unknown target always counts). A move poke-env does not know counts
as neither (it is left out of the passive share).
"""

from __future__ import annotations

from collections.abc import Sequence

from poke_env.data.normalize import to_id_str

from pcsa.battle import game
from pcsa.players.opponent import move_object

FAIL_UNLESS_ATTACKED = frozenset({"suckerpunch", "thunderclap"})
PASSIVE_MAX = 0.5  # the default share of their non-attacking actions above which the move is blocked
BROUGHT = 3  # their team size when the battle did not announce one

TheirAction = tuple[str, str, float]  # (kind "move" | "switch", move id or switch target species id, probability)


def _sides(battle) -> tuple[str, str]:
    ours = battle.player_role or "p1"
    return ours, "p2" if ours == "p1" else "p1"


def ident(position: str) -> str:
    """'p2a: Charizard' -> 'p2: Charizard' (the key of the team dicts)."""
    side, rest = position[:3], position[3:]
    if len(side) == len("p2a") and side[2] in "abc" and rest.startswith(":"):
        return side[:2] + rest
    return position


def _bases(species: str) -> set[str]:
    return {to_id_str(species), game.base_name(species)}


def seen_species(battle) -> set[str]:
    """Species ids (and base species ids) of the opposing Pokemon seen in battle (not the preview-only ones)."""
    out: set[str] = set()
    for key, mon in battle.opponent_team.items():
        if ": " in key:  # preview keys are bare species
            out |= _bases(mon.species)
    return out


def team_closed(battle) -> bool:
    """As many distinct opposing Pokemon seen as they brought: the preview-only ones are not in their team."""
    _, theirs = _sides(battle)
    size = getattr(battle, "_team_size", {}).get(theirs) or BROUGHT
    return sum(1 for key in battle.opponent_team if ": " in key) >= size


def impossible_switch(battle, target: str) -> bool:
    """A switch to `target` (a species id) cannot happen: their team is closed and it was never seen."""
    return bool(target) and team_closed(battle) and not (_bases(target) & seen_species(battle))


def passive_share(their_actions: Sequence[TheirAction], battle=None) -> float:
    """The probability their active does not attack this turn: their status moves plus their switches. With
    `battle`, switches that cannot happen (impossible_switch) are masked and the rest renormalised."""
    total = possible = passive = 0.0
    for kind, target, p in their_actions:
        total += p
        if kind == "switch" and battle is not None and impossible_switch(battle, target):
            continue
        possible += p
        move = move_object(target) if kind == "move" else None
        if kind == "switch" or (move is not None and move.category.name == "STATUS"):
            passive += p
    if possible <= 0:
        return 0.0
    return passive if possible == total else passive * total / possible


def last_one_each(battle) -> bool:
    """One Pokemon left on each side: ours by the team we hold, theirs by the team size the battle announced
    (|teamsize|) minus their fainted. Unknown team size: False."""
    ours, theirs = _sides(battle)
    size = getattr(battle, "_team_size", {}).get(theirs)
    if not size:
        return False
    their_left = size - sum(1 for mon in battle.opponent_team.values() if mon.fainted)
    our_left = sum(1 for mon in battle.team.values() if not mon.fainted)
    return their_left == 1 and our_left == 1


def why(battle, theirs, their_actions: Sequence[TheirAction] | None, passive_max: float = PASSIVE_MAX) -> str | None:
    """Why a move that fails unless the target attacks should not be played into `theirs` now, or None.

    In a 1v1 (one Pokemon left on each side) the chance they do not attack is taken as 0: never blocked (the
    user's rule, 14 Nov 2026)."""
    if last_one_each(battle):
        return None
    if their_actions:
        share = passive_share(their_actions, battle)
        if share > passive_max:
            return f"P(they do not attack) {share:.2f} > {passive_max:g}"
    return None

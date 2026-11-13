# From research/pcsa-rl/rl2/matrix.py (stage 3 of research/pcsa-belief), split into matrix_cells (the cell
# arithmetic and battle readers) and matrix (the lookahead over a belief) for pcsa's file and function limits; the
# arithmetic is unchanged, in the same order (tests/rl_training/test_matrix.py compares it with research's).
"""Matrix blend: a one-turn lookahead per (our action, their action) cell from the belief tracker's damage
calculator, blended with the network's policy logits (FormerPlayer.matrix_blend; 0 = off, nothing here is imported).

Per decision, with our active A and their active B:
  our actions   the legal ones: moves 0-3, the same moves with Mega Evolution 4-7, switches 8-10
  their actions the opponent head's kept columns (former.Net: their K likeliest moves, revealed ones first, then
                their 5 bench rows as switches); weights p_j = the head's kept_p (only p_j >= P_MIN are computed, the
                rest is renormalised away)

Cell value V(i, j), in units of one member's HP (both sides symmetric):
  dealt  = (1 - pD) * min(D, hT) + pD * hT + KO_BONUS * pD       D: belief mean damage x hits x accuracy (fraction of
                                                                  the target's max HP), pD: P(KO) at its HP hT
  taken  = the same for their hit on our target
  move vs move:     V = pf * (dealt - (1 - pD) * taken) + (1 - pf) * ((1 - pT) * dealt - taken)
                    pf = P(our move resolves first): priority, then the belief's P(speed), Trick Room swapping it
  our switch vs their move:  V = -taken (their move on the member coming in, after its hazard damage)
  our move vs their switch:  V = dealt (our move on their incoming member, after its hazard damage)
  both switch:               V = their hazard damage - ours
  Mega actions use our Mega form (stats, types, ability, speed) for both directions.
  Protect-like moves zero the other side's hit; Sucker Punch-like moves fail against a switch or a non-damaging move;
  First Impression / Fake Out fail after the first turn; Focus Sash / Sturdy at full HP stop a single-hit KO (theirs:
  by the belief's item / ability probabilities); Endeavor, Seismic Toss / Night Shade and Super Fang are handled
  as fixed damage.

Q_i = sum_j p_j V(i, j); the player's logits become logits + matrix_blend * Q (masked actions stay masked).
"""

from __future__ import annotations

from dataclasses import dataclass

from poke_env.battle import PokemonType, SideCondition

KO_BONUS = 0.3  # a faint is worth this much beyond the HP it removes
P_MIN = 0.01  # opposing actions below this head probability are not computed
PROTECT = frozenset(
    "protect detect spikyshield kingsshield banefulbunker silktrap burningbulwark obstruct maxguard".split()
)
SUCKER = frozenset({"suckerpunch", "thunderclap"})  # fail unless the target uses a damaging move
FIRST_TURN = frozenset({"fakeout", "firstimpression"})
FIXED_50 = frozenset({"seismictoss", "nightshade"})  # level 50: 50 HP
HALF_HP = frozenset({"superfang", "ruination", "naturesmadness"})
ACTIONS, MEGA_AT, SWITCH_AT = 11, 4, 8


@dataclass
class Hit:
    """One attack on one target: mean damage (fraction of the target's max HP, x hits x accuracy), P(KO) at the
    target's HP before the hit, and the move facts the cell needs."""

    mean: float = 0.0
    ko: float = 0.0
    priority: int = 0
    damaging: bool = False
    protect: bool = False
    sucker: bool = False


NO_HIT = Hit()


def dealt_value(hit: Hit, hp: float, ko_bonus: float = KO_BONUS) -> float:
    """The worth of `hit` on a target at `hp` (fraction of its max HP)."""
    if hp <= 0:
        return 0.0
    return (1.0 - hit.ko) * min(hit.mean, hp) + hit.ko * hp + ko_bonus * hit.ko


def exchange(ours: Hit, theirs: Hit, p_first: float, h_them: float, h_us: float, ko_bonus: float = KO_BONUS) -> float:
    """V of a move-vs-move cell (both on the field): see the module docstring."""
    if ours.protect:
        return 0.0
    if theirs.protect:
        return 0.0
    if ours.sucker and not theirs.damaging:
        ours = NO_HIT
    if theirs.sucker and not ours.damaging:
        theirs = NO_HIT
    d, t = dealt_value(ours, h_them, ko_bonus), dealt_value(theirs, h_us, ko_bonus)
    return p_first * (d - (1.0 - ours.ko) * t) + (1.0 - p_first) * ((1.0 - theirs.ko) * d - t)


def first_prob(our_priority: int, their_priority: int, p_they_faster: float, p_tie: float, trick_room: bool) -> float:
    """P(our move resolves before theirs)."""
    if our_priority != their_priority:
        return 1.0 if our_priority > their_priority else 0.0
    p_we_faster = max(0.0, 1.0 - p_they_faster - p_tie)
    if trick_room:
        p_we_faster, p_they_faster = p_they_faster, p_we_faster
    return p_we_faster + 0.5 * p_tie


def q_values(rows: dict, p: dict) -> dict:
    """Q_i = sum_j p_j V(i, j) over the computed opposing actions (weights renormalised)."""
    tot = sum(p.values())
    if tot <= 0:
        return {i: 0.0 for i in rows}
    return {i: sum(p[j] * v for j, v in cells.items()) / tot for i, cells in rows.items()}


# battle readers ---------------------------------------------------------------------------------------------------
def hp_of(mon) -> float:
    """A member's HP as a fraction of its max (0 when fainted or absent, 1 when unknown)."""
    if mon is None or mon.fainted:
        return 0.0
    if mon.current_hp is None:
        return 1.0
    return float(mon.current_hp_fraction)


def hazard(battle, mon, ours: bool) -> float:
    """Entry hazard damage (fraction of max HP) for `mon` switching in on that side: Stealth Rock and Spikes."""
    side = battle.side_conditions if ours else battle.opponent_side_conditions
    if not side or mon is None:
        return 0.0
    item = (mon.item or "") if ours or (mon.item not in (None, "unknown_item")) else ""
    if item == "heavydutyboots":
        return 0.0
    out = 0.0
    if SideCondition.STEALTH_ROCK in side:
        try:
            out += mon.damage_multiplier(PokemonType.ROCK) / 8.0
        except Exception:  # noqa: BLE001
            out += 1 / 8.0
    layers = side.get(SideCondition.SPIKES, 0)
    types = {t.name.lower() for t in mon.types if t is not None}
    grounded = "flying" not in types and (mon.ability or "") != "levitate" and item != "airballoon"
    if layers and grounded:
        out += {1: 1 / 8, 2: 1 / 6}.get(int(layers), 1 / 4)
    return out

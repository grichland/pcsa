"""Endgame duel chains: expected-value duels and minimax over who fights whom."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pcsa.players.chain import RECOVER, Attack, ChainResult, Combatant, Matchups, chain_options, duel


def hit(fraction: float, ko: float, *, move: str = "hit", priority: int = 0, accuracy: float = 1.0) -> Attack:
    return Attack(move, fraction, priority, accuracy, ko)


def best_line(*args, **kwargs) -> ChainResult:
    """The line the endgame player would take: the winning option with the best margin, else the best margin."""
    return max(chain_options(*args, **kwargs).values(), key=lambda r: (r.wins, r.margin))


def mon(name: str, speed: int, *attacks: Attack, hp: float = 1.0, **extra: float) -> Combatant:
    return replace(Combatant(name, hp, speed, tuple(attacks)), **extra)


def test_faster_ko_wins() -> None:
    result = duel(mon("a", 100, hit(0.6, 0.7)), mon("b", 80, hit(0.6, 0.7)))
    assert result.winner == "a"
    assert result.turns == 2
    assert result.a_hp_left == pytest.approx(0.4)
    assert result.b_hp_left == 0


def test_priority_beats_speed_when_it_kos() -> None:
    slow = mon("a", 50, hit(0.6, 0.7, move="strong"), hit(0.3, 0.35, move="jet", priority=1), hp=0.5)
    fast = mon("b", 100, hit(0.6, 0.7), hp=0.3)
    result = duel(slow, fast)
    assert (result.winner, result.turns, result.a_first_move) == ("a", 1, "jet")
    no_priority = mon("a", 50, hit(0.6, 0.7, move="strong"), hit(0.3, 0.35, move="jet"), hp=0.5)
    assert duel(no_priority, fast).winner == "b"


def test_recovery_out_heals_weak_hits() -> None:
    foe = mon("b", 100, hit(0.2, 0.24))
    healer = duel(mon("a", 50, hit(0.1, 0.12), recover=0.5), foe)
    assert healer.winner == "a"
    assert healer.turns > 5
    assert duel(mon("a", 50, hit(0.1, 0.12)), foe).winner == "b"


def test_recovery_reported_as_move() -> None:
    result = duel(mon("a", 50, hit(0.1, 0.12), hp=0.3, recover=0.5), mon("b", 100, hit(0.2, 0.24)))
    assert result.a_first_move == RECOVER


def test_poison_flips_a_close_duel() -> None:
    fast = mon("b", 100, hit(0.3, 0.35))
    assert duel(mon("a", 50, hit(0.35, 0.4)), fast).winner == "a"
    assert duel(mon("a", 50, hit(0.35, 0.4), residual=-0.125), fast).winner == "b"


def test_speed_tie_goes_to_the_flagged_side() -> None:
    a, b = mon("a", 80, hit(0.6, 1.0)), mon("b", 80, hit(0.6, 1.0))
    assert duel(a, b).winner == "b"
    assert duel(a, b, tie_goes_to="a").winner == "a"


def test_rocky_helmet_can_double_ko() -> None:
    result = duel(mon("a", 100, hit(1.0, 1.0), hp=0.1), mon("b", 50, passive_on_hit=0.17))
    assert result.winner is None
    assert result.a_hp_left == result.b_hp_left == 0


def test_horizon_is_a_draw() -> None:
    result = duel(mon("a", 50, hit(0.0, 0.0)), mon("b", 60, hit(0.0, 0.0)), horizon=12)
    assert (result.winner, result.turns) == (None, 12)
    drawn = best_line([mon("a", 50)], [mon("b", 60)], Matchups({}, {}), active_ours=0, active_theirs=0, horizon=12)
    assert not drawn.wins
    assert drawn.margin == 0


# Our P0, P1 against their Q0, Q1. Q0 beats P0 but P1 walls it; Q1 beats P1 but P0 beats it.
# Indices: ours[0] is a fainted member (we are choosing a replacement), P0 = 1, P1 = 2.
SWEEP = Matchups(
    ours={(1, 0): (hit(0.1, 0.12),), (2, 0): (hit(0.5, 0.55),), (2, 1): (hit(0.1, 0.12),), (1, 1): (hit(0.6, 0.7),)},
    theirs={
        (1, 0): (hit(0.6, 0.7),),
        (2, 0): (hit(0.1, 0.12),),
        (2, 1): (hit(0.5, 0.55),),
        (1, 1): (hit(0.1, 0.12),),
    },
)


def no_swap(name: str, speed: int, hp: float = 1.0) -> Combatant:
    return Combatant(name, hp, speed, can_switch_on_faint=False)


def test_lead_order_decides_the_chain() -> None:
    ours = [no_swap("fainted", 1, hp=0.0), no_swap("P0", 50), no_swap("P1", 50)]
    theirs = [no_swap("Q0", 100), no_swap("Q1", 100)]
    options = chain_options(ours, theirs, SWEEP, active_ours=0, active_theirs=0)
    assert not options[1].wins
    assert options[1].margin == pytest.approx(-0.9)
    assert options[2].wins
    best = best_line(ours, theirs, SWEEP, active_ours=0, active_theirs=0)
    assert (best.wins, best.order, best.first_move) == (True, (2, 1), "hit")
    assert best.margin == pytest.approx(0.8)


def test_opponent_picks_its_order_adversarially() -> None:
    # Q2 first chips P0 into Q1's KO range; Q1 first would lose both duels.
    ours = [no_swap("P0", 100)]
    theirs = [no_swap("fainted", 1, hp=0.0), no_swap("Q1", 150, hp=0.5), no_swap("Q2", 50)]
    matrix = Matchups(
        ours={(0, 1): (hit(0.6, 0.7),), (0, 2): (hit(0.5, 0.55),)},
        theirs={(0, 1): (hit(0.5, 0.6),), (0, 2): (hit(0.4, 0.45),)},
    )
    q1_first = duel(mon("P0", 100, hit(0.6, 0.7)), mon("Q1", 150, hit(0.5, 0.6), hp=0.5))
    then_q2 = duel(mon("P0", 100, hit(0.5, 0.55), hp=q1_first.a_hp_left), mon("Q2", 50, hit(0.4, 0.45)))
    assert q1_first.winner == then_q2.winner == "a"
    result = best_line(ours, theirs, matrix, active_ours=0, active_theirs=0)
    assert not result.wins
    assert result.margin == pytest.approx(-0.5)


def test_switching_out_of_a_lost_matchup() -> None:
    # Q0 KOs P0 at once; P1 takes little from Q0 and beats it. Switching keeps P0 alive.
    ours = [no_swap("P0", 50), no_swap("P1", 50)]
    theirs = [no_swap("Q0", 100)]
    matrix = Matchups(
        ours={(0, 0): (hit(0.1, 0.12),), (1, 0): (hit(0.5, 0.55),)},
        theirs={(0, 0): (hit(1.0, 1.0),), (1, 0): (hit(0.1, 0.12),)},
    )
    options = chain_options(ours, theirs, matrix, active_ours=0, active_theirs=0)
    assert options[0].wins and options[1].wins
    assert options[1].margin > options[0].margin
    best = best_line(ours, theirs, matrix, active_ours=0, active_theirs=0)
    assert (best.order, best.first_move) == ((1,), None)
    assert best.margin == pytest.approx(1.0 + 0.7)


def test_free_swap_after_a_ko() -> None:
    # With swaps allowed, P1 KOs Q0 and hands Q1 to P0 instead of dying to it.
    ours = [Combatant("fainted", 0.0, 1), Combatant("P0", 1.0, 50), Combatant("P1", 1.0, 50)]
    theirs = [no_swap("Q0", 100), no_swap("Q1", 100)]
    best = best_line(ours, theirs, SWEEP, active_ours=0, active_theirs=0)
    assert best.wins
    assert best.order == (2, 1)
    assert best.margin == pytest.approx(0.8 + 0.8)


def test_nobody_left() -> None:
    assert (
        chain_options([mon("a", 50, hp=0.0)], [mon("b", 50)], Matchups({}, {}), active_ours=0, active_theirs=0) == {}
    )

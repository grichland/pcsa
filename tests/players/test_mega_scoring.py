"""A move chosen with Mega Evolution is scored and shown with the Mega form: it evolves before the move."""

from __future__ import annotations

import pytest
from poke_env.battle import Move, PokemonType

from pcsa.battle.stats import current_stats
from pcsa.players.heuristic import HeuristicParams, HeuristicPlayer
from pcsa.players.tactics import Scoring, attack_score, effective_accuracy, mega_form_of
from tests.battle.conftest import translator_for

MEGA_X = "charizardmegax"
STONE = "charizarditex"


def charizard_vs_snorlax():
    t = translator_for(("Charizard", "Charizard", 153), [("Snorlax", "Snorlax")])
    battle = t.battle
    mine = battle.active_pokemon
    assert mine is not None
    mine._item = STONE
    battle._available_moves = [Move("flamethrower", gen=9), Move("dragonclaw", gen=9)]
    return battle


def test_mega_form_of_reads_the_held_stone() -> None:
    battle = charizard_vs_snorlax()
    mine = battle.active_pokemon
    assert mega_form_of(mine) == MEGA_X
    mine._item = "lifeorb"
    assert mega_form_of(mine) is None
    mine._item = None
    assert mega_form_of(mine) is None


def test_attack_score_with_a_mega_form_uses_its_stats_types_and_ability() -> None:
    battle = charizard_vs_snorlax()
    mine, theirs = battle.active_pokemon, battle.opponent_active_pokemon
    flamethrower, dragon_claw = battle.available_moves
    base, mega = current_stats(mine), current_stats(mine, form=MEGA_X)
    assert mega["atk"] > base["atk"] and mega["spa"] > base["spa"] and mega["hp"] == base["hp"]
    scored = Scoring(form=MEGA_X)
    # Dragon Claw: Mega X adds the Dragon type (STAB) on top of the higher Attack.
    expected = attack_score(battle, mine, theirs, dragon_claw) * 1.5 * mega["atk"] / base["atk"]
    assert attack_score(battle, mine, theirs, dragon_claw, scored) == pytest.approx(expected)
    # Flamethrower keeps its Fire STAB: only the Special Attack changes.
    expected = attack_score(battle, mine, theirs, flamethrower) * mega["spa"] / base["spa"]
    assert attack_score(battle, mine, theirs, flamethrower, scored) == pytest.approx(expected)
    # Without a form the score is the current form's.
    assert attack_score(battle, mine, theirs, dragon_claw, Scoring()) == attack_score(
        battle, mine, theirs, dragon_claw
    )
    assert PokemonType.DRAGON not in mine.types
    # The Mega's single ability stands in for the attacker's: Mega Pidgeot's No Guard lands Hurricane.
    hurricane = Move("hurricane", gen=9)
    assert effective_accuracy(battle, mine, theirs, hurricane) == pytest.approx(0.7)
    assert effective_accuracy(battle, mine, theirs, hurricane, "noguard") == 1.0


def test_preferred_scores_the_moves_with_the_mega_form_when_it_will_mega(monkeypatch) -> None:
    battle = charizard_vs_snorlax()
    battle._can_mega_evolve = True
    forms: list[str | None] = []
    real = attack_score

    def recording(battle, attacker, defender, move, scoring=None):
        forms.append(scoring.form if scoring else None)
        return real(battle, attacker, defender, move, scoring)

    monkeypatch.setattr("pcsa.players.heuristic.attack_score", recording)
    order = HeuristicPlayer(HeuristicParams(mega="first")).preferred(battle)
    assert order is not None and order.mega and order.order.id == "dragonclaw"
    assert forms[-2:] == [MEGA_X, MEGA_X]
    forms.clear()
    order = HeuristicPlayer(HeuristicParams(mega="never")).preferred(battle)
    assert order is not None and not order.mega and order.order.id == "flamethrower"
    assert forms[-2:] == [None, None]


def test_preferred_scores_the_current_form_when_the_mega_is_not_offered() -> None:
    battle = charizard_vs_snorlax()
    battle._can_mega_evolve = False
    order = HeuristicPlayer(HeuristicParams(mega="first")).preferred(battle)
    assert order is not None and not order.mega and order.order.id == "flamethrower"

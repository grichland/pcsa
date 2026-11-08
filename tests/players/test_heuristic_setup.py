"""The heuristic's setup rule fires: a self-targeting boost worth two stages in a favourable matchup."""

from __future__ import annotations

from poke_env.battle import Move

from pcsa.players.heuristic import HeuristicParams, HeuristicPlayer
from tests.battle.conftest import translator_for


def charizard_vs_snorlax_with_setup():
    t = translator_for(("Charizard", "Charizard", 153), [("Snorlax", "Snorlax")])
    battle = t.battle
    battle._available_moves = [Move("flamethrower", gen=9), Move("swordsdance", gen=9)]
    return battle


def test_preferred_sets_up_when_faster_and_not_threatened() -> None:
    battle = charizard_vs_snorlax_with_setup()
    player = HeuristicPlayer(HeuristicParams(mega="never"))
    assert player.matchup(battle.active_pokemon, battle.opponent_active_pokemon) > 0
    assert player.preferred(battle).message == "/choose move swordsdance"


def test_preferred_attacks_once_the_boost_is_maxed() -> None:
    battle = charizard_vs_snorlax_with_setup()
    battle.active_pokemon._boosts["atk"] = 6
    player = HeuristicPlayer(HeuristicParams(mega="never"))
    assert player.preferred(battle).message == "/choose move flamethrower"

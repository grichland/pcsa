"""The heuristic decides when a team read left some of our stats unread (None)."""

from __future__ import annotations

from poke_env.battle import Move, Pokemon

from pcsa.players.heuristic import HeuristicParams, HeuristicPlayer
from tests.battle.conftest import translator_for

UNREAD = {"hp": 153, "atk": None, "def": 98, "spa": None, "spd": 105, "spe": 152}


def battle_with_unread_stats():
    battle = translator_for(("Charizard", "Charizard", 153), [("Snorlax", "Snorlax")]).battle
    battle.active_pokemon._stats = dict(UNREAD)
    bench = Pokemon(gen=9, species="blastoise")
    bench._current_hp = bench._max_hp = 154
    bench._stats = {"hp": 154, "atk": None, "def": 120, "spa": 105, "spd": 125, "spe": None}
    battle._available_moves = [Move("flamethrower", gen=9), Move("airslash", gen=9)]
    battle._available_switches = [bench]
    return battle


def test_should_switch_reads_a_missing_attack_stat_as_zero() -> None:
    battle = battle_with_unread_stats()
    player = HeuristicPlayer(HeuristicParams(mega="never", switch_in_min=-10.0))
    assert isinstance(player.should_switch(battle), bool)


def test_preferred_scores_moves_and_switches_with_unread_stats() -> None:
    battle = battle_with_unread_stats()
    player = HeuristicPlayer(HeuristicParams(mega="never", switch_in_min=-100.0))
    assert player.preferred(battle) is not None
    assert player.best_switch(battle) is not None

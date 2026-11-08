"""best_matchup_max_damage: max_damage's attack, a switch out of a really bad matchup, the back-to-back rule."""

from __future__ import annotations

import random

from pcsa.battle.state import replay
from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.config import Config
from pcsa.players import mirror
from pcsa.players.base import DecisionView, Degrade, ScreenFacts
from pcsa.players.best_matchup import BestMatchupMaxDamagePlayer, opponent_switched
from pcsa.players.factory import build_player
from pcsa.players.max_damage import MaxDamagePlayer

PRELUDE = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|teamsize|p1|3", "|teamsize|p2|3", "|start"]
TEAM = {
    "Charizard": (153, ["flamethrower", "airslash"]),
    "Swampert": (175, ["surf", "earthquake"]),
    "Snorlax": (235, ["bodyslam", "crunch"]),
}


def battle(ours: str, foe: str, turns: list[list[str]] | None = None, forced: bool = False):
    """`ours` and their `foe` lead; `turns` holds each finished turn's events. Our active: the last one in."""
    lines = [*PRELUDE, f"|switch|p1a: {ours}|{ours}, L50|{TEAM[ours][0]}/{TEAM[ours][0]}"]
    lines += [f"|switch|p2a: {foe}|{foe}, L50|100/100", "|turn|1"]
    for turn, events in enumerate(turns or [], start=2):
        lines += [*events, f"|turn|{turn}"]
    active = [line.split("|")[2][5:] for line in lines if line.startswith("|switch|p1a")][-1]
    pokemon = [
        {
            "ident": f"p1: {name}",
            "details": f"{name}, L50",
            "condition": "0 fnt" if forced and name == active else f"{hp}/{hp}",
            "active": name == active,
            "stats": {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 100},
            "moves": moves,
            "baseAbility": "",
            "item": "",
        }
        for name, (hp, moves) in TEAM.items()
    ]
    request: dict = {"side": {"name": "me", "id": "p1", "pokemon": pokemon}, "rqid": 1}
    if forced:
        request["forceSwitch"] = [True]
    else:
        request["active"] = [
            {"moves": [{"move": m, "id": m, "pp": 10, "maxpp": 10, "target": "normal"} for m in TEAM[active][1]]}
        ]
    return replay(lines, request)


def view(b) -> DecisionView:
    return DecisionView(mirror.legal_actions(b), ScreenFacts(), Degrade.FULL, random.Random(0), b)


def decide(player, b) -> tuple[str, dict]:
    return player.choose_action(view(b), Deadline.after(FakeClock(), 30.0)).message, player.notes()


def test_attacks_like_max_damage_in_a_neutral_matchup() -> None:
    b = battle("Snorlax", "Snorlax")
    chosen, notes = decide(BestMatchupMaxDamagePlayer(), b)
    assert chosen == decide(MaxDamagePlayer(), b)[0] and chosen.startswith("/choose move")
    assert "switch" not in notes and "switch_blocked" not in notes


def test_switches_out_of_a_bad_matchup_to_the_best_bench_member() -> None:
    chosen, notes = decide(BestMatchupMaxDamagePlayer(), battle("Charizard", "Tyranitar"))
    assert chosen == "/choose switch Swampert"
    assert notes["switch"].startswith("bad matchup: charizard")


# Turn 1 as the log shows it: Charizard switched to Swampert while the opponent switched Tyranitar to Venusaur.
SWITCHED = ["|switch|p1a: Swampert|Swampert, L50|175/175", "|switch|p2a: Venusaur|Venusaur, L50|100/100"]


def test_back_to_back_switch_blocked_when_the_opponent_stayed() -> None:
    player = BestMatchupMaxDamagePlayer()
    assert decide(player, battle("Charizard", "Tyranitar"))[0] == "/choose switch Swampert"
    stayed = ["|move|p2a: Venusaur|Sludge Bomb|p1a: Swampert"]
    after = battle("Swampert", "Venusaur", [stayed])  # a bad matchup on turn 2; they did not switch on turn 1
    assert not opponent_switched(after, 1)
    chosen, notes = decide(player, after)
    assert chosen.startswith("/choose move") and "the opponent did not" in notes["switch_blocked"]


def test_back_to_back_switch_allowed_once_when_the_opponent_switched_too() -> None:
    player = BestMatchupMaxDamagePlayer()
    assert decide(player, battle("Charizard", "Tyranitar"))[0] == "/choose switch Swampert"
    second = battle("Charizard", "Tyranitar", [SWITCHED])
    assert opponent_switched(second, 1)
    chosen, notes = decide(player, second)
    assert chosen == "/choose switch Charizard" and "back-to-back" in notes["switch"]
    back = ["|switch|p1a: Charizard|Charizard, L50|153/153", "|switch|p2a: Tyranitar|Tyranitar, L50|100/100"]
    chosen, notes = decide(player, battle("Charizard", "Tyranitar", [SWITCHED, back]))  # Charizard in again
    assert chosen.startswith("/choose move") and "twice in a row" in notes["switch_blocked"]


def test_a_turn_without_a_switch_restarts_the_count() -> None:
    player = BestMatchupMaxDamagePlayer()
    assert decide(player, battle("Charizard", "Tyranitar"))[0] == "/choose switch Swampert"
    rest = ["|move|p1a: Swampert|Surf|p2a: Venusaur"]
    later = battle("Charizard", "Tyranitar", [SWITCHED, rest])  # turn 3: switched on 1, attacked on 2
    assert decide(player, later)[0] == "/choose switch Charizard"


def test_a_replacement_after_a_faint_is_not_an_opponent_switch() -> None:
    fainted = [SWITCHED[0], "|faint|p2a: Tyranitar", SWITCHED[1]]
    assert not opponent_switched(battle("Charizard", "Tyranitar", [fainted]), 1)


def test_forced_switches_are_max_damage_s_and_never_blocked() -> None:
    player = BestMatchupMaxDamagePlayer()
    assert decide(player, battle("Charizard", "Tyranitar"))[0] == "/choose switch Swampert"
    fainted = ["|move|p2a: Tyranitar|Stone Edge|p1a: Charizard", "|faint|p1a: Charizard"]
    forced = battle("Charizard", "Tyranitar", [fainted], forced=True)
    chosen, notes = decide(player, forced)
    assert chosen == decide(MaxDamagePlayer(), forced)[0] and chosen.startswith("/choose switch")
    assert notes["switch"].startswith("forced")


def test_a_new_battle_resets_the_rule() -> None:
    player = BestMatchupMaxDamagePlayer()
    assert decide(player, battle("Charizard", "Tyranitar"))[0] == "/choose switch Swampert"
    player._last_switch, player._streak = 1, 2
    assert decide(player, battle("Charizard", "Tyranitar"))[0] == "/choose switch Swampert"  # turn went 2 -> 1


def test_factory_builds_it_with_the_config_thresholds(config: Config) -> None:
    player = build_player(config, "best_matchup_max_damage")
    assert isinstance(player, BestMatchupMaxDamagePlayer) and player.name == "best_matchup_max_damage"
    assert player.switch.switch_below < 0 < player.switch.switch_in_above
    assert player.heuristic.p.speed_tier == build_player(config, "custom_heuristic").p.speed_tier  # type: ignore[attr-defined]

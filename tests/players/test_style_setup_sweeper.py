"""style_setup_sweeper: the preview plan, holding and shielding the sweeper, its entry and its relaxed setup gate.

The threat estimates are test_disciplined's FakeThreat table; the opponent's assumed sets are SETS / ABILITIES.
Golisopod (Swords Dance + Aqua Jet) is the only sweeper of SIX; of their six, Skarmory (Roar), Dragonite (resists
Water and Bug) and Corviknight (faster, Brave Bird) check it.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from pcsa.battle.state import replay
from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.config import Config
from pcsa.players import disciplined, free_turn, mirror, opponent, style_setup_sweeper
from pcsa.players.base import DecisionView, Degrade, ScreenFacts, TeamPreviewView
from pcsa.players.factory import KINDS, LIVE_KINDS, SIM_KINDS, build_player
from pcsa.players.free_turn import FreeTurnPlayer
from pcsa.players.style_setup_sweeper import StyleSetupSweeperPlayer, SweeperParams
from tests.players.test_disciplined import FakeThreat, FakeTracker

PRELUDE = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|teamsize|p1|3", "|teamsize|p2|3", "|start"]
SIX = {
    "Golisopod": (150, ["swordsdance", "liquidation", "leechlife", "aquajet"]),
    "Charizard": (153, ["airslash", "flamethrower", "nastyplot", "willowisp"]),
    "Azumarill": (175, ["playrough", "aquajet"]),
    "Slowbro": (170, ["scald", "slackoff"]),
    "Gengar": (135, ["shadowball", "sludgebomb"]),
    "Snorlax": (235, ["bodyslam", "crunch"]),
}
BROUGHT = ("Golisopod", "Charizard", "Azumarill")
THEIRS = ("Metagross", "Dragonite", "Skarmory", "Corviknight", "Milotic", "Espeon")
SETS = {
    "metagross": ("meteormash", "zenheadbutt", "earthquake", "bulletpunch"),
    "dragonite": ("outrage", "extremespeed", "earthquake", "roost"),
    "skarmory": ("bravebird", "roar", "spikes", "roost"),
    "corviknight": ("bravebird", "bodypress", "defog", "roost"),
    "milotic": ("scald", "icebeam", "recover"),
    "espeon": ("psychic", "shadowball", "dazzlinggleam", "morningsun"),
}
ABILITIES = {"espeon": "magicbounce", "metagross": "clearbody"}
FASTER = {("corviknight", "golisopod")}
HITS = {  # supporters on the checks: Charizard 0.5 + 0.4 + 0.1, Azumarill 0.1 + 0.6 + 0.1, the rest 0.3
    ("charizard", "skarmory"): (0.5, "flamethrower"),
    ("charizard", "corviknight"): (0.4, "flamethrower"),
    ("azumarill", "dragonite"): (0.6, "playrough"),
}
CHECK_ALIVE = [("Metagross", 100)]  # Skarmory, Dragonite or Corviknight may still come
NO_CHECK = [("Milotic", 100), ("Espeon", 100), ("Metagross", 100)]  # all three seen, none a check


@pytest.fixture
def fake(monkeypatch):
    def install(hits: dict | None = None) -> FakeThreat:
        estimates = FakeThreat({**HITS, **(hits or {})}, faster=FASTER)
        for module in (disciplined, free_turn, style_setup_sweeper):
            monkeypatch.setattr(module, "threat", estimates)
        monkeypatch.setattr(disciplined, "opponent", SimpleNamespace(Tracker=FakeTracker))
        sets = SimpleNamespace(
            assumed=lambda battle, mon, *known: SimpleNamespace(
                moves=SETS.get(mon.species, ()), ability=ABILITIES.get(mon.species, "")
            ),
            move_object=opponent.move_object,
        )
        monkeypatch.setattr(free_turn, "opponent", sets)
        return estimates

    return install


def _pokemon(team: dict, active: str | None, hp: dict[str, float]) -> list[dict]:
    return [
        {
            "ident": f"p1: {name}",
            "details": f"{name}, L50",
            "condition": f"{round(full * hp.get(name, 1.0))}/{full}" if hp.get(name, 1.0) > 0 else "0 fnt",
            "active": name == active,
            "stats": {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 100},
            "moves": moves,
            "baseAbility": "",
            "item": "",
        }
        for name, (full, moves) in team.items()
    ]


def preview(six: dict | None = None):
    six = six or SIX
    lines = [*PRELUDE[:3], *(f"|poke|p2|{name}, L50|" for name in THEIRS), "|teampreview"]
    request = {"side": {"name": "me", "id": "p1", "pokemon": _pokemon(six, None, {})}, "rqid": 1, "teamPreview": True}
    b = replay(lines, request)
    b._max_team_size = 3  # as the arena sets it from maxChosenTeamSize
    return b


def battle(active: str, foes: list[tuple[str, int]], hp: dict[str, float] | None = None, six: dict | None = None):
    """`active` (fainted in `hp`: a forced replacement) against `foes` revealed in order, the last one active."""
    hp, team = hp or {}, {name: (six or SIX)[name] for name in BROUGHT}
    full = team[active][0]
    lines = [*PRELUDE[:-1], *(f"|poke|p2|{name}, L50|" for name in THEIRS), "|start"]
    lines.append(f"|switch|p1a: {active}|{active}, L50|{max(1, round(full * hp.get(active, 1.0)))}/{full}")
    if hp.get(active, 1.0) <= 0:
        lines.append(f"|faint|p1a: {active}")
    lines += [f"|switch|p2a: {name}|{name}, L50|{percent}/100" for name, percent in foes]
    lines.append("|turn|1")
    request: dict = {"side": {"name": "me", "id": "p1", "pokemon": _pokemon(team, active, hp)}, "rqid": 1}
    if hp.get(active, 1.0) <= 0:
        request["forceSwitch"] = [True]
    else:
        request["active"] = [
            {"moves": [{"move": m, "id": m, "pp": 10, "maxpp": 10, "target": "normal"} for m in team[active][1]]}
        ]
    return replay(lines, request)


def pick(player, six: dict | None = None) -> tuple[list[str], dict]:
    view = TeamPreviewView(preview(six), (None,) * 6, random.Random(0))
    picks = player.choose_team(view, Deadline.after(FakeClock(), 30.0))
    return [list(six or SIX)[i] for i in picks], player.notes()


def decide(player, b) -> tuple[str, dict]:
    view = DecisionView(mirror.legal_actions(b), ScreenFacts(), Degrade.FULL, random.Random(0), b)
    return player.choose_action(view, Deadline.after(FakeClock(), 30.0)).message, player.notes()


def planned() -> StyleSetupSweeperPlayer:
    player = StyleSetupSweeperPlayer(catalog=SimpleNamespace())  # type: ignore[arg-type]
    pick(player)
    return player


# Preview --------------------------------------------------------------------------------------------------
def test_a_fit_brings_the_sweeper_with_the_supporters_that_hit_its_checks(fake) -> None:
    fake()
    picks, notes = pick(StyleSetupSweeperPlayer(catalog=SimpleNamespace()))  # type: ignore[arg-type]
    assert set(picks) == set(BROUGHT) and picks[0] != "Golisopod" and picks[-1] == "Golisopod"
    assert notes["style"] == "setup_sweeper" and notes["sweeper"] == "golisopod"
    assert notes["checks"] == "dragonite (resists), skarmory (roar), corviknight (faster, super effective)"


def test_no_fit_plays_exactly_free_turn(fake) -> None:
    fake({("charizard", "metagross"): (0.1, None), ("metagross", "charizard"): (0.6, "meteormash")})
    no_setup = {**SIX, "Golisopod": (150, ["liquidation", "leechlife", "aquajet"])}
    me, plain = StyleSetupSweeperPlayer(catalog=SimpleNamespace()), FreeTurnPlayer(catalog=SimpleNamespace())  # type: ignore[arg-type]
    picks, notes = pick(me, no_setup)
    assert notes == {"style": "no fit"} and picks == pick(plain, no_setup)[0]
    fake({("metagross", "charizard"): (0.6, "meteormash"), ("golisopod", "metagross"): (0.35, "liquidation")})
    b = battle("Charizard", CHECK_ALIVE, six=no_setup)
    assert decide(me, b) == decide(plain, b)
    assert decide(me, b)[0] == "/choose switch Golisopod"  # free_turn's switch, no sweeper held back


# Holding and shielding ------------------------------------------------------------------------------------
def test_the_sweeper_is_held_back_while_a_check_lives(fake) -> None:
    fake({("metagross", "charizard"): (0.6, "meteormash"), ("golisopod", "metagross"): (0.35, "liquidation")})
    chosen, notes = decide(planned(), battle("Charizard", CHECK_ALIVE))
    assert chosen.startswith("/choose move") and notes["rule"] == "attack"  # disciplined alone would bring it


def test_a_threatened_sweeper_is_shielded_while_a_check_lives(fake) -> None:
    fake(
        {
            ("metagross", "golisopod"): (0.5, "meteormash"),
            ("metagross", "charizard"): (0.3, "meteormash"),  # disciplined wants < 0.25
            ("metagross", "azumarill"): (0.45, "meteormash"),
        }
    )
    chosen, notes = decide(planned(), battle("Golisopod", CHECK_ALIVE))
    assert chosen == "/choose switch Charizard" and notes["rule"] == "sweeper:shield"


def test_no_shield_when_the_sweeper_kos_first(fake) -> None:
    estimates = fake(
        {("metagross", "golisopod"): (0.5, "meteormash"), ("golisopod", "metagross"): (1.0, "liquidation")}
    )
    estimates.faster.add(("golisopod", "metagross"))
    chosen, notes = decide(planned(), battle("Golisopod", CHECK_ALIVE))
    assert chosen == "/choose move liquidation" and notes["rule"] == "attack"


def test_a_forced_switch_avoids_the_sweeper_while_a_check_lives(fake) -> None:
    fake()
    chosen, notes = decide(planned(), battle("Charizard", CHECK_ALIVE, {"Charizard": 0.0}))
    assert chosen == "/choose switch Azumarill" and notes["rule"] == "sweeper:hold"
    chosen, notes = decide(planned(), battle("Charizard", NO_CHECK, {"Charizard": 0.0}))
    assert chosen == "/choose switch Golisopod" and notes["rule"] == "sweeper:enter"


def test_the_sweeper_comes_in_when_it_is_the_last_one(fake) -> None:
    fake()
    b = battle("Charizard", CHECK_ALIVE, {"Charizard": 0.0, "Azumarill": 0.0})
    assert decide(planned(), b) == ("/choose switch Golisopod", {"rule": "forced"})


# Entry and sweep ------------------------------------------------------------------------------------------
def test_the_sweeper_enters_once_no_check_lives_and_the_hit_is_small(fake) -> None:
    fake()
    chosen, notes = decide(planned(), battle("Charizard", NO_CHECK))
    assert chosen == "/choose switch Golisopod" and notes["rule"] == "sweeper:enter"
    fake({("metagross", "golisopod"): (0.45, "meteormash")})
    _, notes = decide(planned(), battle("Charizard", NO_CHECK))
    assert notes["rule"] != "sweeper:enter" and notes["sweeper"] == "no entry: takes 0.45"


def test_the_relaxed_setup_gate_is_the_sweepers_only(fake) -> None:
    fake({("metagross", "golisopod"): (0.5, "meteormash"), ("metagross", "charizard"): (0.5, "meteormash")})
    chosen, notes = decide(planned(), battle("Golisopod", NO_CHECK))
    assert chosen == "/choose move swordsdance" and notes["rule"] == "free:setup"
    _, notes = decide(planned(), battle("Charizard", NO_CHECK))
    assert notes["rule"] == "attack" and notes["free"] == "not free: hit 0.50"


def test_setup_stops_at_plus_two(fake) -> None:
    fake()
    b = battle("Golisopod", NO_CHECK)
    b.active_pokemon._boosts["atk"] = 2
    chosen, notes = decide(planned(), b)
    assert chosen != "/choose move swordsdance" and notes["rule"] == "attack"


def test_supporters_do_not_set_up_while_a_check_lives(fake) -> None:
    fake()
    b = battle("Charizard", CHECK_ALIVE)
    assert decide(FreeTurnPlayer(catalog=SimpleNamespace()), b)[0] == "/choose move nastyplot"  # type: ignore[arg-type]
    chosen, notes = decide(planned(), b)
    assert chosen == "/choose move willowisp" and notes["rule"] == "free:status"
    assert notes["sweeper"] == "supporter: no setup while a check lives"


# Factory --------------------------------------------------------------------------------------------------
def test_factory_builds_it_with_the_config_thresholds(config: Config, fake) -> None:
    fake()
    built = build_player(config, "style_setup_sweeper")
    assert isinstance(built, StyleSetupSweeperPlayer) and built.name == "style_setup_sweeper"
    assert built.sweep == SweeperParams()  # the yaml holds the defaults
    assert built.params()["sweeper"]["sweeper_free_hit_max"] == 0.55 and built.params()["free_hit_max"] == 0.4
    assert "style_setup_sweeper" in SIM_KINDS and "style_setup_sweeper" in KINDS
    assert "style_setup_sweeper" not in LIVE_KINDS


def test_the_preview_plan_runs_on_the_real_estimates() -> None:
    """At preview poke-env keys opponent_team by bare species; the real form lookup must not see those keys."""
    picks, notes = pick(StyleSetupSweeperPlayer())
    assert (
        set(picks) <= set(SIX) and len(picks) == 3 and picks[-1] == "Golisopod" and notes["style"] == "setup_sweeper"
    )

"""free_turn: the free-turn gate and its setup / Stealth Rock / status picks on top of disciplined.

The threat estimates are test_disciplined's FakeThreat table; the opponent's assumed sets are SETS / ABILITIES.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from pcsa.battle.state import replay
from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.config import Config
from pcsa.players import disciplined, free_turn, mirror, opponent
from pcsa.players.base import DecisionView, Degrade, ScreenFacts
from pcsa.players.factory import KINDS, LIVE_KINDS, SIM_KINDS, build_player
from pcsa.players.free_turn import FreeTurnParams, FreeTurnPlayer
from tests.players.test_disciplined import FakeThreat, FakeTracker

PRELUDE = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|teamsize|p1|3", "|teamsize|p2|3", "|start"]
TEAM = {
    "Garchomp": (183, ["swordsdance", "earthquake", "stealthrock", "dragonclaw"]),
    "Slowbro": (170, ["scald", "willowisp", "toxic", "slackoff"]),
    "Azumarill": (175, ["playrough", "aquajet"]),
}
SETS = {
    "metagross": ("meteormash", "zenheadbutt", "earthquake", "bulletpunch"),  # physical, Steel
    "milotic": ("scald", "icebeam", "recover"),  # special
    "dragonite": ("outrage", "extremespeed", "earthquake", "roost"),  # Dragon STAB is SE on Garchomp
    "espeon": ("psychic", "shadowball", "dazzlinggleam", "morningsun"),
    "skarmory": ("bravebird", "roar", "spikes", "roost"),
    "corviknight": ("bravebird", "bodypress", "defog", "roost"),
}
ABILITIES = {"espeon": "magicbounce", "metagross": "clearbody"}
ME_DOWN = ["|faint|p1a: Azumarill", "|switch|p1a: {ours}|{ours}, L50|{hp}/{hp}"]
FOE_DOWN = ["|faint|p2a: {foe}", "|switch|p2a: Dragonite|Dragonite, L50|100/100"]
# Garchomp's own hit when a test threatens it: at or above `trade_hit_min`, so it attacks rather than trades out
HIT_BACK = {("garchomp", "metagross"): (0.3, "earthquake")}


@pytest.fixture
def fake(monkeypatch):
    def install(hits: dict | None = None, **kwargs) -> FakeThreat:
        estimates = FakeThreat(hits or {}, **kwargs)
        for module in (disciplined, free_turn):
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


def battle(ours: str, foe: str = "Metagross", *, moves: list[str] | None = None, phase: str = "", hp: float = 1.0):
    """`ours` (with `moves`) against `foe`; `phase` "behind" (Azumarill fainted) or "ahead" (their lead fainted)."""
    lead = "Azumarill" if phase == "behind" else ours
    their_lead = "Corviknight" if phase == "ahead" else foe
    lines = [*PRELUDE, f"|switch|p1a: {lead}|{lead}, L50|{TEAM[lead][0]}/{TEAM[lead][0]}"]
    lines += [f"|switch|p2a: {their_lead}|{their_lead}, L50|100/100", "|turn|1"]
    if phase == "behind":
        lines += [line.format(ours=ours, hp=TEAM[ours][0]) for line in ME_DOWN] + ["|turn|2"]
    if phase == "ahead":
        down = [line.format(foe=their_lead) for line in FOE_DOWN]
        lines += [down[0], f"|switch|p2a: {foe}|{foe}, L50|100/100", "|turn|2"]
    own = {**{k: v[1] for k, v in TEAM.items()}, ours: moves or TEAM[ours][1]}
    pokemon = [
        {
            "ident": f"p1: {name}",
            "details": f"{name}, L50",
            "condition": _condition(name, full, ours, phase, hp),
            "active": name == ours,
            "stats": {"atk": 120, "def": 120, "spa": 120, "spd": 120, "spe": 100},
            "moves": own[name],
            "baseAbility": "",
            "item": "",
        }
        for name, (full, _) in TEAM.items()
    ]
    request = {
        "side": {"name": "me", "id": "p1", "pokemon": pokemon},
        "rqid": 1,
        "active": [{"moves": [{"move": m, "id": m, "pp": 10, "maxpp": 10, "target": "normal"} for m in own[ours]]}],
    }
    return replay(lines, request)


def _condition(name: str, full: int, ours: str, phase: str, hp: float) -> str:
    if phase == "behind" and name == "Azumarill":
        return "0 fnt"
    return f"{round(full * hp) if name == ours else full}/{full}"


def decide(player, b) -> tuple[str, dict]:
    view = DecisionView(mirror.legal_actions(b), ScreenFacts(), Degrade.FULL, random.Random(0), b)
    return player.choose_action(view, Deadline.after(FakeClock(), 30.0)).message, player.notes()


def player() -> FreeTurnPlayer:
    return FreeTurnPlayer(catalog=SimpleNamespace())  # type: ignore[arg-type]


# The gate -------------------------------------------------------------------------------------------------
def test_a_free_turn_sets_up(fake) -> None:
    fake()
    chosen, notes = decide(player(), battle("Garchomp"))
    assert chosen == "/choose move swordsdance" and notes["rule"] == "free:setup"


@pytest.mark.parametrize(
    ("foe", "hits", "why"),
    [
        ("Metagross", {("metagross", "garchomp"): (0.45, "meteormash"), **HIT_BACK}, "hit 0.45"),  # hits too hard
        ("Dragonite", {}, "SE STAB outrage"),  # Dragon STAB on a Dragon
        ("Skarmory", {}, "foe has roar"),  # a phaze in the assumed set
        ("Metagross", {("garchomp", "metagross"): (1.0, "earthquake")}, "we KO"),  # just attack
    ],
)
def test_the_gate_needs_every_condition(fake, foe: str, hits: dict, why: str) -> None:
    fake(hits)
    chosen, notes = decide(player(), battle("Garchomp", foe))
    assert notes["rule"] == "attack" and notes["free"] == f"not free: {why}"
    assert chosen.startswith("/choose move") and "swordsdance" not in chosen and "stealthrock" not in chosen


def test_they_ko_us_closes_the_gate(fake) -> None:
    fake({("metagross", "garchomp"): (0.3, "meteormash")})
    _, notes = decide(player(), battle("Garchomp", hp=0.25))
    assert notes["free"] == "not free: they KO"


def test_no_note_without_a_free_move(fake) -> None:
    fake({("metagross", "azumarill"): (0.6, "meteormash"), ("azumarill", "metagross"): (0.2, "playrough")})
    chosen, notes = decide(player(), battle("Azumarill"))
    assert chosen == "/choose move playrough" and notes["rule"] == "attack" and "free" not in notes


# Setup ----------------------------------------------------------------------------------------------------
def test_setup_when_behind_not_when_ahead(fake) -> None:
    fake()
    chosen, notes = decide(player(), battle("Garchomp", phase="behind"))
    assert chosen == "/choose move swordsdance" and notes["rule"] == "free:setup"
    chosen, notes = decide(player(), battle("Garchomp", phase="ahead"))
    assert chosen.startswith("/choose move") and notes == {"rule": "attack", "free": "not free: ahead in members"}


def test_setup_stops_at_plus_two_and_at_half_hp(fake) -> None:
    fake()
    boosted = battle("Garchomp")
    boosted.active_pokemon._boosts["atk"] = 2
    assert decide(player(), boosted) == ("/choose move stealthrock", {"rule": "free:rocks", "free": "stealthrock"})
    chosen, notes = decide(player(), battle("Garchomp", hp=0.45))
    assert chosen == "/choose move stealthrock" and notes["rule"] == "free:rocks"


# Stealth Rock ---------------------------------------------------------------------------------------------
def test_rocks_once_per_battle(fake) -> None:
    fake()
    me, rocks = player(), ["earthquake", "stealthrock"]
    assert decide(me, battle("Garchomp", moves=rocks))[0] == "/choose move stealthrock"
    again = battle("Garchomp", moves=rocks)
    again._turn = 2
    assert decide(me, again) == ("/choose move earthquake", {"rule": "attack"})


@pytest.mark.parametrize("foe", ["Espeon", "Corviknight"])
def test_rocks_skipped_vs_magic_bounce_and_hazard_removal(fake, foe: str) -> None:
    fake()
    assert decide(player(), battle("Garchomp", foe, moves=["earthquake", "stealthrock"])) == (
        "/choose move earthquake",
        {"rule": "attack"},
    )


# Status ---------------------------------------------------------------------------------------------------
def test_will_o_wisp_at_a_physical_attacker_toxic_at_a_special_one(fake) -> None:
    fake()
    assert decide(player(), battle("Slowbro")) == (
        "/choose move willowisp",
        {"rule": "free:status", "free": "willowisp"},
    )
    assert decide(player(), battle("Slowbro", "Milotic")) == (
        "/choose move toxic",
        {"rule": "free:status", "free": "toxic"},
    )


def test_toxic_skipped_vs_steel_and_status_once_per_foe(fake) -> None:
    fake()
    assert decide(player(), battle("Slowbro", moves=["scald", "toxic"])) == ("/choose move scald", {"rule": "attack"})
    me = player()
    assert decide(me, battle("Slowbro", "Milotic"))[0] == "/choose move toxic"
    missed = battle("Slowbro", "Milotic")
    missed._turn = 2
    assert decide(me, missed)[0] == "/choose move scald"


def test_not_free_falls_back_to_disciplined(fake) -> None:
    fake({("metagross", "slowbro"): (0.5, "meteormash"), ("slowbro", "metagross"): (0.2, "scald")})
    chosen, notes = decide(player(), battle("Slowbro", hp=0.5))
    assert chosen == "/choose move slackoff" and notes["rule"] == "recover"


# Factory --------------------------------------------------------------------------------------------------
def test_factory_builds_it_with_the_config_thresholds(config: Config, fake) -> None:
    fake()
    built = build_player(config, "free_turn")
    assert isinstance(built, FreeTurnPlayer) and built.name == "free_turn"
    assert built.free == FreeTurnParams()  # the yaml holds the defaults
    assert built.params()["free_hit_max"] == 0.4 and built.params()["recover_below"] == 0.6
    assert "free_turn" in SIM_KINDS and "free_turn" in KINDS and "free_turn" not in LIVE_KINDS

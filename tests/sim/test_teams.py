"""Simulator team sampling against the real Showdown validator (skipped without `pcsa setup --with sim`), and
the p2-to-p1 mirror the arena replays each decision through."""

from __future__ import annotations

import random

import pytest

from pcsa.battle.builds import Catalog
from pcsa.battle.state import replay
from pcsa.common.config import SimConfig
from pcsa.common.errors import ConfigError
from pcsa.players import mirror
from pcsa.sim.engine import Engine
from pcsa.sim.teams import sample_teams
from pcsa.sim.validate import validate


def _engine() -> Engine | None:
    try:
        return Engine.locate(SimConfig().format)
    except ConfigError:
        return None


ENGINE = _engine()


@pytest.mark.skipif(ENGINE is None, reason="the simulator is not installed; run `pcsa setup --with sim`")
def test_sampled_teams_validate() -> None:
    assert ENGINE is not None
    teams = sample_teams(Catalog.load(), ENGINE, random.Random(11), 2)
    assert len(teams) == 2
    assert all(r["ok"] for r in validate(ENGINE, teams))


P2_PROTOCOL = [
    "|player|p1|alice|",
    "|player|p2|bob|",
    "|gen|9",
    "|start",
    "|switch|p1a: Snorlax|Snorlax, L50|100/100",
    "|switch|p2a: Garchomp|Garchomp, L50|185/185",
    "|turn|1",
]
P2_REQUEST = {
    "active": [
        {
            "moves": [
                {"move": "Earthquake", "id": "earthquake", "pp": 16, "maxpp": 16, "target": "allAdjacent"},
                {"move": "Dragon Claw", "id": "dragonclaw", "pp": 24, "maxpp": 24, "target": "normal"},
            ]
        }
    ],
    "side": {
        "name": "bob",
        "id": "p2",
        "pokemon": [
            {
                "ident": "p2: Garchomp",
                "details": "Garchomp, L50",
                "condition": "185/185",
                "active": True,
                "stats": {"atk": 182, "def": 115, "spa": 90, "spd": 105, "spe": 169},
                "moves": ["earthquake", "dragonclaw"],
                "baseAbility": "roughskin",
                "item": "",
                "pokeball": "pokeball",
            },
            {
                "ident": "p2: Rotom",
                "details": "Rotom-Wash, L50",
                "condition": "125/125",
                "active": False,
                "stats": {"atk": 70, "def": 127, "spa": 125, "spd": 127, "spe": 106},
                "moves": ["hydropump"],
                "baseAbility": "levitate",
                "item": "",
                "pokeball": "pokeball",
            },
        ],
    },
    "rqid": 3,
}


def test_mirror_protocol_swaps_the_sides() -> None:
    assert mirror.mirror_protocol(["|switch|p2a: Garchomp|Garchomp, L50|185/185", "|player|p1|alice|"]) == [
        "|switch|p1a: Garchomp|Garchomp, L50|185/185",
        "|player|p2|alice|",
    ]


def test_a_p2_seat_is_mirrored_into_p1() -> None:
    lines, request = mirror.as_p1("p2", P2_PROTOCOL, P2_REQUEST)
    assert lines[:2] == ["|player|p1|me||", "|player|p2|opponent||"]
    assert "|switch|p1a: Garchomp|Garchomp, L50|185/185" in lines
    assert request["side"]["id"] == "p1"
    assert [m["ident"] for m in request["side"]["pokemon"]] == ["p1: Garchomp", "p1: Rotom"]
    assert P2_REQUEST["side"]["id"] == "p2"


def test_a_p1_seat_only_renames_the_players() -> None:
    lines, request = mirror.as_p1("p1", P2_PROTOCOL, P2_REQUEST)
    assert lines[:2] == ["|player|p1|me||", "|player|p2|opponent||"]
    assert lines[4:] == P2_PROTOCOL[4:]
    assert request is P2_REQUEST


def test_legal_actions_of_a_mirrored_request() -> None:
    battle = replay(*mirror.as_p1("p2", P2_PROTOCOL, P2_REQUEST))
    assert battle.active_pokemon.species == "garchomp"
    actions = mirror.legal_actions(battle)
    moves = sorted(a.action.move for a in actions if a.action.kind == "move" and not a.action.mega)
    assert moves == [0, 1]
    assert [a.message for a in actions if a.action.kind == "switch"] == ["/choose switch Rotom"]

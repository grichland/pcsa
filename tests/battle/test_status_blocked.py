"""Taunt and an Assault Vest stop our member choosing a status move; the game's move screen does not show it, so
the request rebuilt from the log has to (run 20261111-190432, battle 7: Slack Off and Yawn offered under Taunt)."""

from __future__ import annotations

from pcsa.battle.state import battle_request, replay
from pcsa.battle.teams import OwnMember, Team
from pcsa.common.types import MoveInfo
from tests.battle.conftest import PRELUDE

MOVES = ("Earthquake", "Stealth Rock", "Yawn", "Slack Off")
START = [
    *PRELUDE,
    "|switch|p1a: Hippowdon|Hippowdon, L50|215/215",
    "|switch|p2a: Rillaboom|Rillaboom, L50|100/100",
    "|turn|1",
]
TAUNT = ["|move|p2a: Rillaboom|Taunt|p1a: Hippowdon", "|-start|p1a: Hippowdon|move: Taunt", "|turn|2"]
SCREEN = [MoveInfo(i, name, (8, 8)) for i, name in enumerate(MOVES)]


def rows(lines: list[str], item: str = "Sitrus Berry") -> dict[str, bool]:
    team = Team(None, (OwnMember(0, "Hippowdon", "Hippowdon", "Sand Stream", item, MOVES, None, None, "Impish"),))
    request = battle_request(team, [0], replay(lines, None), active=0, screen_hp={}, fainted=set(), moves=SCREEN)
    return {m["id"]: m["disabled"] for m in request["active"][0]["moves"]}


def test_under_taunt_only_attacks_stay_usable_and_it_ends_with_the_taunt() -> None:
    free = {"earthquake": False, "stealthrock": False, "yawn": False, "slackoff": False}
    assert rows(START) == free
    assert rows([*START, *TAUNT]) == {"earthquake": False, "stealthrock": True, "yawn": True, "slackoff": True}
    over = [
        *START,
        *TAUNT,
        "|move|p1a: Hippowdon|Earthquake|p2a: Rillaboom",
        "|-end|p1a: Hippowdon|move: Taunt",
        "|turn|3",
    ]
    assert rows(over) == free


def test_an_assault_vest_blocks_status_moves_from_the_start() -> None:
    assert rows(START, "Assault Vest") == {"earthquake": False, "stealthrock": True, "yawn": True, "slackoff": True}

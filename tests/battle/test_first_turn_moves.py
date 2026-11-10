"""Fake Out / First Impression are only offered on the member's first move action, as the simulator's request does.

The game's move screen does not grey them out, so the live request must (the neural player never saw them legal
on a later turn in training)."""

from __future__ import annotations

from pcsa.battle.state import acted_since_entry, battle_request, replay
from pcsa.battle.teams import OwnMember, Team
from pcsa.common.types import MoveInfo
from tests.battle.conftest import PRELUDE

MOVES = ("First Impression", "Iron Head", "Sucker Punch", "Leech Life")
TEAM = Team(
    None, (OwnMember(0, "Golisopod", "Golisopod", "Emergency Exit", "Leftovers", MOVES, None, None, "Adamant"),)
)
START = [
    *PRELUDE,
    "|switch|p1a: Golisopod|Golisopod, L50|182/182",
    "|switch|p2a: Snorlax|Snorlax, L50|100/100",
    "|turn|1",
]
SCREEN = [MoveInfo(i, name, (16, 16)) for i, name in enumerate(MOVES)]


def rows(lines: list[str]) -> dict[str, bool]:
    battle = replay(lines, None)
    request = battle_request(TEAM, [0], battle, active=0, screen_hp={}, fainted=set(), moves=SCREEN)
    return {m["id"]: m["disabled"] for m in request["active"][0]["moves"]}


def test_first_impression_is_offered_once_per_entry() -> None:
    assert rows(START) == dict.fromkeys(("firstimpression", "ironhead", "suckerpunch", "leechlife"), False)
    used = [*START, "|move|p1a: Golisopod|First Impression|p2a: Snorlax", "|turn|2"]
    assert rows(used) == {"firstimpression": True, "ironhead": False, "suckerpunch": False, "leechlife": False}
    other = [*START, "|move|p1a: Golisopod|Iron Head|p2a: Snorlax", "|turn|2"]
    assert rows(other)["firstimpression"] is True  # any move action ends the first turn
    stopped = [*START, "|cant|p1a: Golisopod|flinch", "|turn|2"]
    assert rows(stopped)["firstimpression"] is True and acted_since_entry(replay(stopped, None), "Golisopod")


def test_coming_back_in_offers_it_again() -> None:
    back = [
        *START,
        "|move|p1a: Golisopod|First Impression|p2a: Snorlax",
        "|turn|2",
        "|switch|p1a: Golisopod|Golisopod, L50|182/182",
        "|turn|3",
    ]
    assert rows(back)["firstimpression"] is False
    assert not acted_since_entry(replay(back, None), "Golisopod")
    assert not acted_since_entry(replay(START, None), "Golisopod")

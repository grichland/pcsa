"""A Choice item locks our member into its last move until it leaves the field; the request must say so."""

from __future__ import annotations

from pcsa.battle.state import battle_request, choice_lock, replay
from pcsa.battle.teams import OwnMember, Team
from pcsa.common.types import MoveInfo
from tests.battle.conftest import PRELUDE

MOVES = ("Dark Pulse", "Draco Meteor", "Flamethrower", "Focus Blast")
TEAM = Team(
    None, (OwnMember(0, "Hydreigon", "Hydreigon", "Levitate", "Choice Scarf", True, MOVES, None, None, "Timid"),)
)
START = [
    *PRELUDE,
    "|switch|p1a: Hydreigon|Hydreigon, L50|169/169",
    "|switch|p2a: Snorlax|Snorlax, L50|100/100",
    "|turn|1",
]
SCREEN = [MoveInfo(i, name, (16, 16)) for i, name in enumerate(MOVES)]


def rows(lines: list[str]) -> dict[str, bool]:
    battle = replay(lines, None)
    request = battle_request(TEAM, [0], battle, active=0, screen_hp={}, fainted=set(), moves=SCREEN)
    return {m["id"]: m["disabled"] for m in request["active"][0]["moves"]}


def test_after_a_move_only_that_move_stays_usable() -> None:
    assert rows(START) == dict.fromkeys(("darkpulse", "dracometeor", "flamethrower", "focusblast"), False)
    used = [*START, "|move|p1a: Hydreigon|Dark Pulse|p2a: Snorlax", "|turn|2"]
    assert rows(used) == {"darkpulse": False, "dracometeor": True, "flamethrower": True, "focusblast": True}
    assert choice_lock(replay(used, None), "Hydreigon", "Choice Scarf") == "darkpulse"


def test_leaving_the_field_clears_the_lock_and_a_plain_item_never_locks() -> None:
    back = [
        *START,
        "|move|p1a: Hydreigon|Dark Pulse|p2a: Snorlax",
        "|turn|2",
        "|switch|p1a: Hydreigon|Hydreigon, L50|169/169",
        "|turn|3",
    ]
    assert all(not disabled for disabled in rows(back).values())
    assert choice_lock(replay(back, None), "Hydreigon", "Choice Scarf") is None
    assert (
        choice_lock(replay([*START, "|move|p1a: Hydreigon|Dark Pulse|p2a: Snorlax"], None), "Hydreigon", "Leftovers")
        is None
    )

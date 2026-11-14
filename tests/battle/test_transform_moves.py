"""A Ditto that copies one of ours holds our whole move set once the request is in (battle/adapter.py).

14 Nov 2026 battle 43: the live tracker replays the log before our request, so Ditto, transforming into a Cinderace
that had not moved yet, copied no moves; the opponent head then never offered High Jump Kick, which KO'd us."""

from __future__ import annotations

from pcsa.battle.state import apply_request, battle_request, replay
from pcsa.battle.teams import OwnMember, Team
from pcsa.battle.translate import Translation
from pcsa.common.types import MoveInfo
from tests.battle.conftest import PRELUDE

MOVES = ("Pyro Ball", "High Jump Kick", "Gunk Shot", "U-turn")
TEAM = Team(None, (OwnMember(0, "Cinderace", "Cinderace", "Libero", "Choice Scarf", MOVES, None, None, "Jolly"),))
LINES = [
    *PRELUDE,
    "|switch|p1a: Cinderace|Cinderace, L50|157/157",
    "|switch|p2a: Lucario|Lucario, L50|100/100",
    "|turn|1",
    "|switch|p2a: Ditto|Ditto, L50|100/100",
    "|-transform|p2a: Ditto|p1a: Cinderace",
    "|move|p1a: Cinderace|Pyro Ball|p2a: Ditto",
    "|turn|2",
]
SCREEN = [MoveInfo(i, name, (8, 8)) for i, name in enumerate(MOVES)]


def test_a_transformed_ditto_holds_our_whole_move_set_after_the_request() -> None:
    battle = replay(LINES, None)
    ditto = battle.opponent_active_pokemon
    assert set(ditto.moves) == set()  # the log alone: Cinderace had shown nothing when Ditto copied it
    request = battle_request(TEAM, [0], battle, active=0, screen_hp={}, fainted=set(), moves=SCREEN)
    apply_request(battle, Translation(), request)
    assert set(ditto.moves) == {"pyroball", "highjumpkick", "gunkshot", "uturn"}
    assert battle.active_pokemon.moves.keys() == ditto.moves.keys()

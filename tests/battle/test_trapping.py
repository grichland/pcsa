"""Shadow Tag, Arena Trap, Magnet Pull and trapping moves stop our active switching; the request must say so."""

from __future__ import annotations

from pcsa.battle.state import battle_request, replay
from pcsa.battle.teams import OwnMember, Team
from pcsa.battle.trapping import is_trapped
from pcsa.common.types import MoveInfo
from tests.battle.conftest import PRELUDE

SCREEN = [MoveInfo(0, "Tackle", (8, 8))]


def start(ours: str, theirs: str, extra=()) -> list[str]:
    return [
        *PRELUDE,
        f"|switch|p1a: {ours}|{ours}, L50|200/200",
        f"|switch|p2a: {theirs}|{theirs}, L50|100/100",
        "|turn|1",
        *extra,
    ]


def trapped(ours, theirs, item="", ability="", extra=()) -> bool:
    battle = replay(start(ours, theirs, extra), None)
    return is_trapped(battle, {"item": item, "ability": ability})


def test_shadow_tag_traps_non_ghost_and_not_ghost() -> None:
    assert trapped("Garchomp", "Gengar-Mega")
    assert trapped("Garchomp", "Wobbuffet", extra=["|-ability|p2a: Wobbuffet|Shadow Tag"])
    assert not trapped("Gholdengo", "Gengar-Mega")
    assert not trapped("Gengar", "Gengar-Mega")
    assert not trapped("Garchomp", "Gengar-Mega", item="Shed Shell")
    assert not trapped("Garchomp", "Gengar")


def test_unrevealed_ambiguous_ability_is_not_a_trap() -> None:
    assert not trapped("Garchomp", "Wobbuffet")


def test_arena_trap_needs_a_grounded_target() -> None:
    arena = ["|-ability|p2a: Dugtrio|Arena Trap"]
    assert trapped("Garchomp", "Dugtrio", extra=arena)
    assert not trapped("Dragonite", "Dugtrio", extra=arena)
    assert not trapped("Garchomp", "Dugtrio", item="Air Balloon", extra=arena)
    assert not trapped("Garchomp", "Dugtrio", ability="Levitate", extra=arena)
    assert not trapped("Garchomp", "Dugtrio", extra=[*arena, "|-start|p1a: Garchomp|Magnet Rise"])


def test_magnet_pull_traps_steel_only() -> None:
    pull = ["|-ability|p2a: Magnezone|Magnet Pull"]
    assert trapped("Kingambit", "Magnezone", extra=pull)
    assert not trapped("Garchomp", "Magnezone", extra=pull)


def test_trapping_volatile_lasts_until_the_user_leaves() -> None:
    hit = ["|-start|p1a: Garchomp|move: Infestation"]
    assert trapped("Garchomp", "Toxapex", extra=hit)
    assert not trapped("Gengar", "Toxapex", extra=hit)
    assert not trapped("Garchomp", "Toxapex", item="Shed Shell", extra=hit)
    assert not trapped("Garchomp", "Toxapex", extra=[*hit, "|-end|p1a: Garchomp|move: Infestation"])
    left = [*hit, "|turn|2", "|switch|p2a: Altaria|Altaria, L50|100/100"]
    assert not trapped("Garchomp", "Toxapex", extra=left)
    assert trapped("Garchomp", "Toxapex", extra=[*left, "|-start|p1a: Garchomp|move: Fire Spin"])


def test_request_carries_the_flag_and_pivot_moves_stay() -> None:
    members = (
        OwnMember(0, "Garchomp", "Garchomp", "Rough Skin", "", ("U-turn",), None, None, "Jolly"),
        OwnMember(1, "Gholdengo", "Gholdengo", "Good as Gold", "", ("Make It Rain",), None, None, "Modest"),
    )
    lines = [
        *PRELUDE,
        "|switch|p1a: Garchomp|Garchomp, L50|185/185",
        "|switch|p2a: Gengar|Gengar-Mega, L50|100/100",
        "|turn|1",
    ]
    screen = [MoveInfo(0, "U-turn", (8, 8))]
    team = Team(None, members)
    battle = replay(lines, None)
    request = battle_request(team, [0, 1], battle, active=0, screen_hp={}, fainted=set(), moves=screen)
    assert request["active"][0]["trapped"] is True
    battle.parse_request(request)
    assert battle.trapped and not battle.available_switches
    assert [m.id for m in battle.available_moves] == ["uturn"]
    free = replay([*lines[:5], "|switch|p2a: Snorlax|Snorlax, L50|100/100", "|turn|2"], None)
    request = battle_request(team, [0, 1], free, active=0, screen_hp={}, fainted=set(), moves=screen)
    assert "trapped" not in request["active"][0]
    free.parse_request(request)
    assert [m.species for m in free.available_switches] == ["gholdengo"]

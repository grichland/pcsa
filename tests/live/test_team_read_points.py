"""Stat points recovered from exact stats when the summary screen's point boxes were not read."""

from __future__ import annotations

from pcsa.battle.stats import nature_name
from pcsa.live.team_read import from_sheet
from pcsa.phone.team_sheet import MemberStats, ReadField, RosterMember, TeamSheet

STATS = {"hp": 185, "atk": 93, "def": 143, "spa": 130, "spd": 106, "spe": 120}  # Charizard, Bold, 32/0/32/1/1/0


def field(value: str | None) -> ReadField:
    return ReadField(value, "known" if value else "unknown", 1.0, (0, 0, 1, 1))


def member(points: dict) -> RosterMember:
    stats = MemberStats("Charizard", dict(STATS), points, "def", "atk")
    return RosterMember(
        0, field("Charizard"), field("Blaze"), field("Charizardite Y"), (field("Flamethrower"),) * 4, stats
    )


def test_points_are_solved_from_exact_stats_when_unread() -> None:
    unread = dict.fromkeys(STATS, None)
    team = from_sheet(TeamSheet("Team", (member(unread),)))
    assert team.members[0].points == {"hp": 32, "atk": 0, "def": 32, "spa": 1, "spd": 1, "spe": 0}
    assert team.members[0].nature == "Bold"


def test_read_points_are_kept_as_read() -> None:
    read = {"hp": 32, "atk": 0, "def": 32, "spa": 1, "spd": 1, "spe": 0}
    assert from_sheet(TeamSheet("Team", (member(read),))).members[0].points == read


def test_nature_names() -> None:
    assert nature_name({"atk": 90, "def": 110, "spa": 100, "spd": 100, "spe": 100}) == "Bold"
    assert nature_name({"atk": 110, "def": 100, "spa": 100, "spd": 100, "spe": 90}) == "Brave"
    assert nature_name(dict.fromkeys(("atk", "def", "spa", "spd", "spe"), 100)) in (
        "Hardy",
        "Bashful",
        "Docile",
        "Quirky",
        "Serious",
    )

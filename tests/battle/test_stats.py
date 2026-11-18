"""Stat arithmetic and nature names."""

from __future__ import annotations

from pcsa.battle.stats import hp_value, nature_name, solve_points, stat_value


def test_nature_names() -> None:
    assert nature_name({"atk": 90, "def": 110, "spa": 100, "spd": 100, "spe": 100}) == "Bold"
    assert nature_name({"atk": 110, "def": 100, "spa": 100, "spd": 100, "spe": 90}) == "Brave"
    assert nature_name(dict.fromkeys(("atk", "def", "spa", "spd", "spe"), 100)) == "Bashful"


def test_points_solve_back() -> None:
    base = {"hp": 108, "atk": 130, "def": 95, "spa": 80, "spd": 85, "spe": 102}
    stats = {
        "hp": hp_value(108, 2),
        "atk": stat_value(130, 32, 110),
        "def": stat_value(95, 0),
        "spa": stat_value(80, 0, 90),
    }
    stats |= {"spd": stat_value(85, 0), "spe": stat_value(102, 32)}
    points, nature = solve_points(base, stats)
    assert points == {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}
    assert nature_name(nature) == "Adamant"

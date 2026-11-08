"""Damage formula: the base arithmetic by hand, then each modifier family against that baseline."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pcsa.battle.damage import Conditions, Fighter, build_stats, damage, investment_bounds, nature_percent
from pcsa.battle.moves import base_data

FLAT = {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
WATER = Fighter(FLAT, ("water",))
MOVE = "bodyslam"  # Normal, physical: no STAB for a Water attacker, neutral on Water


def expected(power: int) -> tuple[float, float]:
    base = ((2 * 50 // 5 + 2) * power * 100 // 100) // 50 + 2
    return (base * 85 // 100) / 100, base / 100


def test_base_formula_by_hand() -> None:
    assert damage(WATER, WATER, MOVE) == pytest.approx(expected(base_data(MOVE)["power"]))


def test_status_move_and_immunities_do_nothing() -> None:
    assert damage(WATER, WATER, "swordsdance") == (0.0, 0.0)
    assert damage(WATER, Fighter(FLAT, ("ghost",)), MOVE) == (0.0, 0.0)
    assert damage(WATER, replace(WATER, ability="levitate"), "earthquake") == (0.0, 0.0)
    assert damage(WATER, replace(WATER, ability="waterabsorb"), "surf") == (0.0, 0.0)


def test_modifiers_scale_the_roll() -> None:
    low, high = damage(WATER, WATER, MOVE)
    assert damage(replace(WATER, types=("normal",)), WATER, MOVE)[1] == pytest.approx(high * 1.5)
    assert damage(WATER, WATER, MOVE, Conditions(defender_screens=frozenset({"reflect"})))[1] == pytest.approx(
        high * 0.5
    )
    assert damage(replace(WATER, item="lifeorb"), WATER, MOVE)[1] == pytest.approx(high * 1.3)
    assert damage(replace(WATER, status="brn"), WATER, MOVE)[1] == pytest.approx(high * 0.5)
    assert damage(replace(WATER, boosts={"atk": 2}), WATER, MOVE)[1] > high
    surf_rain = damage(WATER, Fighter(FLAT, ("normal",)), "surf", Conditions(weather="raindance"))
    surf = damage(WATER, Fighter(FLAT, ("normal",)), "surf")
    assert surf_rain[1] > surf[1]
    assert low < high


def test_unknown_spread_brackets_every_known_spread() -> None:
    base = {"hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80}
    attacker = Fighter(build_stats(base, {"atk": 32}, {"atk": 110}), ("fire",))
    weakest, strongest = investment_bounds(base)
    unknown = Fighter(weakest, ("water",), base=base, spread_known=False)
    lo, hi = damage(attacker, unknown, MOVE)
    for points in (0, 16, 32):
        known = Fighter(build_stats(base, {"hp": points, "def": points}, {}), ("water",))
        k_lo, k_hi = damage(attacker, known, MOVE)
        assert lo <= k_lo and k_hi <= hi
    assert strongest["def"] > weakest["def"]


def test_nature_percent_and_power_override() -> None:
    assert nature_percent("Adamant") == {"atk": 110, "def": 100, "spa": 90, "spd": 100, "spe": 100}
    assert nature_percent("nonsense") == {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
    assert damage(WATER, WATER, MOVE, power=170)[1] > damage(WATER, WATER, MOVE)[1]


def test_type_chart_doubles_and_quarters() -> None:
    fire = Fighter(FLAT, ("fire",))
    neutral = damage(fire, WATER, "rockslide")[1]
    assert damage(fire, Fighter(FLAT, ("fire", "flying")), "rockslide")[1] == pytest.approx(neutral * 4, rel=0.05)


def test_attacker_abilities() -> None:
    contact = damage(WATER, WATER, MOVE)[1]  # Body Slam makes contact
    assert damage(replace(WATER, ability="toughclaws"), WATER, MOVE)[1] == pytest.approx(contact * 1.3, rel=0.05)
    assert damage(replace(WATER, ability="ironfist"), WATER, MOVE)[1] == pytest.approx(contact)
    weak = damage(WATER, WATER, "quickattack")[1]
    assert damage(replace(WATER, ability="technician"), WATER, "quickattack")[1] == pytest.approx(weak * 1.5, rel=0.1)
    ghost = Fighter(FLAT, ("ghost",))
    assert damage(replace(WATER, ability="pixilate"), ghost, MOVE)[1] > 0  # Normal turned Fairy hits a Ghost


def test_defender_abilities_and_stat_swaps() -> None:
    fire = Fighter(FLAT, ("fire",))
    surf = damage(WATER, fire, "surf")[1]
    assert damage(WATER, replace(fire, ability="filter"), "surf")[1] == pytest.approx(surf * 0.75, rel=0.05)
    strong_foe = replace(WATER, stats={**FLAT, "atk": 200})
    assert damage(WATER, strong_foe, "foulplay")[1] > damage(WATER, WATER, "foulplay")[1]
    bulky = replace(WATER, stats={**FLAT, "def": 200})
    assert damage(bulky, WATER, "bodypress")[1] > damage(WATER, WATER, "bodypress")[1]

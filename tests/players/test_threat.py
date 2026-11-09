"""HP-fraction threats: Charizard against Tyranitar by type, KO checks, speed order and the threat threshold."""

from __future__ import annotations

import pytest
from poke_env.battle import Field

from pcsa.players import threat
from pcsa.players.opponent import Tracker, default_catalog, speed
from tests.players.test_opponent import CHARIZARD, battle

CATALOG = default_catalog()


def sides(foe: str = "Tyranitar", **kwargs):
    b = battle(foe, **kwargs)
    return b, b.active_pokemon, b.opponent_active_pokemon


def test_fighters_are_exact_for_ours_and_assumed_for_theirs() -> None:
    b, me, them = sides()
    assert threat.fighters(b, me, CATALOG).stats == CHARIZARD
    theirs = threat.fighters(b, them, CATALOG)
    assert theirs.types == ("rock", "dark") and theirs.item == "tyranitarite"


def test_rock_beats_charizard_and_fire_does_not_ko_tyranitar() -> None:
    b, me, them = sides()
    assert threat.hit_fraction(b, them, me, "stoneedge", CATALOG) > 0.5
    share, move = threat.best_hit(b, them, me, CATALOG)
    assert move == "stoneedge" and share > 1.0
    assert threat.hit_fraction(b, me, them, "flamethrower", CATALOG) < 0.5
    assert not threat.can_ko(b, me, them, CATALOG)
    assert threat.can_ko(b, them, me, CATALOG)


def test_immunity_and_status_moves_do_nothing() -> None:
    b, me, them = sides()
    assert threat.hit_fraction(b, them, me, "earthquake", CATALOG) == 0.0
    assert threat.hit_fraction(b, me, them, "roost", CATALOG) == 0.0


def test_ko_on_us_uses_the_max_roll() -> None:
    b, me, snorlax = sides("Snorlax")  # top build: Ice Punch, Yawn, Earthquake (immune), Rest
    mean = threat.hit_fraction(b, snorlax, me, "icepunch", CATALOG)
    me._current_hp = round(CHARIZARD["hp"] * mean * 1.05)  # above the mean roll, below the max
    assert me.current_hp_fraction > mean
    assert threat.can_ko(b, snorlax, me, CATALOG)
    me._current_hp = round(CHARIZARD["hp"] * mean * 1.12)
    assert not threat.can_ko(b, snorlax, me, CATALOG)


def test_outspeeds_and_trick_room() -> None:
    b, me, them = sides()
    assert threat.outspeeds(b, me, them, CATALOG) and not threat.outspeeds(b, them, me, CATALOG)
    b._fields[Field.TRICK_ROOM] = 1
    assert threat.outspeeds(b, them, me, CATALOG)


def test_assumed_scarf_changes_the_speed_order() -> None:
    turn = [
        "|move|p2a: Tyranitar|Crunch|p1a: Charizard",
        "|-damage|p1a: Charizard|110/155",
        "|move|p1a: Charizard|Flamethrower|p2a: Tyranitar",
        "|-damage|p2a: Tyranitar|80/100",
    ]
    b, me, them = sides(events=turn, hp=110)
    tracker = Tracker(CATALOG)
    tracker.observe(b)
    plain = threat.fighters(b, them, CATALOG).stats["spe"]
    scarfed = threat.fighters(b, them, CATALOG, tracker)
    assert scarfed.item == "choicescarf"
    assert speed(b, them, scarfed) == pytest.approx(plain * 1.5, abs=1)


def test_threatened_threshold() -> None:
    b, me, them = sides()
    assert threat.threatened(b, me, them, CATALOG)
    assert not threat.threatened(b, them, me, CATALOG, threshold=0.6)
    assert threat.threatened(b, them, me, CATALOG, threshold=0.1)


def test_priority_ko() -> None:
    b, me, them = sides("Tyranitar")
    assert threat.priority_ko(b, them, me, CATALOG) is None
    b2, me2, them2 = sides("Tyranitar", moves=["flamethrower", "quickattack"])
    them2._current_hp = 1
    assert threat.priority_ko(b2, me2, them2, CATALOG) == "quickattack"


def test_triple_axel_counts_its_escalation_once() -> None:
    from poke_env.battle import Move

    from pcsa.players.threat import ESCALATING_HITS, _hits

    axel = Move("tripleaxel", gen=9)
    assert _hits(axel) == ESCALATING_HITS < 3  # move_power already gives the mean power per hit (40)
    assert _hits(Move("bulletseed", gen=9)) == Move("bulletseed", gen=9).expected_hits

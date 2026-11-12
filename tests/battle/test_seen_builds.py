"""Builds seen on the live ladder: covered by the catalog or added to it, never counted twice, and ranked first in
the players' catalog (and so in the assumed set)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcsa.battle.builds import Catalog
from pcsa.battle.seen_builds import Observed, SeenBuilds
from pcsa.players.opponent import _base_build, fits
from tests.battle.test_builds import CATALOG


@pytest.fixture
def store(tmp_path: Path) -> SeenBuilds:
    return SeenBuilds.load(tmp_path / "seen_builds.json")


def plain() -> Catalog:
    return Catalog(CATALOG)


def moves(build) -> list[str]:
    return list(build.moves)


def test_a_covered_set_marks_the_catalog_build_and_ranks_it_first(store: SeenBuilds) -> None:
    report = store.learn("b1", [Observed("garchomp", ("earthquake", "swordsdance"))], plain(), "2026-11-10")
    assert report[0]["result"] == "covered by the catalog"
    players = Catalog(CATALOG, seen=store)
    assert players.builds("Garchomp")[0].item == "Focus Sash"  # the seen build, ahead of the usage top
    assert len(players.builds("Garchomp")) == 2  # no duplicate
    assert plain().builds("Garchomp")[0].item == "Choice Scarf"  # the plain catalog is unchanged


def test_an_uncovered_set_is_added_and_filled_from_the_closest_build(store: SeenBuilds) -> None:
    report = store.learn("b1", [Observed("garchomp", ("spikes", "dragontail"), "focussash")], plain(), "2026-11-10")
    assert report[0]["result"] == "added"
    added = Catalog(CATALOG, seen=store).builds("Garchomp")[0]
    assert added.item == "Focus Sash" and moves(added)[:2] == ["Spikes", "Dragon Tail"] and len(added.moves) == 4
    assert set(moves(added)[2:]) <= {"Earthquake", "Scale Shot", "Rock Tomb", "Swords Dance"}  # the sash build's


def test_the_same_battle_is_never_counted_twice_and_repeats_rank_higher(store: SeenBuilds) -> None:
    scarf, sash = Observed("garchomp", ("outrage",)), Observed("garchomp", ("swordsdance",))
    store.learn("b1", [sash], plain(), "2026-11-09")
    store.learn("b2", [scarf], plain(), "2026-11-10")
    store.learn("b3", [scarf], plain(), "2026-11-10")
    assert store.learn("b3", [scarf], plain())[0]["result"] == "already learned"
    assert [s.count for s in store.ranked("garchomp")] == [2, 1]
    assert Catalog(CATALOG, seen=store).builds("Garchomp")[0].item == "Choice Scarf"


def test_an_added_build_learns_what_later_sightings_reveal(store: SeenBuilds) -> None:
    store.learn("b1", [Observed("garchomp", ("spikes",))], plain(), "2026-11-10")
    report = store.learn("b2", [Observed("garchomp", ("spikes", "dragontail"))], plain(), "2026-11-10")
    assert report[0]["result"] == "added build extended"
    [only] = store.ranked("garchomp")
    assert only.known_moves == ("spikes", "dragontail") and only.count == 2


def test_impossible_sightings_are_skipped_and_the_file_round_trips(store: SeenBuilds) -> None:
    five = Observed("garchomp", ("earthquake", "outrage", "rocktomb", "stealthrock", "spikes"))
    assert store.learn("b1", [five, Observed("garchomp", ("spikes",))], plain())[0]["result"] == "skipped"
    store.save()
    again = SeenBuilds.load(store.path)
    assert again.battles == ["b1"] and moves(again.ranked("garchomp")[0].build)[0] == "Spikes"


def test_the_assumed_set_starts_from_the_first_build_that_fits_what_was_revealed() -> None:
    builds = plain().builds("Garchomp")
    mon = type("Mon", (), {"species": "garchomp", "item": "unknown_item"})()
    build, _, _ = _base_build(builds, mon, "garchomp", False, None, (["scaleshot"], None))
    assert build.item == "Focus Sash" and fits(build, ["scaleshot"])
    build, _, _ = _base_build(builds, mon, "garchomp", False, None)
    assert build.item == "Choice Scarf"  # nothing revealed: the top build


def test_the_most_recently_seen_build_ranks_first_and_late_learning_keeps_the_dates(store: SeenBuilds) -> None:
    """Today's sighting goes to the top even when an older build was seen more often; a battle from an older run
    learned afterwards adds to the count without making its build look recent."""
    scarf, sash = Observed("garchomp", ("outrage",)), Observed("garchomp", ("swordsdance",))
    store.learn("b1", [scarf], plain(), "2026-11-04")
    store.learn("b2", [scarf], plain(), "2026-11-05")
    store.learn("b3", [sash], plain(), "2026-11-11")
    assert [(s.build.item, s.count) for s in store.ranked("garchomp")] == [("Focus Sash", 1), ("Choice Scarf", 2)]
    assert Catalog(CATALOG, seen=store).builds("Garchomp")[0].item == "Focus Sash"
    store.learn("b0", [sash], plain(), "2026-11-02")  # an older run, learned later
    [top] = [s for s in store.ranked("garchomp") if s.build.item == "Focus Sash"]
    assert (top.count, top.first, top.last) == (2, "2026-11-02", "2026-11-11")

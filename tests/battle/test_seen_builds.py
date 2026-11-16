"""Builds seen on the live ladder: covered by the catalog or added to it, never counted twice, and ranked first in
the players' catalog (and so in the assumed set)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcsa.battle import game
from pcsa.battle.builds import Catalog
from pcsa.battle.seen_builds import Observed, SeenBuilds, observed_from_battle, species_key
from pcsa.battle.state import replay
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


@pytest.mark.parametrize(
    ("form", "key"),
    [
        ("Slowbro-Galar", "slowbrogalar"),  # regional forms keep their own builds, catalog or not
        ("Typhlosion-Hisui", "typhlosionhisui"),
        ("Rotom-Fan", "rotomfan"),
        ("Indeedee-F", "indeedeef"),
        ("Toxtricity-Low-Key", "toxtricitylowkey"),
        ("Tauros-Paldea-Blaze", "taurospaldeablaze"),
        ("Basculegion", "basculegion"),
        ("Furfrou-Dandy", "furfrou"),  # forms that differ only in looks: the species itself
        ("Vivillon-High Plains", "vivillon"),
        ("Maushold-Four", "maushold"),
        ("Charizard-Mega-X", "charizard"),  # Megas and in-battle forms: what they change from
        ("Floette-Mega", "floetteeternal"),
        ("Aegislash-Blade", "aegislash"),
        ("Palafin-Hero", "palafin"),
        ("Garchomp", "garchomp"),
    ],
)
def test_a_form_is_kept_under_its_own_key(form: str, key: str) -> None:
    assert species_key(form, plain()) == key


def slowbro_catalog() -> Catalog:
    """Kanto Slowbro builds only: Scald and Teleport are not in Slowbro-Galar's learnset, nor Oblivious its ability."""
    points = {"hp": 32, "atk": 0, "def": 32, "spa": 2, "spd": 0, "spe": 0}
    first = {"item": "Leftovers", "ability": "Oblivious", "nature": "Bold", "points": points, "score": -1.0}
    second = {"item": "Leftovers", "ability": "Regenerator", "nature": "Bold", "points": points, "score": -2.0}
    builds = [
        first | {"moves": ["Scald", "Teleport", "Slack Off", "Calm Mind"]},
        second | {"moves": ["Scald", "Ice Beam", "Psychic", "Fire Blast"]},
    ]
    return Catalog({"schema": CATALOG["schema"], "species": {"Slowbro": {"usage": 10, "builds": builds}}})


def test_a_regional_form_without_catalog_builds_is_never_filed_under_its_base(store: SeenBuilds) -> None:
    report = store.learn("b1", [Observed("slowbrogalar", ("shellsidearm",))], slowbro_catalog(), "2026-11-15")
    assert report[0]["result"] == "added" and store.ranked("slowbro") == []
    [seen] = store.ranked("slowbrogalar")
    assert seen.build.species == "Slowbro-Galar" and seen.build.ability == "Quick Draw"
    assert list(seen.build.moves) == ["Shell Side Arm", "Slack Off", "Calm Mind", "Ice Beam"]  # learnable fills only
    assert all(game.can_learn("slowbrogalar", m) for m in seen.build.moves)


def test_a_form_is_never_covered_by_its_base_species_catalog_builds(store: SeenBuilds) -> None:
    report = store.learn("b1", [Observed("slowbrogalar", ("slackoff", "calmmind"))], slowbro_catalog(), "2026-11-15")
    assert report[0]["result"] == "added" and store.ranked("slowbrogalar")[0].build.species == "Slowbro-Galar"


def test_the_protocol_form_names_the_observed_species() -> None:
    lines = [
        "|player|p1|me|",
        "|player|p2|opponent|",
        "|gen|9",
        "|poke|p2|Slowbro-Galar, L50|",
        "|poke|p2|Rotom-Wash, L50|",
        "|start",
        "|switch|p2a: Slowbro|Slowbro-Galar, L50|100/100",
        "|move|p2a: Slowbro|Shell Side Arm|",
        "|switch|p2a: Rotom|Rotom-Wash, L50|100/100",
        "|move|p2a: Rotom|Hydro Pump|",
    ]
    observed = observed_from_battle(replay(lines, None), None, plain())
    assert {(o.species, o.moves) for o in observed} == {
        ("slowbrogalar", ("shellsidearm",)),
        ("rotomwash", ("hydropump",)),
    }


def test_an_illusion_user_named_from_its_nickname_takes_the_previewed_form() -> None:
    lines = [
        "|player|p1|me|",
        "|player|p2|opponent|",
        "|gen|9",
        "|poke|p2|Zoroark-Hisui, L50|",
        "|poke|p2|Archaludon, L50|",
        "|start",
        "|switch|p2a: Archaludon|Archaludon, L50|100/100",
        "|-damage|p2a: Zoroark|0 fnt",
        "|detailschange|p2a: Zoroark|zoroarkhisui, L50",
        "|faint|p2a: Zoroark",
    ]
    observed = observed_from_battle(replay(lines, None), {"p2: Zoroark": "Illusion"}, plain())
    assert ("zoroarkhisui", "illusion") in {(o.species, o.ability) for o in observed}

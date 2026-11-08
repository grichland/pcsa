"""Team styles: hand-built sixes that fit or miss each style in config/players/styles.yaml."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from poke_env.data.normalize import to_id_str

from pcsa.battle.builds import Build
from pcsa.common.errors import ConfigError
from pcsa.players import styles


@pytest.fixture(scope="module")
def rules() -> styles.Styles:
    return styles.load()


def b(species: str, ability: str, *moves: str, item: str = "") -> Build:
    return Build(species, item, ability, "Hardy", (0, 0, 0, 0, 0, 0), moves)


GARCHOMP = b("Garchomp", "Rough Skin", "Earthquake", "Dragon Claw")
CLEFABLE = b("Clefable", "Unaware", "Moonblast", "Calm Mind", "Moonlight", "Protect")
GENGAR = b("Gengar", "Cursed Body", "Shadow Ball", "Sludge Bomb")
SNORLAX = b("Snorlax", "Thick Fat", "Body Slam", "Curse", "Rest")
TORKOAL = b("Torkoal", "Drought", "Lava Plume", "Stealth Rock")
VENUSAUR = b("Venusaur", "Chlorophyll", "Giga Drain", "Sludge Bomb", "Sleep Powder")
CHARIZARD = b("Charizard", "Blaze", "Flamethrower", "Air Slash")
KINGDRA = b("Kingdra", "Swift Swim", "Draco Meteor", "Dragon Pulse")
BARRASKEWDA = b("Barraskewda", "Swift Swim", "Close Combat", "Psychic Fangs")


def test_a_sun_six_fits_with_the_setter_leading(rules: styles.Styles) -> None:
    six = [GARCHOMP, VENUSAUR, CLEFABLE, TORKOAL, CHARIZARD, SNORLAX]
    fit = rules.fit("weather", six)
    assert fit is not None
    assert fit.weather == "sun"
    assert fit.roles["setter"] == (3,)
    assert set(fit.roles["abuser"]) == {1, 4}
    assert fit.lead == 3
    assert fit.bring[0] == 3 and set(fit.bring) == {1, 3, 4}


def test_drought_with_swift_swim_abusers_does_not_fit(rules: styles.Styles) -> None:
    six = [TORKOAL, KINGDRA, BARRASKEWDA, GARCHOMP, CLEFABLE, GENGAR]
    assert rules.fit("weather", six) is None


def test_drizzle_with_swift_swim_fits_but_politoed_does_not_lead(rules: styles.Styles) -> None:
    politoed = b("Politoed", "Drizzle", "Scald", "Encore")
    six = [GARCHOMP, politoed, KINGDRA, BARRASKEWDA, CLEFABLE, GENGAR]
    fit = rules.fit("weather", six)
    assert fit is not None and fit.weather == "rain"
    assert fit.roles["setter"] == (1,)
    assert 1 in fit.bring and fit.lead != 1


def test_a_mega_stone_gives_its_forms_weather(rules: styles.Styles) -> None:
    mega_y = b("Charizard", "Blaze", "Flamethrower", "Solar Beam", item="Charizardite Y")
    arcanine = b("Arcanine", "Intimidate", "Flare Blitz", "Extreme Speed")
    six = [GARCHOMP, VENUSAUR, CLEFABLE, mega_y, arcanine, SNORLAX]
    fit = rules.fit("weather", six)
    assert fit is not None and fit.weather == "sun" and fit.roles["setter"] == (3,)
    assert rules.fit("weather", [GARCHOMP, VENUSAUR, CLEFABLE, CHARIZARD, arcanine, SNORLAX]) is None


def test_setup_sweeper_needs_priority_or_bulk(rules: styles.Styles) -> None:
    others = [GARCHOMP, replace(CLEFABLE, moves=("Moonblast", "Moonlight")), GENGAR]
    dragonite = b("Dragonite", "Multiscale", "Dragon Dance", "Extreme Speed", "Earthquake")
    kingambit = b("Kingambit", "Supreme Overlord", "Swords Dance", "Kowtow Cleave", "Iron Head")
    volcarona = b("Volcarona", "Flame Body", "Quiver Dance", "Fiery Dance", "Bug Buzz")
    lopunny = b("Lopunny", "Limber", "Swords Dance", "Fake Out", "Return")
    for sweeper in (dragonite, kingambit):
        fit = rules.fit("setup_sweeper", [*others, sweeper, SNORLAX, VENUSAUR])
        assert fit is not None and fit.roles["sweeper"] == (3,) and len(fit.roles["supporter"]) == 2
    for frail in (volcarona, lopunny):  # no priority (Fake Out does not count) and base bulk under 270
        assert rules.fit("setup_sweeper", [*others, frail, SNORLAX, VENUSAUR]) is None


def test_hazard_lead_needs_rocks_and_two_real_attackers(rules: styles.Styles) -> None:
    setter = b("Garchomp", "Rough Skin", "Stealth Rock", "Earthquake", "Dragon Claw")
    fit = rules.fit("hazard_lead", [CLEFABLE, setter, GENGAR, CHARIZARD, SNORLAX, VENUSAUR])
    assert fit is not None and fit.roles["setter"] == (1,) and fit.lead == 1
    assert rules.fit("hazard_lead", [CLEFABLE, GARCHOMP, GENGAR, CHARIZARD, SNORLAX, VENUSAUR]) is None
    passive = [
        b(name, "Pressure", attack, "Protect", "Rest")
        for name, attack in [("Clefable", "Moonblast"), ("Gengar", "Hex"), ("Snorlax", "Body Slam")]
    ]
    assert rules.fit("hazard_lead", [setter, *passive, b("Blissey", "Natural Cure", "Seismic Toss")]) is None


def test_fit_mons_reads_our_battle_team(rules: styles.Styles) -> None:
    six = [GARCHOMP, VENUSAUR, CLEFABLE, TORKOAL, CHARIZARD, SNORLAX]
    mons = [
        SimpleNamespace(
            species=to_id_str(m.species),
            item=to_id_str(m.item),
            ability=to_id_str(m.ability),
            moves={to_id_str(x): None for x in m.moves},
        )
        for m in six
    ]
    assert rules.fit_mons("weather", mons) == rules.fit("weather", six)


def test_plans_and_bad_configs(rules: styles.Styles) -> None:
    assert {"weather", "setup_sweeper", "hazard_lead"} <= set(rules.names())
    assert sorted(p[0].weather.name for p in rules.plans("weather") if p[0].weather) == [
        "rain",
        "sand",
        "snow",
        "sun",
    ]
    assert [d.label for d in rules.plans("setup_sweeper")[0]] == ["sweeper"]
    with pytest.raises(ConfigError):
        rules.fit("nope", [GARCHOMP])
    with pytest.raises(ConfigError):
        styles.Styles({"styles": {"x": {"roles": {"r": {"min": 1, "bogus": 1}}}}})
    with pytest.raises(ConfigError):
        styles.Styles({"styles": {"x": {"roles": {"r": {"min": 1}}, "lead": "other"}}})

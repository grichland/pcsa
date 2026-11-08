from __future__ import annotations

import pytest

from pcsa.battle.builds import Catalog

CATALOG = {
    "schema": "pcsa-top-builds/1",
    "species": {
        "Garchomp": {
            "usage": 100,
            "builds": [
                {
                    "item": "Choice Scarf",
                    "ability": "Rough Skin",
                    "nature": "Jolly",
                    "points": {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32},
                    "moves": ["Earthquake", "Outrage", "Rock Tomb", "Stealth Rock"],
                    "score": -2.3,
                },
                {
                    "item": "Focus Sash",
                    "ability": "Rough Skin",
                    "nature": "Jolly",
                    "points": {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32},
                    "moves": ["Earthquake", "Scale Shot", "Rock Tomb", "Swords Dance"],
                    "score": -2.9,
                },
            ],
        }
    },
}


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(CATALOG)


def test_builds_load_best_first(catalog: Catalog) -> None:
    builds = catalog.builds("garchomp")
    assert [b.item for b in builds] == ["Choice Scarf", "Focus Sash"]
    assert builds[0].points == (2, 32, 0, 0, 0, 32) and builds[0].score == -2.3
    assert catalog.species() == ["Garchomp"] and catalog.usage == {"garchomp": 100}
    assert builds[0].member().species == "Garchomp"


def test_mega_form_falls_back_to_the_base_species(catalog: Catalog) -> None:
    assert catalog.builds("Garchomp-Mega-Z")[0].item == "Choice Scarf"
    assert catalog.builds("Mewtwo") == []


def test_an_unknown_schema_is_rejected() -> None:
    with pytest.raises(ValueError):
        Catalog({**CATALOG, "schema": "other"})

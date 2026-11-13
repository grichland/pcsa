"""pcsa.players.no_effect: moves that surely do nothing, and only those."""

from __future__ import annotations

from poke_env.battle import Move

from pcsa.players.no_effect import why
from tests.players.test_disciplined import battle
from tests.players.test_neural_rules import TERRAIN


def _why(move: str, foe: str, turns=None, ability: str | None = None) -> str | None:
    b = battle("Azumarill", turns, foe=foe)
    target = b.opponent_active_pokemon
    if ability is not None:
        target.ability = ability
    return why(b, b.active_pokemon, target, Move(move, 9))


def test_immune_types_and_certain_levitate() -> None:
    assert _why("earthquake", "Dragonite")  # Flying
    assert _why("earthquake", "Flygon")  # Levitate is its only ability
    assert _why("earthquake", "Bronzong") is None  # may have Heatproof or Heavy Metal
    assert _why("earthquake", "Bronzong", ability="levitate")  # revealed
    assert _why("thunderwave", "Garchomp") and _why("thunderbolt", "Garchomp")
    assert _why("shadowball", "Snorlax") and _why("bodyslam", "Gengar")
    assert _why("earthquake", "Garchomp") is None and _why("playrough", "Garchomp") is None


def test_absorbing_and_blocking_abilities_only_when_certain() -> None:
    assert _why("thunderbolt", "Jolteon") is None  # Volt Absorb or Quick Feet
    assert _why("thunderbolt", "Jolteon", ability="voltabsorb")
    assert _why("surf", "Vaporeon", ability="waterabsorb")
    assert _why("flamethrower", "Heatran", ability="flashfire")
    assert _why("thunderwave", "Gholdengo")  # Good as Gold (its only ability)
    assert _why("shadowball", "Dragonite", ability="bulletproof")


def test_priority_into_psychic_terrain_and_first_turn_moves() -> None:
    assert _why("suckerpunch", "Indeedee", [TERRAIN])
    assert _why("aquajet", "Indeedee", [TERRAIN]) and _why("firstimpression", "Indeedee", [TERRAIN])
    assert _why("aquajet", "Dragonite", [TERRAIN]) is None  # not grounded
    assert _why("aquajet", "Indeedee") is None
    assert _why("protect", "Indeedee", [TERRAIN]) is None  # targets the user
    later = ["|move|p1a: Azumarill|Play Rough|p2a: Garchomp", "|-damage|p2a: Garchomp|80/100"]
    assert _why("fakeout", "Garchomp", [later])  # not the user's first turn out


def test_status_the_target_cannot_get() -> None:
    assert _why("willowisp", "Heatran") and _why("toxic", "Gholdengo") and _why("toxic", "Toxapex")
    assert _why("spore", "Venusaur") and _why("thunderwave", "Jolteon")
    assert _why("willowisp", "Garchomp") is None and _why("toxic", "Garchomp") is None
    misty = ["|-fieldstart|move: Misty Terrain|[from] ability: Misty Surge|[of] p2a: Garchomp"]
    assert _why("willowisp", "Garchomp", [misty]) and _why("willowisp", "Dragonite", [misty]) is None

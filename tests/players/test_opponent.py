"""Assumed opposing sets: top build under revealed facts, the Choice Scarf and power corrections, the fallback."""

from __future__ import annotations

from pcsa.battle.damage import build_stats, nature_percent
from pcsa.battle.game import species_entry
from pcsa.battle.state import replay
from pcsa.players.opponent import Tracker, assumed, default_catalog

PRELUDE = ["|player|p1|me|", "|player|p2|opponent|", "|gen|9", "|teamsize|p1|1", "|teamsize|p2|1", "|start"]
CHARIZARD_MOVES = ["flamethrower", "airslash", "roost", "willowisp"]


def stats(species: str, points: dict[str, int], nature: str) -> dict[str, int]:
    return build_stats(species_entry(species)["baseStats"], points, nature_percent(nature))


CHARIZARD = stats("charizard", {"hp": 2, "spa": 32, "spe": 32}, "Timid")


def battle(foe: str, events: list[str] | None = None, hp: int | None = None, moves: list[str] | None = None):
    """Our Timid Charizard (exact stats) against `foe`; `events` are turn 1's lines, then turn 2 begins."""
    full = CHARIZARD["hp"]
    lines = [*PRELUDE, f"|switch|p1a: Charizard|Charizard, L50|{full}/{full}"]
    lines += [f"|switch|p2a: {foe}|{foe}, L50|100/100", "|turn|1"]
    if events is not None:
        lines += [*events, "|turn|2"]
    moves = moves or CHARIZARD_MOVES
    member = {
        "ident": "p1: Charizard",
        "details": "Charizard, L50",
        "condition": f"{hp if hp is not None else full}/{full}",
        "active": True,
        "stats": {k: v for k, v in CHARIZARD.items() if k != "hp"},
        "moves": moves,
        "baseAbility": "blaze",
        "item": "heavydutyboots",
    }
    request = {
        "side": {"name": "me", "id": "p1", "pokemon": [member]},
        "rqid": 1,
        "active": [{"moves": [{"move": m, "id": m, "pp": 10, "maxpp": 10, "target": "normal"} for m in moves]}],
    }
    return replay(lines, request)


def test_top_build_overlay() -> None:
    b = battle("Tyranitar")
    chosen = assumed(b, b.opponent_active_pokemon, default_catalog())
    top = default_catalog().builds("tyranitar")[0]
    assert chosen.source == "revealed+top"
    assert (chosen.item, chosen.ability, chosen.nature) == (top.item, top.ability, top.nature)
    assert chosen.moves == ("stoneedge", "knockoff", "superpower", "icepunch")
    assert chosen.points["atk"] == top.points[1]


def test_revealed_moves_and_item_win() -> None:
    b = battle("Tyranitar", ["|move|p2a: Tyranitar|Crunch|p1a: Charizard", "|-damage|p1a: Charizard|100/155"])
    them = b.opponent_active_pokemon
    them._item = "leftovers"
    chosen = assumed(b, them, default_catalog())
    assert chosen.moves[0] == "crunch" and len(chosen.moves) == 4
    assert "crunch" not in chosen.moves[1:] and chosen.item == "leftovers"


def test_scarf_correction_when_it_moved_first() -> None:
    turn = [
        "|move|p2a: Tyranitar|Crunch|p1a: Charizard",
        "|-damage|p1a: Charizard|110/155",
        "|move|p1a: Charizard|Flamethrower|p2a: Tyranitar",
        "|-damage|p2a: Tyranitar|80/100",
    ]
    b = battle("Tyranitar", turn, hp=110)
    tracker = Tracker()
    tracker.observe(b)
    assert "tyranitar" in tracker.fast and "tyranitar" not in tracker.strong
    chosen = assumed(b, b.opponent_active_pokemon, default_catalog(), tracker)
    assert (chosen.item, chosen.source) == ("Choice Scarf", "scarf-corrected")
    assert assumed(b, b.opponent_active_pokemon, default_catalog(), tracker) is chosen  # cached
    tracker.observe(b)  # nothing new to read
    assert tracker.fast == {"tyranitar"}


def test_moving_second_and_a_plain_hit_change_nothing() -> None:
    turn = [
        "|move|p1a: Charizard|Flamethrower|p2a: Tyranitar",
        "|-damage|p2a: Tyranitar|80/100",
        "|move|p2a: Tyranitar|Crunch|p1a: Charizard",
        "|-damage|p1a: Charizard|110/155",
    ]
    tracker = Tracker()
    tracker.observe(battle("Tyranitar", turn, hp=110))
    assert not tracker.fast and not tracker.strong


def test_power_correction_after_an_impossible_hit() -> None:
    turn = [
        "|move|p1a: Charizard|Flamethrower|p2a: Tyranitar",
        "|-damage|p2a: Tyranitar|80/100",
        "|move|p2a: Tyranitar|Crunch|p1a: Charizard",
        "|-damage|p1a: Charizard|5/155",
    ]
    b = battle("Tyranitar", turn, hp=5)
    tracker = Tracker()
    tracker.observe(b)
    assert tracker.strong == {"tyranitar"}
    chosen = assumed(b, b.opponent_active_pokemon, default_catalog(), tracker)
    assert (chosen.item, chosen.source) == ("Life Orb", "power-corrected")


def test_mega_form_keeps_its_stone_and_ability() -> None:
    b = battle("Tyranitar", ["|-mega|p2a: Tyranitar|Tyranitar|Tyranitarite"])
    chosen = assumed(b, b.opponent_active_pokemon, default_catalog())
    assert chosen.item == "Tyranitarite" and chosen.ability == "sandstream"


def test_species_without_builds_falls_back_to_what_was_revealed() -> None:
    b = battle("Magikarp", ["|move|p2a: Magikarp|Tackle|p1a: Charizard", "|-damage|p1a: Charizard|150/155"])
    chosen = assumed(b, b.opponent_active_pokemon, default_catalog())
    assert chosen.source == "fallback" and chosen.moves == ("tackle",)
    assert chosen.item is None and chosen.nature == "Hardy" and set(chosen.points.values()) == {11}

"""Rows the engine check found unwritten: misses, multi-hit counts, and our Mega Evolution when its rows were
not captured. The texts are the OCR'd rows of the recorded battles named in each test."""

from pcsa.battle.log_rows import RawRow, split_category
from pcsa.battle.moves import observed_form
from pcsa.battle.translate import Translator
from tests.battle.conftest import feed, translator_for


def sprite(name: str, base: str) -> dict:
    return {"status": "matched", "name": name, "base_name": base, "score": 0.7}


def feed_with_sprites(t: Translator, rows: list[tuple[str, dict | None]]) -> None:
    for index, (text, found) in enumerate(rows):
        category, message = split_category(text)
        t.row(RawRow(f"T1#{index}", 1, index, text, category, message, sprite=found))


def test_an_avoided_attack_is_a_miss_by_the_latest_mover() -> None:
    # run 20261102-160125, battle 1, turn 4: Glaceon in snow avoided Gholdengo's Shadow Ball
    t = translator_for(("Gholdengo", "Gholdengo", 194), [("Glaceon", "Glaceon")])
    feed(
        t,
        [
            "Move: The opposing Glaceon used Icy Wind!",
            "HP Lost: Gholdengo’s HP: 165 → 130",
            "It’s not very effective..",
            "Gholdengo’s Speed fell!",
            "Move: Gholdengo used Shadow Ball!",
            "The opposing Glaceon avoided the attack!",
        ],
    )
    assert t.out.unmapped == []
    assert t.out.lines[-1] == "|-miss|p1a: Gholdengo|p2a: Glaceon"


def test_a_missed_leech_seed_seeds_no_one() -> None:
    # run 20261102-104355, battle 7, turn 15: the log has no "was seeded" row because the move missed
    t = translator_for(("Charizard", "Charizard", 153), [("Venusaur", "Venusaur")])
    feed(
        t,
        [
            "Move: Charizard used Dragon Claw!",
            "HP Lost: Venusaur’s HP: 72% → 40%",
            "Move: The opposing Venusaur used Leech Seed!",
            "Charizard avoided the attack!",
        ],
    )
    assert t.out.unmapped == []
    assert t.out.lines[-1] == "|-miss|p2a: Venusaur|p1a: Charizard"
    assert not t.battle.active_pokemon.effects


def test_a_miss_without_its_move_row_stays_informational() -> None:
    t = translator_for(("Charizard", "Charizard", 153), [("Venusaur", "Venusaur")], turn=2)
    before = t.out.informational
    feed(t, ["Charizard avoided the attack!"])
    assert t.out.unmapped == []
    assert t.out.informational == before + 1
    assert not any(line.startswith("|-miss|") for line in t.out.lines)


def test_a_hit_count_goes_to_the_latest_move_target() -> None:
    # run 20261102-220602, battle 20, turn 1: Parental Bond Double-Edge
    t = translator_for(("Charizard", "Charizard", 185), [("Kangaskhan", "Kangaskhan")])
    feed(
        t,
        [
            "Move: The opposing Kangaskhan used Double-Edge!",
            "HP Lost: Charizard’s HP: 185 → 89",
            "HP Lost: Charizard’s HP: 89 → 65",
            "The Pokémon was hit 2 times!",
        ],
    )
    assert t.out.unmapped == []
    assert t.out.lines[-1] == "|-hitcount|p1a: Charizard|2"


def test_our_mega_seen_only_in_a_row_sprite_is_written_before_our_move() -> None:
    # run 20261103-015542, battle 14, turn 1 as recorded: the climb stopped before the rows of our Mega
    # Evolution, so the first sign of it is Charizard's Mega Y icon beside its move
    t = translator_for(("Charizard", "Charizard", 185), [("Venusaur", "Venusaur")])
    feed_with_sprites(
        t,
        [
            ("The opposing Venusaur’s Venusaurite is reacting to Opponent’s Omni Ring!", None),
            (
                "Mega Evolution: The opposing Venusaur has Mega Evolved into Mega Venusaur!",
                sprite("Venusaur-Mega", "Venusaur"),
            ),
            ("Move: Charizard used Weather Ball!", sprite("Charizard-Mega-Y", "Charizard")),
            ("HP Lost: Venusaur’s HP: 100% → 42%", sprite("Venusaur-Mega", "Venusaur")),
        ],
    )
    lines = t.out.lines
    assert t.out.unmapped == []
    mega = lines.index("|-mega|p1a: Charizard|Charizard-Mega-Y|Charizardite Y")
    assert mega < lines.index("|move|p1a: Charizard|Weather Ball|p2a: Venusaur")
    assert lines[mega - 2 : mega] == [
        "|-item|p1a: Charizard|Charizardite Y",
        "|detailschange|p1a: Charizard|Charizard-Mega-Y, L50",
    ]
    assert sum(line.startswith("|-mega|p1a") for line in lines) == 1
    assert observed_form(t.battle, t.battle.active_pokemon) == "charizardmegay"


def test_our_mega_rows_when_read_write_one_mega() -> None:
    # the same turn re-read with the last capture of the climb kept
    t = translator_for(("Charizard", "Charizard", 185), [("Venusaur", "Venusaur")])
    feed_with_sprites(
        t,
        [
            ("Charizard’s Charizardite Y is reacting to Trainer’s Omni Ring!", None),
            (
                "Mega Evolution: Charizard has Mega Evolved into Mega Charizard!",
                sprite("Charizard-Mega-Y", "Charizard"),
            ),
            ("Ability: Charizard’s Drought", sprite("Charizard-Mega-Y", "Charizard")),
            ("The sunlight turned harsh!", None),
            ("Move: Charizard used Weather Ball!", sprite("Charizard-Mega-Y", "Charizard")),
        ],
    )
    lines = t.out.lines
    assert t.out.unmapped == []
    assert [line for line in lines if line.startswith("|-mega|p1a")] == [
        "|-mega|p1a: Charizard|Charizard-Mega-Y|Charizardite Y"
    ]
    assert "|-weather|SunnyDay" in lines
    assert lines.index("|-mega|p1a: Charizard|Charizard-Mega-Y|Charizardite Y") < lines.index(
        "|move|p1a: Charizard|Weather Ball|p2a: Venusaur"
    )

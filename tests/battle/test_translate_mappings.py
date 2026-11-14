"""Every live log row maps to protocol: the mechanics first seen unmapped in the 8-14 Nov live runs."""

from poke_env.battle.effect import Effect
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status

from pcsa.battle.log_rows import RawRow, split_category
from pcsa.battle.moves import observed_form
from tests.battle.conftest import feed, parse, translator_for


def _mapped(t) -> None:
    assert t.out.unmapped == []
    assert t.out.warnings == [], t.out.warnings


def test_red_card_row_ends_the_holders_item_once() -> None:
    t = translator_for(("Primarina", "Primarina", 187), [("Kleavor", "Kleavor"), ("Basculegion", "Basculegion")])
    feed(t, ["The opposing Kleavor held up its Red Card against Primarina!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-enditem|p2a: Kleavor|Red Card"
    assert t.battle.opponent_active_pokemon.item is None
    # after the "Held Item" row (which already ended it) the Red Card row writes nothing more
    t = translator_for(("Primarina", "Primarina", 187), [("Kleavor", "Kleavor")])
    feed(t, ["Held Item: Kleavor’s Red Card", "The opposing Kleavor held up its Red Card against Primarina!"])
    _mapped(t)
    assert [line for line in t.out.lines if "Red Card" in line] == ["|-enditem|p2a: Kleavor|Red Card"]


def test_eject_button_row_ends_the_item() -> None:
    t = translator_for(("Mimikyu", "Mimikyu", 131), [("Politoed", "Politoed")])
    feed(t, ["The opposing Politoed is switched out with the Eject Button!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-enditem|p2a: Politoed|Eject Button"


def test_revival_blessing_brings_a_fainted_member_back_at_half_hp() -> None:
    t = translator_for(("Primarina", "Primarina", 187), [("Gardevoir", "Gardevoir"), ("Pawmot", "Pawmot")])
    t.apply("|faint|p2a: Gardevoir")
    t.apply("|switch|p2a: Pawmot|Pawmot, L50|100/100")
    gardevoir = t.battle.opponent_team["p2: Gardevoir"]
    assert gardevoir.fainted
    feed(t, ["The opposing Gardevoir was revived and is ready to fight again!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-heal|p2: Gardevoir|50/100|[from] move: Revival Blessing"
    assert not gardevoir.fainted and gardevoir.current_hp == 50 and not gardevoir.active


def test_revival_blessing_of_ours_uses_half_our_max_hp() -> None:
    t = translator_for(("Garchomp", "Garchomp", 185), [("Pawmot", "Pawmot")])
    t.apply("|faint|p1a: Garchomp")
    feed(t, ["Garchomp was revived and is ready to fight again!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-heal|p1: Garchomp|92/185|[from] move: Revival Blessing"
    assert not t.battle.team["p1: Garchomp"].fainted


def test_disable_names_the_disabled_move(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    feed(t, ["Garchomp’s Earth Power was disabled!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-start|p1a: Garchomp|Disable|Earth Power"
    assert Effect.DISABLE in t.battle.active_pokemon.effects


def test_perish_song_start_and_counts(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    feed(
        t,
        [
            "All Pokémon that heard the song will faint in three turns!",
            "Garchomp’s perish count fell to 3!",
            "The opposing Snorlax’s perish count fell to 2!",
        ],
    )
    _mapped(t)
    assert t.out.lines[-3:] == [
        "|-fieldactivate|move: Perish Song",
        "|-start|p1a: Garchomp|perish3",
        "|-start|p2a: Snorlax|perish2",
    ]
    assert Effect.PERISH3 in t.battle.active_pokemon.effects
    assert Effect.PERISH2 in t.battle.opponent_active_pokemon.effects


def test_poltergeist_reveals_the_targets_item() -> None:
    t = translator_for(("Cinderace", "Cinderace", 155), [("Ceruledge", "Ceruledge")])
    feed(t, ["Cinderace is about to be attacked by its Choice Scarf!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-item|p1a: Cinderace|Choice Scarf"
    t = translator_for(("Ceruledge", "Ceruledge", 151), [("Snorlax", "Snorlax")])
    feed(t, ["The opposing Snorlax is about to be attacked by its Leftovers!"])
    assert t.battle.opponent_active_pokemon.item == "leftovers"


def test_heal_block_end(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    feed(t, ["Garchomp was prevented from healing!", "Garchomp is no longer prevented from healing!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-end|p1a: Garchomp|move: Heal Block"
    assert Effect.HEAL_BLOCK not in t.battle.active_pokemon.effects


def test_destiny_bond_with_and_without_the_final_exclamation() -> None:
    for text in [
        "The opposing Gengar is hoping to take its attacker down with it!",
        "The opposing Gengar is hoping to take its attacker down with it",
    ]:
        t = translator_for(("Kingambit", "Kingambit", 207), [("Gengar", "Gengar")])
        feed(t, [text])
        _mapped(t)
        assert t.out.lines[-1] == "|-singlemove|p2a: Gengar|Destiny Bond"
        assert Effect.DESTINY_BOND in t.battle.opponent_active_pokemon.effects


def test_curse_start_and_damage() -> None:
    t = translator_for(("Golisopod", "Golisopod", 182), [("Mimikyu", "Mimikyu")])
    feed(
        t,
        [
            "The opposing Mimikyu cut its own HP and put a curse on Golisopod!",
            "Golisopod is afflicted by the curse!",
            "HP Lost: Golisopod’s HP: 171 → 126",
        ],
    )
    _mapped(t)
    assert "|-start|p1a: Golisopod|Curse" in t.out.lines
    assert Effect.CURSE in t.battle.active_pokemon.effects
    assert t.battle.active_pokemon.current_hp == 126  # the curse row itself writes nothing; the HP row does


def test_lum_berry_defrost_cures_freeze_and_eats_the_berry() -> None:
    t = translator_for(("Metagross", "Metagross", 187), [("Rillaboom", "Rillaboom")])
    t.apply("|-status|p2a: Rillaboom|frz")
    feed(t, ["The opposing Rillaboom’s Lum Berry defrosted it!"])
    _mapped(t)
    assert t.out.lines[-2:] == ["|-enditem|p2a: Rillaboom|Lum Berry|[eat]", "|-curestatus|p2a: Rillaboom|frz"]
    assert t.battle.opponent_active_pokemon.status is None


def test_sticky_web_champions_wording_and_activation() -> None:
    t = translator_for(("Golisopod", "Golisopod", 182), [("Araquanid", "Araquanid")])
    feed(t, ["A sticky web has been laid out on the ground on your side!"])
    feed(t, ["A sticky web has been laid out on the ground on the opposing side!"])
    _mapped(t)
    assert t.out.lines[-2:] == ["|-sidestart|p1: me|move: Sticky Web", "|-sidestart|p2: opponent|move: Sticky Web"]
    assert t.battle.side_conditions and t.battle.opponent_side_conditions
    feed(t, ["Golisopod was caught in a sticky web!", "Golisopod’s Speed fell!"])
    _mapped(t)
    assert t.out.lines[-2:] == ["|-activate|p1a: Golisopod|move: Sticky Web", "|-unboost|p1a: Golisopod|spe|1"]


def test_rest_puts_the_user_to_sleep(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    feed(t, ["The opposing Snorlax slept and restored its HP!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-status|p2a: Snorlax|slp|[from] move: Rest"
    assert t.battle.opponent_active_pokemon.status == Status.SLP


def test_rows_that_restate_other_rows_write_nothing(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    before = list(t.out.lines)
    feed(
        t,
        [
            "Metagross’s stats were not lowered!",
            "The battlers shared their pain!",
            "Garchomp already has a substitute!",
            "Garchomp is already drowsy.",
            "The opposing Snorlax is already drowsy.",
            "Garchomp was hurt!",
            "Garchomp has no moves left that it can use!",
            "Until Battle Ends",
        ],
    )
    assert t.out.unmapped == []
    assert t.out.lines == before
    assert parse("Garchomp has no moves left that it can use!").data["what"] == "struggle"
    assert parse("Until Battle Ends").kind == "overlay"


def test_freed_from_a_binding_move_ends_it(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    feed(
        t,
        [
            "Garchomp has been afflicted with an infestation by the opposing Snorlax!",
            "Garchomp was freed from Infestation!",
        ],
    )
    _mapped(t)
    assert t.out.lines[-1] == "|-end|p1a: Garchomp|move: Infestation"
    assert Effect.INFESTATION not in t.battle.active_pokemon.effects


def test_zero_to_hero_changes_palafin_for_good() -> None:
    t = translator_for(("Gholdengo", "Gholdengo", 163), [("Palafin", "Palafin"), ("Snorlax", "Snorlax")])
    feed(t, ["The opposing Palafin underwent a heroic transformation!"])
    _mapped(t)
    assert t.out.lines[-2:] == [
        "|-activate|p2a: Palafin|ability: Zero to Hero",
        "|detailschange|p2a: Palafin|Palafin-Hero, L50",
    ]
    assert observed_form(t.battle, t.battle.opponent_active_pokemon) == "palafinhero"
    feed(t, ["opponent sent out Snorlax!", "opponent sent out Palafin!"])
    assert t.out.lines[-1].startswith("|switch|p2a: Palafin|Palafin-Hero, L50|")


def test_tailwind_champions_start_and_end(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    feed(t, ["A tailwind started blowing on the opposing side!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-sidestart|p2: opponent|move: Tailwind"
    assert t.battle.opponent_side_conditions
    feed(t, ["The opposing side’s tailwind petered out!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-sideend|p2: opponent|move: Tailwind"
    assert not t.battle.opponent_side_conditions
    feed(t, ["A tailwind started blowing on your side!", "Your side’s Tailwind petered out!"])
    assert t.out.lines[-2:] == ["|-sidestart|p1: me|move: Tailwind", "|-sideend|p1: me|move: Tailwind"]


def test_healing_wish_cures_status_and_heals_fully() -> None:
    t = translator_for(("Cinderace", "Cinderace", 155), [("Medicham", "Medicham")])
    t.apply("|-damage|p2a: Medicham|23/100")
    t.apply("|-status|p2a: Medicham|brn")
    feed(t, ["The healing wish came true for the opposing Medicham!", "HP Restored: Medicham’s HP: 23% → 100%"])
    _mapped(t)
    assert "|-curestatus|p2a: Medicham|brn|[msg]" in t.out.lines
    assert "|-heal|p2a: Medicham|100/100|[from] move: Healing Wish" in t.out.lines
    mon = t.battle.opponent_active_pokemon
    assert mon.status is None and mon.current_hp == 100


def test_haze_clears_every_boost(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    feed(t, ["Garchomp’s Attack rose sharply!", "The opposing Snorlax’s Defense fell!"])
    assert t.battle.active_pokemon.boosts["atk"] == 2 and t.battle.opponent_active_pokemon.boosts["def"] == -1
    feed(t, ["All stat changes were eliminated!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-clearallboost"
    assert not any(t.battle.active_pokemon.boosts.values())
    assert not any(t.battle.opponent_active_pokemon.boosts.values())


def test_flash_fire_and_fairy_aura_show_the_ability() -> None:
    t = translator_for(("Cinderace", "Cinderace", 155), [("Ceruledge", "Ceruledge")])
    feed(t, ["The power of the opposing Ceruledge’s Fire-type moves rose!"])
    _mapped(t)
    assert t.out.lines[-2:] == ["|-ability|p2a: Ceruledge|Flash Fire", "|-start|p2a: Ceruledge|ability: Flash Fire"]
    assert t.battle.opponent_active_pokemon.ability == "flashfire"
    assert Effect.FLASH_FIRE in t.battle.opponent_active_pokemon.effects
    t = translator_for(("Garchomp", "Garchomp", 185), [("Floette", "Floette-Mega")])
    feed(t, ["The opposing Floette is radiating a fairy aura!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-ability|p2a: Floette|Fairy Aura"


def test_electro_shot_charge_records_the_move() -> None:
    t = translator_for(("Primarina", "Primarina", 187), [("Archaludon", "Archaludon")])
    feed(t, ["Move: The opposing Archaludon used Electro Shot!", "The opposing Archaludon absorbed electricity!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-prepare|p2a: Archaludon|Electro Shot"
    assert t.battle.opponent_active_pokemon.preparing_move.id == "electroshot"


def test_charge_text_without_its_move_row_infers_the_move() -> None:
    t = translator_for(("Primarina", "Primarina", 187), [("Charizard", "Charizard")])
    feed(t, ["The opposing Charizard absorbed light!"])
    _mapped(t)
    assert t.out.lines[-2:] == ["|move|p2a: Charizard|Solar Beam||[still]", "|-prepare|p2a: Charizard|Solar Beam"]
    assert "solarbeam" in t.battle.opponent_active_pokemon.moves
    assert any(i.get("kind") == "charge_move" and i["move"] == "Solar Beam" for i in t.out.inferred)


def test_beak_blast_heats_up(garchomp_vs_snorlax) -> None:
    t = translator_for(("Golisopod", "Golisopod", 182), [("Toucannon", "Toucannon")])
    feed(t, ["The opposing Toucannon started heating up its beak!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-singleturn|p2a: Toucannon|move: Beak Blast"


def test_burn_up_drops_the_fire_type() -> None:
    t = translator_for(("Primarina", "Primarina", 187), [("Skeledirge", "Skeledirge")])
    feed(t, ["The opposing Skeledirge burned itself out!"])
    _mapped(t)
    assert t.out.lines[-1] == "|-start|p2a: Skeledirge|typechange|Ghost|[from] move: Burn Up"
    assert t.battle.opponent_active_pokemon.types == [PokemonType.GHOST]


def test_harvest_regains_the_berry_and_shows_the_ability() -> None:
    t = translator_for(("Archaludon", "Archaludon", 197), [("Arboliva", "Arboliva")])
    feed(t, ["The opposing Arboliva harvested a Sitrus Berry!"])
    _mapped(t)
    assert t.out.lines[-2:] == [
        "|-ability|p2a: Arboliva|Harvest",
        "|-item|p2a: Arboliva|Sitrus Berry|[from] ability: Harvest",
    ]
    mon = t.battle.opponent_active_pokemon
    assert (mon.item, mon.ability) == ("sitrusberry", "harvest")
    assert t.out.revealed_abilities["p2: Arboliva"] == "Harvest"


def test_an_item_written_with_a_quantity_resolves() -> None:
    t = translator_for(("Kingambit", "Kingambit", 207), [("Indeedee", "Indeedee")])
    feed(t, ["The opposing Indeedee obtained a serving of Leftovers."])
    _mapped(t)
    assert t.out.lines[-1] == "|-item|p2a: Indeedee|Leftovers"


def test_withdraw_rows_without_the_final_exclamation() -> None:
    for text, trainer in [("Duckwithdrew Primarina", "Duck"), ("withdrew Pelipper", None)]:
        event = parse(text)
        assert (event.kind, event.side, event.data["trainer"]) == ("withdraw", "p2", trainer)


def test_ocr_damaged_boost_rows_are_repaired() -> None:
    for text, name, stats in [
        ("The onnosing Volcarona’s Sn. Atk Sn. Def, and Sneed rosel", "Volcarona", ["spa", "spd", "spe"]),
        ("I ne opposing voicarona’s Sp. Atk, Sp. Det, ana Speea rose!", "voicarona", ["spa", "spd", "spe"]),
        ("The onnosina Felektross’s Attack Defense, and accuracy rosel", "Felektross", ["atk", "def", "accuracy"]),
    ]:
        event = parse(text)
        assert (event.kind, event.side, event.name) == ("boost", "p2", name)
        assert (event.data["stats"], event.data["amount"], event.data["sign"]) == (stats, 1, 1)
    t = translator_for(("Garchomp", "Garchomp", 185), [("Volcarona", "Volcarona")])
    feed(t, ["I ne opposing voicarona’s Sp. Atk, Sp. Det, ana Speea rose!"])
    _mapped(t)
    assert t.out.lines[-3:] == [
        "|-boost|p2a: Volcarona|spa|1",
        "|-boost|p2a: Volcarona|spd|1",
        "|-boost|p2a: Volcarona|spe|1",
    ]


def test_unreadable_rows_stay_unmapped() -> None:
    for text in ["Theoin eTiE S. Dan Spee!", "Gho", "The opposing Volcarona’s Sp. Atk, Spinach, and Speed rose!"]:
        assert parse(text).kind == "unknown", text


def test_a_sprite_naming_no_member_of_the_rows_side_is_ignored() -> None:
    # 12 Nov: "The opposing Dragapult had its energy drained!" carried our drainer's icon (Archaludon)
    t = translator_for(("Archaludon", "Archaludon", 197), [("Dragapult", "Dragapult")])
    text = "The opposing Dragapult had its energy drained!"
    category, message = split_category(text)
    sprite = {"status": "matched", "name": "Archaludon", "base_name": "Archaludon"}
    t.row(RawRow("T1#0", 1, 0, text, category, message, sprite=sprite))
    assert t.out.unmapped == []
    assert any("ignored sprite 'Archaludon', not on p2's roster" in w for w in t.out.warnings)

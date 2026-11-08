"""Rows -> protocol -> poke-env state, on the mechanics live runs exercised."""

from poke_env.battle.effect import Effect

from pcsa.battle.log_rows import RawRow, split_category
from pcsa.battle.state import LOGGER, USERNAME, ChampionsBattle, build
from pcsa.battle.teams import OwnMember, Team
from pcsa.battle.translate import Member, Translator
from tests.battle.conftest import PRELUDE, feed, translator_for


def test_berry_cure_uses_the_berry_and_cures_the_status(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    t.apply("|-status|p2a: Snorlax|slp")
    feed(t, ["The opposing Snorlax’s Lum Berry woke it up!"])
    assert t.out.unmapped == []
    assert t.out.lines[-2:] == ["|-enditem|p2a: Snorlax|Lum Berry|[eat]", "|-curestatus|p2a: Snorlax|slp"]
    assert t.battle.opponent_active_pokemon.status is None


def test_charge_turn_records_the_move_being_charged(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    feed(t, ["Move: The opposing Snorlax used Bounce!", "The opposing Snorlax sprang up!"])
    assert t.out.unmapped == []
    assert t.out.lines[-1] == "|-prepare|p2a: Snorlax|Bounce"
    assert t.battle.opponent_active_pokemon.preparing_move.id == "bounce"


def test_shed_tail_decoy_passes_to_the_replacement() -> None:
    t = translator_for(("Garchomp", "Garchomp", 185), [("Orthworm", "Orthworm"), ("Snorlax", "Snorlax")])
    feed(
        t,
        [
            "The opposing Orthworm shed its tail to create a decoy!",
            "The opposing Orthworm went back to opponent!",
            "Opponent sent out Snorlax!",
        ],
    )
    assert t.out.unmapped == []
    assert t.out.lines[-1] == "|-start|p2a: Snorlax|Substitute"


def test_held_item_row_in_a_mirror_takes_the_side_of_the_row_after_it() -> None:
    for after, side in [
        ("HP Lost: Gholdengo’s HP: 100% → 90%", "p2"),
        ("Gholdengo floats in the air with its Air Balloon!", "p1"),
    ]:
        t = translator_for(("Gholdengo", "Gholdengo", 164), [("Gholdengo", "Gholdengo")])
        t.turns({1: ["Move: Gholdengo used Protect!", "Held Item: Gholdengo’s Air Balloon", after]}, 2)
        assert t.out.unmapped == []
        assert f"|-item|{side}a: Gholdengo|Air Balloon" in t.out.lines


def test_rough_skin_of_a_pokemon_just_knocked_out_in_a_mirror_is_ours() -> None:
    t = translator_for(("Garchomp", "Garchomp", 185), [("Mimikyu", "Mimikyu"), ("Garchomp", "Garchomp")])
    t.apply("|-damage|p1a: Garchomp|0 fnt")
    category, message = split_category("Ability: Garchomp’s Rough Skin")
    t.row(
        RawRow(
            "T1#0",
            1,
            0,
            "Ability: Garchomp’s Rough Skin",
            category,
            message,
            sprite={"status": "matched", "name": "Garchomp", "base_name": "Garchomp"},
        )
    )
    assert t.out.unmapped == []
    assert "|-ability|p1a: Garchomp|Rough Skin" in t.out.lines


def test_send_out_of_a_pokemon_the_preview_missed_adds_it(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    t.turns({1: ["opponent sent out Weavile!", "Move: The opposing Weavile used Knock Off!"]}, 2)
    assert t.out.unmapped == []
    assert "|switch|p2a: Weavile|Weavile, L50|100/100" in t.out.lines
    assert t.battle.opponent_active_pokemon.species == "weavile"


def _team(*names: str) -> Team:
    return Team(
        None, tuple(OwnMember(i, n, n, None, None, True, (None,) * 4, None, None, None) for i, n in enumerate(names))
    )


def test_a_pokemon_the_preview_missed_takes_the_least_confident_unseen_slot() -> None:
    from types import SimpleNamespace as N

    read = [
        ("Pikachu", 0.18),
        ("Sharpedo", 0.08),
        ("Basculegion", 0.18),
        ("Kleavor", 0.23),
        ("Salamence", 0.13),
        ("Zoroark-Hisui", 0.21),
    ]
    preview = [N(top=[(n, s)], species=[(n.split("-")[0], s)]) for n, s in read]
    rows = {1: ["opponent sent out Pikachu!", "Go! Raichu!", "opponent sent out Floette!"]}
    _, out = build(
        "t", _team("Raichu", "Primarina", "Gholdengo"), preview, picks=[0, 1, 2], rows_by_turn=rows, decision_turn=2
    )
    assert [line.split("|")[3] for line in out.lines if line.startswith("|poke|p2|")] == [
        "Pikachu, L50", "Floette-Eternal, L50", "Basculegion, L50", "Kleavor, L50", "Salamence, L50", "Zoroark-Hisui, L50",
    ]  # fmt: skip
    assert "preview corrected from the log: slot 2 is Floette" in out.warnings


def test_ability_row_in_a_mirror_takes_the_side_of_the_row_after_it() -> None:
    t = translator_for(("Mimikyu", "Mimikyu", 157), [("Mimikyu", "Mimikyu")])
    t.turns({1: ["Ability: Mimikyu’s Disguise", "HP Lost: Mimikyu’s HP: 100% → 88%"]}, 2)
    assert t.out.unmapped == []
    assert "|-ability|p2a: Mimikyu|Disguise" in t.out.lines


def test_ability_misread_resolves_against_the_species(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    t.turns({1: ["Ability: Snorlax’s Thick Fal"]}, 2)
    assert t.out.revealed_abilities.get("p2: Snorlax") == "Thick Fat"


def test_changed_ability_outside_the_species_pool_is_accepted() -> None:
    t = translator_for(("Garchomp", "Garchomp", 183), [("Alakazam", "Alakazam")])
    t.turns({1: ["Ability Changed: Alakazam: Trace → Rough Skin"]}, 2)
    assert t.out.unmapped == []
    assert t.out.revealed_abilities.get("p2: Alakazam") == "Rough Skin"


def test_recoil_row_with_the_bang_read_as_an_l_is_mapped(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    feed(t, ["Move: The opposing Snorlax used Double-Edge!", "The opposing Snorlax was damaged by the recoill"])
    assert t.out.unmapped == []


def test_charge_starts_on_its_row_and_ends_after_an_electric_move(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    t.turns({1: ["The opposing Snorlax began charging power!"]}, 2)
    assert t.out.unmapped == []
    assert Effect.CHARGE in t.battle.opponent_active_pokemon.effects
    feed(t, ["Move: The opposing Snorlax used Body Slam!"])
    assert Effect.CHARGE in t.battle.opponent_active_pokemon.effects
    feed(t, ["Move: The opposing Snorlax used Thunderbolt!"])
    assert Effect.CHARGE not in t.battle.opponent_active_pokemon.effects


def test_trick_gives_each_side_its_new_item(garchomp_vs_snorlax) -> None:
    t = garchomp_vs_snorlax
    feed(
        t,
        [
            "Garchomp switched items with its target!",
            "Garchomp obtained a Choice Scarf.",
            "The opposing Snorlax obtained a Leftovers.",
        ],
    )
    assert t.out.unmapped == []
    assert t.battle.active_pokemon.item == "choicescarf"
    assert t.battle.opponent_active_pokemon.item == "leftovers"


def test_confusion_activation_repeats_damage_and_end_have_distinct_events() -> None:
    t = translator_for(("Garchomp", "Garchomp", 185), [("Volcarona", "Volcarona")])
    feed(
        t,
        [
            "Garchomp became confused due to fatigue!",
            "Garchomp is confused!",
            "It hurt itself in its confusion!",
            "Garchomp snapped out of its confusion!",
        ],
    )
    assert not t.out.unmapped
    assert sum(line.startswith("|-start|") for line in t.out.lines) == 1
    assert "|cant|p1a: Garchomp|confusion" in t.out.lines
    assert not t.battle.active_pokemon.effects


def test_unassigned_confusion_damage_remains_unknown(garchomp_vs_snorlax) -> None:
    feed(garchomp_vs_snorlax, ["It hurt itself in its confusion!"])
    assert len(garchomp_vs_snorlax.out.unmapped) == 1


def test_seed_and_trick_room_change_the_projected_state() -> None:
    t = translator_for(("Garchomp", "Garchomp", 185), [("Volcarona", "Volcarona")])
    feed(
        t,
        [
            "Garchomp was seeded!",
            "Garchomp’s health is sapped by Leech Seed!",
            "The opposing Volcarona twisted the dimensions!",
        ],
    )
    assert not t.out.unmapped
    assert "LEECH_SEED" in {effect.name for effect in t.battle.active_pokemon.effects}
    assert "TRICK_ROOM" in {field.name for field in t.battle.fields}


def test_white_herb_restores_only_negative_boosts() -> None:
    t = translator_for(("Corviknight", "Corviknight", 205), [("Sneasler", "Sneasler")], turn=3)
    feed(
        t,
        [
            "The opposing Sneasler’s Defense and Sp. Def fell!",
            "The opposing Sneasler returned its stats to normal using its White Herb!",
        ],
    )
    assert not t.out.unmapped
    assert "|-clearnegativeboost|p2a: Sneasler" in t.out.lines
    assert t.battle.opponent_active_pokemon.boosts["def"] == 0


# Illusion: Hisuian Zoroark entered as Garchomp and used Bitter Malice (live run 20261030-132239).
ILLUSION_ROWS = {
    1: [
        "U-h sent out Garchomp!",
        "Go! Raichu!",
        "Move: The opposing Garchomp used Bitter Malice!",
        "HP Lost: Raichu’s HP: 135 → 60",
        "Held Item: Garchomp’s Life Orb",
        "HP Lost: Garchomp’s HP: 100% → 90%",
    ],
    2: [
        "Move: Raichu used Focus Blast!",
        "HP Lost: Garchomp’s HP: 90% → 0%",
        "The opposing Zoroark’s illusion wore off!",
        "Fainted: The opposing Zoroark fainted!",
        "U-h sent out Garchomp!",
    ],
}


def _illusion_translate(rows: dict) -> Translator:
    disguises: dict = {}
    for _ in range(4):
        battle = ChampionsBattle("illusion", USERNAME, LOGGER, 9)
        battle.player_role = "p1"
        t = Translator(
            battle,
            [Member("Raichu", "Raichu", 135)],
            [Member("Garchomp", "Garchomp"), Member("Zoroark", "Zoroark-Hisui"), Member("Sneasler", "Sneasler")],
            disguises=disguises,
        )
        for line in [
            *PRELUDE[:3],
            "|clearpoke",
            "|poke|p2|Garchomp, L50|",
            "|poke|p2|Zoroark-Hisui, L50|",
            "|poke|p2|Sneasler, L50|",
            "|teampreview|3",
            "|teamsize|p1|1",
            "|teamsize|p2|3",
            "|start",
        ]:
            t.apply(line)
        t.turns(rows, max(rows) + 1)
        if t.out.illusions.items() <= disguises.items():
            return t
        disguises = {**disguises, **t.out.illusions}
    raise AssertionError("illusion did not converge")


def test_move_the_shown_species_cannot_learn_reveals_the_illusion_user() -> None:
    t = _illusion_translate({1: ILLUSION_ROWS[1]})
    assert t.out.unmapped == []
    assert "|switch|p2a: Zoroark|Zoroark-Hisui, L50|100/100" in t.out.lines
    assert "|-item|p2a: Zoroark|Life Orb" in t.out.lines
    assert "p2: Garchomp" not in t.battle.opponent_team
    assert t.battle.opponent_active_pokemon.species == "zoroarkhisui"
    assert t.out.revealed_abilities["p2: Zoroark"] == "Illusion"


def test_real_pokemon_of_the_disguised_species_enters_fresh_after_the_illusion_ends() -> None:
    t = _illusion_translate(ILLUSION_ROWS)
    assert t.out.unmapped == []
    assert "|faint|p2a: Zoroark" in t.out.lines
    garchomp = t.battle.opponent_team["p2: Garchomp"]
    assert garchomp.active and garchomp.current_hp_fraction == 1.0


def test_illusion_end_row_alone_reveals_the_illusion_user() -> None:
    rows = {
        1: [
            "U-h sent out Garchomp!",
            "Go! Raichu!",
            "Move: Raichu used Focus Blast!",
            "HP Lost: Garchomp’s HP: 100% → 40%",
            "The opposing Zoroark’s illusion wore off!",
        ]
    }
    t = _illusion_translate(rows)
    assert t.out.unmapped == []
    assert t.battle.opponent_active_pokemon.species == "zoroarkhisui"


def test_learnable_moves_raise_no_suspicion() -> None:
    t = _illusion_translate(
        {1: ["U-h sent out Garchomp!", "Go! Raichu!", "Move: The opposing Garchomp used Earthquake!"]}
    )
    assert t.out.illusions == {}
    assert t.battle.opponent_active_pokemon.species == "garchomp"


def test_preview_only_build_before_any_switch_does_not_fail() -> None:
    """Before the first switch poke-env keys the opponent team by species id, not "p2: Name"."""
    from types import SimpleNamespace as N

    preview = [N(top=[(n, 0.2)], species=[(n.split("-")[0], 0.2)]) for n in ("Tauros-Paldea-Blaze", "Clefable")]
    battle, out = build("t", _team("Raichu", "Primarina", "Gholdengo"), preview)
    assert [m.species for m in battle.teampreview_opponent_team] == ["taurospaldeablaze", "clefable"]
    assert not out.warnings
    battle, _ = build("t2", _team("Raichu", "Primarina", "Gholdengo"), preview, picks=[0, 1, 2])
    assert battle.turn == 0


def test_team_record_round_trips_through_json() -> None:
    import json

    from pcsa.battle.teams import team_from_record

    team = _team("Raichu", "Primarina", "Gholdengo")
    again = team_from_record(json.loads(json.dumps(team.record())))
    assert again == team
    legacy = {
        "name": "T",
        "members": [{k: v for k, v in m.items() if k != "item_known"} for m in team.record()["members"]],
    }
    assert all(m.item_known for m in team_from_record(legacy).members)

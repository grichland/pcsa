"""game, moves and stats give the recorded reference answers wherever the data is legal; the known differences are listed."""

from __future__ import annotations

import pytest

from pcsa.battle import game, moves, stats
from pcsa.battle.ids import to_id
from tests.battle.data_files import load, plain

# Legal move names pcsa called incomplete only because an illegal move starts with them.
NOW_COMPLETE = {
    "Clangorous Soul",
    "Feint",
    "Inferno",
    "Roar",
}  # Soulblaze, Feint Attack, Inferno Overdrive, Roar of Time
NOW_KNOWN_ITEMS = {
    "Shed Shell",
    "shed shell",
    "Shed Shel",
    "Shed Shells",
    "hed Shell",
}  # legal, missing from the reference vocab


@pytest.fixture(scope="module")
def ref() -> dict:
    return load("game_parity.json.gz")


def test_species_functions(ref) -> None:
    sp = ref["species"]
    assert [k for k, v in sp["species_entry"].items() if game.species_entry(k) != v] == []
    assert [k for k, v in sp["base_name"].items() if game.base_name(k) != v] == []
    assert [k for k, v in sp["is_mega"].items() if game.is_mega(k) != v] == []
    assert [k for k, v in sp["mega_ability"].items() if game.mega_ability(game.species_entry(k)) != v] == []
    assert [c for c in sp["mega_form_for_stone"] if game.mega_form_for_stone(c[0], c[1]) != c[2]] == []


def test_megas_are_pcsa_megas_less_the_illegal_ones(ref) -> None:
    by_base, by_stone = game.megas()
    legal = game.cg_table("species")
    want_base = {b: [k for k in ks if k in legal] for b, ks in ref["species"]["megas_by_base"].items()}
    assert by_base == {b: ks for b, ks in want_base.items() if ks}
    assert by_stone == {s: k for s, k in ref["species"]["megas_by_stone"].items() if k in legal}


def test_abilities(ref) -> None:
    ab = ref["abilities"]
    assert [k for k, v in ab["canonical_ability_any"].items() if game.canonical_ability(k) != v] == []
    assert [c for c in ab["canonical_ability_species"] if game.canonical_ability(c[1], c[0]) != c[2]] == []
    assert [c for c in ab["mega_form_for_ability"] if plain(game.mega_form_for_ability(c[0], c[1])) != c[2]] == []


def test_items_where_pcsa_answered_with_a_legal_item_or_nothing(ref) -> None:
    legal = game.item_ids()
    cases = ref["items"]["canonical_item"]
    same = {k: v for k, v in cases.items() if v is None or to_id(v) in legal}
    assert {k for k, v in same.items() if game.canonical_item(k) != v} == NOW_KNOWN_ITEMS
    illegal = {k: v for k, v in cases.items() if k not in same}
    assert [k for k in illegal if game.canonical_item(k) not in (None, *legal)] == []


def test_moves(ref) -> None:
    mv = ref["moves"]
    assert [k for k, v in mv["base_data"].items() if moves.base_data(k) != v] == []
    assert [k for k, v in mv["max_pp"].items() if moves.max_pp(k) != v] == []
    names = moves.move_names()
    changed = {k for k, v in mv["complete_move_name"].items() if k in names and moves.complete_move_name(k) != v}
    assert changed == NOW_COMPLETE
    assert [k for k in mv["complete_move_name"] if k not in names and moves.complete_move_name(k)] == []


def test_unknown_move_gets_the_no_data_answer() -> None:
    assert moves.base_data("notamove") == {"type": "", "category": "", "power": 1, "accuracy": 101, "pp": 0}


def test_stats(ref) -> None:
    st = ref["stats"]
    assert [c for c in st["nature_name"] if stats.nature_name(c[0]) != c[1]] == []
    assert [c for c in st["solve_points"] if plain(stats.solve_points(c[0], c[1])) != c[2]] == []
    assert [c for c in st["stat_value"] if stats.stat_value(c[0], c[1], c[2]) != c[3]] == []
    assert [c for c in st["hp_value"] if stats.hp_value(c[0], c[1]) != c[2]] == []


def test_learnsets(ref) -> None:
    assert [c for c in ref["learnsets_sample"] if game.can_learn(c[0], c[1]) != c[2]] == []

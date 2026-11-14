"""Battle-log sentences -> events: rows that reached decisions in live runs, and what stays unknown."""

from unittest.mock import patch

import pytest

from pcsa.battle import log_rows
from pcsa.battle.log_rows import TEMPLATE_EVENTS
from tests.battle.conftest import parse

SAVED_ROWS = [
    ("Corviknight shook off the taunt!", "volatile_end", "p1", {"effect": "move: Taunt"}),
    ("Corviknight’s HP is full!", "trailing", "p1", {"what": "heal_failed"}),
    ("Garchomp is protected by the Psychic Terrain!", "trailing", "p1", {"what": "terrain_block"}),
    ("Garchomp obtained a Choice Scarf.", "item_reveal", "p1", {"item": "Choice Scarf"}),
    ("The opposing Indeedee obtained a Focus Band.", "item_reveal", "p2", {"item": "Focus Band"}),
    ("The opposing Indeedee switched items with its target!", "trailing", "p2", {"what": "item_swap"}),
    ("Garchomp was hurt by the spikes!", "cause", "p1", {"cause": "Spikes"}),
    ("Pointed stones dug into the opposing Azumarill", "cause", "p2", {"cause": "Stealth Rock"}),
    ("Spikes were scattered on the ground all around your side!", "hazard", "p1", {"condition": "Spikes"}),
    (
        "Toxic spikes were scattered on the ground all around your side!",
        "hazard",
        "p1",
        {"condition": "move: Toxic Spikes"},
    ),
    ("It’s a one-hit KO!", "trailing", None, {"what": "ohko"}),
    ("Mimikyu couldn’t move because it’s paralyzed!", "cant", "p1", {"reason": "par"}),
    ("The opposing Volcarona couldn’t move because it’s paralyzed!", "cant", "p2", {"reason": "par"}),
    ("Mimikyu is immobilized by love!", "cant", "p1", {"reason": "attract"}),
    ("Mimikyu is in love with Stunfisk!", "volatile_active", "p1", {"effect": "Attract"}),
    ("The opposing Aggron is overflowing with space power!", "prepare", "p2", {}),
    ("The opposing Banette frisked Raichu and found its Raichunite Y!", "ability", "p2", {"ability": "Frisk"}),
    ("The opposing Basculegion went back to", "trailing", "p2", {"what": "recall"}),
    ("The opposing Froslass took its attacker down with it!", "trailing", "p2", {"what": "destiny_bond"}),
    ("The opposing Garchomp braced itself!", "trailing", "p2", {"what": "endure"}),
    ("The opposing Kingambit gained strength from the fallen!", "ability", "p2", {"ability": "Supreme Overlord"}),
    ("The opposing Snorlax’s Lum Berry woke it up!", "cure", "p2", {"status": "slp", "item": "Lum Berry"}),
    ("The opposing Sylveon’s wish came true!", "cause", "p2", {"cause": "Wish"}),
    ("The opposing Cinderace sprang up!", "prepare", "p2", {}),
    (
        "The opposing Orthworm shed its tail to create a decoy!",
        "volatile",
        "p2",
        {"effect": "Substitute", "passes": True},
    ),
    (
        "The opposing Garchomps Garchompite Z is reacting to7s Omni Ring!",
        "mega_reacting",
        "p2",
        {"stone": "Garchompite Z"},
    ),
    ("Go!Gholdengo!", "send_out", "p1", {}),
    ("HP Lost: Garchomp’s HP: 180 → 95", "hp", "p1", {"before": 180, "after": 95, "unit": "hp"}),
    ("HP Lost: Garchomp’s HP: 100% → 40%", "hp", "p2", {"before": 100, "after": 40, "unit": "pct"}),
    (
        "The opposing Sneasler’s Defense and Sp. Def fell!",
        "boost",
        "p2",
        {"stats": ["def", "spd"], "amount": 1, "sign": -1},
    ),
    ("Garchomp’s Attack rose sharply!", "boost", "p1", {"stats": ["atk"], "amount": 2, "sign": 1}),
]
UNSUPPORTED_ROWS = [
    ("The opposing Inteleon is getting pumped!", "Focus Energy"),
]


@pytest.mark.parametrize("text,kind,side,data", SAVED_ROWS, ids=[r[0][:40] for r in SAVED_ROWS])
def test_saved_rows_parse_to_their_events(text, kind, side, data) -> None:
    event = parse(text)
    assert (event.kind, event.side) == (kind, side)
    assert {k: event.data.get(k) for k in data} == data


@pytest.mark.parametrize("text,name", UNSUPPORTED_ROWS)
def test_recognized_mechanics_without_engine_support_stay_unknown(text, name) -> None:
    event = parse(text)
    assert event.kind == "unknown" and name in event.data["reason"]


@pytest.mark.parametrize(
    "text", ["Theoin eTiE S. Dan Spee!", "Gho", "MoveC: Yefan Thanks! Good night!", "The opposing Garchomp"]
)
def test_unrecognized_rows_stay_unknown(text) -> None:
    assert parse(text).kind == "unknown"


def test_shared_sentences_with_different_effects_stay_unknown() -> None:
    changed = {**TEMPLATE_EVENTS, ("switcheroo", "activate"): ("trailing", {"what": "other"})}
    with patch.object(log_rows, "TEMPLATE_EVENTS", changed):
        event = parse("The opposing Indeedee switched items with its target!")
    assert (event.kind, event.data.get("reason")) == ("unknown", "ambiguous_template")


def test_complete_move_name_survives_a_merged_exclamation() -> None:
    for text, move in [
        ("Move: The opposing Gengar used Shadow Ball", "Shadow Ball"),
        ("Move: The opposing Primarina used Iron Tail", "Iron Tail"),
    ]:
        event = parse(text)
        assert (event.kind, event.side, event.data["move"]) == ("move", "p2", move)


@pytest.mark.parametrize(
    "text",
    [
        "Move: The opposing Raichu used Thunder",
        "Move: The opposing Gengar used Shadow Bal",
        "The opposing Gengar used Shadow Ball",
    ],
)
def test_move_prefix_without_exclamation_stays_unknown(text) -> None:
    assert parse(text).kind == "unknown"


def test_missing_final_exclamation_does_not_hide_a_mega() -> None:
    event = parse("Mega Evolution: The opposing Feraligatr has Mega Evolved into Mega Feraligatr")
    assert event.kind == "mega" and event.data["mega"] == "Mega Feraligatr"


def test_priority_label_is_kept() -> None:
    event = parse("Move +1 Priority: Garchomp used Protect!")
    assert event.kind == "move" and event.data["priority"] == "+1"

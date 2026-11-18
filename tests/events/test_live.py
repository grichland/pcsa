"""Rows -> LIVE.md 0.5 events: one event per row, nameless rows stay nameless, every event validates."""

from __future__ import annotations

import pytest

from pcsa.events.live import convert_rows
from pcsa.events.live_check import problems
from pcsa.events.live_ids import Teams

TEAMS = Teams(own=("Mamoswine", "Garchomp", "Golisopod"), opp_preview=("Glimmora", "Lucario", "Garchomp"))


def one(text: str, teams: Teams = TEAMS, side: str | None = None) -> dict:
    events, _ = convert_rows([{"turn": 1, "index": 0, "text": text, "side": side}], teams)
    assert events[0] == {"k": "turn", "n": 1}
    assert len(events) == 2
    return events[1]


@pytest.mark.parametrize(
    ("text", "event"),
    [
        ("Go! Mamoswine!", {"k": "enter", "side": "own", "mon": "mamoswine"}),
        ("U- sent out Glimmora!", {"k": "enter", "side": "opp", "mon": "glimmora"}),
        ("Mamoswine, come back!", {"k": "withdraw", "side": "own", "mon": "mamoswine"}),
        ("The opposing Lucario went back to U-!", {"k": "withdraw", "side": "opp", "mon": "lucario", "pivot": True}),
        ("Move +1 Priority: Mamoswine used Ice Shard!",
         {"k": "move", "side": "own", "mon": "mamoswine", "move": "iceshard", "priority": 1}),
        ("HP Lost: Glimmora’s HP: 100% → 1%",
         {"k": "hp", "side": "opp", "mon": "glimmora", "dir": "lost", "before_pct": 100, "after_pct": 1}),
        ("HP Lost: Mamoswine’s HP: 195 → 154",
         {"k": "hp", "side": "own", "mon": "mamoswine", "dir": "lost", "before": 195, "after": 154}),
        ("But it failed!", {"k": "fail"}),
        ("It’s extremely effective!", {"k": "result", "how": "extreme"}),
        ("The Pokémon was hit 3 times!", {"k": "hits", "n": 3}),
        ("The opposing Lucario protected itself!", {"k": "protect", "side": "opp", "mon": "lucario"}),
        ("Pointed stones float in the air on your side!", {"k": "side", "side": "own", "effect": "stealthrock",
                                                          "what": "start"}),
        ("The sandstorm subsided.", {"k": "weather", "weather": "sandstorm", "what": "end"}),
        ("The opposing Lucario had its HP restored.", {"k": "cause", "effect": "hp_restored", "side": "opp",
                                                       "mon": "lucario"}),
        ("Until Battle Ends", {"k": "unread", "text": "Until Battle Ends"}),
    ],
)
def test_rows(text: str, event: dict) -> None:
    assert one(text) == event


def test_sideless_rows_take_the_pill_side() -> None:
    for side in ("own", "opp"):  # Garchomp is on both teams: only the pill can tell
        event = one("Held Item: Garchomp’s Focus Sash", side=side)
        assert event == {"k": "item", "side": side, "mon": "garchomp", "item": "focussash", "what": "activated"}
    assert one("Ability: Glimmora’s Toxic Debris", side="opp")["side"] == "opp"


def test_unread_pill_leaves_the_side_out() -> None:
    event = one("Held Item: Glimmora’s Focus Sash")
    assert "side" not in event and event["mon"] == "glimmora"


def test_mega_form_left_out_when_the_row_fits_several() -> None:
    assert "form" not in one("Mega Evolution: Garchomp has Mega Evolved into Mega Garchomp!")
    assert one("Mega Evolution: Golisopod has Mega Evolved into Mega Golisopod!")["form"] == "golisopodmega"


def test_events_validate() -> None:
    for text in ("Go! Mamoswine!", "But it failed!", "Move: The opposing Glimmora used Stealth Rock!"):
        assert problems(one(text)) == []


def test_boost_always_sends_a_stats_list() -> None:
    assert one("Mamoswine’s Attack rose!") == {"k": "boost", "side": "own", "mon": "mamoswine", "stats": ["atk"],
                                               "by": 1}
    assert one("The opposing Lucario’s Sp. Atk and Speed rose!")["stats"] == ["spa", "spe"]


def test_name_with_no_single_legal_id_is_unread() -> None:
    assert one("Held Item: Glimmora’s Qwzxv Thing", side="opp")["k"] == "unread"
    assert one("Move: Mamoswine used Zzqqx Blast!")["k"] == "unread"


def test_pp_lost_row() -> None:
    assert one("The opposing Lucario lost 4 PP from Close Combat!") == {
        "k": "pp", "side": "opp", "mon": "lucario", "move": "closecombat", "lost": 4}
    assert problems(one("Mamoswine lost 3 PP from Earthquake!")) == []


def test_borrowed_team_wording_names_the_species() -> None:
    assert one("sent out ’s Archaludon!") == {"k": "enter", "side": "opp", "mon": "archaludon"}


def test_row_the_reader_flagged_is_unread() -> None:
    events, _ = convert_rows([{"turn": 1, "index": 0, "text": "om", "read": "unread", "reason": "no known wording"}],
                             TEAMS)
    assert events[1] == {"k": "unread", "text": "om", "category": "no known wording"}


@pytest.mark.parametrize(
    ("text", "event"),
    [
        ("Garchomp’s HP is full!", {"k": "fail", "side": "own", "mon": "garchomp", "reason": "hpfull"}),
        ("Garchomp is already drowsy.", {"k": "fail", "side": "own", "mon": "garchomp", "reason": "yawn"}),
        ("Golisopod already has a substitute!", {"k": "fail", "side": "own", "mon": "golisopod",
                                                 "reason": "substitute"}),
        ("Mamoswine is already poisoned!", {"k": "fail", "side": "own", "mon": "mamoswine", "reason": "psn"}),
        ("The opposing Lucario switched items with its target!", {"k": "once", "side": "opp", "mon": "lucario",
                                                                   "effect": "trick"}),
        ("Garchomp obtained a Choice Scarf.", {"k": "item", "side": "own", "mon": "garchomp", "item": "choicescarf",
                                               "what": "received"}),
        ("The opposing Lucario obtained a serving of Leftovers.", {"k": "item", "side": "opp", "mon": "lucario",
                                                                   "item": "leftovers", "what": "received"}),
        ("The opposing Lucario stole Garchomp’s Leftovers!", {
            "k": "item", "side": "opp", "mon": "lucario", "item": "leftovers", "what": "received",
            "of": {"side": "own", "mon": "garchomp"}}),
        ("The opposing Lucario foresaw an attack!", {"k": "once", "side": "opp", "mon": "lucario",
                                                     "effect": "futuresight_start"}),
        ("Garchomp took the Future Sight attack!", {"k": "once", "side": "own", "mon": "garchomp",
                                                    "effect": "futuresight"}),
        ("The opposing Lucario switched all changes to its Attack and Sp. Atk with its target!", {
            "k": "boost_reset", "side": "opp", "mon": "lucario", "what": "swap", "stats": ["atk", "spa"]}),
        ("Garchomp switched stat changes with its target!", {"k": "boost_reset", "side": "own", "mon": "garchomp",
                                                             "what": "swap"}),
        ("The opposing Lucario’s stat changes were all reversed!", {"k": "boost_reset", "side": "opp",
                                                                    "mon": "lucario", "what": "invert"}),
        ("The opposing Lucario switched Speed with its target!", {"k": "once", "side": "opp", "mon": "lucario",
                                                                  "effect": "speedswap"}),
        ("Grass type was added to Garchomp!", {"k": "types", "side": "own", "mon": "garchomp", "add": "grass"}),
        ("The opposing Lucario is loafing around!", {"k": "cant", "side": "opp", "mon": "lucario",
                                                     "reason": "truant"}),
        ("Garchomp became cloaked in mystical moonlight!", {"k": "cause", "effect": "healingwish", "side": "own",
                                                            "mon": "garchomp"}),
        ("The opposing Lucario is being salted!", {"k": "volatile", "side": "opp", "mon": "lucario",
                                                   "effect": "saltcure"}),
        ("Garchomp is hurt by Salt Cure!", {"k": "cause", "effect": "saltcure", "side": "own", "mon": "garchomp"}),
        ("Mamoswine must recharge!", {"k": "cant", "side": "own", "mon": "mamoswine", "reason": "recharge"}),
        ("But there was no PP left for the move!", {"k": "cant", "reason": "nopp"}),  # 0.6, LV5Q-2
        ("The opposing Lucario cannot use Explosion!", {"k": "cant", "side": "opp", "mon": "lucario",
                                                        "reason": "explosion"}),
    ],
)
def test_rows_05(text: str, event: dict) -> None:
    assert one(text) == event
    assert problems(event) == []


def test_values_are_checked() -> None:
    assert problems({"k": "result", "how": "great"}) == ["value:how"]
    assert problems({"k": "boost_reset", "what": "swap", "stats": ["atk"]}) == []
    assert problems({"k": "fail", "reason": "hpfull"}) == []
    assert problems({"k": "types", "side": "own", "mon": "x", "add": "grass"}) == []
    assert problems({"k": "item", "item": "x", "what": "received", "of": "garchomp"}) == ["value:of"]
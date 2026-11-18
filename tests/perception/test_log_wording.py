"""The strict wording check, the snap of a misread name, and the vote across frames."""

from __future__ import annotations

import json

import pytest

from pcsa.phone.log_merge import BattleLog
from pcsa.phone.log_panel import Row, View
from pcsa.phone.log_vote import split_label, text_reading, vote
from pcsa.phone.log_wording import check, snap_check


def checked(text: str):
    category, priority, message = split_label(text)
    return check(category, message)


@pytest.mark.parametrize(
    ("read", "game"),
    [
        ("It’s not very effective..", "It’s not very effective..."),
        ("It doesn't affect the opposing Garchomp.", "It doesn’t affect the opposing Garchomp..."),
        ("Move: The opposing Eelektross used Coil", "Move: The opposing Eelektross used Coil!"),
        ("The opposing Swampert was damaged by the recoill", "The opposing Swampert was damaged by the recoil!"),
        ("HP Lost: Golisopod’s HP: 100% →70%", "HP Lost: Golisopod’s HP: 100% → 70%"),
        ("feat. withdrewCharizard!", "withdrew Charizard!"),
        ("Go!Basculegion!", "Go! Basculegion!"),
        ("The opposing Pelipper went back to!", "The opposing Pelipper went back to!"),
        ("Ann sent out Mimikyuthe Sociable!", "sent out Mimikyu the Sociable!"),
        ("The opposing Charizard’s Charizardite Y is reacting to's Omni Ring!",
         "The opposing Charizard’s Charizardite Y is reacting to ’s Omni Ring!"),
        ("The opposing Volcarona’s Sp. Atk, Sp. Def, and Speed rose!",
         "The opposing Volcarona’s Sp. Atk, Sp. Def, and Speed rose!"),
    ],
)  # fmt: skip
def test_a_known_wording_is_written_in_the_games_spelling(read: str, game: str) -> None:
    result = checked(read)
    assert result.valid and result.text == game


@pytest.mark.parametrize(
    "read",
    ["Until Battle Ends", "Held Item: Skarmory’s Rockv Helmet", "Gholdengo’s Sp. Def harshly fel!", "om",
     "HP Lost: Gholdengo’s HP:", "Move 4 Priority: Primarina used Aqua Jet!", "Ivove: Primarina used Aqua Jet!"],
)  # fmt: skip
def test_a_misread_row_fails(read: str) -> None:
    assert not checked(read).valid


def test_a_misread_name_snaps_only_to_a_unique_legal_name() -> None:
    result = snap_check("Held Item", "Skarmory’s Rockv Helmet")
    assert result.valid and result.text == "Held Item: Skarmory’s Rocky Helmet"
    assert result.snaps == [{"hole": "ITEM", "read": "Rockv Helmet", "to": "Rocky Helmet"}]
    assert not snap_check("", "sent out Xyzzyq!").valid


def reading(text: str, **extra) -> dict:
    return {**text_reading(text), **extra}


def test_the_vote_prefers_readings_that_pass_and_then_the_majority() -> None:
    picked = vote([reading("Move: Garchomp used Earthquake"), reading("Move: Garchomp used Earthquake!")])
    assert picked["read"] == "ok" and picked["text"] == "Move: Garchomp used Earthquake!"
    tied = vote([reading("HP Lost: Garchomp’s HP: 100 → 42"), reading("HP Lost: Garchomp’s HP: 100 → 43")])
    assert tied["read"] == "unread" and tied["reason"].startswith("frames disagree")


def test_a_row_read_only_under_the_timer_overlay_is_unread() -> None:
    picked = vote([reading("It’s super effective!", covered=True)])
    assert picked["read"] == "unread" and picked["text"] == "It’s super effective!"


def test_a_row_no_reading_passes_is_snapped_or_unread() -> None:
    assert vote([reading("Held Item: Skarmory’s Rockv Helmet")])["read"] == "snapped"
    unread = vote([reading("Until Battle Ends")])
    assert unread["read"] == "unread" and unread["text"] == "Until Battle Ends"


def view(texts: list[str], *, settled: bool = True, scroll: float = 1.0, thumb=(500, 958), at_top=False) -> View:
    rows = [Row(t, (130 + 144 * i, 264 + 144 * i), []) for i, t in enumerate(texts)]
    return View("v", "", 1, scroll, thumb, at_top, True, True, rows, [], settled=settled)


def test_the_saved_row_keeps_its_readings_and_vote(tmp_path) -> None:
    log = BattleLog("b")
    log.merge([view(["It’s not very effective..", "Until Battle Ends"])])
    log.save(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"] == "It’s not very effective..." and rows[0]["read"] == "ok"
    assert rows[1]["read"] == "unread" and rows[1]["readings"][0]["text"] == "Until Battle Ends"
    coverage = json.loads((tmp_path / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["unread_rows"][0]["frames"][0]["frame"] == "v"


def test_a_lone_unconfirmed_capture_is_kept_when_every_row_passes() -> None:
    log = BattleLog("b")
    report = log.merge([view(["Fainted: Gholdengo fainted!", "Go! Mamoswine!"], settled=False)])
    assert report.rows == ["Fainted: Gholdengo fainted!", "Go! Mamoswine!"]
    assert any("lone unconfirmed capture" in w for w in report.warnings)


def test_a_one_row_overlap_is_accepted_when_the_scrollbar_agrees() -> None:
    log = BattleLog("b")
    log.merge([view(["sent out Golisopod!", "Go! Archaludon!"], thumb=(134, 958), at_top=True)])
    # a re-visit whose top view has about one row above it: the shared row is the stored last row
    later = view(["Go! Archaludon!", "Move: Archaludon used Thunderbolt!"], scroll=0.12, thumb=(200, 700))
    report = log.merge([later])
    assert report.anchor == "overlap1"
    assert log.turns[1] == ["sent out Golisopod!", "Go! Archaludon!", "Move: Archaludon used Thunderbolt!"]

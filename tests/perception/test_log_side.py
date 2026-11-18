"""Row side from the category pill colour: real crops (local-only), drawn pills, and the merge carrying it."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pcsa.phone.log_merge import BattleLog
from pcsa.phone.log_panel import Row, View
from pcsa.phone.log_side import combine_sides, pill_stats, row_side
from pcsa.phone.ocr import REFERENCE
from tests.perception.test_log_frames import FRAMES, image

CROP_LEFT, CROP_TOP = 550, 300  # crops are a row band, x 550-880 of the frame
PANEL_BGR = (146, 39, 49)
OWN_BGR = (253, 122, 137)
OPP_BGR = (141, 116, 255)
BADGE_BGR = (226, 101, 117)  # the neutral "+1 Priority" badge


def placed(name: str) -> tuple[np.ndarray, tuple[int, int]]:
    """A local-only crop pasted into a blank reference frame, and its row band."""
    crop = image(name)
    frame = np.zeros((REFERENCE[1], REFERENCE[0], 3), np.uint8)
    frame[CROP_TOP : CROP_TOP + crop.shape[0], CROP_LEFT : CROP_LEFT + crop.shape[1]] = crop
    return frame, (CROP_TOP, CROP_TOP + crop.shape[0])


@pytest.mark.parametrize(
    ("name", "side"),
    [
        ("pill-opp-move.png", "opp"),
        ("pill-own-move.png", "own"),
        ("pill-none-plain.png", None),
        ("pill-opp-held-item.png", "opp"),
        ("pill-opp-priority.png", "opp"),
        ("pill-own-priority.png", "own"),
    ],
)
def test_real_pills_name_their_side(name: str, side: str | None) -> None:
    frame, band = placed(name)
    assert row_side(frame, band) == side


def test_the_priority_badge_does_not_dilute_the_red_pill() -> None:
    frame, band = placed("pill-opp-priority.png")
    assert pill_stats(frame, band)["opp_share"] == 1.0


def test_full_frame_rows_read_their_side() -> None:
    raw = json.loads((FRAMES / "views.json").read_text(encoding="utf-8"))["mega-bottom.jpg"]
    frame = image("mega-bottom.jpg")
    sides = [row_side(frame, tuple(r["bounds"])) for r in raw["rows"]]
    expected = [
        "opp" if r["text"].startswith("HP Lost") and "%" in r["text"]
        else "own" if r["text"].startswith("HP Lost")
        else ("opp" if "The opposing" in r["text"] else "own") if ": " in r["text"]
        else None
        for r in raw["rows"]
    ]  # fmt: skip
    assert sides == expected


def drawn(pill: tuple[int, int, int] | None, *, badge: bool = False, text: bool = True) -> np.ndarray:
    """A panel row band (y 300-434) with an optional pill, label text and priority badge."""
    frame = np.full((REFERENCE[1], REFERENCE[0], 3), PANEL_BGR, np.uint8)
    if pill is not None:
        top = 330 if badge else 342
        cv2.rectangle(frame, (580, top), (840, top + 48), pill, -1)
        cv2.circle(frame, (580, top + 24), 24, pill, -1)
        cv2.circle(frame, (840, top + 24), 24, pill, -1)
        if text:
            cv2.putText(frame, "Move", (660, top + 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    if badge:
        cv2.rectangle(frame, (602, 384), (817, 424), BADGE_BGR, -1)
    return frame


@pytest.mark.parametrize(
    ("pill", "badge", "side"),
    [
        (OWN_BGR, False, "own"),
        (OPP_BGR, False, "opp"),
        (OWN_BGR, True, "own"),
        (OPP_BGR, True, "opp"),
        (None, False, None),
    ],
)
def test_drawn_pills(pill, badge: bool, side: str | None) -> None:
    assert row_side(drawn(pill, badge=badge), (300, 434)) == side


def test_a_badge_alone_is_not_a_pill() -> None:
    frame = np.full((REFERENCE[1], REFERENCE[0], 3), PANEL_BGR, np.uint8)
    cv2.rectangle(frame, (602, 340), (817, 390), BADGE_BGR, -1)
    assert row_side(frame, (300, 434)) is None


def test_an_empty_band_has_no_pill() -> None:
    assert row_side(drawn(OPP_BGR), (300, 300)) is None


def test_combining_views_keeps_a_seen_side() -> None:
    assert combine_sides(None, "opp") == "opp"
    assert combine_sides("own", None) == "own"
    assert combine_sides("own", "own") == "own"
    assert combine_sides("own", "opp") == "unknown"
    assert combine_sides(None, None) is None


def view(texts: list[str], sides: list[str | None], scroll: float = 1.0) -> View:
    rows = [
        Row(t, (130 + 144 * i, 260 + 144 * i), [], None, s) for i, (t, s) in enumerate(zip(texts, sides, strict=True))
    ]
    return View("v", "", 1, scroll, (500, 958), False, True, True, rows, [])


def test_the_merged_log_writes_each_rows_side(tmp_path) -> None:
    texts = ["Move: Garchomp used Earthquake!", "HP Lost: Venusaur’s HP: 100% → 42%", "It’s super effective!"]
    log = BattleLog("b")
    log.merge([view(texts, ["own", "opp", None])])
    log.merge([view(texts + ["Move: The opposing Venusaur used Toxic!"], [None, "opp", None, "opp"])])
    assert log.side_history() == {1: ("own", "opp", None, "opp")}
    log.save(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["side"] for r in rows] == ["own", "opp", None, "opp"]

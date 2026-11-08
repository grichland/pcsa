"""Battle-log collection and merging on real log-panel frames (run 20261103-015542, battle 14, visit 2).

The climb toward Turn 1's start ran out of budget: the MOVE TIME warning covered one capture and the last
capture, the only one showing our Mega Evolution, Drought and the sun, was never compared with a second one.
`views.json` caches what the panel reader read from each frame (`-m ocr` re-reads them). The frames are game
captures and stay out of git; the tests that need their pixels skip in a checkout without them.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pcsa.phone.log_merge import BattleLog
from pcsa.phone.log_panel import Row, View

FRAMES = Path(__file__).resolve().parents[1] / "fixtures" / "log_frames"
OUR_MEGA = [
    "Charizard’s Charizardite Y is reacting to Trainer’s Omni Ring!",
    "Mega Evolution: Charizard has Mega Evolved into Mega Charizard!",
    "Ability: Charizard’s Drought",
    "The sunlight turned harsh!",
]
SETTLED_ROWS = [
    "The opposing Venusaur’s Venusaurite is reacting to Opponent’s Omni Ring!",
    "Mega Evolution: The opposing Venusaur has Mega Evolved into Mega Venusaur!",
    "Move: Charizard used Weather Ball!",
    "HP Lost: Venusaur’s HP: 100% → 42%",
    "It’s super effective!",
    "Move: The opposing Venusaur used Toxic!",
    "Charizard was badly poisoned!",
    "Charizard was hurt by its poisoning!",
    "HP Lost: Charizard’s HP: 185 → 174",
]
BUDGET_S = 60.0


def image(name: str) -> np.ndarray:
    path = FRAMES / name
    if not path.is_file():
        pytest.skip(f"{name} is a local-only capture (kept out of the repo)")
    return cv2.imread(str(path))


def cached_view(name: str, *, settled: bool = True) -> View:
    raw = json.loads((FRAMES / "views.json").read_text(encoding="utf-8"))[name]
    rows = [Row(r["text"], tuple(r["bounds"]), r["segments"]) for r in raw["rows"]]
    thumb = tuple(raw["thumb"])
    return View(
        name, "", raw["turn"], raw["scroll"], thumb, raw["at_top"], raw["at_bottom"], raw["has_scrollbar"], rows, [],
        settled=settled,
    )  # fmt: skip


def test_an_unsettled_view_that_joins_the_settled_ones_adds_the_rows_only_it_shows() -> None:
    log = BattleLog("b")
    views = [
        cached_view("mega-bottom.jpg"),
        cached_view("mega-middle.jpg"),
        cached_view("mega-top.jpg", settled=False),
    ]
    report = log.merge(views)
    assert report.rows == OUR_MEGA + SETTLED_ROWS
    assert not any("no verified overlap" in w for w in report.warnings)


def test_an_unsettled_view_with_no_verified_overlap_is_left_out() -> None:
    log = BattleLog("b")
    report = log.merge([cached_view("mega-bottom.jpg"), cached_view("mega-top.jpg", settled=False)])
    assert report.rows == SETTLED_ROWS[3:]
    assert any("unsettled view left out" in w for w in report.warnings)


@pytest.mark.ocr
@pytest.mark.parametrize("name", ["mega-bottom.jpg", "mega-middle.jpg", "mega-top.jpg"])
def test_the_panel_reader_reads_the_cached_rows(name: str, config) -> None:
    from pcsa.phone.log_panel import read_view
    from pcsa.phone.ocr import RapidOcrReader

    view = read_view(image(name), RapidOcrReader(config.agent.ocr), name, sprites=False)
    assert [r.text for r in view.rows] == [r.text for r in cached_view(name).rows]

"""Move rows of the Offer when the screen's names and the sheet disagree; the fainted mark of a party row."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pcsa.live.request import screen_rows
from pcsa.phone.texts import red_fraction
from tests.live.turn_rig import TEAM


def _peek(names: list[str | None]) -> SimpleNamespace:
    moves = [SimpleNamespace(slot=k, name=n, pp=(8, 8), disabled=False) for k, n in enumerate(names)]
    return SimpleNamespace(fields={"moves": moves})


def test_a_partly_read_name_keeps_the_sheets_move() -> None:
    member, notes = TEAM.members[4], []  # Mamoswine: Ice Shard, Earthquake, Icicle Spear, Endeavor
    rows = screen_rows(member, _peek(["Ice Shard", "Earth", None, "Endeavor"]), notes)  # type: ignore[arg-type]
    assert [m.id for m in rows] == ["iceshard", "earthquake", "iciclespear", "endeavor"]
    assert notes == ["row 1: screen earth, sheet earthquake"]


def test_another_move_on_the_screen_replaces_the_sheets() -> None:
    notes: list[str] = []
    rows = screen_rows(TEAM.members[4], _peek(["Ice Shard", "Struggle", None, "Endeavor"]), notes)  # type: ignore[arg-type]
    assert rows[1].id == "struggle" and notes == ["row 1: screen struggle, sheet earthquake"]


def test_a_poison_badge_is_not_the_fainted_badge() -> None:
    red = np.zeros((40, 40, 3), np.uint8)
    red[...] = (40, 40, 220)  # BGR
    purple = red.copy()
    purple[...] = (200, 60, 200)
    assert red_fraction(red, (0, 0, 40, 40)) == 1.0 and red_fraction(purple, (0, 0, 40, 40)) == 0.0

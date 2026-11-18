"""The STOPGAP CG tables: headers, legal counts, and the pure form helpers built on them."""

from __future__ import annotations

import json

import pytest

from pcsa.battle import game, moves
from pcsa.common import paths

KINDS = {"species": 392, "moves": 515, "items": 166, "abilities": 318, "natures": 25, "typechart": 18}


@pytest.mark.parametrize("kind,count", KINDS.items())
def test_header_and_count(kind: str, count: int) -> None:
    data = json.loads((paths.assets_dir() / "game" / "cg" / f"{kind}.json").read_text(encoding="utf-8"))
    assert next(iter(data)) == "_header"
    head = data["_header"]
    assert head["status"] == "STOPGAP" and "CG's JSON export" in head["note"]
    assert head["showdown_commit"] == "c0845f33d381402dfcb18ce324c20a94f18af075"
    assert len(data[kind]) == count


def test_type_chart_is_square() -> None:
    chart = game.cg_table("typechart")
    assert all(len(row) == 18 for row in chart.values())
    assert chart["ground"]["flying"] == 0.0 and chart["water"]["fire"] == 2.0 and chart["fire"]["water"] == 0.5


def test_moves_carry_the_listed_fields() -> None:
    move = game.cg_table("moves")["uturn"]
    assert set(move) >= {"name", "type", "category", "basePower", "accuracy", "pp", "priority", "flags", "target"}
    assert moves.max_pp("uturn") == move["pp"]

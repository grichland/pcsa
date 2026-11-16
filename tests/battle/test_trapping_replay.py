"""Decisions saved from a live game against Mega Gengar (Shadow Tag): the rebuilt request must trap our non-Ghost active."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pcsa.battle.state import replay
from pcsa.battle.trapping import is_trapped

BATTLE = Path(r"C:\Users\geral\Code\pcsa\output\runs\20261112-234559-run\battle-38\decisions")
pytestmark = pytest.mark.skipif(not BATTLE.is_dir(), reason="live decision files not on this machine")


def switches(number: int, *, mark: bool) -> tuple[bool, list[str]]:
    saved = json.loads((BATTLE / f"{number:04d}-action.json").read_text(encoding="utf-8"))
    battle = replay(saved["protocol"], None)
    request = copy.deepcopy(saved["request"])
    trapped = is_trapped(battle, request["side"]["pokemon"][0])
    if mark and trapped:
        request["active"][0]["trapped"] = True
    battle.parse_request(request)
    return trapped, [m.species for m in battle.available_switches]


@pytest.mark.parametrize("number", [5, 6, 7, 8, 9, 11])
def test_trapped_against_mega_gengar_offers_no_switch(number: int) -> None:
    trapped, offered = switches(number, mark=True)
    assert trapped and offered == []


def test_without_the_trap_switches_stay_offered() -> None:
    trapped, offered = switches(2, mark=True)
    assert not trapped and offered

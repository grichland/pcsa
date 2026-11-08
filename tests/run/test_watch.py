"""`pcsa watch` state over a run folder: events plus the newest decision record of the current battle."""

from __future__ import annotations

import json
from pathlib import Path

from pcsa.run.watch import RunReader
from pcsa.run.watch_battle import battle_view, rows_of

SNAPSHOT = """TURN 4   weather: -   fields: -
OURS   side: -
  garchomp       ACTIVE  HP 98/183    Dragon/Ground    status -    boosts -          ability roughskin      item focussash      moves earthquake, scaleshot
  raichu         bench   HP 135/135   Electric         status -    boosts -          ability lightningrod   item raichunitey    moves zapcannon
THEIRS side: -
  basculegion    bench   HP 87%       Water/Ghost      status -    boosts -          ability ?              item ?              moves flipturn
  raichu         ACTIVE  HP 88%       Electric         status -    boosts spe+1      ability ?              item raichunitey    moves fakeout
  preview, not yet seen: blaziken, meowscarada
"""

RECORD = {
    "schema": "pcsa-decision/1",
    "kind": "action",
    "turn": 4,
    "choice": "move0",
    "order": "/choose move earthquake",
    "degrade": "full",
    "reason": "tracked battle",
    "warnings": ["preview corrected"],
    "request": {
        "active": [
            {
                "moves": [
                    {"move": "Earthquake", "id": "earthquake", "pp": 10, "maxpp": 12, "disabled": False},
                    {"move": "Scale Shot", "id": "scaleshot", "pp": 20, "maxpp": 20, "disabled": True},
                ]
            }
        ],
        "side": {
            "pokemon": [
                {"ident": "p1: Garchomp", "condition": "98/183", "active": True, "moves": ["earthquake"]},
                {"ident": "p1: Raichu", "condition": "135/135", "active": False, "moves": ["zapcannon"]},
            ]
        },
    },
    "notes": {"member": "max_damage"},
}


def write_run(tmp_path: Path, *, with_record: bool = True) -> Path:
    run = tmp_path / "20261102-run"
    (run / "battle-1" / "decisions").mkdir(parents=True)
    events = [
        {"ts": "2026-11-02T15:44:00+00:00", "event": "observation", "screen": "lobby", "reason": "Begin Matchmaking"},
        {"ts": "2026-11-02T15:44:47+00:00", "event": "battle_start", "battle": 1, "roster": ["Garchomp"]},
        {"ts": "2026-11-02T15:46:40+00:00", "event": "turn", "battle": 1, "turn": 1, "action": "move0"},
    ]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"player": "mixed"}), encoding="utf-8")
    if with_record:
        (run / "battle-1" / "decisions" / "0003-action.json").write_text(json.dumps(RECORD), encoding="utf-8")
        (run / "battle-1" / "decisions" / "0003-action.txt").write_text(SNAPSHOT, encoding="utf-8")
    return run


def test_rows_of_parses_both_sides() -> None:
    header, ours, theirs, unseen = rows_of(SNAPSHOT)
    assert header["turn"] == "4"
    assert [r["species"] for r in ours] == ["garchomp", "raichu"]
    assert theirs[1]["state"] == "ACTIVE" and theirs[1]["boosts"] == "spe+1" and theirs[1]["moves"] == ["fakeout"]
    assert unseen == "blaziken, meowscarada"


def test_state_fills_the_battle_panels_from_the_record(tmp_path: Path) -> None:
    state = RunReader(write_run(tmp_path)).state()
    assert state["player"] == "mixed" and state["turn"] == 4
    assert state["decision"]["order"] == "/choose move earthquake" and state["decision"]["kind"] == "action"
    assert [m["name"] for m in state["moves"]] == ["Earthquake", "Scale Shot"]
    assert state["moves"][1]["disabled"] is True and state["moves"][0]["pp"] == [10, 12]
    assert state["opponent_moves"] == [
        {"name": "basculegion", "moves": ["flipturn"]},
        {"name": "raichu", "moves": ["fakeout"]},
    ]
    assert state["active"] == {"name": "garchomp", "hp": "98/183"}
    assert state["opponent"]["name"] == "raichu" and "meowscarada" in state["opponent"]["preview"]
    assert (
        state["battle_state"]["team"] == "garchomp, raichu"
        and state["battle_state"]["message"] == "preview corrected"
    )
    assert state["battle_state"]["battle_object"] == SNAPSHOT


def test_state_without_a_record_keeps_the_empty_panels(tmp_path: Path) -> None:
    state = RunReader(write_run(tmp_path, with_record=False)).state()
    assert state["decision"] is None and state["moves"] == []
    assert state["battle_state"] == {"battle": "1", "screen": "lobby"}


def test_view_follows_the_current_battle_folder(tmp_path: Path) -> None:
    run = write_run(tmp_path)
    assert battle_view(run, {"index": 2}, "prep") == {}  # battle 2 has no decisions yet: no stale battle-1 panels
    assert battle_view(run, {"index": 1}, "prep")["turn"] == 4


def test_state_is_recomputed_when_the_record_lands_after_the_event(tmp_path: Path) -> None:
    run = write_run(tmp_path, with_record=False)
    reader = RunReader(run)
    assert reader.state()["moves"] == []
    (run / "battle-1" / "decisions" / "0003-action.json").write_text(json.dumps(RECORD), encoding="utf-8")
    assert [m["name"] for m in reader.state()["moves"]] == ["Earthquake", "Scale Shot"]

"""`pcsa watch` state over a run folder: a fixture cut from a real neural run, then missing and partial files."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from pcsa.common import paths
from pcsa.run.watch import RunReader, latest_frame, player_view
from pcsa.run.watch_battle import battle_view, latest_record
from pcsa.run.watch_decisions import notes_view
from pcsa.run.watch_names import names_of, order_view

FIXTURE = paths.repo_root() / "tests" / "fixtures" / "watch_run"


@pytest.fixture
def run(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURE, tmp_path / "20261118-162900-neural-148"))


def test_run_header_record_and_rating(run: Path) -> None:
    os.utime(run / "events.jsonl", (1_700_000_000, 1_700_000_000))
    state = RunReader(run).state()
    assert state["run_name"] == "20261118-162900-neural-148" and state["player"] == "neural"
    view = state["player_view"]
    assert view["summary"] == "league-20261118-g-iter-00148.pt · update 148 · argmax"
    checkpoint = next(p for p in view["params"] if p["key"] == "checkpoint")
    assert checkpoint["value"] == checkpoint["full"] == "league-20261118-g-iter-00148.pt"
    assert {p["key"] for p in view["params"]} == {"checkpoint", "run", "update", "sampling", "fallback", "cg"}
    assert state["session_record"] == {"wins": 1, "losses": 2, "draws": 0, "forfeits": 0}
    rating = state["rating"]
    assert rating["tier"] == "Master Ball" and rating["rank"] == 4
    assert rating["start"] == 1646.544 and rating["rating"] == 1629.402 and rating["delta"] == -17.142
    assert [p["battle"] for p in rating["series"]] == [None, 1, 2, 3]
    assert state["status"] == "STOPPED"


def test_battle_list_carries_each_battles_result(run: Path) -> None:
    first, second, third = RunReader(run).state()["battles"]
    assert [b["outcome"] for b in (first, second, third)] == ["win", "loss", "loss"]
    assert first["opponent"][:2] == ["Blaziken", "Meganium"] and first["picks"] == [
        "Hippowdon",
        "Metagross",
        "Primarina",
    ]
    assert (first["rating_before"], first["rating_after"]) == (1646.544, 1664.853)
    assert first["decisions"] == 7 and second["decisions"] == 12 and third["turn"] == 9


def test_battle_panels_of_a_finished_battle(run: Path) -> None:
    state = RunReader(run).state(battle=2)
    assert state["battle"]["index"] == 2 and state["turn"] == 10 and state["legacy"] is False
    decision = state["decision"]
    assert decision["label"] == "Earthquake" and decision["order"] == "move earthquake"
    assert (
        decision["degrade"] == "player"
        and decision["kind"] == "action"
        and decision["source"] == "player · player answer"
    )
    notes = decision["notes"]
    assert notes["rule"] == "neural" and notes["fallback"] is False and notes["v"] == -0.951 and notes["ms"] == 9.67
    assert 0 < notes["p"] <= 1 and 0 <= notes["h"] <= 1 and notes["matrix"]["played"] == 0
    assert state["active"] == {"name": "Hippowdon", "hp": "32%", "fraction": 0.3209, "status": "", "boosts": ""}
    assert state["opponent"]["name"] == "Basculegion" and state["opponent"]["hp"] == "30%"
    hippowdon = state["your_team"][0]
    assert hippowdon["item"] == "" and hippowdon["moves"][0] == {"name": "Earthquake", "pp": [11, 12]}
    assert [m["name"] for m in state["your_team"]] == ["Hippowdon", "Dragonite", "Cinderace"]
    assert state["your_team"][1]["fainted"] and state["your_team"][1]["moves"][0]["pp"] is None  # PP: active only
    staraptor = next(f for f in state["foes"] if f["name"] == "Staraptor")
    assert staraptor["item"] == "Staraptite" and "Close Combat" in staraptor["moves"] and staraptor["fainted"]
    assert state["unseen"] == ["Archaludon", "Rotom", "Mimikyu"]
    assert state["field"]["weather"] == {"name": "Sand", "left": 4}
    log = state["log"]
    assert log["source"] == "battle_log.txt" and log["format"] == "standard" and log["header"]["battle_number"] == "2"
    assert (
        state["battle_state"]["battle_object"].startswith("TURN 10")
        and "Basculegion" in state["battle_state"]["battle_object"]
    )


def test_decision_list_has_notes_timing_and_the_preview_pick(run: Path) -> None:
    state = RunReader(run).state(battle=1)
    assert state["pinned"] == 1 and state["battle"]["index"] == 1 and state["turn"] == 6
    rows = state["decisions"]
    assert [r["sequence"] for r in rows] == [0, 1, 2, 6]
    assert [r["label"] for r in rows] == ["Yawn", "Metagross", "Meteor Mash + Mega", "Primarina"]
    assert rows[1]["kind"] == "forced" and rows[1]["order_kind"] == "switch"
    assert all(r["notes"]["matrix"] is None for r in rows)  # the list stays light
    assert rows[0]["notes"]["v"] == 0.5272 and rows[0]["notes"]["ms"] == 7.32
    assert rows[0]["timing"] is not None and len(rows[0]["legal"]) == 6 and rows[0]["dropped"] == []
    preview = state["preview"]
    assert preview["names"] == ["Hippowdon", "Metagross", "Primarina"]
    assert preview["notes"]["v"] == 0.6741 and preview["notes"]["ms"] == 8.61


def test_notes_of_other_players_are_shown_as_written() -> None:
    fallback = notes_view({"rule": "neural:fallback", "why": "checkpoint missing", "fallback": "disciplined"})
    assert fallback is not None and fallback["fallback"] is True and fallback["why"] == "checkpoint missing"
    assert fallback["p"] is None and fallback["rest"] == [{"key": "fallback", "value": "disciplined"}]
    scripted = notes_view({"rule": "attack", "switch_blocked": "threatened: lucario", "mega": "first turn offered"})
    assert scripted is not None and scripted["rule"] == "attack" and scripted["fallback"] is False
    assert [n["key"] for n in scripted["rest"]] == ["switch_blocked", "mega"] and scripted["matrix"] is None
    assert notes_view({}) is None and notes_view(None) is None and notes_view(["odd"]) is None


def test_a_controller_fallback_is_a_fallback_with_its_reason() -> None:
    record = {
        "degrade": "builtin",
        "reason": "no answer within 2.0s",
        "live": {"fallback": {"seq": 4, "reason": "builtin: no answer within 2.0s"}},
    }
    view = notes_view({"fallback": "no answer within 2.0s", "builtin": "first legal"}, record=record)
    assert view is not None and view["fallback"] is True and view["why"] == "builtin: no answer within 2.0s"
    assert {"key": "fallback_rule", "value": "builtin"} in view["rest"]
    assert all(r["key"] != "fallback" for r in view["rest"])
    without_live = notes_view({"fallback": "late"}, record={"degrade": "screen", "reason": "screen order"})
    assert without_live is not None and without_live["why"] == "screen order"


def test_orders_use_the_written_names() -> None:
    names = names_of({"own_export": "Garchomp @ Choice Scarf\nAbility: Rough Skin\n- Swords Dance\n"})
    assert order_view("move swordsdance mega", names) == {
        "order": "move swordsdance mega",
        "label": "Swords Dance + Mega",
        "kind": "move",
    }
    assert order_view("switch ninetalesalola", names)["label"] == "Ninetales-Alola"  # CG's species table
    assert order_view("move unheardof", names)["label"] == "unheardof"
    assert order_view("default", names) == {"order": "default", "label": "default", "kind": "other"}


def test_player_view_of_a_scripted_player() -> None:
    view = player_view({"player": "disciplined", "player_params": {"stab": 1.5, "nested": {"a": 1}}})
    assert view["kind"] == "disciplined" and view["summary"] == ""
    assert view["params"][1] == {"key": "nested", "value": '{"a": 1}', "full": '{"a": 1}'}
    assert player_view({}) == {"kind": None, "params": [], "summary": ""}


# Missing and partial files -------------------------------------------------------------------------


def test_an_empty_or_absent_run_folder_is_a_waiting_page(tmp_path: Path) -> None:
    for folder in (tmp_path / "not-there-yet", tmp_path):
        state = RunReader(folder).state()
        assert state["status"] == "WAITING" and state["battle"] is None and state["decision"] is None
        assert state["battles"] == [] and state["rating"] is None and state["frame_id"] is None
        assert state["decisions"] == [] and state["log"]["turns"] == [] and state["player"] is None
        assert state["log"]["source"] is None and state["winrate"]["rate"] is None


def test_missing_manifest_battle_folder_and_log(run: Path) -> None:
    (run / "manifest.json").unlink()
    shutil.rmtree(run / "battle-3")
    state = RunReader(run).state()
    assert state["player"] is None and state["player_view"]["params"] == []
    assert state["battle"]["index"] == 3 and state["decision"] is None and state["decisions"] == []
    assert state["log"]["turns"] == [] and state["log"]["raw"] == "" and state["session_record"]["losses"] == 2
    assert state["battle_state"]["battle"] == "3"
    assert battle_view(run, {"index": 7}, "prep") == {}


def test_a_half_written_record_shows_the_one_before_it(run: Path) -> None:
    decisions = run / "battle-2" / "decisions"
    whole = (decisions / "0011-action.json").read_text(encoding="utf-8")
    reader = RunReader(run)
    assert reader.state(battle=2)["decision"]["sequence"] == 11
    (decisions / "0012-action.json").write_text(whole[: len(whole) // 2], encoding="utf-8")
    state = reader.state(battle=2)
    assert state["decision"]["sequence"] == 11 and [r["sequence"] for r in state["decisions"]] == [0, 1, 5, 11]
    record = latest_record(run, 2)
    assert record is not None and record.path.name == "0011-action.json"
    (decisions / "0012-action.json").write_text(whole.replace('"sequence":11', '"sequence":12'), encoding="utf-8")
    state = reader.state(battle=2)
    assert state["decision"]["sequence"] == 12 and [r["sequence"] for r in state["decisions"]][-2:] == [11, 12]


def test_records_of_an_unexpected_shape_do_not_break_the_page(run: Path) -> None:
    decisions = run / "battle-2" / "decisions"
    (decisions / "0012-action.json").write_text(
        json.dumps(
            {
                "schema": "pcsa-decision/4",
                "notes": {"cg": {"view": [], "net": {"pi": "x", "belief": 3}}},
                "legal": "x",
            }
        ),
        encoding="utf-8",
    )
    (decisions / "0013-action.json").write_text("[1, 2]", encoding="utf-8")
    (run / "manifest.json").write_text("{", encoding="utf-8")
    state = RunReader(run).state(battle=2)
    assert state["player"] is None and state["session_record"]["losses"] == 2
    assert len(state["decisions"]) == 5 and state["decisions"][-1]["legal"] == []
    assert state["foes"]  # the view of the newest record that has one
    json.dumps(state)


def test_a_decision_without_a_view_uses_the_newest_view_before_it(run: Path) -> None:
    path = run / "battle-2" / "decisions" / "0011-action.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["notes"]["cg"]["view"]
    path.write_text(json.dumps(data), encoding="utf-8")
    state = RunReader(run).state(battle=2)
    assert state["decision"]["label"] == "Earthquake" and state["legacy"] is False
    assert state["turn"] == 10 and state["field"]["turn"] == 5  # the view is the forced switch's, at turn 5
    assert state["foes"] and state["active"]["name"] is not None


def test_newest_settled_frame_is_shown(tmp_path: Path) -> None:
    assert latest_frame(tmp_path) is None
    frames = tmp_path / "frames"
    frames.mkdir()
    assert latest_frame(tmp_path) is None
    (frames / "00001-poll-lobby.jpg").write_bytes(b"jpeg")
    first = latest_frame(tmp_path)
    assert first is not None and first[1].name == "00001-poll-lobby.jpg"
    (frames / "00002-poll-moves.jpg").write_bytes(b"")
    (frames / "notes.txt").write_text("not a frame", encoding="utf-8")
    again = latest_frame(tmp_path)
    assert again is not None and again[1].name == "00001-poll-lobby.jpg"

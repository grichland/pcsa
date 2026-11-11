"""`pcsa watch` state over a run folder: a fixture cut from a real neural run, then missing and partial files."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from pcsa.common import paths
from pcsa.run.watch import RunReader, latest_frame, player_view
from pcsa.run.watch_battle import battle_view, latest_record, rows_of
from pcsa.run.watch_decisions import notes_view
from pcsa.run.watch_names import names_of, order_view

FIXTURE = paths.repo_root() / "tests" / "fixtures" / "watch_run"

SNAPSHOT = """TURN 4   weather: -   fields: -
OURS   side: -
  garchomp       ACTIVE  HP 98/183    Dragon/Ground    status -    boosts -          ability roughskin      item focussash      moves earthquake, scaleshot
  raichu         bench   HP 135/135   Electric         status -    boosts -          ability lightningrod   item raichunitey    moves zapcannon
THEIRS side: -
  basculegion    bench   HP 87%       Water/Ghost      status -    boosts -          ability ?              item ?              moves flipturn
  raichu         ACTIVE  HP 88%       Electric         status -    boosts spe+1      ability ?              item raichunitey    moves fakeout
  preview, not yet seen: blaziken, meowscarada
"""


@pytest.fixture
def run(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURE, tmp_path / "20261109-205551-run"))


def test_rows_of_parses_both_sides() -> None:
    header, ours, theirs, unseen = rows_of(SNAPSHOT)
    assert header["turn"] == "4"
    assert [r["species"] for r in ours] == ["garchomp", "raichu"] and ours[0]["types"] == "Dragon/Ground"
    assert theirs[1]["state"] == "ACTIVE" and theirs[1]["boosts"] == "spe+1" and theirs[1]["moves"] == ["fakeout"]
    assert unseen == "blaziken, meowscarada"


def test_run_header_record_and_rating(run: Path) -> None:
    os.utime(run / "events.jsonl", (1_700_000_000, 1_700_000_000))
    state = RunReader(run).state()
    assert state["run_name"] == "20261109-205551-run" and state["player"] == "neural"
    view = state["player_view"]
    assert view["summary"] == "team2-fine-tuned.pt · update 50 · greedy"
    checkpoint = next(p for p in view["params"] if p["key"] == "checkpoint")
    assert checkpoint["value"] == "team2-fine-tuned.pt" and checkpoint["full"].endswith("models\\team2-fine-tuned.pt")
    assert all(p["key"] != "seed" for p in view["params"])  # null settings are left out
    assert state["session_record"] == {"wins": 0, "losses": 2, "draws": 0, "forfeits": 0}
    rating = state["rating"]
    assert rating["tier"] == "Master Ball" and rating["rank"] == 4
    assert rating["start"] == 1572.834 and rating["rating"] == 1542.007 and rating["delta"] == -30.827
    assert [(p["battle"], p["rating"]) for p in rating["series"]] == [(None, 1572.834), (1, 1555.487), (2, 1542.007)]
    assert state["status"] == "STALE"  # the fixture run was cut off: no run_end, and its last event is old


def test_battle_list_carries_each_battles_result(run: Path) -> None:
    first, second, third = RunReader(run).state()["battles"]
    assert (third["index"], third["outcome"], third["rating_before"]) == (3, None, 1542.007)  # still playing
    assert (first["index"], first["outcome"], second["index"], second["outcome"]) == (1, "loss", 2, "loss")
    assert first["opponent"][:2] == ["Dragalge", "Corviknight"] and first["picks"] == [
        "Kingambit",
        "Garchomp",
        "Mimikyu",
    ]
    assert (first["rating_before"], first["rating_after"]) == (1572.834, 1555.487)
    assert (second["rating_before"], second["rating_after"]) == (1555.487, 1542.007)
    assert first["disconnects"] == 3 and second["disconnects"] == 1 and first["decisions"] == 9


def test_battle_panels_of_a_finished_battle(run: Path) -> None:
    state = RunReader(run).state(battle=2)
    assert state["battle"]["index"] == 2 and state["turn"] == 8 and state["screen"] == "wait"
    decision = state["decision"]
    assert decision["label"] == "Close Combat" and decision["order"] == "move closecombat"
    assert decision["degrade"] == "tracked" and decision["kind"] == "action"
    assert {"order": "move extremespeed", "label": "Extreme Speed", "kind": "move"} in decision["legal"]
    notes = decision["notes"]
    assert notes["rule"] == "neural" and notes["fallback"] is False and 0 < notes["p"] <= 1 and 0 <= notes["h"] <= 1
    assert state["active"]["name"] == "Lucario" and state["active"]["fraction"] == 1.0
    assert state["opponent"] == {"name": "Swampert", "hp": "45%", "fraction": 0.45, "status": "", "boosts": ""}
    swampert = next(f for f in state["foes"] if f["name"] == "Swampert")
    assert swampert["active"] and swampert["item"] == "Swampertite" and "Earthquake" in swampert["moves"]
    assert state["unseen"] == ["Kingambit", "Mimikyu", "Raichu"]
    assert state["field"]["weather"]["name"] == "Rain"
    lucario = state["your_team"][0]
    assert lucario["item"] == "Lucarionite" and lucario["moves"][1] == {"name": "Close Combat", "pp": [8, 8]}
    log = state["log"]
    assert log["source"] == "battle_log.txt" and log["format"] == "standard" and log["header"]["battle_number"] == "2"
    assert [t["turn"] for t in log["turns"]] == [1, 2, 3, 4, 5, 6, 7, 8] and log["raw"] == ""
    assert state["battle_state"]["battle_object"].startswith("TURN 8")


def test_decision_list_has_notes_timing_and_the_preview_pick(run: Path) -> None:
    state = RunReader(run).state(battle=1)
    assert state["pinned"] == 1 and state["battle"]["index"] == 1 and state["turn"] == 4
    rows = state["decisions"]
    assert [r["sequence"] for r in rows] == [0, 1, 2, 3]
    assert [r["label"] for r in rows][-1] == "Earthquake" and rows[2]["order_kind"] == "switch"
    assert rows[0]["notes"]["p"] == 0.7017 and rows[0]["notes"]["h"] == 0.4401 and rows[0]["notes"]["ms"] == 5.0
    assert rows[0]["timing"] == {"turn_s": 10.0, "log_s": 4.12, "tap_s": 10.03}
    assert len(rows[0]["legal"]) == 6 and rows[0]["dropped"] == []
    preview = state["preview"]
    assert preview["names"] == ["Kingambit", "Garchomp", "Mimikyu"]
    assert (
        preview["notes"]["v"] == -0.5858 and preview["notes"]["h"] == 0.7901 and preview["notes"]["rule"] == "neural"
    )
    assert state["foes"][0]["name"] == "Ninetales-Alola" and state["foes"][0]["ability"] == "Snow Warning"


def test_value_is_shown_only_where_the_record_has_one(run: Path) -> None:
    """Battle 2's records carry the critic's value (notes v); battle 1's were recorded before it existed."""
    reader = RunReader(run)
    latest = reader.state(battle=2)
    assert latest["decision"]["notes"]["v"] == -0.8
    assert [r["notes"]["v"] for r in latest["decisions"]] == [-1.02, -0.8]
    assert latest["preview"]["notes"]["v"] == -0.0496  # the team pick always had one
    earlier = reader.state(battle=1)
    assert [r["notes"]["v"] for r in earlier["decisions"]] == [None, None, None, None]
    assert earlier["decision"]["notes"]["v"] is None and earlier["decision"]["notes"]["p"] == 0.9776
    scripted = notes_view({"rule": "attack"})
    assert scripted is not None and scripted["v"] is None and scripted["p"] is None


def test_warnings_and_disconnects(run: Path) -> None:
    state = RunReader(run).state()
    kinds = [w["kind"] for w in state["warnings"]]
    assert (
        state["disconnects"] == 4
        and kinds.count("disconnect_start") == 4
        and kinds.count("disconnect_recovered") == 4
    )
    assert state["warning_count"] == len(kinds) == 9 and "stray_surface" in kinds
    assert all(e["kind"] != "observation" for e in state["timeline"])


def test_notes_are_shown_as_written() -> None:
    fallback = notes_view(
        {"rule": "neural:fallback", "why": "checkpoint missing", "fallback": "disciplined", "ms": 1.5}
    )
    assert fallback is not None and fallback["fallback"] is True and fallback["why"] == "checkpoint missing"
    assert fallback["p"] is None and fallback["rest"] == [{"key": "fallback", "value": "disciplined"}]
    scripted = notes_view({"rule": "attack", "switch_blocked": "threatened: lucario", "mega": "first turn offered"})
    assert scripted is not None and scripted["rule"] == "attack" and scripted["fallback"] is False
    assert [n["key"] for n in scripted["rest"]] == ["switch_blocked", "mega"]
    assert notes_view({}) is None and notes_view(None) is None and notes_view(["odd"]) is None


def test_orders_use_the_names_the_record_spells_out() -> None:
    names = names_of(
        {"protocol": ["|move|p2a: Ceruledge|Swords Dance|p1a: Garchomp", "|poke|p2|Ninetales-Alola, L50|"]}
    )
    assert order_view("move swordsdance mega", names) == {
        "order": "move swordsdance mega",
        "label": "Swords Dance + Mega",
        "kind": "move",
    }
    assert order_view("switch ninetalesalola", names)["label"] == "Ninetales-Alola"
    assert order_view("move unheardof", names)["label"] == "unheardof"  # never guessed
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
        assert (
            state["log"]["source"] is None and state["winrate"]["rate"] is None and state["winrate"]["battles"] == 0
        )


def test_missing_manifest_battle_folder_and_log(run: Path) -> None:
    (run / "manifest.json").unlink()
    shutil.rmtree(run / "battle-3")
    state = RunReader(run).state()
    assert state["player"] is None and state["player_view"]["params"] == []
    assert state["battle"]["index"] == 3 and state["decision"] is None and state["decisions"] == []
    assert state["log"]["turns"] == [] and state["log"]["raw"] == "" and state["session_record"]["losses"] == 2
    assert state["battle_state"] == {"battle": "3", "screen": "wait"}
    assert battle_view(run, {"index": 7}, "prep") == {}  # a battle without a folder: no stale panels of another


def test_a_half_written_record_shows_the_one_before_it(run: Path) -> None:
    decisions = run / "battle-2" / "decisions"
    whole = (decisions / "0010-action.json").read_text(encoding="utf-8")
    reader = RunReader(run)
    assert reader.state(battle=2)["decision"]["sequence"] == 10
    (decisions / "0011-action.json").write_text(whole[: len(whole) // 2], encoding="utf-8")
    state = reader.state(battle=2)
    assert state["decision"]["sequence"] == 10 and [r["sequence"] for r in state["decisions"]] == [9, 10]
    record = latest_record(run, 2)
    assert record is not None and record.path.name == "0010-action.json"
    (decisions / "0011-action.json").write_text(whole.replace('"sequence": 10', '"sequence": 11'), encoding="utf-8")
    state = reader.state(battle=2)
    assert state["decision"]["sequence"] == 11 and [r["sequence"] for r in state["decisions"]] == [9, 10, 11]


def test_records_of_an_unexpected_shape_do_not_break_the_page(run: Path) -> None:
    decisions = run / "battle-2" / "decisions"
    (decisions / "0011-action.json").write_text(
        json.dumps({"request": [], "legal": "x", "notes": 3}), encoding="utf-8"
    )
    (decisions / "0012-action.json").write_text("[1, 2]", encoding="utf-8")
    (run / "manifest.json").write_text("{", encoding="utf-8")
    state = RunReader(run).state(battle=2)
    assert state["player"] is None and state["session_record"]["losses"] == 2
    assert len(state["decisions"]) == 3 and state["decisions"][-1]["legal"] == []
    json.dumps(state)  # still serialisable


def test_a_decision_without_a_snapshot_still_fills_the_decision_panel(run: Path) -> None:
    (run / "battle-2" / "decisions" / "0010-action.txt").unlink()
    state = RunReader(run).state(battle=2)
    assert state["decision"]["label"] == "Close Combat" and state["foes"] == [] and state["opponent"]["name"] is None
    assert state["active"]["name"] == "Lucario"  # ours comes from the game's request
    assert "no tracked battle snapshot" in state["battle_state"]["battle_object"]


def test_newest_settled_frame_is_shown(tmp_path: Path) -> None:
    assert latest_frame(tmp_path) is None
    frames = tmp_path / "frames"
    frames.mkdir()
    assert latest_frame(tmp_path) is None
    (frames / "00001-poll-lobby.jpg").write_bytes(b"jpeg")
    first = latest_frame(tmp_path)
    assert first is not None and first[1].name == "00001-poll-lobby.jpg"
    (frames / "00002-poll-moves.jpg").write_bytes(b"")  # being written: empty
    (frames / "notes.txt").write_text("not a frame", encoding="utf-8")
    again = latest_frame(tmp_path)
    assert again is not None and again[1].name == "00001-poll-lobby.jpg"

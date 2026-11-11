"""`pcsa watch` panels of the new neural player: the decision matrix, overrides, win rates and the battle log."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pcsa.common import paths
from pcsa.run.watch import ASSETS, STATIC, RunReader
from pcsa.run.watch_decisions import notes_view
from pcsa.run.watch_events import EventScan
from pcsa.run.watch_log import log_view, parse, side_of
from pcsa.run.watch_matrix import matrix_view, pretty, row_of_order

FIXTURE = paths.repo_root() / "tests" / "fixtures" / "watch_run"
DECISIONS = FIXTURE / "battle-3" / "decisions"


@pytest.fixture
def run(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURE, tmp_path / "20261109-205551-run"))


def notes_of(name: str) -> dict:
    return json.loads((DECISIONS / name).read_text(encoding="utf-8"))["notes"]


def drop(run: Path, *sequences: int) -> None:
    for sequence in sequences:
        for path in (run / "battle-3" / "decisions").glob(f"{sequence:04d}-*"):
            path.unlink()


# The matrix -----------------------------------------------------------------------------------------


def test_latest_decision_carries_the_matrix_with_written_names(run: Path) -> None:
    state = RunReader(run).state()
    decision = state["decision"]
    assert state["battle"]["index"] == 3 and state["turn"] == 4 and decision["label"] == "Extreme Speed"
    notes = decision["notes"]
    assert notes["rule"] == "neural" and notes["fallback"] is False and notes["v"] == 0.52 and notes["ms"] == 12.1
    assert notes["override"] is None and notes["network"] is None
    assert notes["rest"] == [{"key": "decisions", "value": "3"}]  # the matrix is not a generic note
    matrix = notes["matrix"]
    assert matrix["rows"][:4] == ["crunch", "closecombat", "meteormash", "extremespeed"]
    assert matrix["row_labels"][3] == "Extreme Speed" and matrix["row_labels"][5] == "Close Combat + Mega"
    assert matrix["row_labels"][8:] == ["Garchomp", "Mimikyu", ""]
    assert matrix["col_labels"][0] == "Earthquake" and matrix["col_labels"][5] == "Rotom-Wash"  # the record's names
    assert matrix["col_labels"][8] == "" and matrix["col_kind"] == ["move"] * 4 + ["switch"] * 5
    assert matrix["shown"] == [True] + [False] * 8 and matrix["legal"][4:8] == [False] * 4
    assert len(matrix["cells"]) == 11 and all(len(line) == 9 for line in matrix["cells"])
    assert abs(sum(matrix["p_cols"]) - 1) < 0.01 and abs(sum(matrix["probs"]) - 1) < 0.01
    assert matrix["chosen"] == matrix["played"] == 3 and matrix["mega_p"] == 0.71
    assert state["foes"][1]["name"] == "Tyranitar" and state["foes"][1]["moves"] == ["Earthquake"]  # revealed only
    assert state["field"]["weather"]["name"] == "Sand" and state["your_team"][0]["boosts"] == "def-1 spd-1"


def test_decision_list_leaves_the_matrix_out_and_keeps_the_values(run: Path) -> None:
    state = RunReader(run).state()
    rows = state["decisions"]
    assert [r["sequence"] for r in rows] == [0, 1, 2, 3]
    assert all(r["notes"]["matrix"] is None for r in rows) and state["decision"]["notes"]["matrix"] is not None
    assert [r["notes"]["v"] for r in rows] == [0.18, 0.37, None, 0.52]
    assert rows[1]["notes"]["override"] == "priority_ko" and rows[2]["notes"]["fallback"] is True
    assert rows[0]["timing"] == {"turn_s": 10.4, "log_s": 4.1, "tap_s": 10.4}


def test_the_decision_event_does_not_carry_the_matrix() -> None:
    events = [json.loads(raw) for raw in (FIXTURE / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    decisions = [e for e in events if e["event"] == "decision" and e.get("battle") == 3 and e["kind"] == "action"]
    assert len(decisions) == 4 and all("matrix" not in e for e in decisions)
    assert decisions[1]["override"] == "priority_ko" and decisions[1]["network"] == "/choose move closecombat"
    assert decisions[2]["rule"] == "neural:fallback" and decisions[0]["decisions"] == 1


def test_an_override_marks_the_networks_row_and_the_row_played(run: Path) -> None:
    drop(run, 2, 3)
    decision = RunReader(run).state()["decision"]
    notes = decision["notes"]
    assert decision["sequence"] == 1 and decision["label"] == "Extreme Speed"
    assert notes["override"] == "priority_ko" and notes["override_why"].startswith("Extreme Speed knocks out")
    assert notes["network"] == {"order": "move closecombat", "label": "Close Combat", "kind": "move"}
    assert notes["matrix"]["chosen"] == 1 and notes["matrix"]["played"] == 3
    assert all(n["key"] not in ("override", "override_why", "network", "matrix") for n in notes["rest"])


def test_a_fallback_decision_has_no_matrix(run: Path) -> None:
    drop(run, 3)
    decision = RunReader(run).state()["decision"]
    notes = decision["notes"]
    assert decision["sequence"] == 2 and decision["label"] == "Close Combat" and len(decision["legal"]) == 6
    assert notes["fallback"] is True and notes["matrix"] is None and notes["why"].startswith("the network answered")
    assert {"key": "fallback_rule", "value": "attack"} in notes["rest"]


def test_a_mega_turn_keeps_all_eleven_rows(run: Path) -> None:
    drop(run, 1, 2, 3)
    matrix = RunReader(run).state()["decision"]["notes"]["matrix"]
    assert matrix["legal"] == [True] * 10 + [False] and matrix["chosen"] == matrix["played"] == 6
    assert matrix["row_labels"][6] == "Meteor Mash + Mega" and matrix["mega_p"] is None


def test_notes_view_passes_the_matrix_through_and_names_the_override() -> None:
    raw = notes_of("0001-action.json")
    view = notes_view(raw, {"closecombat": "Close Combat"}, "/choose move extremespeed")
    assert view is not None
    matrix = view["matrix"]
    for key in ("rows", "legal", "cols", "col_kind", "shown", "p_cols", "cells", "probs", "mega_p"):
        assert matrix[key] == raw["matrix"][key]
    assert (matrix["chosen"], matrix["played"]) == (1, 3) and matrix["row_labels"][1] == "Close Combat"
    assert view["override"] == "priority_ko" and view["network"]["label"] == "Close Combat"
    plain = notes_view(raw)  # no record names, no order: the note's own row stands
    assert plain is not None and plain["matrix"]["played"] == 1 and plain["network"]["order"] == "move closecombat"


def test_a_matrix_of_another_shape_is_left_out() -> None:
    good = notes_of("0003-action.json")["matrix"]
    assert matrix_view(good) is not None
    for broken in (
        None,
        "matrix",
        {},
        {**good, "rows": good["rows"][:10]},
        {**good, "cells": good["cells"][:5]},
        {**good, "cells": [line[:8] for line in good["cells"]]},
        {**good, "probs": ["high"] * 11},
        {**good, "cols": [1] * 9},
    ):
        assert matrix_view(broken) is None
        view = notes_view({"rule": "neural", "v": 0.1, "matrix": broken})
        assert view is not None and view["matrix"] is None and view["v"] == 0.1
    loose = matrix_view({**good, "chosen": 99, "legal": "x", "shown": None, "col_kind": 4, "mega_p": "no"})
    assert loose is not None and loose["chosen"] is None and loose["played"] is None and loose["mega_p"] is None
    assert loose["legal"] == [True] * 11 and loose["shown"] == [False] * 9 and loose["col_kind"][4] == "switch"


def test_rows_of_orders_and_display_names() -> None:
    rows = notes_of("0000-action.json")["matrix"]["rows"]
    assert row_of_order("/choose move meteormash mega", rows) == 6 and row_of_order("move crunch", rows) == 0
    assert row_of_order("/choose switch Mimikyu", rows) == 9 and row_of_order("switch Garchomp-Mega", rows) == 8
    assert row_of_order("move surf", rows) is None and row_of_order("default", rows) is None
    assert row_of_order(None, rows) is None and row_of_order("switch Lucario", rows) is None
    assert pretty("rotomwash", {"rotomwash": "Rotom-Wash"}) == "Rotom-Wash" and pretty("") == ""
    assert pretty("notathingatall") == "Notathingatall"  # an id nothing names: shown with a capital


# Win rates ------------------------------------------------------------------------------------------


def line(**event) -> str:
    return json.dumps({"ts": "2026-11-10T02:00:00.000+00:00", "level": "INFO", **event}) + "\n"


def test_win_rates_over_the_run_and_the_last_ten(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    scan = EventScan(path)
    scan.refresh()
    assert scan.winrate_view() == {
        "battles": 0,
        "wins": 0,
        "rate": None,
        "recent": {"battles": 0, "wins": 0, "rate": None},
        "recent_size": 10,
        "streak": 0,
    }
    outcomes = ["win", "loss", "loss", "loss", "win", "opponent_forfeit", "draw", "win", "loss", "win", "win", "win"]
    text = "".join(line(event="battle_end", battle=i + 1, outcome=o) for i, o in enumerate(outcomes))
    path.write_text(text + line(event="battle_start", battle=13), encoding="utf-8")  # battle 13 is being played
    scan.refresh()
    view = scan.winrate_view()
    assert (view["battles"], view["wins"], view["rate"]) == (12, 7, 0.5833)
    assert view["recent"] == {"battles": 10, "wins": 6, "rate": 0.6} and view["streak"] == 3
    path.write_text(text + line(event="battle_end", battle=13, outcome="loss"), encoding="utf-8")
    scan = EventScan(path)
    scan.refresh()
    assert scan.winrate_view()["streak"] == -1 and scan.winrate_view()["recent"]["wins"] == 6


def test_fixture_run_win_rate(run: Path) -> None:
    state = RunReader(run).state()
    assert state["winrate"]["battles"] == 2 and state["winrate"]["rate"] == 0.0 and state["winrate"]["streak"] == -2
    assert state["session_record"] == {"wins": 0, "losses": 2, "draws": 0, "forfeits": 0}


# The battle log -------------------------------------------------------------------------------------


def test_standard_log_is_parsed_into_turns_with_sides() -> None:
    log = log_view(FIXTURE / "battle-3")
    assert log["source"] == "battle_log.txt" and log["format"] == "standard" and log["raw"] == ""
    assert log["header"]["lead_observed"] == "Lucario" and log["header"]["outcome"] == "unknown"
    assert [t["turn"] for t in log["turns"]] == [1, 2, 3] and log["notes"] == ["none"]
    first = log["turns"][0]["lines"]
    assert first[0] == {"kind": "event", "category": "", "text": "sent out Dragonite!", "side": "theirs"}
    assert first[1]["side"] == "ours" and first[1]["text"] == "Go! Lucario!"
    assert first[3]["category"] == "Mega Evolution" and first[3]["side"] == "ours"
    assert first[5] == {
        "kind": "event",
        "category": "HP Lost",
        "text": "Dragonite’s HP: 100% → 38%",
        "side": "theirs",
    }
    assert first[6]["category"] == "Move" and first[6]["side"] == "theirs"
    assert first[-1] == {
        "kind": "action",
        "category": "",
        "text": "chose move 3 (Meteor Mash) with Mega Evolution",
        "side": "ours",
    }
    third = {line["text"]: line["side"] for line in log["turns"][2]["lines"]}
    assert third["Lucario’s HP: 147 → 31"] == "ours" and third["The sandstorm is raging."] is None
    assert third["Lucario’s Defense fell!"] == "ours" and third["Tyranitar’s HP: 100% → 22%"] == "theirs"


def test_log_of_a_finished_battle_keeps_notes_and_uncollected_turns() -> None:
    log = log_view(FIXTURE / "battle-2")
    assert [t["turn"] for t in log["turns"]] == [1, 2, 3, 4, 5, 6, 7, 8]
    last = log["turns"][-1]["lines"]
    assert last[0]["kind"] == "note" and last[0]["text"].startswith("(no log for this turn")
    assert last[1]["kind"] == "action" and len(log["notes"]) == 2 and log["notes"][0].startswith("failure:")
    sides = {line["text"]: line["side"] for turn in log["turns"] for line in turn["lines"]}
    assert sides["Indeedee’s Psychic Surge"] == "theirs"  # theirs only at the preview
    assert sides["Kingambit’s Leftovers"] is None  # both teams have a Kingambit: the text does not tell


def test_plain_and_raw_logs(tmp_path: Path) -> None:
    assert log_view(None)["turns"] == [] and log_view(tmp_path / "nowhere")["source"] is None
    (tmp_path / "log.txt").write_text(
        "Turn1:\nGo! Garchomp!\nThe opposing Raichu used Fake Out!\n\nTurn2:\nIt started to rain!\n", encoding="utf-8"
    )
    log = log_view(tmp_path)
    assert log["source"] == "log.txt" and log["format"] == "plain" and [t["turn"] for t in log["turns"]] == [1, 2]
    assert [line["side"] for line in log["turns"][0]["lines"]] == ["ours", "theirs"]
    assert log["turns"][1]["lines"] == [
        {"kind": "event", "category": "", "text": "It started to rain!", "side": None}
    ]
    (tmp_path / "log.txt").write_text("something else entirely\n", encoding="utf-8")
    raw = log_view(tmp_path)
    assert raw["format"] == "raw" and raw["turns"] == [] and raw["raw"].strip() == "something else entirely"
    (tmp_path / "battle_log.txt").write_bytes(b"=== Turn 1 ===\n[Move] \xff\xfe used Tackle!\n=== Turn x ===\n")
    odd = log_view(tmp_path)  # battle_log.txt wins; bad bytes and a broken header do not break it
    assert odd["source"] == "battle_log.txt" and len(odd["turns"]) == 1 and len(odd["turns"][0]["lines"]) == 1
    assert parse("") == {"format": None, "header": {}, "turns": [], "notes": [], "raw": ""}
    assert side_of("Fainted", "Garchomp fainted!", set(), set()) == "ours"
    assert side_of("", "The battlefield got weird!", {"garchomp"}, {"raichu"}) is None


def test_log_follows_the_battle_shown(run: Path) -> None:
    reader = RunReader(run)
    assert reader.state()["log"]["header"]["battle_number"] == "3"
    assert reader.state(battle=1)["log"]["header"]["battle_number"] == "1"
    (run / "battle-3" / "battle_log.txt").unlink()
    assert reader.state()["log"] == {
        "source": None,
        "format": None,
        "header": {},
        "turns": [],
        "notes": [],
        "raw": "",
    }


# The page -------------------------------------------------------------------------------------------


def test_every_script_of_the_page_is_served() -> None:
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    for path in ASSETS:
        assert (STATIC / path.lstrip("/")).is_file() and path in page
    matrix = (STATIC / "matrix.js").read_text(encoding="utf-8")
    assert "mxSeen" in matrix and "is-playing" in matrix  # a decision animates once, by its battle and sequence
    assert "guess" not in page.lower() and "assumed" not in page.lower()  # nothing about guessed opposing builds

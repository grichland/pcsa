"""`pcsa watch` panels of the neural player: the decision matrix folded from the network's 13 columns, win rates and
the battle log."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pcsa.common import paths
from pcsa.run.watch import ASSETS, STATIC, RunReader
from pcsa.run.watch_decisions import notes_view, policy_numbers
from pcsa.run.watch_events import EventScan
from pcsa.run.watch_log import log_view, parse, side_of
from pcsa.run.watch_matrix import fold, matrix_view, row_of_order

FIXTURE = paths.repo_root() / "tests" / "fixtures" / "watch_run"
RECORDS = sorted(FIXTURE.glob("battle-*/decisions/*.json"))


@pytest.fixture
def run(tmp_path: Path) -> Path:
    return Path(shutil.copytree(FIXTURE, tmp_path / "20261118-162900-neural-148"))


def record_of(name: str) -> dict:
    return json.loads((FIXTURE / name).read_text(encoding="utf-8"))


def net_of(name: str) -> dict:
    return record_of(name)["notes"]["cg"]["net"]


def small_net() -> dict:
    """A net whose fold is easy to check by hand: one move with a Mega share, one switch."""
    p_opp = [0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    matrix = [[1.0, 0, 0, 0, -1.0, 0, 0, 0, 0.5, 0, 0, 0, 0] for _ in range(11)]
    return {
        "legal": [0, 1, 8], "pi": [0.6, 0.3] + [0.0] * 6 + [0.1, 0.0, 0.0], "tau": 0.05, "value": 0.1,
        "rows": ["earthquake", "swordsdance", None, None] + [None] * 4 + ["garchomp", "rotomwash", None],
        "cols": ["flamethrower", "", None, None, "flamethrower", "", None, None, "gholdengo", None, None, None, None],
        "p_opp": p_opp, "matrix": matrix, "score": [0.15] * 11,
    }  # fmt: skip


# The fold (D8) --------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_the_fold_keeps_every_rows_score(path: Path) -> None:
    net = json.loads(path.read_text(encoding="utf-8"))["notes"]["cg"]["net"]
    view = matrix_view(net)
    assert view is not None and len(view["cells"]) == 11 and all(len(line) == 9 for line in view["cells"])
    for row, line in enumerate(view["cells"]):
        folded = sum(p * x for p, x in zip(view["p_cols"], line, strict=True))
        assert folded == pytest.approx(net["score"][row], abs=1e-3)
    assert sum(view["p_cols"]) == pytest.approx(sum(net["p_opp"]))
    assert view["mega_p"] == pytest.approx(sum(net["p_opp"][4:8]))


def test_a_move_and_its_mega_variant_fold_into_one_column() -> None:
    view = matrix_view(small_net())
    assert view is not None
    assert view["p_cols"][0] == pytest.approx(0.5) and view["cells"][0][0] == pytest.approx((0.2 - 0.3) / 0.5)
    assert view["cells"][0][1] == 0 and view["p_cols"][1] == 0  # no chance at all: the plain move's cell
    assert view["p_cols"][4] == 0.5 and view["cells"][0][4] == 0.5 and view["mega_p"] == pytest.approx(0.3)
    assert view["cols"][:2] == ["flamethrower", "unseen"] and view["col_labels"][:2] == [
        "Flamethrower",
        "Unrevealed move",
    ]
    assert view["shown"][:4] == [True, False, False, False] and view["col_kind"] == ["move"] * 4 + ["switch"] * 5
    assert view["col_legal"] == [True, True, False, False, True, False, False, False, False]
    assert view["legal"] == [True, True] + [False] * 6 + [True, False, False]
    p_cols, cells, mega = fold([0.0] * 13, [[2.0] * 13])
    assert p_cols == [0.0] * 9 and cells == [[2.0] * 9] and mega == 0


def test_rows_names_and_the_row_played() -> None:
    view = matrix_view(net_of("battle-1/decisions/0002-action.json"), order="/choose move meteormash mega", action=6)
    assert view is not None
    assert view["rows"][6] == "meteormash + mega" and view["row_labels"][6] == "Meteor Mash + Mega"
    assert view["row_labels"][8:] == ["Hippowdon", "Metagross", "Primarina"]
    assert view["chosen"] == 6 and view["played"] == 6 and view["tau"] is not None and view["blend"] is None
    switch = matrix_view(net_of("battle-1/decisions/0006-action.json"), order="/choose switch Primarina", action=10)
    assert switch is not None and switch["played"] == 10 and switch["chosen"] == 10
    assert switch["rows"][9] == "metagrossmega" and switch["row_labels"][9] == "Metagross"  # a Mega switch target
    assert (
        row_of_order("move unknownmove", switch["rows"]) is None and row_of_order("default", switch["rows"]) is None
    )


def test_the_decision_notes_carry_the_matrix_and_the_policy_numbers() -> None:
    record = record_of("battle-1/decisions/0002-action.json")
    notes = notes_view(record["notes"], None, record["order"], record)
    assert notes is not None and notes["matrix"]["played"] == 6 and notes["rule"] == "neural"
    pi = record["notes"]["cg"]["net"]["pi"]
    assert notes["p"] == pi[6] and 0 <= notes["h"] <= 1
    assert {r["key"] for r in notes["rest"]} >= {"cg_ms", "sent", "attempt"}
    assert policy_numbers([0.5, 0.5, 0.0], 0) == (0.5, 1.0) and policy_numbers([1.0, 0.0], 0) == (1.0, 0.0)
    assert policy_numbers("x", 0) == (None, None) and policy_numbers([0.2, "a"], 0) == (None, None)


def test_a_model_fallback_has_no_matrix_and_says_who_played() -> None:
    cg = {
        "action": 2,
        "player": "neural:x.pt",
        "model": {"fallback": {"reason": "timeout", "player": "tower"}, "ms": 2001.0},
    }
    notes = notes_view({"cg_ms": 3.0, "cg": cg})
    assert notes is not None and notes["fallback"] is True and notes["why"] == "timeout" and notes["matrix"] is None
    assert {"key": "fallback", "value": "tower"} in notes["rest"] and notes["v"] is None and notes["ms"] == 2001.0
    builtin = notes_view({"cg": {"player": "tower", "model": {"fallback": False, "value": 0.3}}})
    assert builtin is not None and builtin["rule"] == "tower" and builtin["fallback"] is False and builtin["v"] == 0.3


def test_a_net_of_another_shape_is_left_out() -> None:
    good = small_net()
    assert matrix_view(good) is not None
    for broken in (
        {**good, "pi": good["pi"][:10]},
        {**good, "p_opp": good["p_opp"][:9]},
        {**good, "matrix": good["matrix"][:10]},
        {**good, "matrix": [[*line[:12], "x"] for line in good["matrix"]]},
        {**good, "rows": good["rows"][:10]},
        {**good, "cols": [3] * 13},
        {**good, "pi": [float("nan")] * 11},
        None,
        [1, 2],
    ):
        assert matrix_view(broken) is None
    assert matrix_view({**good, "tau": -1})["tau"] is None  # type: ignore[index]


def test_the_decision_list_leaves_the_matrix_out(run: Path) -> None:
    state = RunReader(run).state(battle=3)
    assert state["decision"]["notes"]["matrix"] is not None
    assert all(r["notes"]["matrix"] is None for r in state["decisions"])
    assert [r["notes"]["v"] for r in state["decisions"]] == [0.6028, 0.7276, -0.4769, -0.6285]


# Win rates ------------------------------------------------------------------------------------------


def line(**event) -> str:
    return json.dumps({"ts": "2026-11-18T22:00:00.000+00:00", "level": "INFO", **event}) + "\n"


def test_win_rates_over_the_run_and_the_last_ten(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    outcomes = ["win", "loss", "loss", "loss", "win", "opponent_forfeit", "draw", "win", "loss", "win", "win", "win"]
    path.write_text("".join(line(event="battle_end", battle=i + 1, outcome=o) for i, o in enumerate(outcomes)))
    scan = EventScan(path)
    scan.refresh()
    view = scan.winrate_view()
    assert (view["battles"], view["wins"], view["rate"]) == (12, 7, 0.5833)
    assert view["recent"] == {"battles": 10, "wins": 6, "rate": 0.6} and view["streak"] == 3


def test_fixture_run_win_rate(run: Path) -> None:
    state = RunReader(run).state()
    assert (
        state["winrate"]["battles"] == 3 and state["winrate"]["rate"] == 0.3333 and state["winrate"]["streak"] == -2
    )


# The battle log -------------------------------------------------------------------------------------


def test_standard_log_is_parsed_into_turns_with_sides() -> None:
    log = log_view(FIXTURE / "battle-1")
    assert log["source"] == "battle_log.txt" and log["format"] == "standard" and log["raw"] == ""
    assert log["header"]["lead_observed"] == "Hippowdon" and log["header"]["outcome"] == "win"
    assert [t["turn"] for t in log["turns"]][:3] == [1, 2, 3]
    first = log["turns"][0]["lines"]
    assert first[0] == {"kind": "event", "category": "", "text": "sent out Meganium!", "side": "theirs"}
    assert first[1]["side"] == "ours" and first[1]["text"] == "Go! Hippowdon!"
    assert any(x["category"] == "Mega Evolution" and x["side"] == "theirs" for x in first)
    assert any(x["kind"] == "action" for x in first)


def test_plain_and_raw_logs(tmp_path: Path) -> None:
    assert log_view(None)["turns"] == [] and log_view(tmp_path / "nowhere")["source"] is None
    (tmp_path / "log.txt").write_text(
        "Turn1:\nGo! Garchomp!\nThe opposing Raichu used Fake Out!\n\nTurn2:\nIt started to rain!\n", encoding="utf-8"
    )
    log = log_view(tmp_path)
    assert log["source"] == "log.txt" and log["format"] == "plain" and [t["turn"] for t in log["turns"]] == [1, 2]
    assert [x["side"] for x in log["turns"][0]["lines"]] == ["ours", "theirs"]
    (tmp_path / "log.txt").write_text("something else entirely\n", encoding="utf-8")
    raw = log_view(tmp_path)
    assert raw["format"] == "raw" and raw["turns"] == [] and raw["raw"].strip() == "something else entirely"
    assert parse("") == {"format": None, "header": {}, "turns": [], "notes": [], "raw": ""}
    assert side_of("Fainted", "Garchomp fainted!", set(), set()) == "ours"


def test_log_follows_the_battle_shown(run: Path) -> None:
    reader = RunReader(run)
    assert reader.state()["log"]["header"]["battle_number"] == "3"
    assert reader.state(battle=1)["log"]["header"]["battle_number"] == "1"
    (run / "battle-3" / "battle_log.txt").unlink()
    assert reader.state()["log"]["source"] is None


# The page -------------------------------------------------------------------------------------------


def test_every_script_of_the_page_is_served() -> None:
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    for path in ASSETS:
        assert (STATIC / path.lstrip("/")).is_file() and path in page
    assert "guess" not in page.lower() and "assumed" not in page.lower()


def test_the_page_reads_the_belief_and_the_new_badge() -> None:
    panels = (STATIC / "panels.js").read_text(encoding="utf-8")
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    style = (STATIC / "style.css").read_text(encoding="utf-8")
    matrix = (STATIC / "matrix.js").read_text(encoding="utf-8")
    for marker in ("m.belief", "unseen_belief", "is-believed", "brought", "lead", "belief-key", "state.legacy"):
        assert marker in panels
    assert "player:" in app and '.badge[data-degrade="player"]' in style and ".chip.is-believed" in style
    assert "predates the Champions Gym port" in matrix and "state.legacy" in matrix
    for label in ('"SCORE"', '"FOR THEM"', "WOULD PICK", "pcsa-mx-top", "pcsa-mx-opp", "m.tau", "col_legal"):
        assert label in matrix

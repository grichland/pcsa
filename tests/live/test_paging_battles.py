"""A battle on the fake device whose log heading jumps past turns with no action screen of ours. Back-paging
is on in config/agent.yaml; the real LogReader pages back through fake_log's book of recorded page frames (the
ones test_log_pages uses), and the missed turns' rows reach the events in turn order."""

from __future__ import annotations

from pathlib import Path

import pytest

from pcsa.common.clock import FakeClock
from pcsa.common.config import Config
from pcsa.common.types import Screen
from pcsa.live.fake import Step
from pcsa.live.fake_log import FakeLogBook, LogPage
from tests.live.battle_rig import MOVE_TAPS, Script, end_of_battle, live_problems, lobby_to_team_select, run_battles
from tests.live.test_log_pages import MANIFEST, plain
from tests.live.turn_rig import FakeLogs
from tests.perception.test_log_frames import image

pytestmark = pytest.mark.ocr
SEND_OUTS = ["Kai sent out Garchomp!", "Go! Gholdengo!"]
PAGES = {t: MANIFEST[str(t)]["rows"] for t in range(2, 7)}


def same(text: str) -> str:
    """Rows compared as test_log_pages does, and with either apostrophe (the manifest has both)."""
    return plain(text).replace("’", "'")


@pytest.fixture(scope="module")
def ocr():
    from pcsa.common.config import load_config
    from pcsa.phone.ocr import RapidOcrReader
    from tests.conftest import REPO

    return RapidOcrReader(load_config(REPO / "config" / "agent.yaml", environ={}).agent.ocr)


class FirstVisitScripted:
    """The first action screen's log (turn 1, send-outs only) is scripted: no recorded frames of it exist."""

    def __init__(self, real) -> None:
        self.real, self.first = real, FakeLogs([{1: SEND_OUTS}])

    def collect_log(self, b, visit, deadline):
        return (self.first if visit == 1 else self.real).collect_log(b, visit, deadline)


def book(clock: FakeClock, current: int) -> FakeLogBook:
    pages = {t: LogPage([image(f"pages/{n}") for n in MANIFEST[str(t)]["frames"]]) for t in range(2, current + 1)}
    return FakeLogBook(pages, current, clock)


def action(log_book: FakeLogBook | None = None) -> list[Step]:
    return [
        Step(Screen.ACTION, {"move_time": 45}, accept=MOVE_TAPS, advance_on=("move_rows",), log_book=log_book),
        Step(Screen.WAIT, accept=("move_rows",), advance_on=(), after_s=12),
    ]


def test_skipped_turns_are_paged_back_into_the_events(config: Config, tmp_path: Path, monkeypatch, ocr):
    assert config.agent.log.paging.enabled  # shipped on
    clock = FakeClock(1000.0)
    books = [book(clock, 2), book(clock, 4), book(clock, 6)]  # headings 2, then 4 (3 skipped), then 6 (5 skipped)
    steps = lobby_to_team_select() + action() + [s for b in books for s in action(b)] + end_of_battle("win")
    played = run_battles(config, steps, tmp_path, monkeypatch, player=Script((0, 3, 1)), reader=ocr,
                         logs=FirstVisitScripted, clock=clock)  # fmt: skip
    assert played.ending == "completed" and played.state.outcomes == ["win"] and played.state.restarts == 0
    assert [b.taps for b in books] == [
        ["view_log", "log_back"],
        ["view_log", "log_prev", "log_next", "log_back"],
        ["view_log", "log_prev", "log_next", "log_back"],
    ]
    last = played.requests()[-1]
    by_turn: dict[int, list[str]] = {}
    for row in last.rows:
        by_turn.setdefault(row.turn, []).append(same(row.text))
    assert by_turn == {t: [same(r) for r in rows] for t, rows in {1: SEND_OUTS, **PAGES}.items()}
    assert [r.row_id for r in last.rows] == [f"{r.turn}-{r.index}" for r in last.rows]
    assert [(r.turn, r.index) for r in last.rows] == sorted((r.turn, r.index) for r in last.rows)

    kinds = [e.get("k") for e in last.events[1:]]
    turns = [i for i, k in enumerate(kinds) if k == "turn"]
    assert [last.events[1 + i]["n"] for i in turns] == [1, 2, 3, 4, 5, 6, 7]  # 7: the turn being chosen
    ends = [*turns[1:], len(kinds)]
    per_turn = [len([k for k in kinds[a + 1 : b] if k != "gap"]) for a, b in zip(turns, ends, strict=True)]
    assert per_turn == [len(by_turn[t]) for t in range(1, 7)] + [0]  # each turn's rows sit after its own turn event
    assert "gap" not in kinds and live_problems(last.events) == []

    records = played.records()
    assert [r["turn"] for r in records] == [1, 3, 5, 7]
    assert [r["peek"] for r in records] == ["skipped"] * 4  # a turn read by paging back is no reason to look
    assert played.device.rejected == []
    assert all(r["log_quality"] == "full" for r in records[1:])
    coverage = played.battle(1)["coverage"]
    assert sorted(n["turn"] for n in coverage if n["kind"] == "back_paged") == [3, 5]
    assert not [n for n in coverage if n["kind"] == "no_own_action_screen"]
    frames = played.recorder.dir / "battle-1" / "log-frames"
    assert (frames / "visit03-page3-0.png").is_file() and (frames / "visit04-page5-0.png").is_file()


def test_with_paging_off_the_skipped_turns_are_gaps(tmp_path: Path, monkeypatch, ocr) -> None:
    from pcsa.common.config import load_config
    from tests.conftest import REPO

    config = load_config(REPO / "config" / "agent.yaml", ["log.paging.enabled=false"], environ={})
    clock = FakeClock(1000.0)
    books = [book(clock, 2), book(clock, 4)]
    steps = lobby_to_team_select() + action() + [s for b in books for s in action(b)] + end_of_battle("win")
    played = run_battles(config, steps, tmp_path, monkeypatch, player=Script((0, 3, 1)), reader=ocr,
                         logs=FirstVisitScripted, clock=clock)  # fmt: skip
    assert played.state.outcomes == ["win"] and [b.taps for b in books] == [["view_log", "log_back"]] * 2
    last = played.requests()[-1]
    assert sorted({r.turn for r in last.rows}) == [1, 2, 4]
    assert {"k": "gap", "what": "turn", "n": 3} in last.events and live_problems(last.events) == []
    assert [r["peek"] for r in played.records()][-1] == "log_skipped_turn"
    coverage = played.battle(1)["coverage"]
    assert [n["turn"] for n in coverage if n["kind"] == "no_own_action_screen"] == [3]

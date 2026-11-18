"""Back-paging on the fake log book: real recorded pages per turn, the real collector, OCR and merge."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.config import Config, Viewport
from pcsa.common.types import Screen
from pcsa.live.fake import ScriptedDevice, Step
from pcsa.live.fake_log import FakeLogBook, LogPage
from pcsa.live.log_reader import LogReader, LogVisit
from pcsa.live.taps import Executor, TapPolicy
from pcsa.phone.log_merge import BattleLog
from pcsa.phone.profile import DeviceProfile
from pcsa.run.battle_state import BattleRecord
from pcsa.run.recorder import Recorder
from tests.perception.test_log_frames import FRAMES, anonymous, image

pytestmark = pytest.mark.ocr
MANIFEST = json.loads((FRAMES / "pages.json").read_text(encoding="utf-8"))["pages"]
ROOMY_S = 90.0


@pytest.fixture
def config() -> Config:
    """agent.yaml with back-paging on, whatever the shipped default."""
    from pcsa.common.config import load_config
    from tests.conftest import REPO

    return load_config(REPO / "config" / "agent.yaml", ["log.paging.enabled=true"], environ={})


@pytest.fixture(scope="module")
def ocr():
    from pcsa.common.config import load_config
    from pcsa.phone.ocr import RapidOcrReader
    from tests.conftest import REPO

    return RapidOcrReader(load_config(REPO / "config" / "agent.yaml", environ={}).agent.ocr)


def rows(turn: int) -> list[str]:
    return MANIFEST[str(turn)]["rows"]


def book(clock: FakeClock, current: int, oldest: int, **faults) -> FakeLogBook:
    opens_at = faults.pop("opens_at", {})
    pages = {
        t: LogPage([image(f"pages/{n}") for n in MANIFEST[str(t)]["frames"]], opens_at.get(t, "bottom"))
        for t in range(oldest, current + 1)
    }
    return FakeLogBook(pages, current, clock, **faults)


class Rig:
    def __init__(self, config: Config, tmp_path: Path, ocr, log_book: FakeLogBook, last_turn: int) -> None:
        self.clock = log_book.clock
        self.book = log_book
        device = ScriptedDevice([Step(Screen.ACTION, log_book=log_book)], config.screen, self.clock)
        executor = Executor(
            config.screen,
            Viewport.of_frame(2340, 1080),
            config.agent.taps,
            TapPolicy(config.screen),
            random.Random(0),
        )
        profile = DeviceProfile.full_frame("fake", 2340, 1080)
        phone = SimpleNamespace(executor=executor, device=device, reader=ocr, profile=profile)
        self.recorder = Recorder(tmp_path / "run")
        self.reader = LogReader(phone, self.recorder, config.agent, self.clock)
        self.battle = BattleRecord(1, "b", (), BattleLog("b"), last_turn=last_turn, action_visits=3)
        self.tmp = tmp_path

    def visit(self, seconds: float = ROOMY_S) -> LogVisit:
        info = self.reader.collect_log(self.battle, 4, Deadline.after(self.clock, seconds))
        self.recorder.close()
        return info

    def notes(self, kind: str) -> list[int]:
        return [n["turn"] for n in self.battle.log.coverage if n["kind"] == kind]

    def saved(self) -> tuple[list[dict], dict, dict]:
        out = self.tmp / "saved"
        self.battle.log.save(out)
        lines = [json.loads(x) for x in (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
        return lines, *(
            json.loads((out / n).read_text(encoding="utf-8")) for n in ("coverage.json", "ocr_audit.json")
        )


def plain(text: str) -> str:
    """The recorded run wrote "It doesn't affect X.." where the strict wording check now writes "..."."""
    return re.sub(r"\.{2,}$", "...", anonymous(text))


def same_rows(log: BattleLog, turn: int) -> bool:
    return [plain(r) for r in log.turns.get(turn, [])] == [plain(r) for r in rows(turn)]


def test_a_jump_of_one_turn_reads_the_skipped_page_under_its_turn(config, tmp_path, ocr) -> None:
    rig = Rig(config, tmp_path, ocr, book(FakeClock(), 6, 5), last_turn=4)
    info = rig.visit()
    assert rig.book.taps == ["view_log", "log_prev", "log_next", "log_back"]
    assert info.turn == 6 and info.back_paged == [5] and info.paging == "done" and info.complete
    assert same_rows(rig.battle.log, 5) and same_rows(rig.battle.log, 6)
    assert rig.notes("no_own_action_screen") == [] and rig.notes("back_paged") == [5]
    lines, coverage, audit = rig.saved()
    assert {r["turn"] for r in lines if r.get("back_paged")} == {5}
    assert coverage["back_paged_turns"] == [5]
    assert [(v["turn"], v.get("page")) for v in audit["visits"]] == [(5, "back"), (6, None)]
    assert set(audit["visits"][0]["view_turns"].values()) == {5}
    assert all("-page5-" in s for s in audit["visits"][0]["view_turns"])
    assert audit["visits"][1]["paging"]["returned"] is True
    assert (rig.tmp / "run" / "battle-1" / "log-frames" / "visit04-page5-0.png").is_file()


def test_a_jump_of_three_turns_reads_them_oldest_first_and_returns(config, tmp_path, ocr) -> None:
    rig = Rig(config, tmp_path, ocr, book(FakeClock(), 6, 3), last_turn=2)
    info = rig.visit()
    assert rig.book.taps == ["view_log", *["log_prev"] * 3, *["log_next"] * 3, "log_back"]
    assert info.back_paged == [3, 4, 5] and rig.battle.last_turn == 6
    assert all(same_rows(rig.battle.log, t) for t in (3, 4, 5, 6))
    assert rig.notes("no_own_action_screen") == []
    _, _, audit = rig.saved()
    assert [v["turn"] for v in audit["visits"]] == [3, 4, 5, 6]


def test_a_page_that_opens_at_its_top_is_read_downwards(config, tmp_path, ocr) -> None:
    rig = Rig(config, tmp_path, ocr, book(FakeClock(), 5, 4, opens_at={4: "top"}), last_turn=3)
    info = rig.visit()
    assert info.back_paged == [4] and same_rows(rig.battle.log, 4)
    assert rig.battle.log.failures == []


def test_a_deadline_cuts_paging_short_and_leaves_the_rest_as_gaps(config, tmp_path, ocr) -> None:
    rig = Rig(config, tmp_path, ocr, book(FakeClock(), 6, 3), last_turn=2)
    info = rig.visit(seconds=30.0)
    assert info.paging == "deadline" and info.seconds <= 30.0
    assert 0 < len(info.back_paged) < 3 and all(same_rows(rig.battle.log, t) for t in info.back_paged)
    assert rig.notes("no_own_action_screen") == [t for t in (3, 4, 5) if t not in info.back_paged]
    assert rig.battle.log.visits[-1]["paging"]["returned"] and rig.book.taps[-1] == "log_back"


def test_a_deadline_that_fits_fewer_pages_aims_at_the_newest_missed_turns(config, tmp_path, ocr) -> None:
    rig = Rig(config, tmp_path, ocr, book(FakeClock(), 6, 3), last_turn=2)
    info = rig.visit(seconds=20.0)
    assert rig.battle.log.visits[-1]["paging"]["planned"] == [5] and info.back_paged == [5]
    assert rig.book.taps == ["view_log", "log_prev", "log_next", "log_back"]
    assert rig.notes("no_own_action_screen") == [3, 4]


def test_no_time_for_any_page_taps_no_arrow(config, tmp_path, ocr) -> None:
    rig = Rig(config, tmp_path, ocr, book(FakeClock(), 6, 3), last_turn=2)
    info = rig.visit(seconds=14.0)
    assert info.paging == "deadline" and info.back_paged == []
    assert rig.book.taps == ["view_log", "log_back"] and rig.notes("no_own_action_screen") == [3, 4, 5]


def test_an_arrow_that_does_nothing_is_retried_once(config, tmp_path, ocr) -> None:
    rig = Rig(config, tmp_path, ocr, book(FakeClock(), 6, 5, stuck_taps=1), last_turn=4)
    info = rig.visit()
    assert rig.book.taps == ["view_log", "log_prev", "log_prev", "log_next", "log_back"]
    assert info.back_paged == [5] and same_rows(rig.battle.log, 5)


def test_an_arrow_that_still_does_nothing_gives_up_and_keeps_the_gap(config, tmp_path, ocr) -> None:
    rig = Rig(config, tmp_path, ocr, book(FakeClock(), 6, 5, stuck_taps=2), last_turn=4)
    info = rig.visit()
    assert rig.book.taps == ["view_log", "log_prev", "log_prev", "log_back"]
    assert info.paging == "arrow_failed" and info.back_paged == [] and 5 not in rig.battle.log.turns
    assert rig.notes("no_own_action_screen") == [5] and same_rows(rig.battle.log, 6)


def test_a_heading_past_the_expected_turn_is_stepped_back_to_it(config, tmp_path, ocr) -> None:
    rig = Rig(config, tmp_path, ocr, book(FakeClock(), 6, 3, double_taps=1), last_turn=3)
    info = rig.visit()
    assert rig.book.taps == ["view_log", "log_prev", "log_next", "log_prev", "log_next", "log_next", "log_back"]
    assert info.back_paged == [4, 5] and all(same_rows(rig.battle.log, t) for t in (4, 5))


def test_a_heading_that_keeps_skipping_is_never_accepted(config, tmp_path, ocr) -> None:
    rig = Rig(config, tmp_path, ocr, book(FakeClock(), 6, 3, double_taps=99), last_turn=3)
    info = rig.visit()
    steps = rig.battle.log.visits[-1]["paging"]["steps"]
    assert [(s["expected"], s["heading"]) for s in steps[:2]] == [(5, 4), (5, 6)]
    assert info.paging == "arrow_failed" and info.back_paged == []
    assert not {4, 5} & set(rig.battle.log.turns) and rig.notes("no_own_action_screen") == [4, 5]

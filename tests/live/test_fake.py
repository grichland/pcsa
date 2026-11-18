from pcsa.common.clock import FakeClock
from pcsa.common.config import Config
from pcsa.common.types import Screen
from pcsa.live.fake import ScriptedDevice, Step


def test_scripted_device_advances_only_on_accepted_taps(config: Config, fake_clock: FakeClock) -> None:
    device = ScriptedDevice(
        [Step(Screen.LOBBY, accept=("begin_matchmaking",)), Step(Screen.MATCHMAKING, after_s=3), Step(Screen.LOBBY)],
        config.screen,
        fake_clock,
    )
    device.tap(100, 100)
    assert device.index == 0 and device.rejected == [(0, 100, 100)]
    device.tap(1600, 930)
    assert device.index == 1
    device.capture()
    assert device.index == 1
    fake_clock.advance(3)
    device.capture()
    assert device.index == 2 and device.step.screen == Screen.LOBBY


def test_region_lookup_handles_lists(config: Config, fake_clock: FakeClock) -> None:
    device = ScriptedDevice([Step(Screen.MOVES)], config.screen, fake_clock)
    assert device.region_at(1900, 560) == "move_rows0"
    assert (
        device.region_at(1900, 950) == "move_rows3"
    )  # begin_matchmaking overlaps; the moves screen prefers its rows
    device = ScriptedDevice([Step(Screen.ACTION)], config.screen, fake_clock)
    assert device.region_at(2094, 690) == "fight"


def test_the_fake_log_book_pages_with_the_arrows_and_injects_faults(config: Config, fake_clock: FakeClock) -> None:
    import numpy as np

    from pcsa.live.fake_log import FakeLogBook, LogPage

    pages = {t: LogPage([np.full((4, 4, 3), t * 10 + i, np.uint8) for i in range(2)]) for t in (3, 4, 5)}
    book = FakeLogBook(pages, 5, fake_clock, stuck_taps=1, double_taps=1)
    device = ScriptedDevice([Step(Screen.ACTION, log_book=book)], config.screen, fake_clock)
    assert int(device.capture().image[0, 0, 0]) == 40  # closed: a plain frame
    device.tap(250, 378)  # View Log
    assert book.open and book.turn == 5 and int(device.capture().image[0, 0, 0]) == 50
    device.drag(1300, 270, 920)  # toward older rows
    assert book.position == 1
    device.tap(995, 70)  # stuck
    assert book.turn == 5
    device.tap(995, 70)  # moves two pages
    assert book.turn == 3
    device.tap(1345, 70)
    assert book.turn == 4 and book.taps == ["view_log", "log_prev", "log_prev", "log_next"]
    device.tap(1810, 1015)  # Back
    assert not book.open and fake_clock.now() > 1000.0


def test_the_pager_plans_the_newest_missed_turns_that_fit(config: Config, fake_clock: FakeClock) -> None:
    from pcsa.common.clock import Deadline
    from pcsa.live.log_pages import Pager

    pager = Pager(None, None, None, None, fake_clock, config.agent.log.paging)  # type: ignore[arg-type]
    cfg = config.agent.log.paging
    room = 2 * 2 * cfg.step_s + 2 * cfg.page_min_s + 0.01  # turns 7 and 8 from page 9
    assert pager.plan([6, 7, 8], 9, Deadline.after(fake_clock, room)) == [7, 8]
    assert pager.plan([6, 7, 8], 9, Deadline.after(fake_clock, 100)) == [6, 7, 8]
    assert pager.plan([6, 7, 8], 9, Deadline.after(fake_clock, cfg.page_min_s)) == []


def test_missed_turns_include_a_lock_in_from_the_first_move() -> None:
    from pcsa.live.log_reader import missed_turns

    assert missed_turns(6, 9, {1: (), 6: ()}) == [7, 8]
    assert missed_turns(None, 3, {}) == [1, 2]  # nothing read yet: turns 1-2 locked, heading now Turn 3
    assert missed_turns(None, 1, {}) == []
    assert missed_turns(4, 5, {}) == []

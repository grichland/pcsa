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
    device.tap(100, 100)  # nowhere
    assert device.index == 0 and device.rejected == [(0, 100, 100)]
    device.tap(1600, 930)  # inside begin_matchmaking
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

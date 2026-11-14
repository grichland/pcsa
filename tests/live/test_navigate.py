import random

import pytest

from pcsa.common.clock import Deadline, FakeClock
from pcsa.common.config import Config, Viewport
from pcsa.common.errors import Recoverable, SafetyStop
from pcsa.common.types import Screen
from pcsa.live.fake import ScriptedDevice, ScriptedPerception, Step
from pcsa.live.navigate import Navigator
from pcsa.live.taps import BLIND_FORBIDDEN_TEXT, BLIND_TEXT_MARGIN, Executor, TapPolicy
from pcsa.phone.ocr import TextBox
from pcsa.phone.texts import Texts

BOX = {"name": "x", "center": (1400, 300), "box": (1300, 280, 1500, 320)}


def make(config: Config, clock: FakeClock, steps: list[Step]) -> tuple[Navigator, ScriptedDevice]:
    device = ScriptedDevice(steps, config.screen, clock)
    perception = ScriptedPerception(device)
    executor = Executor(
        config.screen, Viewport.of_frame(2340, 1080), config.agent.taps, TapPolicy(config.screen), random.Random(0)
    )

    def observe(tag: str):
        frame = device.capture()
        return perception.observe(frame.image, tag), frame

    return Navigator(observe, executor, device, clock, config.agent.navigator, 0.5), device


def test_title_to_lobby_through_the_menus(config: Config, fake_clock: FakeClock) -> None:
    steps = [
        Step(Screen.TITLE, {"target": BOX}, accept=("@target",)),
        Step(Screen.HOME, {"target": {**BOX, "box": (1330, 300, 1590, 520)}}, accept=("@target",)),
        Step(Screen.BATTLE_MODES, {"target": {**BOX, "box": (1280, 215, 1800, 285)}}, accept=("@target",)),
        Step(Screen.RANKED_FORMATS, {"target": {**BOX, "box": (1250, 455, 1800, 610)}}, accept=("@target",)),
        Step(Screen.TRANSITION, after_s=2.0),
        Step(Screen.LOBBY),
    ]
    navigator, device = make(config, fake_clock, steps)
    obs = navigator.reach(Screen.LOBBY, Deadline.after(fake_clock, 120))
    assert obs.screen == Screen.LOBBY
    assert device.rejected == []


def test_options_and_stray_surfaces(config: Config, fake_clock: FakeClock) -> None:
    steps = [Step(Screen.OPTIONS, accept=("continue_battling",)), Step(Screen.LOBBY)]
    navigator, device = make(config, fake_clock, steps)
    assert navigator.reach(Screen.LOBBY, Deadline.after(fake_clock, 30)).screen == Screen.LOBBY
    steps = [Step(Screen.LOG, accept=("log_back",)), Step(Screen.ACTION)]
    navigator, device = make(config, fake_clock, steps)
    obs = navigator.reach(Screen.LOBBY, Deadline.after(fake_clock, 30))
    assert obs.screen == Screen.LOG  # in-battle surfaces are returned to the caller, never navigated away from
    navigator.back_out(obs)
    assert device.step.screen == Screen.ACTION


def test_the_move_info_view_is_turned_off_before_back(config: Config, fake_clock: FakeClock) -> None:
    steps = [
        Step(Screen.MOVES, {"info_open": True}, accept=("move_info",)),
        Step(Screen.MOVES, {"info_open": False}, accept=("move_back",)),
        Step(Screen.ACTION),
    ]
    navigator, device = make(config, fake_clock, steps)
    navigator.back_out(navigator.observe("t")[0])
    assert device.step.screen == Screen.ACTION and device.rejected == []
    steps = [  # the toggle did nothing: the panel's Close
        Step(Screen.MOVES, {"info_open": True}, accept=("move_info", "info_close"), advance_on=("info_close",)),
        Step(Screen.MOVES, {"info_open": False}, accept=("move_back",)),
        Step(Screen.ACTION),
    ]
    navigator, device = make(config, fake_clock, steps)
    navigator.back_out(navigator.observe("t")[0])
    assert device.step.screen == Screen.ACTION
    assert [name for _, name in device.taps] == ["move_info", "info_close", "move_back"]


def test_unknown_screen_is_dismissed_then_gives_up(config: Config, fake_clock: FakeClock) -> None:
    steps = [Step(Screen.UNKNOWN, {"dismiss": BOX}, accept=("@dismiss",)), Step(Screen.LOBBY)]
    navigator, device = make(config, fake_clock, steps)
    assert navigator.reach(Screen.LOBBY, Deadline.after(fake_clock, 120)).screen == Screen.LOBBY
    steps = [Step(Screen.UNKNOWN, {}, accept=())]
    navigator, device = make(config, fake_clock, steps)
    with pytest.raises(Recoverable, match="unknown screen"):
        navigator.reach(Screen.LOBBY, Deadline.after(fake_clock, 600))


def test_a_screen_that_never_changes_is_recoverable(config: Config, fake_clock: FakeClock) -> None:
    steps = [Step(Screen.OPTIONS, accept=())]  # taps land but nothing happens
    navigator, device = make(config, fake_clock, steps)
    with pytest.raises(Recoverable, match="did not change"):
        navigator.reach(Screen.LOBBY, Deadline.after(fake_clock, 600))
    assert all(name == "continue_battling" for _, name in device.taps)


def test_the_centre_tap_is_allowed_only_on_disconnect_screens_and_never_near_forbidden_words(config: Config) -> None:
    policy = TapPolicy(config.screen)
    centre = config.screen.resolve("screen_centre", Viewport.of_frame(2340, 1080))
    cx, cy = centre.center
    for screen in (Screen.UNKNOWN, Screen.COMM_ERROR, Screen.DIALOG, Screen.TRANSITION):
        policy.check(screen, "screen_centre", cx, cy, Texts([]))
    for screen in (Screen.ACTION, Screen.MOVES, Screen.PARTY, Screen.LOBBY, Screen.OPTIONS, Screen.RANKED_FORMATS):
        with pytest.raises(SafetyStop):
            policy.check(screen, "screen_centre", cx, cy)
    for corner in ((centre.x1, centre.y1), (centre.x2, centre.y2)):
        assert policy.blocked("screen_centre", *corner, None) is None  # clear of every forbidden circle
    for word in ("Forfeit", "Quit Battling", "Edit Team", "Rating", "Shop", "Double Battle"):
        texts = Texts([TextBox(word, 0.6, (cx + 30, cy + 30, cx + 200, cy + 70))], 0.5)
        assert policy.blocked("screen_centre", cx, cy, texts, BLIND_FORBIDDEN_TEXT, BLIND_TEXT_MARGIN) is not None

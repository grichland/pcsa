import random

import pytest

from pcsa.common.clock import FakeClock
from pcsa.common.config import Config, Viewport
from pcsa.common.errors import SafetyStop
from pcsa.common.types import Screen, SemanticAction
from pcsa.live.device import Transport
from pcsa.live.taps import Executor, TapPolicy
from pcsa.phone.ocr import TextBox
from pcsa.phone.texts import Texts


class RecordingDevice:
    serial = "X"

    def __init__(self) -> None:
        self.taps: list[tuple[int, int]] = []

    def tap(self, x, y):
        self.taps.append((x, y))
        return Transport("tap", (x, y), 0.01)


@pytest.fixture
def executor(config: Config) -> Executor:
    return Executor(
        config.screen, Viewport.of_frame(2340, 1080), config.agent.taps, TapPolicy(config.screen), random.Random(1)
    )


def test_plans_land_inside_their_regions(executor: Executor, config: Config) -> None:
    plan = executor.plan_move(2, mega=True, focused_row=None)
    assert [t.name for t in plan] == ["fight", "mega", "move_rows2", "move_rows2_confirm"]
    for tap in plan:
        name = tap.name.removesuffix("_confirm").rstrip("0123456789")
        index = 2 if name == "move_rows" else None
        assert config.screen.resolve(name, Viewport.of_frame(2340, 1080), index).contains(tap.rx, tap.ry)


def test_focused_row_needs_one_tap(executor: Executor) -> None:
    assert [t.name for t in executor.plan_move(1, False, focused_row=1)] == ["fight", "move_rows1"]
    assert [t.name for t in executor.plan_switch(2)] == ["pokemon", "party_rows2", "party_rows2_open", "switch_in"]
    assert [t.name for t in executor.plan_forced(1, focused_row=1)] == ["party_rows1", "switch_in"]
    assert [t.name for t in executor.plan_team([2, 0, 1])] == ["team_rows2", "team_rows0", "team_rows1", "team_done"]


def test_switch_target_bounds(executor: Executor) -> None:
    with pytest.raises(ValueError):
        executor.plan_switch(0)  # the active Pokémon is row 0 on the party menu
    assert executor.plan_action(SemanticAction("switch", slot=2), [0, 2, 1], None)[1].name == "party_rows1"


def test_points_scale_through_the_viewport(config: Config) -> None:
    letterboxed = Executor(
        config.screen, Viewport(x=160, y=0, w=1600, h=738), config.agent.taps, TapPolicy(config.screen)
    )
    tap = letterboxed.point("fight")
    assert 160 + 0.87 * 1600 < tap.x < 160 + 0.92 * 1600
    assert 0.59 * 738 < tap.y < 0.69 * 738


def test_policy_refuses_taps_outside_the_screen_whitelist(executor: Executor, fake_clock: FakeClock) -> None:
    device = RecordingDevice()
    with pytest.raises(SafetyStop, match="not allowed"):
        executor.execute(executor.plan_move(0, False, None), Screen.LOBBY, device, fake_clock, 0.5)
    assert device.taps == []  # the whole plan is checked before any tap is sent


def test_policy_refuses_forbidden_geometry_and_text(executor: Executor) -> None:
    policy = executor.policy
    with pytest.raises(SafetyStop, match="forbidden area"):
        policy.check(Screen.LOBBY, "team_details_back", 302, 77)
    texts = Texts([TextBox("Forfeit", 0.95, (1500, 900, 1800, 960))])
    with pytest.raises(SafetyStop, match="Forfeit"):
        policy.check(Screen.LOBBY, "begin_matchmaking", 1600, 930, texts)
    policy.check(
        Screen.LOBBY,
        "begin_matchmaking",
        1600,
        930,
        Texts([TextBox("Begin Matchmaking", 0.95, (1500, 900, 1800, 960))]),
    )


def test_execute_sends_taps_with_spacing(executor: Executor, fake_clock: FakeClock) -> None:
    device = RecordingDevice()
    result = executor.execute(executor.plan_move(0, False, None), Screen.ACTION, device, fake_clock, 0.5)
    assert len(device.taps) == 3 and result.error is None
    assert fake_clock.sleeps == [0.5, 0.5]
    assert result.first_tap_at is not None and result.last_tap_at is not None
